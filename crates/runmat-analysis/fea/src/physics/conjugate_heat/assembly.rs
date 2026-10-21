//! Element, boundary and interface contributions of the conjugate energy equation.
use super::{
    velocity::{lift, CellVelocity},
    ConjugateHeatError as Error, ConjugateHeatInput,
};
use crate::{
    algebra::{
        ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix, SparsePattern,
    },
    excitation::{
        realize_step_excitations, sample_prescribed_heat_flux, sample_prescribed_temperature,
        RealizedExcitationKind, RealizedExcitationTarget,
    },
    topology::{AnalysisProjection, ProjectedBoundaryFace},
};
use runmat_analysis_core::{
    AnalysisConstraint, AnalysisInterfaceKind, BoundaryOperator, ThermalConstraintKind,
    ThermalInterfaceTransfer,
};
use runmat_meshing_core::BoundaryFaceRole;
use std::collections::BTreeMap;

/// Node degrees of freedom. A node on an interface with a temperature jump has a second,
/// side-owned degree of freedom used by the elements of that interface's second side.
#[derive(Debug)]
pub(super) struct Dofs {
    nodes: usize,
    /// (projection node, second-side dof), sorted by node.
    second: Vec<(usize, usize)>,
}

impl Dofs {
    pub(super) fn count(&self) -> usize {
        self.nodes + self.second.len()
    }
    fn second_of(&self, node: usize) -> Option<usize> {
        self.second
            .binary_search_by_key(&node, |(n, _)| *n)
            .ok()
            .map(|i| self.second[i].1)
    }
    pub(super) fn split(&self, values: &[f64]) -> Result<(Vec<f64>, Vec<(usize, f64)>), Error> {
        if values.len() != self.count() {
            return Err(Error::Invalid("temperature dof inventory"));
        }
        Ok((
            values[..self.nodes].to_vec(),
            self.second.iter().map(|&(n, d)| (n, values[d])).collect(),
        ))
    }
    pub(super) fn join(&self, nodes: &[f64], second: &[(usize, f64)]) -> Result<Vec<f64>, Error> {
        if nodes.len() != self.nodes
            || second.len() != self.second.len()
            || second
                .iter()
                .zip(&self.second)
                .any(|((a, _), (b, _))| a != b)
            || nodes
                .iter()
                .chain(second.iter().map(|(_, v)| v))
                .any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("retained temperature inventory"));
        }
        let mut values = nodes.to_vec();
        values.extend(second.iter().map(|(_, v)| *v));
        Ok(values)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum InterfaceLaw {
    Perfect,
    /// Interface conductance h = 1 / R, W/m^2/K.
    Conductance(f64),
}

#[derive(Debug)]
pub(super) struct InterfacePlan {
    pub interface_id: String,
    pub law: InterfaceLaw,
    /// (projection boundary face, first-side element, second-side element).
    pub faces: Vec<(usize, usize, usize)>,
}

#[derive(Debug)]
pub(super) struct ElementCoefficients {
    pub conductivity_w_per_mk: f64,
    pub volumetric_capacity_j_per_m3k: f64,
    pub second_side: bool,
}

/// Exterior boundary contribution attributed to the side of its adjacent element.
#[derive(Debug)]
pub(super) enum BoundaryTerm {
    /// Applied heat flux into the domain, W/m^2.
    HeatFlux { face: usize, value: f64 },
    /// Robin loss h (T − T_ambient).
    Convection {
        face: usize,
        coefficient: f64,
        ambient_k: f64,
    },
}

pub(super) struct AssembledEnergy {
    pub dofs: Dofs,
    pub elements: Vec<ElementCoefficients>,
    pub velocity: Vec<Option<CellVelocity>>,
    pub interfaces: Vec<InterfacePlan>,
    pub boundary: Vec<BoundaryTerm>,
    /// Volumetric generation W/m^3 per element (zero where none applies).
    pub generation_w_per_m3: Vec<f64>,
    pub dirichlet: Vec<Option<f64>>,
    pub full: RealSparseMatrix,
    /// Heat capacity of every dof (transient energy storage).
    pub capacity: RealSparseMatrix,
    pub rhs: Vec<f64>,
    pub constrained: RealSparseMatrix,
    pub constrained_rhs: Vec<f64>,
}

pub(super) fn dof(
    dofs: &Dofs,
    elements: &[ElementCoefficients],
    element: usize,
    node: usize,
) -> usize {
    if elements[element].second_side {
        if let Some(second) = dofs.second_of(node) {
            return second;
        }
    }
    node
}

/// Local element operator split by physical term, so evidence can attribute each part.
pub(super) struct ElementParts {
    pub dofs: usize,
    pub conduction: Vec<f64>,
    pub advection: Vec<f64>,
    pub stabilization: Vec<f64>,
    /// ρc ∫ (N_i + τ u·∇N_i) N_j: consistent heat capacity with its streamline test term.
    pub capacity: Vec<f64>,
    pub load: Vec<f64>,
}

impl ElementParts {
    pub(super) fn total(&self, i: usize, j: usize) -> f64 {
        let k = i * self.dofs + j;
        self.conduction[k] + self.advection[k] + self.stabilization[k]
    }
}

/// Conduction, conservative Galerkin advection and SUPG on one element.
pub(super) fn element_operator(
    projection: &AnalysisProjection,
    element: usize,
    coefficients: &ElementCoefficients,
    velocity: Option<&CellVelocity>,
    generation_w_per_m3: f64,
) -> Result<ElementParts, Error> {
    let e = &projection.elements[element];
    let n = e.node_indices.len();
    let mut parts = ElementParts {
        dofs: n,
        conduction: vec![0.0; n * n],
        advection: vec![0.0; n * n],
        stabilization: vec![0.0; n * n],
        capacity: vec![0.0; n * n],
        load: vec![0.0; n],
    };
    let k = coefficients.conductivity_w_per_mk;
    let rho_c = coefficients.volumetric_capacity_j_per_m3k;
    let tau = match velocity {
        Some(v) => supg_tau(projection, element, v, k, rho_c)?,
        None => 0.0,
    };
    projection
        .visit_element_quadrature(e, |point| {
            let grads = point.physical_shape_gradients();
            let shapes = point.shape_values();
            let w = point.volume_weight_m3;
            let u = velocity.map(|v| v.at(point.position_m));
            for i in 0..n {
                parts.load[i] += generation_w_per_m3 * shapes[i] * w;
                let streamline_i = u.map_or(0.0, |u| dot(u, grads[i]));
                for j in 0..n {
                    parts.capacity[i * n + j] +=
                        rho_c * (shapes[i] + tau * streamline_i) * shapes[j] * w;
                    parts.conduction[i * n + j] += k * dot(grads[i], grads[j]) * w;
                    if let Some(u) = u {
                        let advective_j = dot(u, grads[j]);
                        parts.advection[i * n + j] += rho_c * shapes[i] * advective_j * w;
                        parts.stabilization[i * n + j] +=
                            rho_c * tau * streamline_i * advective_j * w;
                    }
                }
            }
        })
        .map_err(|e| Error::InvalidDetail(e.to_string()))?;
    if parts
        .conduction
        .iter()
        .chain(&parts.advection)
        .chain(&parts.stabilization)
        .chain(&parts.capacity)
        .chain(&parts.load)
        .any(|v| !v.is_finite())
    {
        return Err(Error::Invalid("nonfinite conjugate element operator"));
    }
    Ok(parts)
}

/// Streamline-upwind parameter with the optimal one-dimensional Péclet weighting.
fn supg_tau(
    projection: &AnalysisProjection,
    element: usize,
    velocity: &CellVelocity,
    conductivity: f64,
    rho_c: f64,
) -> Result<f64, Error> {
    let e = &projection.elements[element];
    let u = velocity.at(e.centroid_m);
    let speed = norm(u);
    if speed == 0.0 {
        return Ok(0.0);
    }
    let corners: [[f64; 3]; 4] =
        std::array::from_fn(|i| projection.nodes[e.node_indices[i]].coordinates_m);
    let gradients = barycentric_gradients(corners)?;
    let projected = gradients.iter().map(|g| dot(u, *g).abs()).sum::<f64>();
    if projected <= 0.0 || !projected.is_finite() {
        return Ok(0.0);
    }
    let h = 2.0 * speed / projected;
    let diffusivity = conductivity / rho_c;
    let peclet = speed * h / (2.0 * diffusivity);
    let weight = if peclet < 1e-6 {
        peclet / 3.0
    } else {
        1.0 / peclet.tanh() - 1.0 / peclet
    };
    let tau = h / (2.0 * speed) * weight;
    if !tau.is_finite() || tau < 0.0 {
        return Err(Error::Invalid("streamline stabilization parameter"));
    }
    Ok(tau)
}

fn barycentric_gradients(x: [[f64; 3]; 4]) -> Result<[[f64; 3]; 4], Error> {
    let a = sub(x[1], x[0]);
    let b = sub(x[2], x[0]);
    let c = sub(x[3], x[0]);
    let det = dot(a, cross(b, c));
    if !det.is_finite() || det == 0.0 {
        return Err(Error::Invalid("degenerate fluid element"));
    }
    let g1 = scale(cross(b, c), 1.0 / det);
    let g2 = scale(cross(c, a), 1.0 / det);
    let g3 = scale(cross(a, b), 1.0 / det);
    let g0 = scale(add(add(g1, g2), g3), -1.0);
    Ok([g0, g1, g2, g3])
}

/// Boundary contribution on the face's own nodes (consistent mass for Robin terms).
pub(super) fn boundary_operator(
    projection: &AnalysisProjection,
    term: &BoundaryTerm,
) -> Result<(usize, Vec<f64>, Vec<f64>), Error> {
    let (face, coefficient, source) = match *term {
        BoundaryTerm::HeatFlux { face, value } => (face, 0.0, value),
        BoundaryTerm::Convection {
            face,
            coefficient,
            ambient_k,
        } => (face, coefficient, coefficient * ambient_k),
    };
    let f = &projection.boundary_faces[face];
    let n = f.node_indices.len();
    let mut matrix = vec![0.0; n * n];
    let mut load = vec![0.0; n];
    projection
        .visit_boundary_quadrature(f, |point| {
            let shapes = point.shape_values();
            for i in 0..n {
                load[i] += source * shapes[i] * point.area_weight_m2;
                for j in 0..n {
                    matrix[i * n + j] += coefficient * shapes[i] * shapes[j] * point.area_weight_m2;
                }
            }
        })
        .map_err(|e| Error::InvalidDetail(e.to_string()))?;
    Ok((face, matrix, load))
}

/// Consistent interface mass ∫ N_i N_j dA on one conformal face.
pub(super) fn interface_mass(
    projection: &AnalysisProjection,
    face: &ProjectedBoundaryFace,
) -> Result<Vec<f64>, Error> {
    let n = face.node_indices.len();
    let mut mass = vec![0.0; n * n];
    projection
        .visit_boundary_quadrature(face, |point| {
            let shapes = point.shape_values();
            for i in 0..n {
                for j in 0..n {
                    mass[i * n + j] += shapes[i] * shapes[j] * point.area_weight_m2;
                }
            }
        })
        .map_err(|e| Error::InvalidDetail(e.to_string()))?;
    Ok(mass)
}

pub(super) fn assemble(input: &ConjugateHeatInput<'_>) -> Result<AssembledEnergy, Error> {
    let p = input.projection;
    let velocity = lift(p, input.fluid_topology, input.face_volume_flux_m3_per_s)?;
    // Material ownership: fluid cells come from the cell role, solids from the solid role.
    let mut elements = Vec::with_capacity(p.elements.len());
    for (index, e) in p.elements.iter().enumerate() {
        let fluid = velocity[index].is_some();
        let owner = if fluid {
            input.properties.cell_material(e.stable_identity)
        } else {
            input.properties.solid_material(e.stable_identity)
        }
        .ok_or(Error::Invalid(
            "every element needs a thermal material of its fluid or solid role",
        ))?;
        let thermal = owner
            .material
            .require_thermal()
            .map_err(|e| Error::InvalidDetail(e.to_string()))?;
        let density = owner
            .material
            .require_mass_density()
            .map_err(|e| Error::InvalidDetail(e.to_string()))?;
        let capacity = density * thermal.specific_heat_j_per_kgk;
        if !thermal.conductivity_w_per_mk.is_finite()
            || thermal.conductivity_w_per_mk <= 0.0
            || !capacity.is_finite()
            || capacity <= 0.0
        {
            return Err(Error::Invalid(
                "positive finite conductivity and heat capacity",
            ));
        }
        elements.push(ElementCoefficients {
            conductivity_w_per_mk: thermal.conductivity_w_per_mk,
            volumetric_capacity_j_per_m3k: capacity,
            second_side: false,
        });
    }
    if input.properties.cell_materials().len() != input.fluid_topology.cells.len() {
        return Err(Error::Invalid(
            "every fluid-role element must belong to the solved fluid cell complex",
        ));
    }
    let interfaces = plan_interfaces(input, &velocity)?;
    // Second-side ownership: the elements on the second side of every jump interface.
    let mut jump_nodes = BTreeMap::<usize, ()>::new();
    for plan in &interfaces {
        if plan.law == InterfaceLaw::Perfect {
            continue;
        }
        for &(face, _, second) in &plan.faces {
            elements[second].second_side = true;
            for &node in &p.boundary_faces[face].node_indices {
                jump_nodes.insert(node, ());
            }
        }
    }
    // A second-side element that also touches a first-side jump node across another
    // interface would need two jump identities at one node; that topology is rejected.
    for plan in &interfaces {
        if plan.law == InterfaceLaw::Perfect {
            continue;
        }
        for &(_, first, _) in &plan.faces {
            if elements[first].second_side {
                return Err(Error::Unsupported(
                    "an element on both sides of temperature-jump interfaces",
                ));
            }
        }
    }
    let dofs = Dofs {
        nodes: p.nodes.len(),
        second: jump_nodes
            .keys()
            .enumerate()
            .map(|(i, &n)| (n, p.nodes.len() + i))
            .collect(),
    };
    let sample = input.sample;
    let excitations = realize_step_excitations(input.model, input.step, sample)
        .map_err(|e| Error::InvalidDetail(e.to_string()))?;
    let mut generation = vec![0.0; p.elements.len()];
    let mut boundary = Vec::new();
    for excitation in excitations.iter() {
        match &excitation.kind {
            RealizedExcitationKind::HeatGeneration(value) => {
                if value.imaginary != 0.0 {
                    return Err(Error::Unsupported("complex heat generation"));
                }
                let mut matched = 0;
                for (index, e) in p.elements.iter().enumerate() {
                    let selected = match &excitation.target {
                        RealizedExcitationTarget::WholeModel => true,
                        RealizedExcitationTarget::Region(target) => {
                            target.matches_persistent_entity(&e.region_id)
                        }
                        _ => return Err(Error::Unsupported("heat generation target")),
                    };
                    if selected {
                        generation[index] += value.real;
                        matched += 1;
                    }
                }
                if matched == 0 {
                    return Err(Error::Invalid("heat generation selects no element"));
                }
            }
            RealizedExcitationKind::HeatFlux(value) => {
                if value.imaginary != 0.0 {
                    return Err(Error::Unsupported("complex heat flux"));
                }
                let RealizedExcitationTarget::Region(target) = &excitation.target else {
                    return Err(Error::Unsupported("heat flux target"));
                };
                let faces = exterior_faces(p, target)?;
                for face in faces {
                    boundary.push(BoundaryTerm::HeatFlux {
                        face,
                        value: value.real,
                    });
                }
            }
            // Flow sources were realized by the finite-volume flow solve.
            RealizedExcitationKind::MassFlow(_)
            | RealizedExcitationKind::BodyForce(_)
            | RealizedExcitationKind::InletVelocity(_) => {}
            _ => return Err(Error::Unsupported("excitation in a conjugate heat study")),
        }
    }
    for b in &input.model.boundaries {
        match &b.operator {
            BoundaryOperator::ThermalHeatFlux { value } => {
                let sampled = sample_prescribed_heat_flux(value, &input.model.signals, sample)
                    .map_err(|e| Error::InvalidDetail(e.to_string()))?;
                if sampled.imaginary != 0.0 {
                    return Err(Error::Unsupported("complex heat flux"));
                }
                for face in exterior_faces(p, &b.target)? {
                    boundary.push(BoundaryTerm::HeatFlux {
                        face,
                        value: sampled.real,
                    });
                }
            }
            BoundaryOperator::ThermalConvection {
                ambient_temperature,
                coefficient,
            } => {
                let ambient = sample_prescribed_temperature(
                    ambient_temperature,
                    &input.model.signals,
                    sample,
                )
                .map_err(|e| Error::InvalidDetail(e.to_string()))?;
                if ambient.imaginary != 0.0 {
                    return Err(Error::Unsupported("complex ambient temperature"));
                }
                let h = coefficient.watts_per_square_meter_kelvin;
                if !h.is_finite() || h < 0.0 {
                    return Err(Error::Invalid("convection coefficient"));
                }
                for face in exterior_faces(p, &b.target)? {
                    boundary.push(BoundaryTerm::Convection {
                        face,
                        coefficient: h,
                        ambient_k: ambient.real,
                    });
                }
            }
            BoundaryOperator::FluidOutletPressure { .. } => {}
            _ => {
                return Err(Error::Unsupported(
                    "boundary operator in a conjugate heat study",
                ))
            }
        }
    }
    let mut dirichlet = vec![None; dofs.count()];
    for constraint in &input.model.constraints {
        match constraint {
            AnalysisConstraint::Thermal(c) => {
                let ThermalConstraintKind::Prescribed { target, value } = &c.kind else {
                    return Err(Error::Unsupported("thermal constraint kind"));
                };
                let t = sample_prescribed_temperature(value, &input.model.signals, sample)
                    .map_err(|e| Error::InvalidDetail(e.to_string()))?;
                if t.imaginary != 0.0 || !t.real.is_finite() {
                    return Err(Error::Invalid("prescribed temperature"));
                }
                for face in exterior_faces(p, target)? {
                    let f = &p.boundary_faces[face];
                    let element = f.adjacent_element_indices[0];
                    for &node in &f.node_indices {
                        let d = dof(&dofs, &elements, element, node);
                        if dirichlet[d].is_some_and(|old| old != t.real) {
                            return Err(Error::Invalid("conflicting prescribed temperatures"));
                        }
                        dirichlet[d] = Some(t.real);
                    }
                }
            }
            AnalysisConstraint::Fluid(_) => {}
            _ => return Err(Error::Unsupported("constraint in a conjugate heat study")),
        }
    }
    require_inflow_temperature(input, &dofs, &elements, &dirichlet)?;
    // Sparse support from element, interface and boundary dof sets.
    let mut rows = vec![Vec::<usize>::new(); dofs.count()];
    let element_dofs = |index: usize| -> Vec<usize> {
        p.elements[index]
            .node_indices
            .iter()
            .map(|&n| dof(&dofs, &elements, index, n))
            .collect()
    };
    for index in 0..p.elements.len() {
        let d = element_dofs(index);
        for &r in &d {
            rows[r].extend_from_slice(&d);
        }
    }
    for plan in &interfaces {
        if plan.law == InterfaceLaw::Perfect {
            continue;
        }
        for &(face, first, second) in &plan.faces {
            let nodes = &p.boundary_faces[face].node_indices;
            let mut d = nodes
                .iter()
                .map(|&n| dof(&dofs, &elements, first, n))
                .collect::<Vec<_>>();
            d.extend(nodes.iter().map(|&n| dof(&dofs, &elements, second, n)));
            for &r in &d {
                rows[r].extend_from_slice(&d);
            }
        }
    }
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    let mut columns = Vec::new();
    offsets.push(0u64);
    for row in &mut rows {
        row.sort_unstable();
        row.dedup();
        columns.extend(row.iter().map(|&c| c as u64));
        offsets.push(columns.len() as u64);
    }
    let pattern = SparsePattern::try_new(rows.len() as u64, rows.len() as u64, offsets, columns)
        .map_err(|e| Error::Operator(e.to_string()))?;
    let mut full = new_assembler(&pattern)?;
    let mut capacity = new_assembler(&pattern)?;
    let mut rhs = vec![0.0; dofs.count()];
    for index in 0..p.elements.len() {
        let parts = element_operator(
            p,
            index,
            &elements[index],
            velocity[index].as_ref(),
            generation[index],
        )?;
        let d = element_dofs(index);
        let n = d.len();
        for i in 0..n {
            rhs[d[i]] += parts.load[i];
            for j in 0..n {
                full.add(d[i], d[j], parts.total(i, j));
                capacity.add(d[i], d[j], parts.capacity[i * n + j]);
            }
        }
    }
    for term in &boundary {
        let (face, matrix, load) = boundary_operator(p, term)?;
        let f = &p.boundary_faces[face];
        let element = f.adjacent_element_indices[0];
        let d = f
            .node_indices
            .iter()
            .map(|&n| dof(&dofs, &elements, element, n))
            .collect::<Vec<_>>();
        let n = d.len();
        for i in 0..n {
            rhs[d[i]] += load[i];
            for j in 0..n {
                full.add(d[i], d[j], matrix[i * n + j]);
            }
        }
    }
    for plan in &interfaces {
        let InterfaceLaw::Conductance(h) = plan.law else {
            continue;
        };
        for &(face, first, second) in &plan.faces {
            let f = &p.boundary_faces[face];
            let mass = interface_mass(p, f)?;
            let n = f.node_indices.len();
            let a = f
                .node_indices
                .iter()
                .map(|&node| dof(&dofs, &elements, first, node))
                .collect::<Vec<_>>();
            let b = f
                .node_indices
                .iter()
                .map(|&node| dof(&dofs, &elements, second, node))
                .collect::<Vec<_>>();
            for i in 0..n {
                for j in 0..n {
                    let m = h * mass[i * n + j];
                    full.add(a[i], a[j], m);
                    full.add(a[i], b[j], -m);
                    full.add(b[i], b[j], m);
                    full.add(b[i], a[j], -m);
                }
            }
        }
    }
    let full = full.finish().map_err(|e| Error::Operator(e.to_string()))?;
    let capacity = capacity
        .finish()
        .map_err(|e| Error::Operator(e.to_string()))?;
    // Essential elimination: prescribed dofs become identity rows and their known columns
    // move to the right-hand side of the free equations.
    let mask = dirichlet.iter().map(Option::is_some).collect::<Vec<_>>();
    let constrained = full
        .with_identity_constraints(&mask)
        .map_err(|e| Error::Operator(e.to_string()))?;
    let prescribed = dirichlet
        .iter()
        .map(|v| v.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let lifted =
        crate::algebra::spmv(&full, &prescribed).map_err(|e| Error::Operator(e.to_string()))?;
    let constrained_rhs = (0..rhs.len())
        .map(|i| match dirichlet[i] {
            Some(value) => value,
            None => rhs[i] - lifted[i],
        })
        .collect::<Vec<_>>();
    Ok(AssembledEnergy {
        dofs,
        elements,
        velocity,
        interfaces,
        boundary,
        generation_w_per_m3: generation,
        dirichlet,
        full,
        capacity,
        rhs,
        constrained,
        constrained_rhs,
    })
}

fn new_assembler(pattern: &SparsePattern) -> Result<RealSparseAssembler, Error> {
    let mut a = RealSparseAssembler::with_pattern(
        pattern
            .try_clone()
            .map_err(|e| Error::Operator(e.to_string()))?,
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
    )
    .map_err(|e| Error::Operator(e.to_string()))?;
    a.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Thermal,
        contribution_id: ContributionId("conjugate-energy-conduction-advection".into()),
    });
    Ok(a)
}

/// Exterior faces of an authored boundary target. Interface faces carry no boundary law.
fn exterior_faces(
    p: &AnalysisProjection,
    target: &runmat_analysis_core::RegionTarget,
) -> Result<Vec<usize>, Error> {
    let faces = p
        .boundary_faces
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            f.provenance
                .iter()
                .any(|entity| target.matches_persistent_entity(entity))
        })
        .collect::<Vec<_>>();
    if faces.is_empty() {
        return Err(Error::InvalidDetail(format!(
            "thermal boundary target `{}` selects no face",
            target.region_id
        )));
    }
    if faces
        .iter()
        .any(|(_, f)| f.role != BoundaryFaceRole::Exterior)
    {
        return Err(Error::InvalidDetail(format!(
            "thermal boundary target `{}` selects an interface face; interfaces carry their own law",
            target.region_id
        )));
    }
    Ok(faces.into_iter().map(|(i, _)| i).collect())
}

/// Every conformal fluid–solid face belongs to exactly one declared conjugate interface.
fn plan_interfaces(
    input: &ConjugateHeatInput<'_>,
    velocity: &[Option<CellVelocity>],
) -> Result<Vec<InterfacePlan>, Error> {
    let p = input.projection;
    let mut claimed = vec![false; p.boundary_faces.len()];
    let mut plans = Vec::new();
    for interface in &input.model.interfaces {
        let AnalysisInterfaceKind::ConjugateHeatTransfer(model) = &interface.kind else {
            return Err(Error::Unsupported(
                "non-thermal interface in a conjugate heat study",
            ));
        };
        let law = match model.transfer {
            ThermalInterfaceTransfer::Perfect => InterfaceLaw::Perfect,
            ThermalInterfaceTransfer::Conductance(h) => {
                InterfaceLaw::Conductance(h.watts_per_square_meter_kelvin)
            }
            ThermalInterfaceTransfer::Resistance(r) => {
                InterfaceLaw::Conductance(1.0 / r.square_meter_kelvin_per_watt)
            }
        };
        if let InterfaceLaw::Conductance(h) = law {
            if !h.is_finite() || h <= 0.0 {
                return Err(Error::Invalid(
                    "interface conductance must be positive and finite",
                ));
            }
        }
        let mut faces = Vec::new();
        for (index, f) in p.boundary_faces.iter().enumerate() {
            let selected = f.provenance.iter().any(|entity| {
                interface.primary.matches_persistent_entity(entity)
                    || interface.secondary.matches_persistent_entity(entity)
            });
            if !selected {
                continue;
            }
            let [a, b] = f.adjacent_element_indices[..] else {
                return Err(Error::InvalidDetail(format!(
                    "interface `{}` selects a face that is not a conformal interface between two bodies",
                    interface.interface_id
                )));
            };
            if f.role != BoundaryFaceRole::ConformalInterface {
                return Err(Error::Invalid("interface face role"));
            }
            if std::mem::replace(&mut claimed[index], true) {
                return Err(Error::InvalidDetail(format!(
                    "interface `{}` overlaps another declared interface",
                    interface.interface_id
                )));
            }
            // The fluid side is the second side; between two solids, the greater region.
            let (first, second) = match (velocity[a].is_some(), velocity[b].is_some()) {
                (false, true) => (a, b),
                (true, false) => (b, a),
                (false, false) if p.elements[a].region_id <= p.elements[b].region_id => (a, b),
                (false, false) => (b, a),
                (true, true) => {
                    return Err(Error::Unsupported("thermal interface inside the fluid"))
                }
            };
            faces.push((index, first, second));
        }
        if faces.is_empty() {
            return Err(Error::InvalidDetail(format!(
                "interface `{}` selects no conformal interface face",
                interface.interface_id
            )));
        }
        plans.push(InterfacePlan {
            interface_id: interface.interface_id.clone(),
            law,
            faces,
        });
    }
    for (index, f) in p.boundary_faces.iter().enumerate() {
        if f.role != BoundaryFaceRole::ConformalInterface || claimed[index] {
            continue;
        }
        let fluid = f
            .adjacent_element_indices
            .iter()
            .filter(|&&e| velocity[e].is_some())
            .count();
        if fluid == 1 {
            return Err(Error::Invalid(
                "a fluid–solid face has no declared conjugate heat transfer interface",
            ));
        }
    }
    Ok(plans)
}

/// Flow entering the fluid must carry an authored inlet temperature.
fn require_inflow_temperature(
    input: &ConjugateHeatInput<'_>,
    dofs: &Dofs,
    elements: &[ElementCoefficients],
    dirichlet: &[Option<f64>],
) -> Result<(), Error> {
    let p = input.projection;
    for (i, face) in input.fluid_topology.faces.iter().enumerate() {
        if face.neighbor.is_some() || input.face_volume_flux_m3_per_s[i] >= 0.0 {
            continue;
        }
        let Some(source) = face.boundary_source else {
            return Err(Error::Invalid("fluid inflow face without boundary source"));
        };
        let f = &p.boundary_faces[source.projection_boundary_index];
        let element = input.fluid_topology.cells[face.owner].source_element_index;
        if f.node_indices
            .iter()
            .any(|&n| dirichlet[dof(dofs, elements, element, n)].is_none())
        {
            return Err(Error::Invalid(
                "fluid inflow boundary needs a prescribed inlet temperature",
            ));
        }
    }
    Ok(())
}

pub(super) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
