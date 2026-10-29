//! Small-strain linear elastic structure on the solid-role elements of a coupled mesh.
use super::FluidStructureError as Error;
use crate::{
    algebra::{
        spmv, ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties,
        OperatorRole, RealSparseAssembler, RealSparseMatrix, SparsePattern,
    },
    assembly::elements::solid::{
        global_stiffness_matrix, tetrahedron10_consistent_mass_matrix,
        tetrahedron10_global_stiffness_matrix, tetrahedron4_consistent_mass_matrix, SolidMaterial,
        Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry,
    },
    properties::CompiledPropertyMap,
    topology::AnalysisProjection,
};
use runmat_analysis_core::{AnalysisConstraint, AnalysisModel, StructuralConstraintKind};
use runmat_meshing_core::{BoundaryFaceRole, ElementOrder};

pub(super) struct ElasticStructure {
    /// Projection node → structural node ordinal (three dofs each), `None` off the solid.
    pub node_ordinal: Vec<Option<usize>>,
    pub nodes: Vec<usize>,
    pub elements: Vec<usize>,
    pub stiffness: RealSparseMatrix,
    /// Consistent mass, for structural dynamics.
    pub mass: RealSparseMatrix,
    pub constrained: Vec<bool>,
    pub reduced: super::SpdSystem,
}

impl ElasticStructure {
    pub(super) fn dofs(&self) -> usize {
        3 * self.nodes.len()
    }

    pub(super) fn assemble(
        model: &AnalysisModel,
        projection: &AnalysisProjection,
        properties: &CompiledPropertyMap,
    ) -> Result<Self, Error> {
        let mut node_ordinal = vec![None; projection.nodes.len()];
        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        let mut materials = Vec::new();
        for (index, element) in projection.elements.iter().enumerate() {
            let Some(owner) = properties.solid_material(element.stable_identity) else {
                continue;
            };
            let mechanical = owner
                .material
                .require_mechanical()
                .map_err(|e| Error::InvalidDetail(e.to_string()))?;
            let density = owner
                .material
                .require_mass_density()
                .map_err(|e| Error::InvalidDetail(e.to_string()))?;
            materials.push(SolidMaterial {
                youngs_modulus_pa: mechanical.youngs_modulus_pa,
                poisson_ratio: mechanical.poisson_ratio,
                density_kg_per_m3: density,
            });
            elements.push(index);
            for &node in &element.node_indices {
                if node_ordinal[node].is_none() {
                    node_ordinal[node] = Some(nodes.len());
                    nodes.push(node);
                }
            }
        }
        if elements.is_empty() {
            return Err(Error::Invalid(
                "fluid–structure study has no solid-role element",
            ));
        }
        let n = 3 * nodes.len();
        let element_dofs = |e: usize| -> Vec<usize> {
            projection.elements[e]
                .node_indices
                .iter()
                .flat_map(|&node| {
                    let base = 3 * node_ordinal[node].expect("solid node");
                    [base, base + 1, base + 2]
                })
                .collect()
        };
        let mut rows = vec![Vec::<usize>::new(); n];
        for &e in &elements {
            let d = element_dofs(e);
            for &r in &d {
                rows[r].extend_from_slice(&d);
            }
        }
        let mut offsets = vec![0u64];
        let mut columns = Vec::new();
        for row in &mut rows {
            row.sort_unstable();
            row.dedup();
            columns.extend(row.iter().map(|&c| c as u64));
            offsets.push(columns.len() as u64);
        }
        let pattern = SparsePattern::try_new(n as u64, n as u64, offsets, columns)
            .map_err(|e| Error::Operator(e.to_string()))?;
        let mut mass_assembler = RealSparseAssembler::with_pattern(
            pattern
                .try_clone()
                .map_err(|e| Error::Operator(e.to_string()))?,
            OperatorRole::Mass,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .map_err(|e| Error::Operator(e.to_string()))?;
        mass_assembler.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Structural,
            contribution_id: ContributionId("fluid-structure-consistent-mass".into()),
        });
        let mut assembler = RealSparseAssembler::with_pattern(
            pattern,
            OperatorRole::EffectiveTangent,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|e| Error::Operator(e.to_string()))?;
        assembler.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Structural,
            contribution_id: ContributionId("fluid-structure-small-strain-solid".into()),
        });
        for (&e, &material) in elements.iter().zip(&materials) {
            let element = &projection.elements[e];
            let coordinates = element
                .node_indices
                .iter()
                .map(|&node| projection.nodes[node].coordinates_m.canonical_value())
                .collect::<Vec<_>>();
            let rows = |m: &[[f64; 12]; 12]| m.iter().map(|r| r.to_vec()).collect::<Vec<_>>();
            let rows30 = |m: &[[f64; 30]; 30]| m.iter().map(|r| r.to_vec()).collect::<Vec<_>>();
            let (local, local_mass): (Vec<Vec<f64>>, Vec<Vec<f64>>) = match element.order {
                ElementOrder::Tet4 => {
                    let geometry = Tetrahedron4ElementGeometry {
                        nodes_m: std::array::from_fn(|i| coordinates[i]),
                    };
                    (
                        rows(
                            &global_stiffness_matrix(material, geometry)
                                .map_err(|e| Error::InvalidDetail(e.to_string()))?,
                        ),
                        rows(
                            &tetrahedron4_consistent_mass_matrix(material, geometry)
                                .map_err(|e| Error::InvalidDetail(e.to_string()))?,
                        ),
                    )
                }
                ElementOrder::Tet10 => {
                    let geometry = Tetrahedron10ElementGeometry {
                        nodes_m: std::array::from_fn(|i| coordinates[i]),
                    };
                    (
                        rows30(
                            &tetrahedron10_global_stiffness_matrix(material, geometry)
                                .map_err(|e| Error::InvalidDetail(e.to_string()))?,
                        ),
                        rows30(
                            &tetrahedron10_consistent_mass_matrix(material, geometry)
                                .map_err(|e| Error::InvalidDetail(e.to_string()))?,
                        ),
                    )
                }
            };
            let d = element_dofs(e);
            for (i, (row, mass_row)) in local.iter().zip(&local_mass).enumerate() {
                for (j, (value, mass)) in row.iter().zip(mass_row).enumerate() {
                    assembler.add(d[i], d[j], *value);
                    mass_assembler.add(d[i], d[j], *mass);
                }
            }
        }
        let stiffness = assembler
            .finish()
            .map_err(|e| Error::Operator(e.to_string()))?;
        let mass = mass_assembler
            .finish()
            .map_err(|e| Error::Operator(e.to_string()))?;
        let mut constrained = vec![false; n];
        let mut anchored = false;
        for constraint in &model.constraints {
            let AnalysisConstraint::Structural(c) = constraint else {
                continue;
            };
            match &c.kind {
                StructuralConstraintKind::Fixed { target } => {
                    for face in exterior_faces(projection, target)? {
                        for &node in &projection.boundary_faces[face].node_indices {
                            let base = 3 * node_ordinal[node]
                                .ok_or(Error::Invalid("structural support off the solid"))?;
                            constrained[base..base + 3].fill(true);
                        }
                    }
                    anchored = true;
                }
                StructuralConstraintKind::Symmetry { target } => {
                    for face in exterior_faces(projection, target)? {
                        let f = &projection.boundary_faces[face];
                        let normal = f
                            .oriented_area_vector_m2
                            .map(|v| v / f.area_m2.canonical_value());
                        let axis = (0..3)
                            .find(|&k| (normal[k].abs() - 1.0).abs() < 1e-9)
                            .ok_or(Error::Unsupported(
                                "structural symmetry planes must be normal to a global axis",
                            ))?;
                        for &node in &f.node_indices {
                            let base = 3 * node_ordinal[node]
                                .ok_or(Error::Invalid("structural support off the solid"))?;
                            constrained[base + axis] = true;
                        }
                    }
                }
                _ => {
                    return Err(Error::Unsupported(
                        "structural constraint in a fluid–structure study",
                    ))
                }
            }
        }
        if !anchored {
            return Err(Error::Invalid(
                "the structure needs a fixed support to remove rigid motion",
            ));
        }
        let reduced = stiffness
            .with_identity_constraints(&constrained)
            .map_err(|e| Error::Operator(e.to_string()))?;
        Ok(Self {
            node_ordinal,
            nodes,
            elements,
            stiffness,
            mass,
            constrained,
            reduced: super::SpdSystem::new(reduced),
        })
    }

    /// Displacement of the supported structure under nodal forces (zero at supports).
    pub(super) fn solve(
        &self,
        load: &[f64],
        policy: super::LinearPolicy,
    ) -> Result<(Vec<f64>, u64), Error> {
        let rhs = load
            .iter()
            .zip(&self.constrained)
            .map(|(f, &c)| if c { 0.0 } else { *f })
            .collect::<Vec<_>>();
        self.reduced.solve(&rhs, policy)
    }

    /// ||K u − f|| / ||f|| over the unsupported dofs: equilibrium of a retained displacement.
    pub(super) fn equilibrium_residual(&self, u: &[f64], load: &[f64]) -> Result<f64, Error> {
        let action = spmv(&self.stiffness, u).map_err(|e| Error::Operator(e.to_string()))?;
        let mut numerator = 0.0f64;
        let mut denominator = 0.0f64;
        for i in 0..u.len() {
            if self.constrained[i] {
                if u[i] != 0.0 {
                    return Err(Error::Invalid("retained displacement moves a support"));
                }
                continue;
            }
            numerator = numerator.hypot(action[i] - load[i]);
            denominator = denominator.hypot(load[i]);
        }
        Ok(super::relative(numerator, denominator))
    }

    /// Average-acceleration Newmark operator K + 4 M / Δt², with the supports eliminated.
    pub(super) fn newmark_operator(&self, dt: f64) -> Result<super::SpdSystem, Error> {
        let effective = RealSparseMatrix::scaled_sum(
            &[(1.0, &self.stiffness), (4.0 / (dt * dt), &self.mass)],
            OperatorRole::EffectiveTangent,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .map_err(|e| Error::Operator(e.to_string()))?;
        effective
            .with_identity_constraints(&self.constrained)
            .map(super::SpdSystem::new)
            .map_err(|e| Error::Operator(e.to_string()))
    }

    /// One average-acceleration Newmark step: the displacement at the arrival time under
    /// `load`, from the departure state (u, v, a). Returns (u, v, a) at arrival.
    pub(super) fn newmark_step(
        &self,
        operator: &super::SpdSystem,
        dt: f64,
        departure: &NewmarkState,
        load: &[f64],
        policy: super::LinearPolicy,
    ) -> Result<NewmarkState, Error> {
        let predictor = departure
            .displacement
            .iter()
            .zip(&departure.velocity)
            .zip(&departure.acceleration)
            .map(|((u, v), a)| 4.0 / (dt * dt) * u + 4.0 / dt * v + a)
            .collect::<Vec<_>>();
        let inertia = spmv(&self.mass, &predictor).map_err(|e| Error::Operator(e.to_string()))?;
        let rhs = load
            .iter()
            .zip(&inertia)
            .zip(&self.constrained)
            .map(|((f, m), &c)| if c { 0.0 } else { f + m })
            .collect::<Vec<_>>();
        let (displacement, _) = operator.solve(&rhs, policy)?;
        Ok(departure.advance(dt, displacement))
    }

    /// Kinetic energy ½ vᵀ M v.
    pub(super) fn kinetic_energy(&self, v: &[f64]) -> Result<f64, Error> {
        let action = spmv(&self.mass, v).map_err(|e| Error::Operator(e.to_string()))?;
        Ok(0.5 * v.iter().zip(&action).map(|(a, b)| a * b).sum::<f64>())
    }

    /// ||M a + K u − f|| / ||f|| over the unsupported dofs: dynamic equilibrium at arrival.
    pub(super) fn dynamic_residual(
        &self,
        state: &NewmarkState,
        load: &[f64],
    ) -> Result<f64, Error> {
        let stiff = spmv(&self.stiffness, &state.displacement)
            .map_err(|e| Error::Operator(e.to_string()))?;
        let inertia =
            spmv(&self.mass, &state.acceleration).map_err(|e| Error::Operator(e.to_string()))?;
        let mut numerator = 0.0f64;
        let mut denominator = 0.0f64;
        for i in 0..load.len() {
            if self.constrained[i] {
                continue;
            }
            numerator = numerator.hypot(stiff[i] + inertia[i] - load[i]);
            denominator = denominator.hypot(load[i]).hypot(inertia[i]);
        }
        Ok(super::relative(numerator, denominator))
    }

    /// Strain energy ½ uᵀ K u.
    pub(super) fn strain_energy(&self, u: &[f64]) -> Result<f64, Error> {
        let action = spmv(&self.stiffness, u).map_err(|e| Error::Operator(e.to_string()))?;
        Ok(0.5 * u.iter().zip(&action).map(|(a, b)| a * b).sum::<f64>())
    }
}

fn exterior_faces(
    projection: &AnalysisProjection,
    target: &runmat_analysis_core::RegionTarget,
) -> Result<Vec<usize>, Error> {
    let faces = projection
        .boundary_faces
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            f.role == BoundaryFaceRole::Exterior
                && f.provenance
                    .iter()
                    .any(|entity| target.matches_persistent_entity(entity))
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if faces.is_empty() {
        return Err(Error::InvalidDetail(format!(
            "structural target `{}` selects no exterior face",
            target.region_id
        )));
    }
    Ok(faces)
}

/// Structural displacement, velocity and acceleration at one instant, per structural dof.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewmarkState {
    pub displacement: Vec<f64>,
    pub velocity: Vec<f64>,
    pub acceleration: Vec<f64>,
}

impl NewmarkState {
    pub(super) fn rest(dofs: usize) -> Self {
        Self {
            displacement: vec![0.0; dofs],
            velocity: vec![0.0; dofs],
            acceleration: vec![0.0; dofs],
        }
    }
    /// Average-acceleration kinematics for a given arrival displacement.
    pub(super) fn advance(&self, dt: f64, displacement: Vec<f64>) -> Self {
        let acceleration = displacement
            .iter()
            .zip(&self.displacement)
            .zip(&self.velocity)
            .zip(&self.acceleration)
            .map(|(((u1, u0), v0), a0)| 4.0 / (dt * dt) * (u1 - u0) - 4.0 / dt * v0 - a0)
            .collect::<Vec<_>>();
        let velocity = self
            .velocity
            .iter()
            .zip(&self.acceleration)
            .zip(&acceleration)
            .map(|((v0, a0), a1)| v0 + 0.5 * dt * (a0 + a1))
            .collect();
        Self {
            displacement,
            velocity,
            acceleration,
        }
    }
}
