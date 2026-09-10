use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, ExcitationAction, FluidConstraintKind, PhysicalSignal,
    SignalId, SignalOrdinate, SignalRepresentation, SignalValue,
};
use runmat_meshing_core::SolverMeshArtifact;

pub fn reynolds_number_for_velocity(
    domain: &runmat_analysis_core::CfdDomain,
    inlet_velocity_m_per_s: f64,
) -> f64 {
    domain.density.kilograms_per_cubic_meter * inlet_velocity_m_per_s.abs()
        / domain.dynamic_viscosity.pascal_seconds
}

pub(super) fn profile_scale(profile_scales: &[f64], step_index: usize) -> f64 {
    profile_scales
        .get(step_index)
        .copied()
        .filter(|scale| scale.is_finite() && *scale >= 0.0)
        .unwrap_or(1.0)
}

fn node_count_from_model(model: &AnalysisModel) -> usize {
    model.excitations.len().saturating_mul(3).clamp(3, 512)
}

#[derive(Clone, Debug)]
pub struct CfdDomainTopology {
    pub basis: CfdDomainTopologyBasis,
    pub geometry_source: CfdDomainGeometrySource,
    pub node_count: usize,
    pub control_volume_count: usize,
    pub control_volume_face_count: usize,
    pub control_volume_internal_face_count: usize,
    pub control_volume_boundary_face_count: usize,
    pub control_volume_connectivity_coverage_ratio: f64,
    pub domain_length_m: f64,
    pub hydraulic_diameter_m: f64,
    pub face_area_m2: f64,
    pub dx_m: f64,
    pub active_dimension_count: usize,
    pub element_geometry_node_count: usize,
    pub element_geometry_edge_count: usize,
    pub element_geometry_coverage_ratio: f64,
    pub element_topology_edge_nodes: Vec<[u32; 2]>,
    pub element_topology_element_edges: Vec<[u32; 3]>,
}

impl CfdDomainTopology {
    pub fn from_model(model: &AnalysisModel, solver_mesh: Option<&SolverMeshArtifact>) -> Self {
        solver_mesh
            .map(Self::from_solver_mesh)
            .unwrap_or_else(|| Self::implicit_channel(node_count_from_model(model)))
    }

    pub fn from_solver_mesh(mesh: &SolverMeshArtifact) -> Self {
        let topology = crate::topology::FlowTopologyProjection::derive(mesh);
        Self {
            basis: CfdDomainTopologyBasis::SolverMesh,
            geometry_source: CfdDomainGeometrySource::SolverMesh,
            node_count: topology.node_count,
            control_volume_count: topology.control_volume_count,
            control_volume_face_count: topology.face_count,
            control_volume_internal_face_count: topology.internal_face_count,
            control_volume_boundary_face_count: topology.boundary_face_count,
            control_volume_connectivity_coverage_ratio: 1.0,
            domain_length_m: topology.domain_length_m,
            hydraulic_diameter_m: topology.hydraulic_diameter_m,
            face_area_m2: topology.mean_boundary_face_area_m2,
            dx_m: topology.dx_m,
            active_dimension_count: topology.active_dimension_count,
            element_geometry_node_count: topology.node_count,
            element_geometry_edge_count: topology.edge_nodes.len(),
            element_geometry_coverage_ratio: 1.0,
            element_topology_edge_nodes: topology.edge_nodes,
            element_topology_element_edges: topology.boundary_face_edges,
        }
    }

    pub fn implicit_channel(node_count: usize) -> Self {
        let node_count = node_count.max(2);
        let control_volume_count = node_count.saturating_sub(1).max(1);
        Self {
            basis: CfdDomainTopologyBasis::ImplicitChannel,
            geometry_source: CfdDomainGeometrySource::ImplicitChannel,
            node_count,
            control_volume_count,
            control_volume_face_count: control_volume_count.saturating_add(1),
            control_volume_internal_face_count: control_volume_count.saturating_sub(1),
            control_volume_boundary_face_count: 2,
            control_volume_connectivity_coverage_ratio: 0.0,
            domain_length_m: 1.0,
            hydraulic_diameter_m: 1.0,
            face_area_m2: 1.0,
            dx_m: 1.0 / control_volume_count as f64,
            active_dimension_count: 1,
            element_geometry_node_count: 0,
            element_geometry_edge_count: 0,
            element_geometry_coverage_ratio: 0.0,
            element_topology_edge_nodes: Vec::new(),
            element_topology_element_edges: Vec::new(),
        }
    }

    pub fn face_area_m2(&self) -> f64 {
        self.face_area_m2.max(1.0e-12)
    }

    pub fn control_volume_volume_m3(&self) -> f64 {
        self.face_area_m2() * self.dx_m.max(1.0e-12)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CfdDomainTopologyBasis {
    ImplicitChannel,
    SolverMesh,
}

impl CfdDomainTopologyBasis {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ImplicitChannel => "implicit_channel",
            Self::SolverMesh => "solver_mesh",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CfdDomainGeometrySource {
    ImplicitChannel,
    SolverMesh,
}

impl CfdDomainGeometrySource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ImplicitChannel => "implicit_channel",
            Self::SolverMesh => "solver_mesh",
        }
    }
}
pub struct CfdBoundarySummary {
    pub inlet_boundary_count: usize,
    pub outlet_boundary_count: usize,
    pub no_slip_wall_boundary_count: usize,
    pub slip_wall_boundary_count: usize,
    pub symmetry_boundary_count: usize,
    pub authored_boundary_count: usize,
    pub boundary_coverage_ratio: f64,
    pub wall_boundary_coverage_ratio: f64,
    pub nominal_inlet_velocity_m_per_s: f64,
    pub outlet_pressure_pa: f64,
}

impl CfdBoundarySummary {
    pub fn implicit_channel(model: &AnalysisModel, node_count: usize) -> Self {
        Self {
            inlet_boundary_count: 1,
            outlet_boundary_count: 1,
            no_slip_wall_boundary_count: node_count.saturating_sub(1).max(1),
            slip_wall_boundary_count: 0,
            symmetry_boundary_count: 0,
            authored_boundary_count: 0,
            boundary_coverage_ratio: 1.0,
            wall_boundary_coverage_ratio: 1.0,
            nominal_inlet_velocity_m_per_s: nominal_inlet_velocity_m_per_s(model).unwrap_or(0.0),
            outlet_pressure_pa: 0.0,
        }
    }

    pub fn from_model(model: &AnalysisModel, node_count: usize) -> Self {
        let mut inlet_velocity_sum = 0.0_f64;
        let mut outlet_pressure_sum = 0.0_f64;
        let mut summary = Self {
            inlet_boundary_count: 0,
            outlet_boundary_count: 0,
            no_slip_wall_boundary_count: 0,
            slip_wall_boundary_count: 0,
            symmetry_boundary_count: 0,
            authored_boundary_count: 0,
            boundary_coverage_ratio: 0.0,
            wall_boundary_coverage_ratio: 0.0,
            nominal_inlet_velocity_m_per_s: nominal_inlet_velocity_m_per_s(model).unwrap_or(0.0),
            outlet_pressure_pa: 0.0,
        };

        for constraint in &model.constraints {
            let AnalysisConstraint::Fluid(constraint) = constraint else {
                continue;
            };
            match &constraint.kind {
                FluidConstraintKind::PrescribedVelocity { value, .. } => {
                    let Some(velocity) = value.study_constant(&model.signals) else {
                        continue;
                    };
                    summary.inlet_boundary_count += 1;
                    summary.authored_boundary_count += 1;
                    inlet_velocity_sum += velocity.meters_per_second;
                }
                FluidConstraintKind::PressureReference { value, .. } => {
                    let Some(pressure) = value.study_constant(&model.signals) else {
                        continue;
                    };
                    summary.outlet_boundary_count += 1;
                    summary.authored_boundary_count += 1;
                    outlet_pressure_sum += pressure.pascals;
                }
                FluidConstraintKind::NoSlipWall { .. } => {
                    summary.no_slip_wall_boundary_count += 1;
                    summary.authored_boundary_count += 1;
                }
                FluidConstraintKind::SlipWall { .. } => {
                    summary.slip_wall_boundary_count += 1;
                    summary.authored_boundary_count += 1;
                }
                FluidConstraintKind::Symmetry { .. } => {
                    summary.symmetry_boundary_count += 1;
                    summary.authored_boundary_count += 1;
                }
                _ => {}
            }
        }

        if summary.authored_boundary_count == 0 {
            return Self::implicit_channel(model, node_count);
        }

        if summary.inlet_boundary_count > 0 {
            summary.nominal_inlet_velocity_m_per_s =
                inlet_velocity_sum / summary.inlet_boundary_count as f64;
        }
        if summary.outlet_boundary_count > 0 {
            summary.outlet_pressure_pa = outlet_pressure_sum / summary.outlet_boundary_count as f64;
        }

        let wall_like_count = summary.no_slip_wall_boundary_count
            + summary.slip_wall_boundary_count
            + summary.symmetry_boundary_count;
        let required_groups_present = usize::from(summary.inlet_boundary_count > 0)
            + usize::from(summary.outlet_boundary_count > 0)
            + usize::from(wall_like_count > 0);
        summary.boundary_coverage_ratio = required_groups_present as f64 / 3.0;
        summary.wall_boundary_coverage_ratio = if wall_like_count > 0 { 1.0 } else { 0.0 };
        summary
    }

    pub fn wall_boundary_count(&self) -> usize {
        self.no_slip_wall_boundary_count + self.slip_wall_boundary_count
    }
}

pub fn validate_authored_boundary_conditions(model: &AnalysisModel) -> Result<(), String> {
    let mut inlet_boundary_count = 0usize;
    let mut outlet_boundary_count = 0usize;
    let mut wall_like_boundary_count = 0usize;
    let mut authored_boundary_count = 0usize;

    for constraint in &model.constraints {
        let AnalysisConstraint::Fluid(constraint) = constraint else {
            continue;
        };
        match &constraint.kind {
            FluidConstraintKind::PrescribedVelocity { value, .. } => {
                authored_boundary_count += 1;
                inlet_boundary_count += 1;
                let valid = value
                    .study_constant(&model.signals)
                    .is_some_and(|velocity| {
                        velocity.is_finite() && velocity.meters_per_second >= 0.0
                    });
                if !valid {
                    return Err(format!(
                        "cfd inlet boundary {} requires finite non-negative velocity_m_per_s",
                        constraint.constraint_id
                    ));
                }
            }
            FluidConstraintKind::PressureReference { value, .. } => {
                authored_boundary_count += 1;
                outlet_boundary_count += 1;
                if !value
                    .study_constant(&model.signals)
                    .is_some_and(|pressure| pressure.is_finite())
                {
                    return Err(format!(
                        "cfd outlet boundary {} requires finite pressure_pa",
                        constraint.constraint_id
                    ));
                }
            }
            FluidConstraintKind::NoSlipWall { .. }
            | FluidConstraintKind::SlipWall { .. }
            | FluidConstraintKind::Symmetry { .. } => {
                authored_boundary_count += 1;
                wall_like_boundary_count += 1;
            }
            _ => {}
        }
    }

    if authored_boundary_count == 0 {
        return Ok(());
    }
    if inlet_boundary_count == 0 || outlet_boundary_count == 0 || wall_like_boundary_count == 0 {
        return Err(format!(
            "authored cfd boundaries require at least one inlet, outlet, and wall/symmetry boundary; got inlet={} outlet={} wall_like={}",
            inlet_boundary_count, outlet_boundary_count, wall_like_boundary_count,
        ));
    }
    Ok(())
}

pub fn velocity_scales(model: &AnalysisModel) -> Vec<f64> {
    normalized_scales(inlet_velocity_values(model))
}

pub fn nominal_inlet_velocity_m_per_s(model: &AnalysisModel) -> Option<f64> {
    maximum_absolute(inlet_velocity_values(model))
}

fn inlet_velocity_values(model: &AnalysisModel) -> Vec<f64> {
    model
        .excitations
        .iter()
        .filter(|action| matches!(action, ExcitationAction::InletVelocity(_)))
        .filter_map(|action| signal(model, &action.excitation().signal_id))
        .filter_map(|signal| match signal {
            PhysicalSignal::Velocity(signal) => {
                scalar_or_vector_magnitudes(&signal.representation, |value| value.meters_per_second)
            }
            _ => None,
        })
        .flatten()
        .collect()
}

fn signal<'a>(model: &'a AnalysisModel, id: &SignalId) -> Option<&'a PhysicalSignal> {
    model.signals.iter().find(|signal| signal.signal_id() == id)
}

fn normalized_scales(values: Vec<f64>) -> Vec<f64> {
    let Some(nominal) = maximum_absolute(values.clone()) else {
        return Vec::new();
    };
    if nominal <= f64::EPSILON {
        return Vec::new();
    }
    values.into_iter().map(|value| value / nominal).collect()
}

fn maximum_absolute(values: Vec<f64>) -> Option<f64> {
    values
        .into_iter()
        .filter(|value| value.is_finite())
        .map(f64::abs)
        .reduce(f64::max)
}

fn scalar_or_vector_magnitudes<Q>(
    representation: &SignalRepresentation<Q>,
    component: impl Fn(&Q) -> f64,
) -> Option<Vec<f64>> {
    signal_values(representation)?
        .map(|value| match value {
            SignalValue::Scalar(value) => Some(component(value)),
            SignalValue::Vector3(values) => Some(
                values
                    .iter()
                    .map(|value| component(value).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            ),
            SignalValue::Tensor { .. } => None,
        })
        .collect()
}

fn signal_values<'a, Q>(
    representation: &'a SignalRepresentation<Q>,
) -> Option<Box<dyn Iterator<Item = &'a SignalValue<Q>> + 'a>> {
    match representation {
        SignalRepresentation::Constant(value) => Some(Box::new(std::iter::once(value))),
        SignalRepresentation::Tabulated(table) => Some(Box::new(
            table
                .knots
                .iter()
                .flat_map(knot_ordinates)
                .filter_map(|ordinate| match ordinate {
                    SignalOrdinate::Real(value) => Some(value),
                    SignalOrdinate::Complex { .. } => None,
                }),
        )),
        SignalRepresentation::Waveform(waveform) => {
            Some(Box::new(std::iter::once(&waveform.amplitude)))
        }
        SignalRepresentation::Artifact(_) | SignalRepresentation::Generator(_) => None,
    }
}

fn knot_ordinates<Q>(
    knot: &runmat_analysis_core::SignalKnot<Q>,
) -> impl Iterator<Item = &SignalOrdinate<Q>> {
    match &knot.value {
        runmat_analysis_core::SignalKnotValue::Continuous(value) => [Some(value), None],
        runmat_analysis_core::SignalKnotValue::Jump {
            left_limit,
            right_limit,
            ..
        } => [Some(left_limit), Some(right_limit)],
    }
    .into_iter()
    .flatten()
}
