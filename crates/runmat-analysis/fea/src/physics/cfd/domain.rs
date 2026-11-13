use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, AnalysisProcedure, ExcitationAction, FluidConstraintKind,
    PhysicsFamily,
};
use runmat_meshing_core::SolverMeshArtifact;

#[cfg(test)]
mod tests;

pub fn reynolds_number_for_velocity(
    domain: &runmat_analysis_core::CfdDomain,
    inlet_velocity_m_per_s: f64,
) -> f64 {
    domain.density.canonical_value() * inlet_velocity_m_per_s.abs()
        / domain.dynamic_viscosity.canonical_value()
}

pub(super) fn profile_scale(profile_scales: &[f64], step_index: usize) -> f64 {
    profile_scales
        .get(step_index)
        .copied()
        .filter(|scale| scale.is_finite() && *scale >= 0.0)
        .unwrap_or(1.0)
}

#[derive(Clone, Debug, PartialEq)]
pub struct CfdVelocitySchedule {
    pub nominal_velocity_m_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Velocity, f64>,
    pub profile_scales: Vec<f64>,
    pub has_inlet_excitation: bool,
    pub nominal_time_step_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Time, f64>,
}

pub fn realize_velocity_schedule(
    model: &AnalysisModel,
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    step_count: usize,
) -> Result<CfdVelocitySchedule, String> {
    validate_authored_boundary_conditions(model)?;
    if model.excitations.iter().any(|action| {
        matches!(
            action,
            ExcitationAction::InletVelocity(_) | ExcitationAction::WallMotion(_)
        )
    }) {
        return Err("reduced CFD cannot realize directional, framed, per-region velocity excitations; canonical finite-volume execution is required".to_string());
    }
    let matching_steps = model
        .sequence
        .steps
        .iter()
        .filter(|step| step.family == family && step.procedure == procedure)
        .collect::<Vec<_>>();
    let [step] = matching_steps.as_slice() else {
        return Err(format!(
            "CFD excitation realization requires exactly one matching procedure step, found {}",
            matching_steps.len()
        ));
    };
    let (sample_points, nominal_time_step_s) = if procedure == AnalysisProcedure::TransientResponse
    {
        let grid = crate::excitation::ProcedureTimeGrid::from_sample_count(step, step_count)?;
        (
            grid.points_s()
                .iter()
                .copied()
                .map(crate::excitation::ExcitationSamplePoint::time)
                .collect::<Vec<_>>(),
            grid.nominal_interval_s(),
        )
    } else {
        (vec![crate::excitation::ExcitationSamplePoint::Static], 0.0)
    };
    let sample_count = sample_points.len();
    Ok(CfdVelocitySchedule {
        nominal_velocity_m_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Velocity,
            _,
        >::new(0.0),
        profile_scales: vec![1.0; sample_count],
        has_inlet_excitation: false,
        nominal_time_step_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Time,
            _,
        >::new(nominal_time_step_s),
    })
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
    pub domain_length_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Length, f64>,
    pub hydraulic_diameter_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Length, f64>,
    pub face_area_m2: runmat_types::physical::Quantity<runmat_types::physical::kind::Area, f64>,
    pub dx_m: runmat_types::physical::Quantity<runmat_types::physical::kind::Length, f64>,
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
            domain_length_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(topology.domain_length_m.canonical_value()),
            hydraulic_diameter_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(
                topology.hydraulic_diameter_m.canonical_value()
            ),
            face_area_m2:
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Area, _>::new(
                    topology.mean_boundary_face_area_m2,
                ),
            dx_m: runmat_types::physical::Quantity::<runmat_types::physical::kind::Length, _>::new(
                topology.dx_m.canonical_value(),
            ),
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
            domain_length_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(1.0),
            hydraulic_diameter_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(1.0),
            face_area_m2:
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Area, _>::new(1.0),
            dx_m: runmat_types::physical::Quantity::<runmat_types::physical::kind::Length, _>::new(
                1.0 / control_volume_count as f64,
            ),
            active_dimension_count: 1,
            element_geometry_node_count: 0,
            element_geometry_edge_count: 0,
            element_geometry_coverage_ratio: 0.0,
            element_topology_edge_nodes: Vec::new(),
            element_topology_element_edges: Vec::new(),
        }
    }

    pub fn face_area_m2(&self) -> f64 {
        self.face_area_m2.canonical_value().max(1.0e-12)
    }

    pub fn control_volume_volume_m3(&self) -> f64 {
        self.face_area_m2() * self.dx_m.canonical_value().max(1.0e-12)
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
pub(super) struct CfdBoundarySummary {
    pub nominal_inlet_velocity_m_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Velocity, f64>,
    pub outlet_pressure_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, f64>,
}
impl CfdBoundarySummary {
    pub fn implicit_channel() -> Self {
        Self {
            nominal_inlet_velocity_m_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Velocity,
                _,
            >::new(0.0),
            outlet_pressure_pa: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Pressure,
                _,
            >::new(0.0),
        }
    }
}

/// Admission for the retired reduced channel adapter, not canonical vector CFD.
pub fn validate_authored_boundary_conditions(model: &AnalysisModel) -> Result<(), String> {
    if model.boundaries.iter().any(|boundary| {
        matches!(
            boundary.operator,
            runmat_analysis_core::BoundaryOperator::FluidOutletPressure { .. }
        )
    }) {
        return Err("reduced CFD cannot realize exact physical outlet pressure boundaries; canonical finite-volume execution is required".to_string());
    }
    for constraint in &model.constraints {
        let AnalysisConstraint::Fluid(constraint) = constraint else {
            continue;
        };
        match &constraint.kind {
            FluidConstraintKind::PrescribedVelocity { .. }
            | FluidConstraintKind::MovingWall { .. } => return Err(format!(
                "reduced CFD cannot realize vector velocity constraint {}; canonical finite-volume execution is required",
                constraint.constraint_id,
            )),
            FluidConstraintKind::PressureReference { .. } => return Err(format!(
                "reduced CFD cannot realize pressure gauge {}; a pressure reference is not a physical outlet",
                constraint.constraint_id,
            )),
            _ => return Err(format!(
                "reduced CFD cannot realize exact fluid boundary constraint {}; canonical finite-volume execution is required",
                constraint.constraint_id,
            )),
        }
    }

    Ok(())
}
