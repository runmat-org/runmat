use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, AnalysisProcedure, ExcitationAction, FluidConstraintKind,
    PhysicsFamily,
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

#[derive(Clone, Debug, PartialEq)]
pub struct CfdVelocitySchedule {
    pub nominal_velocity_m_per_s: f64,
    pub profile_scales: Vec<f64>,
    pub has_inlet_excitation: bool,
    pub nominal_time_step_s: f64,
}

pub fn realize_velocity_schedule(
    model: &AnalysisModel,
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    step_count: usize,
) -> Result<CfdVelocitySchedule, String> {
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
    let has_inlet_excitation = model
        .excitations
        .iter()
        .any(|action| matches!(action, ExcitationAction::InletVelocity(_)));
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
    let mut velocities = Vec::with_capacity(sample_count);
    for sample in sample_points {
        let realized = crate::excitation::realize_step_excitations(model, step, sample)
            .map_err(|error| format!("CFD inlet excitation realization failed: {error}"))?;
        let velocity = realized
            .iter()
            .filter_map(|excitation| match &excitation.kind {
                crate::excitation::RealizedExcitationKind::InletVelocity(value) => Some(value),
                _ => None,
            })
            .map(|value| {
                if value.imaginary.iter().any(|component| *component != 0.0) {
                    return Err(
                        "CFD inlet velocity must be real in steady and transient solves"
                            .to_string(),
                    );
                }
                Ok(value
                    .real
                    .iter()
                    .map(|component| component * component)
                    .sum::<f64>()
                    .sqrt())
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .fold(0.0_f64, f64::max);
        velocities.push(velocity);
    }
    if !has_inlet_excitation {
        let constrained = CfdBoundarySummary::from_model(model, 2).nominal_inlet_velocity_m_per_s;
        velocities.fill(constrained);
    }
    let nominal_velocity_m_per_s = velocities.iter().copied().fold(0.0_f64, f64::max);
    let profile_scales = if nominal_velocity_m_per_s > f64::EPSILON {
        velocities
            .into_iter()
            .map(|velocity| velocity / nominal_velocity_m_per_s)
            .collect()
    } else {
        vec![1.0; sample_count]
    };
    Ok(CfdVelocitySchedule {
        nominal_velocity_m_per_s,
        profile_scales,
        has_inlet_excitation,
        nominal_time_step_s,
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
    pub fn implicit_channel(node_count: usize) -> Self {
        Self {
            inlet_boundary_count: 1,
            outlet_boundary_count: 1,
            no_slip_wall_boundary_count: node_count.saturating_sub(1).max(1),
            slip_wall_boundary_count: 0,
            symmetry_boundary_count: 0,
            authored_boundary_count: 0,
            boundary_coverage_ratio: 1.0,
            wall_boundary_coverage_ratio: 1.0,
            nominal_inlet_velocity_m_per_s: 0.0,
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
            nominal_inlet_velocity_m_per_s: 0.0,
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
            return Self::implicit_channel(node_count);
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
