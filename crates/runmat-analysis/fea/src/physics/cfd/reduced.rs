use runmat_analysis_core::{AnalysisModel, AnalysisProcedure, CfdDomain};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::backend::kind::LinearAlgebraBackendKind,
    AnalysisRunIdentity, ComputeBackend, ExecutionBackend, FeaRunResult, SolverMethod,
    SolverPreconditioner,
};

use super::{
    diagnostics,
    domain::{self, CfdBoundarySummary, CfdDomainTopology},
    recovery, solve,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducedCfdSolveOptions {
    pub time_step_s: f64,
    pub step_count: usize,
    pub max_linear_iters: usize,
    pub tolerance: f64,
    pub residual_warn_threshold: f64,
}

pub fn solve_reduced_finite_volume(
    model: &AnalysisModel,
    domain: &CfdDomain,
    backend: ComputeBackend,
    identity: AnalysisRunIdentity,
    options: &ReducedCfdSolveOptions,
    solver_mesh: Option<&SolverMeshArtifact>,
) -> FeaRunResult {
    let topology = CfdDomainTopology::from_model(model, solver_mesh);
    let node_count = topology.node_count;
    let step_count = options.step_count.max(1);
    let transient = identity.procedure() == AnalysisProcedure::TransientResponse;
    let field_step = if transient {
        step_count.saturating_sub(1)
    } else {
        0
    };
    let boundary_summary = CfdBoundarySummary::from_model(model, node_count);
    let profile_scales = domain::velocity_scales(model);
    let solution = solve::solve_velocity_pressure(
        domain,
        &boundary_summary,
        &topology,
        field_step,
        &profile_scales,
        solve::CfdIterationControl {
            step_count,
            max_linear_iters: options.max_linear_iters,
            tolerance: options.tolerance,
        },
    );
    let max_momentum_residual = solution
        .residual_momentum
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let max_continuity_residual = solution
        .residual_continuity
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let known_answer_metrics = diagnostics::known_answer_metrics(&solution);
    let fields = recovery::build_run_fields(domain, &solution);
    let residual_severity = if max_momentum_residual <= options.residual_warn_threshold
        && max_continuity_residual <= options.residual_warn_threshold
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };
    FeaRunResult {
        identity,
        requested_backend: backend,
        execution_backend: ExecutionBackend::Cpu,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        solver_device_apply_k_ratio: 0.0,
        solver_method: SolverMethod::CfdVelocityPressureFiniteVolume,
        preconditioner: SolverPreconditioner::FiniteVolumePressureBalance,
        solver_host_sync_count: 0,
        accelerator_evidence: Vec::new(),
        spatial_projection_evidence: Vec::new(),
        observations: Vec::new(),
        diagnostics: vec![
            FeaDiagnostic {
                code: "FEA_CFD_RESIDUAL".to_string(),
                severity: residual_severity,
                message: format!(
                    "max_momentum_residual={} max_continuity_residual={} residual_warn_threshold={} cfd_node_count={} cfd_step_count={}",
                    max_momentum_residual,
                    max_continuity_residual,
                    options.residual_warn_threshold,
                    node_count,
                    step_count,
                ),
            },
            diagnostics::assembly_diagnostic(
                &solution.topology,
                domain,
                solution.nominal_inlet_velocity_m_per_s,
                options.time_step_s,
                solution.pressure_drop_pa,
                solution.mass_balance_residual,
                options.residual_warn_threshold,
            ),
            FeaDiagnostic {
                code: "FEA_CFD_BOUNDARY_CONDITIONS".to_string(),
                severity: if solution.boundary_coverage_ratio >= 1.0
                    && solution.wall_boundary_coverage_ratio >= 1.0
                    && solution.inlet_velocity_realization_ratio.is_finite()
                {
                    FeaDiagnosticSeverity::Info
                } else {
                    FeaDiagnosticSeverity::Warning
                },
                message: format!(
                    "boundary_source={} authored_boundary_count={} inlet_boundary_count={} outlet_boundary_count={} wall_boundary_count={} no_slip_wall_boundary_count={} slip_wall_boundary_count={} symmetry_boundary_count={} boundary_coverage_ratio={} wall_boundary_coverage_ratio={} nominal_inlet_velocity_m_per_s={} outlet_pressure_pa={} inlet_velocity_realization_ratio={}",
                    if solution.authored_boundary_count > 0 {
                        "authored"
                    } else {
                        "implicit_channel"
                    },
                    solution.authored_boundary_count,
                    solution.inlet_boundary_count,
                    solution.outlet_boundary_count,
                    solution.wall_boundary_count,
                    solution.no_slip_wall_boundary_count,
                    solution.slip_wall_boundary_count,
                    solution.symmetry_boundary_count,
                    solution.boundary_coverage_ratio,
                    solution.wall_boundary_coverage_ratio,
                    solution.nominal_inlet_velocity_m_per_s,
                    solution.outlet_pressure_pa,
                    solution.inlet_velocity_realization_ratio,
                ),
            },
            FeaDiagnostic {
                code: "FEA_CFD_PRESSURE_CORRECTION".to_string(),
                severity: if solution.pressure_correction_residual_ratio <= options.tolerance
                    && solution.velocity_correction_residual_ratio <= options.tolerance
                {
                    FeaDiagnosticSeverity::Info
                } else {
                    FeaDiagnosticSeverity::Warning
                },
                message: format!(
                    "iteration_count={} max_linear_iters={} tolerance={} pressure_correction_residual_ratio={} velocity_correction_residual_ratio={}",
                    solution.pressure_correction_iteration_count,
                    options.max_linear_iters,
                    options.tolerance,
                    solution.pressure_correction_residual_ratio,
                    solution.velocity_correction_residual_ratio,
                ),
            },
            FeaDiagnostic {
                code: "FEA_CFD_TRANSIENT_EVOLUTION".to_string(),
                severity: FeaDiagnosticSeverity::Info,
                message: format!(
                    "solve_family={} step_count={} time_step_s={} transient_profile_point_count={} transient_scale_min={} transient_scale_max={} transient_scale_variation={}",
                    if transient { "transient" } else { "steady_state" },
                    step_count,
                    options.time_step_s,
                    profile_scales.len(),
                    solution.transient_scale_min,
                    solution.transient_scale_max,
                    solution.transient_scale_variation,
                ),
            },
            diagnostics::known_answer_diagnostic(&known_answer_metrics, &topology),
            FeaDiagnostic {
                code: "FEA_CFD_FLOW".to_string(),
                severity: FeaDiagnosticSeverity::Info,
                message: format!(
                    "density={} viscosity={} inlet_velocity={} turbulence_intensity={} reynolds_number={} solve_family={} profile_point_count={} topology_basis={} control_volume_count={} control_volume_face_count={} control_volume_internal_face_count={} control_volume_boundary_face_count={} control_volume_connectivity_coverage_ratio={} domain_length_m={} hydraulic_diameter_m={}",
                    domain.density.kilograms_per_cubic_meter,
                    domain.dynamic_viscosity.pascal_seconds,
                    boundary_summary.nominal_inlet_velocity_m_per_s,
                    domain.turbulence_intensity.ratio,
                    domain::reynolds_number_for_velocity(
                        domain,
                        boundary_summary.nominal_inlet_velocity_m_per_s,
                    ),
                    if transient { "transient" } else { "steady_state" },
                    profile_scales.len(),
                    topology.basis.as_str(),
                    topology.control_volume_count,
                    topology.control_volume_face_count,
                    topology.control_volume_internal_face_count,
                    topology.control_volume_boundary_face_count,
                    topology.control_volume_connectivity_coverage_ratio,
                    topology.domain_length_m,
                    topology.hydraulic_diameter_m,
                ),
            },
        ],
        fields,
    }
}
