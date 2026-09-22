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
    pub sample_count: usize,
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
) -> Result<FeaRunResult, String> {
    let topology = CfdDomainTopology::from_model(model, solver_mesh);
    let node_count = topology.node_count;
    let sample_count = if identity.procedure() == AnalysisProcedure::TransientResponse {
        options.sample_count
    } else {
        1
    };
    let transient = identity.procedure() == AnalysisProcedure::TransientResponse;
    let field_step = if transient {
        sample_count.saturating_sub(1)
    } else {
        0
    };
    let schedule = domain::realize_velocity_schedule(
        model,
        identity.family(),
        identity.procedure(),
        sample_count,
    )?;
    let mut boundary_summary = CfdBoundarySummary::from_model(model, node_count);
    if schedule.has_inlet_excitation {
        boundary_summary.nominal_inlet_velocity_m_per_s = schedule.nominal_velocity_m_per_s;
    }
    let profile_scales = schedule.profile_scales;
    let solution = solve::solve_velocity_pressure(
        domain,
        &boundary_summary,
        &topology,
        field_step,
        &profile_scales,
        solve::CfdIterationControl {
            step_count: sample_count,
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
    let mut run_diagnostics = diagnostics::coupling_diagnostics(
        domain,
        &topology,
        diagnostics::CfdCouplingEvidence {
            inlet_velocity_m_per_s: solution.nominal_inlet_velocity_m_per_s,
            executed_sample_count: profile_scales.len(),
            nominal_time_step_s: schedule.nominal_time_step_s,
            transient,
            pressure_drop_pa: solution.pressure_drop_pa,
            mass_balance_residual: solution.mass_balance_residual,
            max_momentum_residual,
            max_continuity_residual,
            residual_warn_threshold: options.residual_warn_threshold,
        },
    );
    let run = FeaRunResult {
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
        structural_static_balance: None,
        structural_constraint_result: None,
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        diagnostics: {
            run_diagnostics.extend([
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
                    "solve_family={} executed_sample_count={} nominal_time_step_s={} transient_scale_min={} transient_scale_max={} transient_scale_variation={}",
                    if transient { "transient" } else { "steady_state" },
                    profile_scales.len(),
                    schedule.nominal_time_step_s,
                    solution.transient_scale_min,
                    solution.transient_scale_max,
                    solution.transient_scale_variation,
                ),
            },
                diagnostics::known_answer_diagnostic(&known_answer_metrics, &topology),
            ]);
            run_diagnostics
        },
        fields,
    };
    run.validate_solver_formulation()
        .map_err(|error| error.to_string())?;
    Ok(run)
}
