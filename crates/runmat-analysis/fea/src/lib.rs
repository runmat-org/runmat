//! FEA pipeline scaffolding for assembly, linear solve, and field postprocessing.

pub mod assembly;
pub mod diagnostics;
pub mod fixtures;
pub mod operator;
pub mod parity;
pub mod post;
pub mod solve;
pub(crate) mod thermo;

use runmat_analysis_core::{
    validate_model, AnalysisField, AnalysisModel, EvidenceConfidence, MaterialAssignment,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    assembly::assemble_linear_system,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    post::fields::recover_result_fields,
    solve::{
        backend::{build_backend, kind::LinearAlgebraBackendKind},
        linear::solve_linear_system,
        modal::solve_modal_system,
        nonlinear::{solve_nonlinear_system, NonlinearSolveOptions},
        preconditioner::SpdPreconditionerKind,
        transient::{solve_transient_system, TransientSolveOptions},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackend {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaRunResult {
    pub backend: ComputeBackend,
    pub solver_backend: String,
    pub solver_device_apply_k_ratio: f64,
    pub solver_method: String,
    pub preconditioner: String,
    pub solver_host_sync_count: u32,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub displacement_field: AnalysisField,
    pub von_mises_field: AnalysisField,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FeaPrepContext {
    pub prepared_mesh_count: usize,
    pub prepared_node_count: usize,
    pub prepared_element_count: usize,
    pub mapped_region_count: usize,
    pub min_scaled_jacobian: f64,
    pub mean_aspect_ratio: f64,
    pub inverted_element_count: usize,
    pub mapped_load_count: usize,
    pub mapped_bc_count: usize,
    pub layout_seed: u64,
    pub topology_dof_multiplier: f64,
    pub topology_bandwidth_proxy: u32,
    pub mapped_region_participation_ratio: f64,
    pub topology_surface_patch_ratio: f64,
    pub topology_volume_core_ratio: f64,
    pub topology_mixed_family_ratio: f64,
    pub topology_region_span_mean: f64,
    pub topology_region_block_count: usize,
    pub topology_region_mesh_mean: f64,
    pub topology_region_mesh_variance: f64,
    pub topology_triangle_family_ratio: f64,
    pub topology_quad_family_ratio: f64,
    pub topology_tet_family_ratio: f64,
    pub topology_hex_family_ratio: f64,
    pub calibration_profile_override: Option<FeaPrepCalibrationProfile>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermoRegionTemperatureDelta {
    pub region_id: String,
    pub temperature_delta_k: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermoTimeProfilePoint {
    pub normalized_time: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaThermoFieldInterpolationMode {
    Linear,
    Step,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermoFieldSource {
    pub source_id: String,
    pub revision: u32,
    pub interpolation_mode: Option<FeaThermoFieldInterpolationMode>,
    pub expected_region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaThermoMechanicalContext {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_temperature_delta_k: f64,
    pub thermal_expansion_coefficient: f64,
    pub field_source: Option<FeaThermoFieldSource>,
    pub region_temperature_deltas: Vec<FeaThermoRegionTemperatureDelta>,
    pub time_profile: Vec<FeaThermoTimeProfilePoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaPrepCalibrationProfile {
    Fast,
    Balanced,
    Conservative,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearStaticSolveOptions {
    pub preconditioner_kind: SpdPreconditionerKind,
    pub algebra_backend_kind: LinearAlgebraBackendKind,
    pub prep_context: Option<FeaPrepContext>,
    pub thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModalSolveOptions {
    pub mode_count: usize,
    pub prep_context: Option<FeaPrepContext>,
    pub thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
}

impl Default for ModalSolveOptions {
    fn default() -> Self {
        Self {
            mode_count: 3,
            prep_context: None,
            thermo_mechanical_context: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaModalRunResult {
    pub run: FeaRunResult,
    pub eigenvalues_hz: Vec<f64>,
    pub mode_shapes: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaTransientRunResult {
    pub run: FeaRunResult,
    pub time_points_s: Vec<f64>,
    pub displacement_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaNonlinearRunResult {
    pub run: FeaRunResult,
    pub load_factors: Vec<f64>,
    pub displacement_snapshots: Vec<AnalysisField>,
    pub residual_norms: Vec<f64>,
    pub increment_norms: Vec<f64>,
    pub iteration_counts: Vec<usize>,
    pub failed_increments: usize,
    pub line_search_backtracks: usize,
    pub max_line_search_backtracks_per_increment: usize,
    pub tangent_rebuild_count: usize,
    pub iteration_spike_count: usize,
    pub convergence_stall_count: usize,
    pub backtrack_burst_count: usize,
}

impl Default for LinearStaticSolveOptions {
    fn default() -> Self {
        Self {
            preconditioner_kind: SpdPreconditionerKind::Jacobi,
            algebra_backend_kind: LinearAlgebraBackendKind::CpuReference,
            prep_context: None,
            thermo_mechanical_context: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum FeaRunError {
    #[error("FEA_MODEL_INVALID: {0}")]
    InvalidModel(String),
}

pub fn run_linear_static(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaRunResult, FeaRunError> {
    run_linear_static_with_options(model, backend, LinearStaticSolveOptions::default())
}

pub fn run_linear_static_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
) -> Result<FeaRunResult, FeaRunError> {
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;

    let summary = assemble_linear_system(
        model,
        options.prep_context,
        options.thermo_mechanical_context,
    );
    let algebra_backend = build_backend(options.algebra_backend_kind);
    let solve_result = solve_linear_system(
        &summary,
        options.preconditioner_kind,
        options.algebra_backend_kind,
        algebra_backend.as_ref(),
    );
    let fields = recover_result_fields(&summary, &solve_result);
    let solver_device_apply_k_ratio = if solve_result.device_apply_k_attempt_count == 0 {
        0.0
    } else {
        solve_result.device_apply_k_count as f64 / solve_result.device_apply_k_attempt_count as f64
    };

    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_CONVERGENCE".to_string(),
        severity: if solve_result.converged {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "iterations={} residual_norm={} converged={} solver_method={} preconditioner={}",
            solve_result.iterations,
            solve_result.residual_norm,
            solve_result.converged,
            solve_result.solver_method,
            solve_result.preconditioner,
        ),
    }];
    diagnostics.extend(material_assignment_diagnostics(&model.material_assignments));
    if let Some(prep) = options.prep_context {
        diagnostics.push(prep_diagnostic(prep));
    }
    if let Some(prep_summary) = summary.prep_assembly.as_ref() {
        diagnostics.push(prep_assembly_diagnostic(prep_summary));
        if let Some(prep) = options.prep_context {
            diagnostics.push(prep_topology_diagnostic(prep, summary.dof_count));
        }
    }
    if let Some(operator_topology) = summary.prep_operator_topology.as_ref() {
        diagnostics.push(prep_operator_topology_diagnostic(operator_topology));
    }
    if let Some(region_topology) = summary.prep_region_topology.as_ref() {
        diagnostics.push(prep_region_topology_diagnostic(region_topology));
    }
    if let Some(element_assembly) = summary.prep_element_assembly.as_ref() {
        diagnostics.push(prep_element_assembly_diagnostic(element_assembly));
    }
    if let Some(element_connectivity) = summary.prep_element_connectivity.as_ref() {
        diagnostics.push(prep_element_connectivity_diagnostic(element_connectivity));
    }
    if let Some(graph_assembly) = summary.prep_graph_assembly.as_ref() {
        diagnostics.push(prep_graph_assembly_diagnostic(graph_assembly));
        diagnostics.push(prep_graph_solver_diagnostic(
            graph_assembly,
            solve_result.iterations as f64,
            solve_result.residual_norm,
            options.preconditioner_kind.as_str(),
            &solve_result.preconditioner,
        ));
    }
    if let Some(calibration) = summary.prep_calibration.as_ref() {
        diagnostics.push(prep_calibration_diagnostic(calibration));
    }
    if let Some(acceptance) = summary.prep_acceptance.as_ref() {
        diagnostics.push(prep_acceptance_diagnostic(acceptance));
    }
    if let Some(thermo_mechanical) = summary.thermo_mechanical.as_ref() {
        diagnostics.push(thermo_mechanical_diagnostic(thermo_mechanical));
    }
    diagnostics.extend(solve_result.diagnostics);

    Ok(FeaRunResult {
        backend,
        solver_backend: solve_result.solver_backend,
        solver_device_apply_k_ratio,
        solver_method: solve_result.solver_method,
        preconditioner: solve_result.preconditioner,
        solver_host_sync_count: solve_result.host_sync_count,
        diagnostics,
        displacement_field: fields.displacement_field,
        von_mises_field: fields.von_mises_field,
    })
}

pub fn run_modal(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaModalRunResult, FeaRunError> {
    run_modal_with_options(model, backend, ModalSolveOptions::default())
}

pub fn run_modal_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ModalSolveOptions,
) -> Result<FeaModalRunResult, FeaRunError> {
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;

    let summary = assemble_linear_system(
        model,
        options.prep_context,
        options.thermo_mechanical_context,
    );
    let modal = solve_modal_system(&summary, options.mode_count, backend);
    let mut diagnostics = modal.diagnostics.clone();
    diagnostics.extend(material_assignment_diagnostics(&model.material_assignments));
    if let Some(prep) = options.prep_context {
        diagnostics.push(prep_diagnostic(prep));
    }
    if let Some(prep_summary) = summary.prep_assembly.as_ref() {
        diagnostics.push(prep_assembly_diagnostic(prep_summary));
        if let Some(prep) = options.prep_context {
            diagnostics.push(prep_topology_diagnostic(prep, summary.dof_count));
        }
    }
    if let Some(operator_topology) = summary.prep_operator_topology.as_ref() {
        diagnostics.push(prep_operator_topology_diagnostic(operator_topology));
    }
    if let Some(region_topology) = summary.prep_region_topology.as_ref() {
        diagnostics.push(prep_region_topology_diagnostic(region_topology));
    }
    if let Some(element_assembly) = summary.prep_element_assembly.as_ref() {
        diagnostics.push(prep_element_assembly_diagnostic(element_assembly));
    }
    if let Some(element_connectivity) = summary.prep_element_connectivity.as_ref() {
        diagnostics.push(prep_element_connectivity_diagnostic(element_connectivity));
    }
    if let Some(graph_assembly) = summary.prep_graph_assembly.as_ref() {
        diagnostics.push(prep_graph_assembly_diagnostic(graph_assembly));
        diagnostics.push(prep_graph_solver_diagnostic(
            graph_assembly,
            mode_shapes_iteration_proxy(&modal.residual_norms),
            modal.residual_norms.iter().copied().fold(0.0_f64, f64::max),
            "auto",
            if backend == ComputeBackend::Gpu {
                "jacobi"
            } else {
                "none"
            },
        ));
    }
    if let Some(calibration) = summary.prep_calibration.as_ref() {
        diagnostics.push(prep_calibration_diagnostic(calibration));
    }
    if let Some(acceptance) = summary.prep_acceptance.as_ref() {
        diagnostics.push(prep_acceptance_diagnostic(acceptance));
    }
    if let Some(thermo_mechanical) = summary.thermo_mechanical.as_ref() {
        diagnostics.push(thermo_mechanical_diagnostic(thermo_mechanical));
    }

    let displacement = modal
        .mode_shapes
        .first()
        .cloned()
        .unwrap_or_else(|| vec![0.0; summary.dof_count.max(3)]);
    let von_mises = displacement
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        * 1.0e11;

    let run = FeaRunResult {
        backend,
        solver_backend: modal.solver_backend,
        solver_device_apply_k_ratio: if modal.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            modal.device_apply_k_count as f64 / modal.device_apply_k_attempt_count as f64
        },
        solver_method: modal.solver_method,
        preconditioner: if backend == ComputeBackend::Gpu {
            "jacobi".to_string()
        } else {
            "none".to_string()
        },
        solver_host_sync_count: modal.solver_host_sync_count,
        diagnostics,
        displacement_field: AnalysisField::host_f64(
            "displacement",
            vec![displacement.len()],
            displacement,
        ),
        von_mises_field: AnalysisField::host_f64("von_mises", vec![1], vec![von_mises]),
    };

    let mode_shapes = modal
        .mode_shapes
        .into_iter()
        .enumerate()
        .map(|(index, shape)| {
            AnalysisField::host_f64(
                format!("mode_shape_{}", index + 1),
                vec![shape.len()],
                shape,
            )
        })
        .collect();

    Ok(FeaModalRunResult {
        run,
        eigenvalues_hz: modal.eigenvalues_hz,
        mode_shapes,
        residual_norms: modal.residual_norms,
    })
}

pub fn run_transient(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaTransientRunResult, FeaRunError> {
    run_transient_with_options(model, backend, TransientSolveOptions::default())
}

pub fn run_transient_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: TransientSolveOptions,
) -> Result<FeaTransientRunResult, FeaRunError> {
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    let prep_context = options.prep_context;
    let thermo_context = options.thermo_mechanical_context.clone();

    let summary = assemble_linear_system(model, prep_context, thermo_context);
    let transient = solve_transient_system(&summary, options.clone(), backend);
    let mut diagnostics = transient.diagnostics.clone();
    diagnostics.extend(material_assignment_diagnostics(&model.material_assignments));
    if let Some(prep) = prep_context {
        diagnostics.push(prep_diagnostic(prep));
    }
    if let Some(prep_summary) = summary.prep_assembly.as_ref() {
        diagnostics.push(prep_assembly_diagnostic(prep_summary));
        if let Some(prep) = prep_context {
            diagnostics.push(prep_topology_diagnostic(prep, summary.dof_count));
        }
    }
    if let Some(operator_topology) = summary.prep_operator_topology.as_ref() {
        diagnostics.push(prep_operator_topology_diagnostic(operator_topology));
    }
    if let Some(region_topology) = summary.prep_region_topology.as_ref() {
        diagnostics.push(prep_region_topology_diagnostic(region_topology));
    }
    if let Some(element_assembly) = summary.prep_element_assembly.as_ref() {
        diagnostics.push(prep_element_assembly_diagnostic(element_assembly));
    }
    if let Some(element_connectivity) = summary.prep_element_connectivity.as_ref() {
        diagnostics.push(prep_element_connectivity_diagnostic(element_connectivity));
    }
    if let Some(graph_assembly) = summary.prep_graph_assembly.as_ref() {
        diagnostics.push(prep_graph_assembly_diagnostic(graph_assembly));
        diagnostics.push(prep_graph_solver_diagnostic(
            graph_assembly,
            transient.converged_steps as f64,
            transient
                .residual_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            "auto",
            &transient.preconditioner,
        ));
    }
    if let Some(calibration) = summary.prep_calibration.as_ref() {
        diagnostics.push(prep_calibration_diagnostic(calibration));
    }
    if let Some(acceptance) = summary.prep_acceptance.as_ref() {
        diagnostics.push(prep_acceptance_diagnostic(acceptance));
    }
    if let Some(thermo_mechanical) = summary.thermo_mechanical.as_ref() {
        diagnostics.push(thermo_mechanical_diagnostic(thermo_mechanical));
    }

    let displacement = transient
        .displacement_snapshots
        .last()
        .cloned()
        .unwrap_or_else(|| vec![0.0; summary.dof_count.max(3)]);
    let von_mises = displacement
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        * 1.0e11;

    let run = FeaRunResult {
        backend,
        solver_backend: transient.solver_backend,
        solver_device_apply_k_ratio: if transient.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            transient.device_apply_k_count as f64 / transient.device_apply_k_attempt_count as f64
        },
        solver_method: transient.solver_method,
        preconditioner: transient.preconditioner,
        solver_host_sync_count: transient.solver_host_sync_count,
        diagnostics,
        displacement_field: AnalysisField::host_f64(
            "displacement",
            vec![displacement.len()],
            displacement,
        ),
        von_mises_field: AnalysisField::host_f64("von_mises", vec![1], vec![von_mises]),
    };

    let displacement_snapshots = transient
        .displacement_snapshots
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                format!("displacement_t{}", index),
                vec![snapshot.len()],
                snapshot,
            )
        })
        .collect();

    Ok(FeaTransientRunResult {
        run,
        time_points_s: transient.time_points_s,
        displacement_snapshots,
        residual_norms: transient.residual_norms,
    })
}

pub fn run_nonlinear(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaNonlinearRunResult, FeaRunError> {
    run_nonlinear_with_options(model, backend, NonlinearSolveOptions::default())
}

pub fn run_nonlinear_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: NonlinearSolveOptions,
) -> Result<FeaNonlinearRunResult, FeaRunError> {
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    let prep_context = options.prep_context;
    let thermo_context = options.thermo_mechanical_context.clone();

    let summary = assemble_linear_system(model, prep_context, thermo_context);
    let nonlinear = solve_nonlinear_system(&summary, options.clone(), backend);
    let mut diagnostics = nonlinear.diagnostics.clone();
    diagnostics.extend(material_assignment_diagnostics(&model.material_assignments));
    if let Some(prep) = prep_context {
        diagnostics.push(prep_diagnostic(prep));
    }
    if let Some(prep_summary) = summary.prep_assembly.as_ref() {
        diagnostics.push(prep_assembly_diagnostic(prep_summary));
        if let Some(prep) = prep_context {
            diagnostics.push(prep_topology_diagnostic(prep, summary.dof_count));
        }
    }
    if let Some(operator_topology) = summary.prep_operator_topology.as_ref() {
        diagnostics.push(prep_operator_topology_diagnostic(operator_topology));
    }
    if let Some(region_topology) = summary.prep_region_topology.as_ref() {
        diagnostics.push(prep_region_topology_diagnostic(region_topology));
    }
    if let Some(element_assembly) = summary.prep_element_assembly.as_ref() {
        diagnostics.push(prep_element_assembly_diagnostic(element_assembly));
    }
    if let Some(element_connectivity) = summary.prep_element_connectivity.as_ref() {
        diagnostics.push(prep_element_connectivity_diagnostic(element_connectivity));
    }
    if let Some(graph_assembly) = summary.prep_graph_assembly.as_ref() {
        diagnostics.push(prep_graph_assembly_diagnostic(graph_assembly));
        diagnostics.push(prep_graph_solver_diagnostic(
            graph_assembly,
            nonlinear
                .iteration_counts
                .iter()
                .copied()
                .max()
                .unwrap_or(0) as f64,
            nonlinear
                .residual_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            "auto",
            &nonlinear.preconditioner,
        ));
    }
    if let Some(calibration) = summary.prep_calibration.as_ref() {
        diagnostics.push(prep_calibration_diagnostic(calibration));
    }
    if let Some(acceptance) = summary.prep_acceptance.as_ref() {
        diagnostics.push(prep_acceptance_diagnostic(acceptance));
    }
    if let Some(thermo_mechanical) = summary.thermo_mechanical.as_ref() {
        diagnostics.push(thermo_mechanical_diagnostic(thermo_mechanical));
    }

    let displacement = nonlinear
        .displacement_snapshots
        .last()
        .cloned()
        .unwrap_or_else(|| vec![0.0; summary.dof_count.max(3)]);
    let von_mises = displacement
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        * 1.0e11;

    let run = FeaRunResult {
        backend,
        solver_backend: nonlinear.solver_backend,
        solver_device_apply_k_ratio: if nonlinear.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            nonlinear.device_apply_k_count as f64 / nonlinear.device_apply_k_attempt_count as f64
        },
        solver_method: nonlinear.solver_method,
        preconditioner: nonlinear.preconditioner,
        solver_host_sync_count: nonlinear.solver_host_sync_count,
        diagnostics,
        displacement_field: AnalysisField::host_f64(
            "displacement",
            vec![displacement.len()],
            displacement,
        ),
        von_mises_field: AnalysisField::host_f64("von_mises", vec![1], vec![von_mises]),
    };

    let displacement_snapshots = nonlinear
        .displacement_snapshots
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                format!("nonlinear_displacement_inc{}", index),
                vec![snapshot.len()],
                snapshot,
            )
        })
        .collect();

    Ok(FeaNonlinearRunResult {
        run,
        load_factors: nonlinear.load_factors,
        displacement_snapshots,
        residual_norms: nonlinear.residual_norms,
        increment_norms: nonlinear.increment_norms,
        iteration_counts: nonlinear.iteration_counts,
        failed_increments: nonlinear.failed_increments,
        line_search_backtracks: nonlinear.line_search_backtracks,
        max_line_search_backtracks_per_increment: nonlinear
            .max_line_search_backtracks_per_increment,
        tangent_rebuild_count: nonlinear.tangent_rebuild_count,
        iteration_spike_count: nonlinear.iteration_spike_count,
        convergence_stall_count: nonlinear.convergence_stall_count,
        backtrack_burst_count: nonlinear.backtrack_burst_count,
    })
}

fn material_assignment_diagnostics(assignments: &[MaterialAssignment]) -> Vec<FeaDiagnostic> {
    let mut out = Vec::new();
    for assignment in assignments {
        if assignment.expected_material_id == assignment.assigned_material_id {
            continue;
        }

        let (code, severity) = match assignment.confidence {
            EvidenceConfidence::Verified => (
                "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_VERIFIED",
                FeaDiagnosticSeverity::Error,
            ),
            EvidenceConfidence::Probable => (
                "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_PROBABLE",
                FeaDiagnosticSeverity::Warning,
            ),
            EvidenceConfidence::Inferred => (
                "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_INFERRED",
                FeaDiagnosticSeverity::Warning,
            ),
        };

        out.push(FeaDiagnostic {
            code: code.to_string(),
            severity,
            message: format!(
                "region={} expected_material={} assigned_material={} confidence={:?}",
                assignment.region_id,
                assignment.expected_material_id,
                assignment.assigned_material_id,
                assignment.confidence
            ),
        });
    }
    out
}

fn prep_diagnostic(prep: FeaPrepContext) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_CONTEXT".to_string(),
        severity: if prep.inverted_element_count == 0 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "prepared_mesh_count={} prepared_node_count={} prepared_element_count={} mapped_region_count={} mapped_load_count={} mapped_bc_count={} min_scaled_jacobian={} mean_aspect_ratio={} inverted_element_count={} topology_dof_multiplier={} topology_bandwidth_proxy={} mapped_region_participation_ratio={} topology_surface_patch_ratio={} topology_volume_core_ratio={} topology_mixed_family_ratio={} topology_region_span_mean={} topology_region_block_count={} topology_region_mesh_mean={} topology_region_mesh_variance={} topology_triangle_family_ratio={} topology_quad_family_ratio={} topology_tet_family_ratio={} topology_hex_family_ratio={} calibration_profile_override={}",
            prep.prepared_mesh_count,
            prep.prepared_node_count,
            prep.prepared_element_count,
            prep.mapped_region_count,
            prep.mapped_load_count,
            prep.mapped_bc_count,
            prep.min_scaled_jacobian,
            prep.mean_aspect_ratio,
            prep.inverted_element_count,
            prep.topology_dof_multiplier,
            prep.topology_bandwidth_proxy,
            prep.mapped_region_participation_ratio,
            prep.topology_surface_patch_ratio,
            prep.topology_volume_core_ratio,
            prep.topology_mixed_family_ratio,
            prep.topology_region_span_mean,
            prep.topology_region_block_count,
            prep.topology_region_mesh_mean,
            prep.topology_region_mesh_variance,
            prep.topology_triangle_family_ratio,
            prep.topology_quad_family_ratio,
            prep.topology_tet_family_ratio,
            prep.topology_hex_family_ratio,
            prep.calibration_profile_override
                .map(|profile| match profile {
                    FeaPrepCalibrationProfile::Fast => "fast",
                    FeaPrepCalibrationProfile::Balanced => "balanced",
                    FeaPrepCalibrationProfile::Conservative => "conservative",
                })
                .unwrap_or("auto"),
        ),
    }
}

fn prep_assembly_diagnostic(summary: &assembly::PrepAssemblySummary) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_ASSEMBLY".to_string(),
        severity: if summary.mapped_load_ratio > 0.0 || summary.constrained_prep_ratio > 0.0 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "active_region_count={} mapped_load_count={} mapped_bc_count={} mapped_load_ratio={} constrained_prep_ratio={} layout_seed={}",
            summary.active_region_count,
            summary.mapped_load_count,
            summary.mapped_bc_count,
            summary.mapped_load_ratio,
            summary.constrained_prep_ratio,
            summary.layout_seed
        ),
    }
}

fn prep_topology_diagnostic(prep: FeaPrepContext, dof_count: usize) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_TOPOLOGY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "effective_dof_multiplier={} effective_dof_count={} coupling_bandwidth_proxy={} mapped_region_participation_ratio={} surface_patch_ratio={} volume_core_ratio={} mixed_family_ratio={} mean_region_span={}",
            prep.topology_dof_multiplier,
            dof_count,
            prep.topology_bandwidth_proxy,
            prep.mapped_region_participation_ratio,
            prep.topology_surface_patch_ratio,
            prep.topology_volume_core_ratio,
            prep.topology_mixed_family_ratio,
            prep.topology_region_span_mean,
        ),
    }
}

fn prep_operator_topology_diagnostic(
    summary: &assembly::PrepOperatorTopologySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_OPERATOR_TOPOLOGY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "stiffness_scale={} mass_scale={} damping_scale={} rhs_scale={} coupling_nonzero_ratio={} stiffness_spread_ratio={} topology_fingerprint={}",
            summary.stiffness_scale,
            summary.mass_scale,
            summary.damping_scale,
            summary.rhs_scale,
            summary.coupling_nonzero_ratio,
            summary.stiffness_spread_ratio,
            summary.topology_fingerprint,
        ),
    }
}

fn prep_region_topology_diagnostic(summary: &assembly::PrepRegionTopologySummary) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_REGION_TOPOLOGY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "region_block_count={} inter_block_edge_count={} coupling_nonzero_ratio={} block_size_min={} block_size_max={} block_size_mean={} region_topology_fingerprint={}",
            summary.region_block_count,
            summary.inter_block_edge_count,
            summary.coupling_nonzero_ratio,
            summary.block_size_min,
            summary.block_size_max,
            summary.block_size_mean,
            summary.region_topology_fingerprint,
        ),
    }
}

fn prep_element_assembly_diagnostic(
    summary: &assembly::PrepElementAssemblySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_ELEMENT_ASSEMBLY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "assembled_element_count={} triangle_element_count={} quad_element_count={} tet_element_count={} hex_element_count={} mixed_element_count={} scatter_nnz_count={} assembly_fingerprint={}",
            summary.assembled_element_count,
            summary.triangle_element_count,
            summary.quad_element_count,
            summary.tet_element_count,
            summary.hex_element_count,
            summary.mixed_element_count,
            summary.scatter_nnz_count,
            summary.assembly_fingerprint,
        ),
    }
}

fn prep_element_connectivity_diagnostic(
    summary: &assembly::PrepElementConnectivitySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_ELEMENT_CONNECTIVITY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "assembled_element_count={} stiffness_offdiag_nnz_count={} mass_offdiag_proxy_nnz_count={} damping_offdiag_proxy_nnz_count={} triangle_contrib_share={} quad_contrib_share={} tet_contrib_share={} hex_contrib_share={} mixed_contrib_share={} mean_connectivity_hop={} connectivity_fingerprint={}",
            summary.assembled_element_count,
            summary.stiffness_offdiag_nnz_count,
            summary.mass_offdiag_proxy_nnz_count,
            summary.damping_offdiag_proxy_nnz_count,
            summary.triangle_contrib_share,
            summary.quad_contrib_share,
            summary.tet_contrib_share,
            summary.hex_contrib_share,
            summary.mixed_contrib_share,
            summary.mean_connectivity_hop,
            summary.connectivity_fingerprint,
        ),
    }
}

fn prep_graph_assembly_diagnostic(summary: &assembly::PrepGraphAssemblySummary) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_GRAPH_ASSEMBLY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "node_count={} edge_count={} degree_min={} degree_max={} degree_mean={} degree_p95={} fill_ratio={} connected_component_count={} ordering_bandwidth_before={} ordering_bandwidth_after={} ordering_reduction_ratio={} ordering_fingerprint={} recommend_ilu0={} graph_fingerprint={}",
            summary.node_count,
            summary.edge_count,
            summary.degree_min,
            summary.degree_max,
            summary.degree_mean,
            summary.degree_p95,
            summary.fill_ratio,
            summary.connected_component_count,
            summary.ordering_bandwidth_before,
            summary.ordering_bandwidth_after,
            summary.ordering_reduction_ratio,
            summary.ordering_fingerprint,
            summary.recommend_ilu0,
            summary.graph_fingerprint,
        ),
    }
}

fn prep_graph_solver_diagnostic(
    summary: &assembly::PrepGraphAssemblySummary,
    iteration_metric: f64,
    residual_metric: f64,
    requested_preconditioner: &str,
    effective_preconditioner: &str,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_GRAPH_SOLVER".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "ordering_bandwidth_before={} ordering_bandwidth_after={} ordering_reduction_ratio={} ordering_fingerprint={} recommend_ilu0={} requested_preconditioner={} effective_preconditioner={} iteration_metric={} residual_metric={} graph_fingerprint={}",
            summary.ordering_bandwidth_before,
            summary.ordering_bandwidth_after,
            summary.ordering_reduction_ratio,
            summary.ordering_fingerprint,
            summary.recommend_ilu0,
            requested_preconditioner,
            effective_preconditioner,
            iteration_metric,
            residual_metric,
            summary.graph_fingerprint,
        ),
    }
}

fn prep_calibration_diagnostic(summary: &assembly::PrepCalibrationSummary) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_CALIBRATION".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "profile={} triangle_weight={} quad_weight={} tet_weight={} hex_weight={} mixed_weight={} stiffness_calibration_scale={} mass_calibration_scale={} damping_calibration_scale={} calibration_fingerprint={}",
            summary.profile,
            summary.triangle_weight,
            summary.quad_weight,
            summary.tet_weight,
            summary.hex_weight,
            summary.mixed_weight,
            summary.stiffness_calibration_scale,
            summary.mass_calibration_scale,
            summary.damping_calibration_scale,
            summary.calibration_fingerprint,
        ),
    }
}

fn prep_acceptance_diagnostic(summary: &assembly::PrepAcceptanceSummary) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_ACCEPTANCE".to_string(),
        severity: if summary.accepted {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "profile={} accepted={} bounded_displacement_scale={} bounded_stress_scale={} bounded_connectivity_fill={} acceptance_score={} acceptance_fingerprint={}",
            summary.profile,
            summary.accepted,
            summary.bounded_displacement_scale,
            summary.bounded_stress_scale,
            summary.bounded_connectivity_fill,
            summary.acceptance_score,
            summary.acceptance_fingerprint,
        ),
    }
}

fn thermo_mechanical_diagnostic(
    summary: &assembly::ThermoMechanicalAssemblySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_TM_COUPLING".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "enabled={} reference_temperature_k={} applied_temperature_delta_k={} thermal_expansion_coefficient={} thermal_strain_scale={} thermal_load_scale={} constitutive_temperature_factor={} constitutive_poisson_coupling={} effective_modulus_scale={} constitutive_material_spread_ratio={} assignment_heterogeneity_index={} spatial_gradient_index={} spatial_coverage_ratio={} temporal_profile_variation={} region_delta_count={} coupling_fingerprint={}",
            summary.enabled,
            summary.reference_temperature_k,
            summary.applied_temperature_delta_k,
            summary.thermal_expansion_coefficient,
            summary.thermal_strain_scale,
            summary.thermal_load_scale,
            summary.constitutive_temperature_factor,
            summary.constitutive_poisson_coupling,
            summary.effective_modulus_scale,
            summary.constitutive_material_spread_ratio,
            summary.assignment_heterogeneity_index,
            summary.spatial_gradient_index,
            summary.spatial_coverage_ratio,
            summary.temporal_profile_variation,
            summary.region_delta_count,
            summary.coupling_fingerprint,
        ),
    }
}

fn mode_shapes_iteration_proxy(residual_norms: &[f64]) -> f64 {
    residual_norms.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fixtures::{fixture_model, FixtureId},
        parity::{assert_vectors_within_tolerance, ParityTolerance},
    };

    #[test]
    fn canonical_cantilever_benchmark_runs() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let result = run_linear_static(&model, ComputeBackend::Cpu).expect("solve should succeed");

        assert_eq!(result.displacement_field.element_count(), 3);
        assert_eq!(result.von_mises_field.element_count(), 1);
        let displacement = result
            .displacement_field
            .as_host_f64()
            .expect("scaffold emits host displacement field");
        assert!(displacement[1] < 0.0);
        assert!(displacement[1] < -8.0e-6 && displacement[1] > -1.2e-5);

        let stress = result
            .von_mises_field
            .as_host_f64()
            .expect("stress field should be host-backed");
        assert!(stress[0] > 8.0e5 && stress[0] < 1.2e6);
    }

    #[test]
    fn convergence_diagnostics_are_emitted() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let result = run_linear_static(&model, ComputeBackend::Cpu).expect("solve should succeed");
        assert!(result
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_CONVERGENCE"));
    }

    #[test]
    fn deterministic_replay_for_fixture_is_stable() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let first =
            run_linear_static(&model, ComputeBackend::Cpu).expect("first run should succeed");
        let second =
            run_linear_static(&model, ComputeBackend::Cpu).expect("second run should succeed");

        assert_eq!(first.displacement_field, second.displacement_field);
        assert_eq!(first.von_mises_field, second.von_mises_field);
        assert_eq!(first.diagnostics, second.diagnostics);
    }

    #[test]
    fn cpu_gpu_parity_respects_tolerance_policy() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let cpu = run_linear_static(&model, ComputeBackend::Cpu).expect("cpu run should succeed");
        let gpu = run_linear_static(&model, ComputeBackend::Gpu).expect("gpu run should succeed");

        let tol = ParityTolerance::strict();
        let cpu_displacement = cpu
            .displacement_field
            .as_host_f64()
            .expect("cpu displacement should be host-backed in scaffold");
        let gpu_displacement = gpu
            .displacement_field
            .as_host_f64()
            .expect("gpu displacement should be host-backed in scaffold");
        assert_vectors_within_tolerance(cpu_displacement, gpu_displacement, tol);

        let cpu_stress = cpu
            .von_mises_field
            .as_host_f64()
            .expect("cpu stress should be host-backed in scaffold");
        let gpu_stress = gpu
            .von_mises_field
            .as_host_f64()
            .expect("gpu stress should be host-backed in scaffold");
        assert_vectors_within_tolerance(cpu_stress, gpu_stress, tol);
    }

    #[test]
    fn fixture_missing_materials_is_rejected() {
        let model = fixture_model(FixtureId::MissingMaterials);
        let err = run_linear_static(&model, ComputeBackend::Cpu)
            .expect_err("fixture should fail validation");
        assert!(err
            .to_string()
            .contains("ANALYSIS_VALIDATION_MISSING_MATERIALS"));
    }

    #[test]
    fn fixture_missing_loads_is_rejected() {
        let model = fixture_model(FixtureId::MissingLoads);
        let err = run_linear_static(&model, ComputeBackend::Cpu)
            .expect_err("fixture should fail validation");
        assert!(err
            .to_string()
            .contains("ANALYSIS_VALIDATION_MISSING_LOADS"));
    }

    #[test]
    fn modal_solver_emits_modes_for_modal_step_fixture() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.steps = vec![runmat_analysis_core::AnalysisStep {
            step_id: "modal_1".to_string(),
            kind: runmat_analysis_core::AnalysisStepKind::Modal,
        }];
        let result = run_modal(&model, ComputeBackend::Cpu).expect("modal solve should succeed");

        assert!(!result.eigenvalues_hz.is_empty());
        assert_eq!(result.eigenvalues_hz.len(), result.mode_shapes.len());
        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_MODAL_CONVERGENCE"));
    }

    #[test]
    fn transient_solver_emits_time_snapshots_for_transient_step_fixture() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.steps = vec![runmat_analysis_core::AnalysisStep {
            step_id: "transient_1".to_string(),
            kind: runmat_analysis_core::AnalysisStepKind::Transient,
        }];
        let result =
            run_transient(&model, ComputeBackend::Cpu).expect("transient solve should succeed");

        assert!(!result.time_points_s.is_empty());
        assert_eq!(
            result.time_points_s.len(),
            result.displacement_snapshots.len()
        );
        assert!(!result.residual_norms.is_empty());
        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_TRANSIENT_CONVERGENCE"));
    }

    #[test]
    fn modal_large_fixture_emits_orthogonality_and_separation_diagnostics() {
        let model = fixture_model(FixtureId::ModalLarge);
        let result = run_modal_with_options(
            &model,
            ComputeBackend::Cpu,
            ModalSolveOptions {
                mode_count: 8,
                prep_context: None,
                thermo_mechanical_context: None,
            },
        )
        .expect("modal large fixture should solve");

        assert!(!result.eigenvalues_hz.is_empty());
        assert_eq!(result.eigenvalues_hz.len(), result.mode_shapes.len());
        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_MODAL_ORTHOGONALITY"));
        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_MODAL_SEPARATION"));
    }

    #[test]
    fn transient_long_fixture_emits_stability_diagnostics() {
        let model = fixture_model(FixtureId::TransientLong);
        let result = run_transient_with_options(
            &model,
            ComputeBackend::Cpu,
            TransientSolveOptions {
                step_count: 24,
                ..TransientSolveOptions::default()
            },
        )
        .expect("transient long fixture should solve");

        assert!(result.time_points_s.len() > 8);
        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_TRANSIENT_STABILITY"));
        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_TRANSIENT_ENERGY"));
    }

    #[test]
    fn transient_shock_fixture_emits_adaptivity_and_physics_diagnostics() {
        let model = fixture_model(FixtureId::TransientShock);
        let result = run_transient_with_options(
            &model,
            ComputeBackend::Cpu,
            TransientSolveOptions {
                step_count: 48,
                ..TransientSolveOptions::default()
            },
        )
        .expect("transient shock fixture should solve");

        assert!(result.time_points_s.len() > 24);
        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_TRANSIENT_ADAPTIVITY"));
        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_TRANSIENT_PHYSICS"));
    }

    #[test]
    fn thermo_mechanical_transient_emits_coupled_solve_profile_diagnostic() {
        let model = fixture_model(FixtureId::ThermoMechanicalKickoff);
        let result = run_transient_with_options(
            &model,
            ComputeBackend::Cpu,
            TransientSolveOptions {
                step_count: 24,
                thermo_mechanical_context: Some(FeaThermoMechanicalContext {
                    enabled: true,
                    reference_temperature_k: 293.15,
                    applied_temperature_delta_k: 65.0,
                    thermal_expansion_coefficient: 1.2e-5,
                    field_source: None,
                    region_temperature_deltas: Vec::new(),
                    time_profile: Vec::new(),
                }),
                ..TransientSolveOptions::default()
            },
        )
        .expect("thermo-mechanical transient solve should succeed");

        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_TM_COUPLING"));
        let coupling = result
            .run
            .diagnostics
            .iter()
            .find(|diag| diag.code == "FEA_TM_COUPLING")
            .expect("thermo coupling diagnostic should be present");
        assert!(coupling.message.contains("effective_modulus_scale="));
        assert!(coupling
            .message
            .contains("constitutive_material_spread_ratio="));
        assert!(coupling.message.contains("assignment_heterogeneity_index="));
        let profile = result
            .run
            .diagnostics
            .iter()
            .find(|diag| diag.code == "FEA_TM_TRANSIENT")
            .expect("thermo transient profile diagnostic should be present");
        assert!(profile.message.contains("effective_residual_target_peak="));
        assert!(profile.message.contains("growth_limit_min="));
    }

    #[test]
    fn nonlinear_fixture_emits_incremental_payload_and_diagnostics() {
        let mut model = fixture_model(FixtureId::TransientShock);
        model.steps = vec![runmat_analysis_core::AnalysisStep {
            step_id: "nonlinear_1".to_string(),
            kind: runmat_analysis_core::AnalysisStepKind::Nonlinear,
        }];
        let result =
            run_nonlinear(&model, ComputeBackend::Cpu).expect("nonlinear solve should succeed");

        assert!(!result.load_factors.is_empty());
        assert_eq!(result.load_factors.len(), result.residual_norms.len());
        assert_eq!(result.residual_norms.len(), result.increment_norms.len());
        assert_eq!(result.residual_norms.len(), result.iteration_counts.len());
        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_NONLINEAR_CONVERGENCE"));
        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_NONLINEAR_COST"));
        let convergence = result
            .run
            .diagnostics
            .iter()
            .find(|diag| diag.code == "FEA_NONLINEAR_CONVERGENCE")
            .expect("nonlinear convergence diagnostic should be present");
        assert!(convergence.message.contains("iteration_spike_count="));
        assert!(convergence.message.contains("convergence_stall_count="));
        assert!(convergence.message.contains("backtrack_burst_count="));
    }

    #[test]
    fn thermo_mechanical_nonlinear_emits_coupled_convergence_profile_diagnostic() {
        let model = fixture_model(FixtureId::NonlinearLoadPathMix);
        let result = run_nonlinear_with_options(
            &model,
            ComputeBackend::Cpu,
            NonlinearSolveOptions {
                thermo_mechanical_context: Some(FeaThermoMechanicalContext {
                    enabled: true,
                    reference_temperature_k: 293.15,
                    applied_temperature_delta_k: 90.0,
                    thermal_expansion_coefficient: 1.4e-5,
                    field_source: None,
                    region_temperature_deltas: Vec::new(),
                    time_profile: Vec::new(),
                }),
                ..NonlinearSolveOptions::default()
            },
        )
        .expect("thermo-mechanical nonlinear solve should succeed");

        let profile = result
            .run
            .diagnostics
            .iter()
            .find(|diag| diag.code == "FEA_TM_NONLINEAR")
            .expect("thermo nonlinear profile diagnostic should be present");
        assert!(profile
            .message
            .contains("convergence_residual_target_peak="));
        assert!(profile
            .message
            .contains("convergence_increment_target_peak="));
    }

    #[test]
    fn nonlinear_harder_fixtures_emit_difficulty_profile_signals() {
        for fixture in [
            FixtureId::NonlinearSofteningProxy,
            FixtureId::NonlinearLoadPathMix,
        ] {
            let model = fixture_model(fixture);
            let result =
                run_nonlinear(&model, ComputeBackend::Cpu).expect("hard nonlinear fixture solves");
            assert!(!result.load_factors.is_empty());
            assert!(result.backtrack_burst_count > 0);
            assert!(result.iteration_spike_count > 0);
            assert!(result.max_line_search_backtracks_per_increment > 0);
            assert!(result
                .run
                .diagnostics
                .iter()
                .any(|diag| diag.code == "FEA_NONLINEAR_CONVERGENCE"));
        }
    }

    #[test]
    fn load_sweep_fixture_uses_operator_solver_path() {
        let baseline = run_linear_static(
            &fixture_model(FixtureId::CantileverLinearStatic),
            ComputeBackend::Cpu,
        )
        .expect("baseline solve should succeed");
        let model = fixture_model(FixtureId::CantileverLoadSweep);
        let result = run_linear_static(&model, ComputeBackend::Cpu).expect("solve should succeed");

        let convergence = result
            .diagnostics
            .iter()
            .find(|diag| diag.code == "FEA_CONVERGENCE")
            .expect("convergence diagnostic should be present");
        assert!(convergence.message.contains("residual_norm="));
        assert!(result
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_SOLVER_METHOD"));
        assert!(result.displacement_field.element_count() >= 384);

        let baseline_max = baseline
            .displacement_field
            .as_host_f64()
            .expect("baseline displacement should be host-backed")
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let sweep_max = result
            .displacement_field
            .as_host_f64()
            .expect("sweep displacement should be host-backed")
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        assert!(sweep_max > baseline_max);
    }

    #[test]
    fn large_load_sweep_fixture_scales_dof_count() {
        let model = fixture_model(FixtureId::CantileverLargeLoadSweep);
        let result = run_linear_static(&model, ComputeBackend::Cpu).expect("solve should succeed");

        assert!(result.displacement_field.element_count() >= 1536);
        assert!(result
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_SOLVER_METHOD"));
    }

    #[test]
    fn multi_material_fixture_has_distinct_response_profile() {
        let baseline = run_linear_static(
            &fixture_model(FixtureId::CantileverLinearStatic),
            ComputeBackend::Cpu,
        )
        .expect("baseline solve should succeed");
        let multi_material = run_linear_static(
            &fixture_model(FixtureId::MultiMaterialAssembly),
            ComputeBackend::Cpu,
        )
        .expect("multi-material solve should succeed");

        assert!(multi_material.displacement_field.element_count() >= 9);
        assert!(multi_material
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_SOLVER_METHOD"));

        let baseline_peak = baseline
            .displacement_field
            .as_host_f64()
            .expect("baseline displacement should be host-backed")
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let multi_peak = multi_material
            .displacement_field
            .as_host_f64()
            .expect("multi displacement should be host-backed")
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(multi_peak > baseline_peak);

        assert!(multi_material
            .diagnostics
            .iter()
            .any(|diag| diag.code == "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_INFERRED"));
    }
}
