//! FEA pipeline scaffolding for assembly, linear solve, and field postprocessing.

pub mod assembly;
pub mod diagnostics;
pub mod fixtures;
pub mod operator;
pub mod parity;
pub mod post;
pub mod solve;

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
        preconditioner::SpdPreconditionerKind,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearStaticSolveOptions {
    pub preconditioner_kind: SpdPreconditionerKind,
    pub algebra_backend_kind: LinearAlgebraBackendKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModalSolveOptions {
    pub mode_count: usize,
}

impl Default for ModalSolveOptions {
    fn default() -> Self {
        Self { mode_count: 3 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaModalRunResult {
    pub run: FeaRunResult,
    pub eigenvalues_hz: Vec<f64>,
    pub mode_shapes: Vec<AnalysisField>,
}

impl Default for LinearStaticSolveOptions {
    fn default() -> Self {
        Self {
            preconditioner_kind: SpdPreconditionerKind::Jacobi,
            algebra_backend_kind: LinearAlgebraBackendKind::CpuReference,
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

    let summary = assemble_linear_system(model);
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

    let summary = assemble_linear_system(model);
    let modal = solve_modal_system(&summary, options.mode_count);
    let mut diagnostics = modal.diagnostics.clone();
    diagnostics.extend(material_assignment_diagnostics(&model.material_assignments));

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
        solver_backend: "cpu_reference".to_string(),
        solver_device_apply_k_ratio: 0.0,
        solver_method: modal.solver_method,
        preconditioner: "none".to_string(),
        solver_host_sync_count: 0,
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
