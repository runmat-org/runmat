//! FEA pipeline scaffolding for assembly, linear solve, and field postprocessing.

pub mod assembly;
pub mod diagnostics;
pub mod fixtures;
pub mod operator;
pub mod parity;
pub mod post;
pub mod solve;

use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    assembly::assemble_linear_system,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    post::fields::recover_result_fields,
    solve::{
        backend::{build_backend, kind::LinearAlgebraBackendKind},
        linear::solve_linear_system,
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
    pub solver_method: String,
    pub preconditioner: String,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub displacement_field: AnalysisField,
    pub von_mises_field: AnalysisField,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearStaticSolveOptions {
    pub preconditioner_kind: SpdPreconditionerKind,
    pub algebra_backend_kind: LinearAlgebraBackendKind,
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
    diagnostics.extend(solve_result.diagnostics);

    Ok(FeaRunResult {
        backend,
        solver_backend: options.algebra_backend_kind.as_str().to_string(),
        solver_method: solve_result.solver_method,
        preconditioner: solve_result.preconditioner,
        diagnostics,
        displacement_field: fields.displacement_field,
        von_mises_field: fields.von_mises_field,
    })
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
}
