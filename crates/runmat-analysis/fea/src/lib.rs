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
    solve::linear::solve_linear_system,
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
    pub diagnostics: Vec<FeaDiagnostic>,
    pub displacement_field: AnalysisField,
    pub von_mises_field: AnalysisField,
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
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;

    let summary = assemble_linear_system(model);
    let solve_result = solve_linear_system(&summary);
    let fields = recover_result_fields(&summary, &solve_result);

    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_CONVERGENCE".to_string(),
        severity: if solve_result.converged {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "iterations={} residual_norm={} converged={}",
            solve_result.iterations, solve_result.residual_norm, solve_result.converged
        ),
    }];
    diagnostics.extend(solve_result.diagnostics);

    Ok(FeaRunResult {
        backend,
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
}
