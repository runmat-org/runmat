//! FEA pipeline scaffolding for assembly, linear solve, and field postprocessing.

pub mod assembly;
pub mod diagnostics;
pub mod post;
pub mod solve;

use runmat_analysis_core::{validate_model, AnalysisModel};
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
    pub displacement_field: Vec<f64>,
    pub von_mises_field: Vec<f64>,
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
    use runmat_analysis_core::{
        AnalysisModel, AnalysisModelId, AnalysisStep, AnalysisStepKind, BoundaryCondition,
        BoundaryConditionKind, LoadCase, LoadKind, MaterialModel, ReferenceFrame,
    };
    use runmat_geometry_core::UnitSystem;

    use super::*;

    fn cantilever_model() -> AnalysisModel {
        AnalysisModel {
            model_id: AnalysisModelId("cantilever".to_string()),
            geometry_id: "geo:cantilever".to_string(),
            geometry_revision: 1,
            units: UnitSystem::Meter,
            frame: ReferenceFrame::Global,
            materials: vec![MaterialModel {
                material_id: "mat_steel".to_string(),
                name: "Steel".to_string(),
                youngs_modulus_pa: 200e9,
                poisson_ratio: 0.3,
            }],
            boundary_conditions: vec![BoundaryCondition {
                bc_id: "bc_root".to_string(),
                region_id: "root".to_string(),
                kind: BoundaryConditionKind::Fixed,
            }],
            loads: vec![LoadCase {
                load_id: "tip_load".to_string(),
                region_id: "tip".to_string(),
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -1000.0,
                    fz: 0.0,
                },
            }],
            steps: vec![AnalysisStep {
                step_id: "static_1".to_string(),
                kind: AnalysisStepKind::Static,
            }],
        }
    }

    #[test]
    fn canonical_cantilever_benchmark_runs() {
        let model = cantilever_model();
        let result = run_linear_static(&model, ComputeBackend::Cpu).expect("solve should succeed");

        assert_eq!(result.displacement_field.len(), 3);
        assert_eq!(result.von_mises_field.len(), 1);
        assert!(result.displacement_field[1] < 0.0);
    }

    #[test]
    fn convergence_diagnostics_are_emitted() {
        let model = cantilever_model();
        let result = run_linear_static(&model, ComputeBackend::Cpu).expect("solve should succeed");
        assert!(result
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_CONVERGENCE"));
    }

    #[test]
    fn cpu_gpu_parity_for_scaffold_pipeline() {
        let model = cantilever_model();
        let cpu = run_linear_static(&model, ComputeBackend::Cpu).expect("cpu run should succeed");
        let gpu = run_linear_static(&model, ComputeBackend::Gpu).expect("gpu run should succeed");

        assert_eq!(cpu.displacement_field, gpu.displacement_field);
        assert_eq!(cpu.von_mises_field, gpu.von_mises_field);
    }
}
