use std::collections::BTreeMap;

use runmat_analysis_core::{
    validate_model_against_geometry, AnalysisModel, AnalysisValidationError, ReferenceFrame,
};
use runmat_analysis_fea::{run_linear_static, ComputeBackend, FeaRunResult};
use runmat_geometry_core::UnitSystem;

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

const ANALYSIS_VALIDATE_OPERATION: &str = "analysis.validate";
const ANALYSIS_VALIDATE_OP_VERSION: &str = "analysis.validate/v1";
const ANALYSIS_RUN_OPERATION: &str = "analysis.run_linear_static";
const ANALYSIS_RUN_OP_VERSION: &str = "analysis.run_linear_static/v1";

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisValidateResult {
    pub valid: bool,
}

pub fn analysis_validate(
    model: &AnalysisModel,
    geometry_units: UnitSystem,
    geometry_frame: &ReferenceFrame,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisValidateResult>, OperationErrorEnvelope> {
    validate_model_against_geometry(model, geometry_units, geometry_frame)
        .map_err(|err| map_validate_error(err, model, &context))?;

    Ok(OperationEnvelope::new(
        ANALYSIS_VALIDATE_OPERATION,
        ANALYSIS_VALIDATE_OP_VERSION,
        &context,
        AnalysisValidateResult { valid: true },
    ))
}

pub fn analysis_run_linear_static_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    context: OperationContext,
) -> Result<OperationEnvelope<FeaRunResult>, OperationErrorEnvelope> {
    let result = run_linear_static(model, backend).map_err(|err| {
        operation_error(
            ANALYSIS_RUN_OPERATION,
            ANALYSIS_RUN_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "SOLVER_MODEL_INVALID",
                error_type: OperationErrorType::Validation,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            err.to_string(),
            BTreeMap::from([
                ("analysis_model_id".to_string(), model.model_id.0.clone()),
                ("geometry_id".to_string(), model.geometry_id.clone()),
            ]),
        )
    })?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_OPERATION,
        ANALYSIS_RUN_OP_VERSION,
        &context,
        result,
    ))
}

fn map_validate_error(
    error: AnalysisValidationError,
    model: &AnalysisModel,
    context: &OperationContext,
) -> OperationErrorEnvelope {
    let (error_code, message, mut error_context) = match error {
        AnalysisValidationError::MissingMaterials => (
            "ANALYSIS_VALIDATION_MISSING_MATERIALS",
            "analysis model must include at least one material".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::MissingBoundaryConditions => (
            "ANALYSIS_VALIDATION_MISSING_BCS",
            "analysis model must include at least one boundary condition".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::MissingLoads => (
            "ANALYSIS_VALIDATION_MISSING_LOADS",
            "analysis model must include at least one load".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::UnitMismatch { model, geometry } => (
            "ANALYSIS_VALIDATION_UNIT_MISMATCH",
            format!("model units {model:?} do not match geometry units {geometry:?}"),
            BTreeMap::from([
                ("model_units".to_string(), format!("{model:?}")),
                ("geometry_units".to_string(), format!("{geometry:?}")),
            ]),
        ),
        AnalysisValidationError::FrameMismatch { model, geometry } => (
            "ANALYSIS_VALIDATION_FRAME_MISMATCH",
            format!("model frame {model:?} does not match geometry frame {geometry:?}"),
            BTreeMap::from([
                ("model_frame".to_string(), format!("{model:?}")),
                ("geometry_frame".to_string(), format!("{geometry:?}")),
            ]),
        ),
    };

    error_context.insert("analysis_model_id".to_string(), model.model_id.0.clone());
    error_context.insert("geometry_id".to_string(), model.geometry_id.clone());

    operation_error(
        ANALYSIS_VALIDATE_OPERATION,
        ANALYSIS_VALIDATE_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code,
            error_type: OperationErrorType::Validation,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        message,
        error_context,
    )
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AnalysisModel, AnalysisModelId, AnalysisStep, AnalysisStepKind, BoundaryCondition,
        BoundaryConditionKind, LoadCase, LoadKind, MaterialModel,
    };

    use super::*;

    fn sample_model() -> AnalysisModel {
        AnalysisModel {
            model_id: AnalysisModelId("beam_model".to_string()),
            geometry_id: "geo:beam".to_string(),
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
                load_id: "load_tip".to_string(),
                region_id: "tip".to_string(),
                kind: LoadKind::Force {
                    fx: 0.0,
                    fy: -1000.0,
                    fz: 0.0,
                },
            }],
            steps: vec![AnalysisStep {
                step_id: "step_static".to_string(),
                kind: AnalysisStepKind::Static,
            }],
        }
    }

    #[test]
    fn analysis_validate_returns_typed_envelope() {
        let model = sample_model();
        let context =
            OperationContext::new(Some("trace-a1".to_string()), Some("request-a1".to_string()));
        let envelope =
            analysis_validate(&model, UnitSystem::Meter, &ReferenceFrame::Global, context)
                .expect("validation should pass");

        assert_eq!(envelope.operation, "analysis.validate");
        assert_eq!(envelope.op_version, "analysis.validate/v1");
        assert!(envelope.data.valid);
        assert_eq!(envelope.trace_id.as_deref(), Some("trace-a1"));
    }

    #[test]
    fn analysis_validate_maps_typed_error_code() {
        let mut model = sample_model();
        model.materials.clear();
        let context = OperationContext::new(None, None);
        let error = analysis_validate(&model, UnitSystem::Meter, &ReferenceFrame::Global, context)
            .expect_err("validation should fail");

        assert_eq!(error.error_code, "ANALYSIS_VALIDATION_MISSING_MATERIALS");
        assert_eq!(error.operation, "analysis.validate");
        assert_eq!(error.op_version, "analysis.validate/v1");
    }

    #[test]
    fn analysis_run_linear_static_returns_typed_envelope() {
        let model = sample_model();
        let context =
            OperationContext::new(Some("trace-a2".to_string()), Some("request-a2".to_string()));
        let envelope = analysis_run_linear_static_op(&model, ComputeBackend::Cpu, context)
            .expect("run should pass");

        assert_eq!(envelope.operation, "analysis.run_linear_static");
        assert_eq!(envelope.op_version, "analysis.run_linear_static/v1");
        assert_eq!(envelope.data.backend, ComputeBackend::Cpu);
        assert!(!envelope.data.displacement_field.is_empty());
    }
}
