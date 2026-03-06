use std::collections::BTreeMap;

use runmat_analysis_core::{
    validate_model_against_geometry, AnalysisModel, AnalysisValidationError, ReferenceFrame,
};
use runmat_analysis_fea::{run_linear_static, ComputeBackend};
use runmat_geometry_core::UnitSystem;

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

mod contracts;
mod promotion;

pub use contracts::{
    AnalysisRunOptions, AnalysisRunResult, AnalysisValidateResult, PrecisionMode, QualityGate,
    RunProvenance, RunStatus,
};

const ANALYSIS_VALIDATE_OPERATION: &str = "analysis.validate";
const ANALYSIS_VALIDATE_OP_VERSION: &str = "analysis.validate/v1";
const ANALYSIS_RUN_OPERATION: &str = "analysis.run_linear_static";
const ANALYSIS_RUN_OP_VERSION: &str = "analysis.run_linear_static/v1";

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
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    analysis_run_linear_static_with_options(model, backend, AnalysisRunOptions::default(), context)
}

pub fn analysis_run_linear_static_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let run = run_linear_static(model, backend).map_err(|err| {
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

    let mut run = run;
    let mut fallback_events = Vec::new();
    promotion::promote_run_fields_to_device_refs(&mut run, &mut fallback_events);

    let solver_convergence = if run.diagnostics.iter().any(|item| {
        item.code == "FEA_CONVERGENCE"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    }) {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };

    let result_quality = if !run.displacement_field.is_empty() && !run.von_mises_field.is_empty() {
        QualityGate::Pass
    } else {
        QualityGate::Fail
    };

    let publishable =
        solver_convergence == QualityGate::Pass && result_quality == QualityGate::Pass;
    let run_status = if publishable {
        RunStatus::Publishable
    } else if result_quality == QualityGate::Fail {
        RunStatus::Rejected
    } else {
        RunStatus::Degraded
    };

    let result = AnalysisRunResult {
        run,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        publishable,
        provenance: RunProvenance {
            backend,
            precision_mode: contracts::format_precision_mode(options.precision_mode),
            deterministic_mode: options.deterministic_mode,
            fallback_events,
        },
    };

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
mod tests;
