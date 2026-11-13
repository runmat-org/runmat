//! Typed core validation failures at the runtime reporting boundary.
use super::{ANALYSIS_VALIDATE_OPERATION, ANALYSIS_VALIDATE_OP_VERSION};
use crate::operations::{
    operation_error, OperationContext, OperationErrorEnvelope, OperationErrorSeverity,
    OperationErrorSpec, OperationErrorType,
};
use runmat_analysis_core::{AnalysisModel, AnalysisValidationError};
use std::collections::BTreeMap;

pub(super) fn map_validate_error(
    error: AnalysisValidationError,
    model: &AnalysisModel,
    context: &OperationContext,
) -> OperationErrorEnvelope {
    let error_type = match &error {
        AnalysisValidationError::DependencyGraphResourceLimit { .. }
        | AnalysisValidationError::DependencyGraphResourceOverflow => OperationErrorType::Capacity,
        _ => OperationErrorType::Validation,
    };
    let (error_code, message, mut error_context) = match error {
        AnalysisValidationError::MissingMaterials => (
            "RM.FEA.VALIDATE.MISSING_MATERIALS",
            "FEA model must include at least one material".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::MissingConstraintsOrBoundaries => (
            "RM.FEA.VALIDATE.MISSING_CONDITIONS",
            "FEA model must include at least one constraint or natural boundary".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::InvalidId { kind } => (
            "RM.FEA.VALIDATE.INVALID_ID",
            format!("{kind} id must be non-empty"),
            BTreeMap::from([("kind".to_string(), kind.to_string())]),
        ),
        AnalysisValidationError::DuplicateId { kind, id } => (
            "RM.FEA.VALIDATE.DUPLICATE_ID",
            format!("duplicate {kind} id `{id}`"),
            BTreeMap::from([
                ("kind".to_string(), kind.to_string()),
                ("id".to_string(), id),
            ]),
        ),
        AnalysisValidationError::InvalidTarget { owner_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_TARGET",
            format!("{owner_id} has an invalid target: {detail}"),
            BTreeMap::from([
                ("owner_id".to_string(), owner_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidFrame => (
            "RM.FEA.VALIDATE.INVALID_FRAME",
            "coordinate frame must be finite and non-degenerate".to_string(),
            BTreeMap::new(),
        ),
        AnalysisValidationError::InvalidMaterial {
            material_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_MATERIAL",
            format!("material `{material_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("material_id".to_string(), material_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidMaterialAssignment { region_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_MATERIAL_ASSIGNMENT",
            format!("material assignment for region `{region_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("region_id".to_string(), region_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidConstraint {
            constraint_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_CONSTRAINT",
            format!("constraint `{constraint_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("constraint_id".to_string(), constraint_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidBoundary {
            boundary_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_BOUNDARY",
            format!("boundary `{boundary_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("boundary_id".to_string(), boundary_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidConnector {
            connector_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_CONNECTOR",
            format!("connector `{connector_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("connector_id".to_string(), connector_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidInterface {
            interface_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_INTERFACE",
            format!("interface `{interface_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("interface_id".to_string(), interface_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidSignal { signal_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_SIGNAL",
            format!("signal `{signal_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("signal_id".to_string(), signal_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidDomain { domain, detail } => (
            "RM.FEA.VALIDATE.INVALID_DOMAIN",
            format!("{domain} domain is invalid: {detail}"),
            BTreeMap::from([
                ("domain".to_string(), domain.to_string()),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidExcitation {
            excitation_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_EXCITATION",
            format!("excitation `{excitation_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("excitation_id".to_string(), excitation_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidExcitationCase { case_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_EXCITATION_CASE",
            format!("excitation case `{case_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("case_id".to_string(), case_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidInitialCondition {
            initial_condition_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_INITIAL_CONDITION",
            format!("initial condition `{initial_condition_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("initial_condition_id".to_string(), initial_condition_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidResponseSpectrum {
            spectrum_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_RESPONSE_SPECTRUM",
            format!("response spectrum `{spectrum_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("spectrum_id".to_string(), spectrum_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidRandomExcitation {
            excitation_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_RANDOM_EXCITATION",
            format!("random excitation `{excitation_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("excitation_id".to_string(), excitation_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidSequence {
            sequence_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_SEQUENCE",
            format!("analysis sequence `{sequence_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("sequence_id".to_string(), sequence_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidProcedureStep { step_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_PROCEDURE_STEP",
            format!("procedure step `{step_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("step_id".to_string(), step_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidStateTransition {
            transition_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_STATE_TRANSITION",
            format!("state transition `{transition_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("transition_id".to_string(), transition_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidObservation {
            observation_id,
            detail,
        } => (
            "RM.FEA.VALIDATE.INVALID_OBSERVATION",
            format!("observation `{observation_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("observation_id".to_string(), observation_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidMonitor { monitor_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_MONITOR",
            format!("monitor `{monitor_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("monitor_id".to_string(), monitor_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::InvalidAccuracyPlan { plan_id, detail } => (
            "RM.FEA.VALIDATE.INVALID_ACCURACY_PLAN",
            format!("accuracy plan `{plan_id}` is invalid: {detail}"),
            BTreeMap::from([
                ("plan_id".to_string(), plan_id),
                ("detail".to_string(), detail),
            ]),
        ),
        AnalysisValidationError::DependencyGraphResourceLimit { requested_bytes, maximum_bytes } => (
            "RM.FEA.VALIDATE.DEPENDENCY_GRAPH_RESOURCE_LIMIT",
            format!("dependency graph requests {requested_bytes} bytes, exceeding its {maximum_bytes}-byte workspace budget"),
            BTreeMap::from([("requested_bytes".into(), requested_bytes.to_string()), ("maximum_bytes".into(), maximum_bytes.to_string())]),
        ),
        AnalysisValidationError::DependencyGraphResourceOverflow => (
            "RM.FEA.VALIDATE.DEPENDENCY_GRAPH_RESOURCE_OVERFLOW",
            "dependency graph workspace inventory overflows".into(), BTreeMap::new(),
        ),
        AnalysisValidationError::FrameMismatch { model, geometry } => (
            "RM.FEA.VALIDATE.FRAME_MISMATCH",
            format!("model frame {model:?} does not match geometry frame {geometry:?}"),
            BTreeMap::from([
                ("model_frame".to_string(), model.context_label()),
                ("geometry_frame".to_string(), geometry.context_label()),
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
            error_type,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        message,
        error_context,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dependency_capacity_diagnostics_preserve_typed_budget_and_model_context() {
        let model = runmat_analysis_fea::fixtures::fixture_model(
            runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
        );
        let context =
            OperationContext::new(Some("graph-capacity".into()), Some("request-graph".into()));
        let error = map_validate_error(
            AnalysisValidationError::DependencyGraphResourceLimit {
                requested_bytes: 4096,
                maximum_bytes: 4095,
            },
            &model,
            &context,
        );
        assert_eq!(error.error_type, OperationErrorType::Capacity);
        assert_eq!(
            error.error_code,
            "RM.FEA.VALIDATE.DEPENDENCY_GRAPH_RESOURCE_LIMIT"
        );
        assert_eq!(error.context["requested_bytes"], "4096");
        assert_eq!(error.context["maximum_bytes"], "4095");
        assert_eq!(error.context["analysis_model_id"], model.model_id.0);
        assert!(!error.retryable);
        let overflow = map_validate_error(
            AnalysisValidationError::DependencyGraphResourceOverflow,
            &model,
            &context,
        );
        assert_eq!(overflow.error_type, OperationErrorType::Capacity);
        assert_eq!(
            overflow.error_code,
            "RM.FEA.VALIDATE.DEPENDENCY_GRAPH_RESOURCE_OVERFLOW"
        );
        let legacy =
            map_validate_error(AnalysisValidationError::MissingMaterials, &model, &context);
        assert_eq!(legacy.error_type, OperationErrorType::Validation);
        assert_eq!(legacy.error_code, "RM.FEA.VALIDATE.MISSING_MATERIALS");
    }
}
