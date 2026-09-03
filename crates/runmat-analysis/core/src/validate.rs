use runmat_geometry_core::UnitSystem;
use thiserror::Error;

use crate::problem::{coordinate_frame::CoordinateFrame, model::AnalysisModel};

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AnalysisValidationError {
    #[error(
        "ANALYSIS_VALIDATION_MISSING_MATERIALS: analysis model must include at least one material"
    )]
    MissingMaterials,
    #[error("ANALYSIS_VALIDATION_MISSING_CONDITIONS: analysis model must include at least one constraint or natural boundary")]
    MissingConstraintsOrBoundaries,
    #[error("ANALYSIS_VALIDATION_INVALID_ID: {kind} id must be non-empty")]
    InvalidId { kind: &'static str },
    #[error("ANALYSIS_VALIDATION_DUPLICATE_ID: duplicate {kind} id `{id}`")]
    DuplicateId { kind: &'static str, id: String },
    #[error("ANALYSIS_VALIDATION_INVALID_TARGET: {owner_id} has an invalid target: {detail}")]
    InvalidTarget { owner_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_FRAME: coordinate frame is not finite and non-degenerate"
    )]
    InvalidFrame,
    #[error(
        "ANALYSIS_VALIDATION_INVALID_CONSTRAINT: constraint `{constraint_id}` is invalid: {detail}"
    )]
    InvalidConstraint {
        constraint_id: String,
        detail: String,
    },
    #[error("ANALYSIS_VALIDATION_INVALID_BOUNDARY: boundary `{boundary_id}` is invalid: {detail}")]
    InvalidBoundary { boundary_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_CONNECTOR: connector `{connector_id}` is invalid: {detail}"
    )]
    InvalidConnector {
        connector_id: String,
        detail: String,
    },
    #[error("ANALYSIS_VALIDATION_INVALID_SIGNAL: signal `{signal_id}` is invalid: {detail}")]
    InvalidSignal { signal_id: String, detail: String },
    #[error("ANALYSIS_VALIDATION_INVALID_DOMAIN: {domain} domain is invalid: {detail}")]
    InvalidDomain {
        domain: &'static str,
        detail: String,
    },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_EXCITATION: excitation `{excitation_id}` is invalid: {detail}"
    )]
    InvalidExcitation {
        excitation_id: String,
        detail: String,
    },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_EXCITATION_CASE: excitation case `{case_id}` is invalid: {detail}"
    )]
    InvalidExcitationCase { case_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_INITIAL_CONDITION: initial condition `{initial_condition_id}` is invalid: {detail}"
    )]
    InvalidInitialCondition {
        initial_condition_id: String,
        detail: String,
    },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_RESPONSE_SPECTRUM: response spectrum `{spectrum_id}` is invalid: {detail}"
    )]
    InvalidResponseSpectrum { spectrum_id: String, detail: String },
    #[error("ANALYSIS_VALIDATION_INVALID_SEQUENCE: sequence `{sequence_id}` is invalid: {detail}")]
    InvalidSequence { sequence_id: String, detail: String },
    #[error("ANALYSIS_VALIDATION_INVALID_PROCEDURE_STEP: procedure step `{step_id}` is invalid: {detail}")]
    InvalidProcedureStep { step_id: String, detail: String },
    #[error("ANALYSIS_VALIDATION_INVALID_STATE_TRANSITION: state transition `{transition_id}` is invalid: {detail}")]
    InvalidStateTransition {
        transition_id: String,
        detail: String,
    },
    #[error("ANALYSIS_VALIDATION_INVALID_OBSERVATION: observation `{observation_id}` is invalid: {detail}")]
    InvalidObservation {
        observation_id: String,
        detail: String,
    },
    #[error("ANALYSIS_VALIDATION_INVALID_MONITOR: monitor `{monitor_id}` is invalid: {detail}")]
    InvalidMonitor { monitor_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_ACCURACY_PLAN: accuracy plan `{plan_id}` is invalid: {detail}"
    )]
    InvalidAccuracyPlan { plan_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_UNIT_MISMATCH: model units {model:?} do not match geometry units {geometry:?}"
    )]
    UnitMismatch {
        model: UnitSystem,
        geometry: UnitSystem,
    },
    #[error(
        "ANALYSIS_VALIDATION_FRAME_MISMATCH: model frame {model:?} does not match geometry frame {geometry:?}"
    )]
    FrameMismatch {
        model: Box<CoordinateFrame>,
        geometry: Box<CoordinateFrame>,
    },
}

pub fn validate_model(model: &AnalysisModel) -> Result<(), AnalysisValidationError> {
    if model.materials.is_empty() {
        return Err(AnalysisValidationError::MissingMaterials);
    }
    if model.constraints.is_empty() && model.boundaries.is_empty() {
        return Err(AnalysisValidationError::MissingConstraintsOrBoundaries);
    }
    crate::validate_authored::validate_authored_model(model)?;
    Ok(())
}

pub fn validate_model_against_geometry(
    model: &AnalysisModel,
    geometry_units: UnitSystem,
    geometry_frame: &CoordinateFrame,
) -> Result<(), AnalysisValidationError> {
    validate_model(model)?;

    if model.units != geometry_units {
        return Err(AnalysisValidationError::UnitMismatch {
            model: model.units,
            geometry: geometry_units,
        });
    }
    if &model.frame != geometry_frame {
        return Err(AnalysisValidationError::FrameMismatch {
            model: Box::new(model.frame.clone()),
            geometry: Box::new(geometry_frame.clone()),
        });
    }
    Ok(())
}
