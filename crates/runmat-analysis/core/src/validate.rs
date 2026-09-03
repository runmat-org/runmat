use runmat_geometry_core::UnitSystem;
use thiserror::Error;

use crate::problem::{coordinate_frame::CoordinateFrame, loads::LoadKind, model::AnalysisModel};

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AnalysisValidationError {
    #[error(
        "ANALYSIS_VALIDATION_MISSING_MATERIALS: analysis model must include at least one material"
    )]
    MissingMaterials,
    #[error("ANALYSIS_VALIDATION_MISSING_CONDITIONS: analysis model must include at least one constraint or natural boundary")]
    MissingConstraintsOrBoundaries,
    #[error("ANALYSIS_VALIDATION_MISSING_LOADS: analysis model must include at least one load")]
    MissingLoads,
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
    #[error(
        "ANALYSIS_VALIDATION_INVALID_MOMENT: moment load {load_id} must have finite components"
    )]
    InvalidMomentVector { load_id: String },
    #[error("ANALYSIS_VALIDATION_ZERO_MOMENT: moment load {load_id} must have nonzero magnitude")]
    ZeroMomentVector { load_id: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_WRENCH: wrench load {load_id} must have finite force, moment, and point components"
    )]
    InvalidWrench { load_id: String },
    #[error(
        "ANALYSIS_VALIDATION_ZERO_WRENCH: wrench load {load_id} must have nonzero force or moment"
    )]
    ZeroWrench { load_id: String },
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
    if model.loads.is_empty() {
        return Err(AnalysisValidationError::MissingLoads);
    }
    crate::validate_authored::validate_authored_model(model)?;
    for load in &model.loads {
        if let LoadKind::Moment { mx, my, mz } = load.kind {
            if !mx.is_finite() || !my.is_finite() || !mz.is_finite() {
                return Err(AnalysisValidationError::InvalidMomentVector {
                    load_id: load.load_id.clone(),
                });
            }
            if mx == 0.0 && my == 0.0 && mz == 0.0 {
                return Err(AnalysisValidationError::ZeroMomentVector {
                    load_id: load.load_id.clone(),
                });
            }
        }
        if let LoadKind::Wrench {
            fx,
            fy,
            fz,
            mx,
            my,
            mz,
            px,
            py,
            pz,
        } = load.kind
        {
            if !fx.is_finite()
                || !fy.is_finite()
                || !fz.is_finite()
                || !mx.is_finite()
                || !my.is_finite()
                || !mz.is_finite()
                || !px.is_finite()
                || !py.is_finite()
                || !pz.is_finite()
            {
                return Err(AnalysisValidationError::InvalidWrench {
                    load_id: load.load_id.clone(),
                });
            }
            if fx == 0.0 && fy == 0.0 && fz == 0.0 && mx == 0.0 && my == 0.0 && mz == 0.0 {
                return Err(AnalysisValidationError::ZeroWrench {
                    load_id: load.load_id.clone(),
                });
            }
        }
    }
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
