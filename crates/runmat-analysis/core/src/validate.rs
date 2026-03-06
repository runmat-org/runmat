use runmat_geometry_core::UnitSystem;
use thiserror::Error;

use crate::problem::model::{AnalysisModel, ReferenceFrame};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AnalysisValidationError {
    #[error(
        "ANALYSIS_VALIDATION_MISSING_MATERIALS: analysis model must include at least one material"
    )]
    MissingMaterials,
    #[error("ANALYSIS_VALIDATION_MISSING_BCS: analysis model must include at least one boundary condition")]
    MissingBoundaryConditions,
    #[error("ANALYSIS_VALIDATION_MISSING_LOADS: analysis model must include at least one load")]
    MissingLoads,
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
        model: ReferenceFrame,
        geometry: ReferenceFrame,
    },
}

pub fn validate_model(model: &AnalysisModel) -> Result<(), AnalysisValidationError> {
    if model.materials.is_empty() {
        return Err(AnalysisValidationError::MissingMaterials);
    }
    if model.boundary_conditions.is_empty() {
        return Err(AnalysisValidationError::MissingBoundaryConditions);
    }
    if model.loads.is_empty() {
        return Err(AnalysisValidationError::MissingLoads);
    }
    Ok(())
}

pub fn validate_model_against_geometry(
    model: &AnalysisModel,
    geometry_units: UnitSystem,
    geometry_frame: &ReferenceFrame,
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
            model: model.frame.clone(),
            geometry: geometry_frame.clone(),
        });
    }
    Ok(())
}
