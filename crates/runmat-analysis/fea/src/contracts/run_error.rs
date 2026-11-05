//! Typed failures at the public FEA run boundary.

use thiserror::Error;

use super::SolverFormulationCompatibilityError;

#[derive(Debug, Error)]
pub enum FeaRunError {
    #[error("FEA_NUMERICAL_USAGE: {0}")]
    Usage(#[source] runmat_simulation_usage::LedgerError),
    #[error("FEA_MODEL_INVALID: {0}")]
    InvalidModel(String),
    #[error("FEA_ASSEMBLY_FAILED: {0}")]
    Assembly(String),
    #[error("FEA_SOLVE_FAILED: {0}")]
    Solve(String),
    #[error("FEA_OBSERVATION_FAILED: {0}")]
    Observation(String),
    #[error(transparent)]
    Resource(#[from] crate::planning::resource::ResourcePlanningError),
    #[error(transparent)]
    FrequencyPlan(#[from] crate::frequency::FrequencyPlanError),
    #[error("FEA_ELECTROMAGNETIC_SOURCE_MISSING: no nonzero source at {frequency_hz} Hz")]
    MissingElectromagneticSource { frequency_hz: f64 },
    #[error(transparent)]
    FormulationAdmission(#[from] crate::admission::FormulationAdmissionRejection),
    #[error(transparent)]
    FormulationAvailability(#[from] crate::admission::FormulationAvailabilityRejection),
    #[error(transparent)]
    SolverFormulationCompatibility(#[from] SolverFormulationCompatibilityError),
    #[error(transparent)]
    AcceleratorRequired(#[from] super::accelerator_admission::AcceleratorRequiredError),
    #[error("FEA_CANCELLED: execution cancelled by user")]
    Cancelled,
}

impl From<runmat_simulation_usage::LedgerError> for FeaRunError {
    fn from(error: runmat_simulation_usage::LedgerError) -> Self {
        Self::Usage(error)
    }
}

impl From<crate::post::StructuralFieldRecoveryError> for FeaRunError {
    fn from(error: crate::post::StructuralFieldRecoveryError) -> Self {
        use crate::{
            assembly::elements::{
                beam::BeamElementError,
                shell::ShellElementError,
                solid::{
                    material::SolidMaterialError, Tetrahedron10ElementError,
                    Tetrahedron4ElementError,
                },
            },
            post::StructuralFieldRecoveryError as Recovery,
        };
        match error {
            Recovery::Accounting(cause)
            | Recovery::Beam {
                source: BeamElementError::Accounting(cause),
                ..
            }
            | Recovery::Tetrahedron4 {
                source: Tetrahedron4ElementError::Accounting(cause),
                ..
            }
            | Recovery::Tetrahedron4 {
                source:
                    Tetrahedron4ElementError::InvalidMaterial(SolidMaterialError::Accounting(cause)),
                ..
            }
            | Recovery::Tetrahedron10 {
                source: Tetrahedron10ElementError::Accounting(cause),
                ..
            }
            | Recovery::Tetrahedron10 {
                source:
                    Tetrahedron10ElementError::InvalidMaterial(SolidMaterialError::Accounting(cause)),
                ..
            }
            | Recovery::Shell {
                source: ShellElementError::Accounting(cause),
                ..
            } => Self::Usage(cause),
            Recovery::Cancelled => Self::Cancelled,
            other => Self::Assembly(other.to_string()),
        }
    }
}

impl From<crate::constraints::ConstraintResultError> for FeaRunError {
    fn from(error: crate::constraints::ConstraintResultError) -> Self {
        match error {
            crate::constraints::ConstraintResultError::Accounting(cause) => Self::Usage(cause),
            crate::constraints::ConstraintResultError::Cancelled => Self::Cancelled,
            other => Self::Assembly(other.to_string()),
        }
    }
}
