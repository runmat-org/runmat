//! Typed failures at the public FEA run boundary.

use thiserror::Error;

use super::SolverFormulationCompatibilityError;

#[derive(Debug, Error)]
pub enum FeaRunError {
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
    #[error("FEA_CANCELLED: execution cancelled by user")]
    Cancelled,
}
