//! Typed thermal assembly and physical radiation evaluation failures.
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub(crate) enum ThermalAssemblyError {
    #[error("thermal assembly numerical accounting failed: {0}")]
    Accounting(#[from] runmat_simulation_usage::LedgerError),
    #[error("thermal sparse operator construction failed: {0}")]
    SparseOperator(String),
    #[error("thermal analysis projection is invalid: {0}")]
    InvalidProjection(String),
    #[error("volume region {region_id} has no thermal material assignment")]
    MissingMaterial { region_id: String },
    #[error("thermal material coverage is invalid: {0}")]
    InvalidMaterialCoverage(String),
    #[error("thermal excitation {excitation_id} uses an unsupported complex ordinate")]
    ComplexSource { excitation_id: String },
    #[error("thermal excitation {excitation_id} does not resolve any {entity_kind}")]
    UnmappedSource {
        excitation_id: String,
        entity_kind: &'static str,
    },
    #[error("thermal excitation {excitation_id} failed scalar conservation with relative error {relative_error} above tolerance {relative_tolerance}")]
    ProjectionConservation {
        excitation_id: String,
        relative_error: f64,
        relative_tolerance: f64,
    },
    #[error(
        "thermal boundary or constraint {operator_id} signal could not be realized: {message}"
    )]
    SignalRealization {
        operator_id: String,
        message: String,
    },
    #[error("thermal boundary or constraint {operator_id} uses an unsupported complex ordinate")]
    ComplexBoundary { operator_id: String },
    #[error("thermal boundary or constraint {operator_id} does not resolve any boundary faces")]
    UnmappedBoundary { operator_id: String },
    #[error("thermal radiation boundary {operator_id} requires the nonlinear radiation solver, which is not yet admitted")]
    RadiationNotAdmitted { operator_id: String },
    #[error(transparent)]
    Radiation(#[from] crate::physics::thermal::radiation::RadiationError),
    #[error("thermal constraints prescribe conflicting temperatures at solver node {node_id}")]
    ConflictingTemperature { node_id: u64 },
    #[error("temperature-field excitation {excitation_id} uses an unsupported complex ordinate")]
    ComplexTemperatureField { excitation_id: String },
    #[error("temperature-field excitation {excitation_id} does not resolve any solver nodes")]
    UnmappedTemperatureField { excitation_id: String },
    #[error(
        "temperature-field excitations {first_excitation_id} and {second_excitation_id} prescribe conflicting temperatures at {location}"
    )]
    ConflictingTemperatureFields {
        first_excitation_id: String,
        second_excitation_id: String,
        location: String,
    },
}

/// A projection traversal that ran out of simulation work keeps its usage type; every other
/// projection failure is an invalid projection.
impl From<crate::topology::AnalysisProjectionError> for ThermalAssemblyError {
    fn from(error: crate::topology::AnalysisProjectionError) -> Self {
        match error {
            crate::topology::AnalysisProjectionError::Accounting(cause) => Self::Accounting(cause),
            error => Self::InvalidProjection(error.to_string()),
        }
    }
}
