mod cancellation;
mod error;
mod failure;
mod identity;
mod metric;
mod request;
mod tolerance;

pub use cancellation::{CancellationPolicyV2, MeshingCancellationSignal, NeverCancelled};
pub use error::MeshingContractError;
pub use failure::{
    GeometricWitness, MeshingDiagnosticEntry, MeshingDiagnosticValue, MeshingFailure,
    MeshingFailureCategory, MeshingOperationV2, MeshingStageV2, MESHING_FAILURE_SCHEMA_VERSION,
};
pub use identity::{GeometryRevisionRef, PersistentEntityId, PersistentEntityKind, StableDigest};
pub use metric::{
    MetricCombinationRule, MetricContribution, MetricContributionScope, MetricFieldRequestV2,
    MetricSourceKind, MetricTensor3,
};
pub use request::{
    AlgorithmVersionSet, MeshElementOrderV2, MeshingQualityTargetsV2, MeshingRequestV2,
    MeshingResourceBudgetV2, SurfaceQualityTargetsV2, VolumeQualityTargetsV2,
    MESHING_REQUEST_SCHEMA_VERSION,
};
pub use tolerance::GeometryTolerancePolicy;

pub(super) fn validate_token(
    field: &str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), MeshingContractError> {
    if value.is_empty()
        || value.len() > maximum_bytes
        || !value.is_ascii()
        || value.chars().any(char::is_control)
        || value.trim() != value
    {
        return Err(MeshingContractError::invalid(
            field,
            format!("must be 1..={maximum_bytes} printable ASCII bytes without surrounding space"),
        ));
    }
    Ok(())
}

pub(super) fn validate_finite(field: &str, value: f64) -> Result<(), MeshingContractError> {
    if !value.is_finite() {
        return Err(MeshingContractError::invalid(field, "must be finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
