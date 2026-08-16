mod artifact;
mod artifact_classification;
mod artifact_types;
mod cancellation;
mod error;
mod evidence;
mod failure;
mod identity;
mod metric;
mod request;
mod tolerance;

pub use artifact_types::{
    AnalysisBoundaryEdgeV2, AnalysisBoundaryFaceV2, AnalysisMeshArtifactV2, AnalysisMeshNodeV2,
    AnalysisMeshTopologyV2, AnalysisVolumeElementV2, BoundaryFaceRoleV2, BoundaryTriangleOrderV2,
    ContactPairV2, FieldTopologyLocationV2, FieldTopologyMapV2, MaterialInterfaceV2,
    MeshNeighborV2, MeshRegionV2, ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
};
pub use cancellation::{CancellationPolicyV2, MeshingCancellationSignal, NeverCancelled};
pub use error::MeshingContractError;
pub use evidence::{
    CacheAdmissionDecisionV2, ErrorDistributionV2, InvariantEvidenceV2, MeshingEvidenceV2,
    MeshingResourceUsageV2, PlatformBuildIdentityV2, SizingResolutionEvidenceV2, StageEvidenceV2,
    MESHING_EVIDENCE_SCHEMA_VERSION,
};
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
mod artifact_tests;
#[cfg(test)]
mod tests;
