use sha2::{Digest as _, Sha256};

use super::{encode_contract, CanonicalMeshingContract, MeshingCanonicalLimits};
use crate::contracts::v2::{
    AlgorithmVersionSet, AnalysisMeshArtifactV2, GeometryTolerancePolicy, MeshingContractError,
    MeshingEvidenceV2, MeshingFailure, MeshingJoinIdentityV2, MeshingPartitionDescriptorV2,
    MeshingPartitionIdentityV2, MeshingProgressV2, MeshingRequestV2, MeshingStageIdentityV2,
    MeshingStageManifestV2, MeshingStageResultIdentityV2, MeshingValidationIdentityV2,
    MeshingWorkloadRequestV2, MeshingWorkloadResultV2, MetricFieldRequestV2, StableDigest,
};

macro_rules! canonical_contract {
    ($type:ty, $domain:literal, $limits:expr, $validator:path) => {
        impl CanonicalMeshingContract for $type {
            const DOMAIN: &'static str = $domain;
            const LIMITS: MeshingCanonicalLimits = $limits;

            fn validate_canonical(&self) -> Result<(), MeshingContractError> {
                $validator(self).map_err(Into::into)
            }
        }
    };
}

canonical_contract!(
    GeometryTolerancePolicy,
    "analysis.mesh.tolerance-policy/v2",
    MeshingCanonicalLimits::IDENTITY,
    GeometryTolerancePolicy::validate
);
canonical_contract!(
    MetricFieldRequestV2,
    "analysis.mesh.metric-field-request/v2",
    MeshingCanonicalLimits::REQUEST,
    MetricFieldRequestV2::validate
);
canonical_contract!(
    AlgorithmVersionSet,
    "analysis.mesh.algorithm-versions/v2",
    MeshingCanonicalLimits::IDENTITY,
    AlgorithmVersionSet::validate
);
canonical_contract!(
    MeshingRequestV2,
    "analysis.mesh.request/v2",
    MeshingCanonicalLimits::REQUEST,
    MeshingRequestV2::validate
);
canonical_contract!(
    MeshingFailure,
    "analysis.mesh.failure/v2",
    MeshingCanonicalLimits::MANIFEST,
    MeshingFailure::validate
);
canonical_contract!(
    MeshingPartitionDescriptorV2,
    "analysis.mesh.partition-descriptor/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingPartitionDescriptorV2::validate
);
canonical_contract!(
    MeshingStageIdentityV2,
    "analysis.mesh.stage-request-identity/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingStageIdentityV2::validate
);
canonical_contract!(
    MeshingPartitionIdentityV2,
    "analysis.mesh.partition-identity/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingPartitionIdentityV2::validate
);
canonical_contract!(
    MeshingJoinIdentityV2,
    "analysis.mesh.join-identity/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingJoinIdentityV2::validate
);
canonical_contract!(
    MeshingStageResultIdentityV2,
    "analysis.mesh.stage-result-identity/v2",
    MeshingCanonicalLimits::MANIFEST,
    MeshingStageResultIdentityV2::validate
);
canonical_contract!(
    MeshingValidationIdentityV2,
    "analysis.mesh.validation-identity/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingValidationIdentityV2::validate
);
canonical_contract!(
    MeshingStageManifestV2,
    "analysis.mesh.stage-manifest/v2",
    MeshingCanonicalLimits::MANIFEST,
    MeshingStageManifestV2::validate
);
canonical_contract!(
    MeshingWorkloadRequestV2,
    "analysis.mesh.workload-request/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingWorkloadRequestV2::validate
);
canonical_contract!(
    MeshingProgressV2,
    "analysis.mesh.progress/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingProgressV2::validate
);

impl CanonicalMeshingContract for MeshingWorkloadResultV2 {
    const DOMAIN: &'static str = "analysis.mesh.workload-result/v2";
    const LIMITS: MeshingCanonicalLimits = MeshingCanonicalLimits::MANIFEST;

    fn validate_canonical(&self) -> Result<(), MeshingContractError> {
        self.validate_standalone()
    }
}

impl CanonicalMeshingContract for MeshingEvidenceV2 {
    const DOMAIN: &'static str = "analysis.mesh.evidence/v2";
    const LIMITS: MeshingCanonicalLimits = MeshingCanonicalLimits::MANIFEST;

    fn validate_canonical(&self) -> Result<(), MeshingContractError> {
        self.validate_standalone()
    }
}

impl CanonicalMeshingContract for AnalysisMeshArtifactV2 {
    const DOMAIN: &'static str = "analysis.mesh.artifact/v2";
    const LIMITS: MeshingCanonicalLimits = MeshingCanonicalLimits::ARTIFACT;

    fn validate_canonical(&self) -> Result<(), MeshingContractError> {
        self.validate_payload()?;
        self.canonical_digest
            .validate_nonzero("artifact.canonical_digest")?;
        if self.canonical_digest != artifact_identity_digest(self)? {
            return Err(MeshingContractError::invalid(
                "artifact.canonical_digest",
                "does not match the canonical artifact payload",
            ));
        }
        Ok(())
    }

    fn canonical_digest(&self) -> Result<StableDigest, MeshingContractError> {
        self.validate_canonical()?;
        artifact_identity_digest(self)
    }
}

impl AnalysisMeshArtifactV2 {
    pub fn seal_canonical_digest(&mut self) -> Result<StableDigest, MeshingContractError> {
        self.validate_payload()?;
        let digest = artifact_identity_digest(self)?;
        self.canonical_digest = digest;
        Ok(digest)
    }
}

fn artifact_identity_digest(
    artifact: &AnalysisMeshArtifactV2,
) -> Result<StableDigest, MeshingContractError> {
    let mut projection = artifact.clone();
    projection.canonical_digest = StableDigest::ZERO;
    let encoded = encode_contract(
        <AnalysisMeshArtifactV2 as CanonicalMeshingContract>::DOMAIN,
        &projection,
        <AnalysisMeshArtifactV2 as CanonicalMeshingContract>::LIMITS,
    )?;
    Ok(StableDigest::from_bytes(Sha256::digest(encoded).into()))
}
