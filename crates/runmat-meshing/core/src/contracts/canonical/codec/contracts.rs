use sha2::{Digest as _, Sha256};

use super::{encode_contract, CanonicalMeshingContract, MeshingCanonicalLimits};
use crate::contracts::canonical::{
    AlgorithmVersionSet, GeometryTolerancePolicy, MeshingContractError, MeshingEvidence,
    MeshingFailure, MeshingJoinIdentity, MeshingPartitionDescriptor, MeshingPartitionIdentity,
    MeshingProgress, MeshingRequest, MeshingStageEvidence, MeshingStageIdentity,
    MeshingStageManifest, MeshingStageResultIdentity, MeshingValidationIdentity,
    MeshingWorkloadRequest, MeshingWorkloadResult, MetricFieldRequest, SolverMeshAdaptationLineage,
    SolverMeshArtifact, SolverMeshProjection, SolverMeshTransferMap, SolverMeshValidation,
    StableDigest,
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
    SolverMeshTransferMap,
    "analysis.mesh.solver-transfer-map/v1",
    MeshingCanonicalLimits::ARTIFACT,
    SolverMeshTransferMap::validate
);
canonical_contract!(
    SolverMeshAdaptationLineage,
    "analysis.mesh.solver-adaptation-lineage/v1",
    MeshingCanonicalLimits::ARTIFACT,
    SolverMeshAdaptationLineage::validate
);
canonical_contract!(
    MetricFieldRequest,
    "analysis.mesh.metric-field-request/v2",
    MeshingCanonicalLimits::REQUEST,
    MetricFieldRequest::validate
);
canonical_contract!(
    AlgorithmVersionSet,
    "analysis.mesh.algorithm-versions/v2",
    MeshingCanonicalLimits::IDENTITY,
    AlgorithmVersionSet::validate
);
canonical_contract!(
    SolverMeshProjection,
    "analysis.mesh.solver-projection/v1",
    MeshingCanonicalLimits::ARTIFACT,
    SolverMeshProjection::validate
);
canonical_contract!(
    SolverMeshValidation,
    "analysis.mesh.solver-validation/v1",
    MeshingCanonicalLimits::MANIFEST,
    SolverMeshValidation::validate
);
canonical_contract!(
    MeshingRequest,
    "analysis.mesh.request/v2",
    MeshingCanonicalLimits::REQUEST,
    MeshingRequest::validate
);
canonical_contract!(
    MeshingFailure,
    "analysis.mesh.failure/v2",
    MeshingCanonicalLimits::MANIFEST,
    MeshingFailure::validate
);
canonical_contract!(
    MeshingPartitionDescriptor,
    "analysis.mesh.partition-descriptor/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingPartitionDescriptor::validate
);
canonical_contract!(
    MeshingStageIdentity,
    "analysis.mesh.stage-request-identity/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingStageIdentity::validate
);
canonical_contract!(
    MeshingPartitionIdentity,
    "analysis.mesh.partition-identity/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingPartitionIdentity::validate
);
canonical_contract!(
    MeshingJoinIdentity,
    "analysis.mesh.join-identity/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingJoinIdentity::validate
);
canonical_contract!(
    MeshingStageResultIdentity,
    "analysis.mesh.stage-result-identity/v2",
    MeshingCanonicalLimits::MANIFEST,
    MeshingStageResultIdentity::validate
);
canonical_contract!(
    MeshingValidationIdentity,
    "analysis.mesh.validation-identity/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingValidationIdentity::validate
);
canonical_contract!(
    MeshingStageManifest,
    "analysis.mesh.stage-manifest/v2",
    MeshingCanonicalLimits::MANIFEST,
    MeshingStageManifest::validate
);
canonical_contract!(
    MeshingWorkloadRequest,
    "analysis.mesh.workload-request/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingWorkloadRequest::validate
);
canonical_contract!(
    MeshingProgress,
    "analysis.mesh.progress/v2",
    MeshingCanonicalLimits::IDENTITY,
    MeshingProgress::validate
);
canonical_contract!(
    MeshingStageEvidence,
    "analysis.mesh.stage-evidence/v2",
    MeshingCanonicalLimits::MANIFEST,
    MeshingStageEvidence::validate
);

impl CanonicalMeshingContract for MeshingWorkloadResult {
    const DOMAIN: &'static str = "analysis.mesh.workload-result/v2";
    const LIMITS: MeshingCanonicalLimits = MeshingCanonicalLimits::MANIFEST;

    fn validate_canonical(&self) -> Result<(), MeshingContractError> {
        self.validate_standalone()
    }
}

impl CanonicalMeshingContract for MeshingEvidence {
    const DOMAIN: &'static str = "analysis.mesh.evidence/v2";
    const LIMITS: MeshingCanonicalLimits = MeshingCanonicalLimits::MANIFEST;

    fn validate_canonical(&self) -> Result<(), MeshingContractError> {
        self.validate_standalone()
    }
}

impl CanonicalMeshingContract for SolverMeshArtifact {
    const DOMAIN: &'static str = "analysis.mesh.artifact/v4";
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

impl SolverMeshArtifact {
    pub fn seal_canonical_digest(&mut self) -> Result<StableDigest, MeshingContractError> {
        self.validate_payload()?;
        let digest = artifact_identity_digest(self)?;
        self.canonical_digest = digest;
        Ok(digest)
    }
}

fn artifact_identity_digest(
    artifact: &SolverMeshArtifact,
) -> Result<StableDigest, MeshingContractError> {
    let mut projection = artifact.clone();
    projection.canonical_digest = StableDigest::ZERO;
    // The validation manifest authenticates one physical stage/chunk layout. It remains in the
    // artifact for closure loading, but legal partition and chunk layouts must converge on the
    // same logical solver-mesh identity.
    projection.root_stage_manifest_digest = StableDigest::ZERO;
    let encoded = encode_contract(
        <SolverMeshArtifact as CanonicalMeshingContract>::DOMAIN,
        &projection,
        <SolverMeshArtifact as CanonicalMeshingContract>::LIMITS,
    )?;
    Ok(StableDigest::from_bytes(Sha256::digest(encoded).into()))
}
