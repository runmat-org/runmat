use sha2::{Digest as _, Sha256};

use super::{encode_contract, CanonicalMeshingContract, MeshingCanonicalLimits};
use crate::contracts::canonical::{
    AlgorithmVersionSet, GeometryRevisionRef, GeometryTolerancePolicy, MeshingCancellationSignal,
    MeshingContractError, MeshingEvidence, MeshingFailure, MeshingJoinIdentity,
    MeshingPartitionDescriptor, MeshingPartitionIdentity, MeshingProgress, MeshingRequest,
    MeshingStageEvidence, MeshingStageIdentity, MeshingStageManifest, MeshingStageResultIdentity,
    MeshingValidationIdentity, MeshingWorkloadRequest, MeshingWorkloadResult, MetricFieldRequest,
    SolverMeshAdaptationLineage, SolverMeshArtifact, SolverMeshProjection, SolverMeshTopology,
    SolverMeshTransferMap, SolverMeshValidation, StableDigest,
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
    /// Decodes and independently validates a solver artifact under an execution-owned
    /// cancellation signal. This retains the canonical byte and sealed-identity checks while
    /// allowing large topology validation to satisfy the execution cadence contract.
    pub fn canonical_decode_with_cancellation(
        bytes: &[u8],
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, MeshingContractError> {
        let artifact: Self = super::decode_contract_with_cancellation(
            Self::DOMAIN,
            bytes,
            Self::LIMITS,
            cancellation,
        )?;
        artifact.validate_payload_with_cancellation(cancellation)?;
        artifact
            .canonical_digest
            .validate_nonzero("artifact.canonical_digest")?;
        if artifact.canonical_digest
            != artifact_identity_digest_with_cancellation(&artifact, cancellation)?
        {
            return Err(MeshingContractError::invalid(
                "artifact.canonical_digest",
                "does not match the canonical artifact payload",
            ));
        }
        Ok(artifact)
    }

    pub fn seal_canonical_digest(&mut self) -> Result<StableDigest, MeshingContractError> {
        self.validate_payload()?;
        let digest = artifact_identity_digest(self)?;
        self.canonical_digest = digest;
        Ok(digest)
    }
}

fn artifact_identity_digest_with_cancellation(
    artifact: &SolverMeshArtifact,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<StableDigest, MeshingContractError> {
    let projection = SolverMeshArtifactIdentity::from(artifact);
    let encoded = super::encode_contract_with_cancellation(
        <SolverMeshArtifact as CanonicalMeshingContract>::DOMAIN,
        &projection,
        <SolverMeshArtifact as CanonicalMeshingContract>::LIMITS,
        cancellation,
    )?;
    Ok(StableDigest::from_bytes(Sha256::digest(encoded).into()))
}

fn artifact_identity_digest(
    artifact: &SolverMeshArtifact,
) -> Result<StableDigest, MeshingContractError> {
    let projection = SolverMeshArtifactIdentity::from(artifact);
    let encoded = encode_contract(
        <SolverMeshArtifact as CanonicalMeshingContract>::DOMAIN,
        &projection,
        <SolverMeshArtifact as CanonicalMeshingContract>::LIMITS,
    )?;
    Ok(StableDigest::from_bytes(Sha256::digest(encoded).into()))
}

/// Borrowed identity projection: the validation manifest authenticates one physical stage/chunk
/// layout, while legal layouts must converge on the same logical solver-mesh identity.
#[derive(serde::Serialize)]
struct SolverMeshArtifactIdentity<'a> {
    schema_version: u16,
    canonical_digest: StableDigest,
    root_stage_manifest_digest: StableDigest,
    geometry: &'a GeometryRevisionRef,
    resolved_request: &'a MeshingRequest,
    topology: &'a SolverMeshTopology,
}

impl<'a> From<&'a SolverMeshArtifact> for SolverMeshArtifactIdentity<'a> {
    fn from(artifact: &'a SolverMeshArtifact) -> Self {
        Self {
            schema_version: artifact.schema_version,
            canonical_digest: StableDigest::ZERO,
            root_stage_manifest_digest: StableDigest::ZERO,
            geometry: &artifact.geometry,
            resolved_request: &artifact.resolved_request,
            topology: &artifact.topology,
        }
    }
}
