use super::{CanonicalMeshingContract, MeshingCanonicalLimits};
use crate::contracts::canonical::{
    AlgorithmVersionSet, GeometryRevisionRef, GeometryTolerancePolicy, MeshingCancellationSignal,
    MeshingContractError, MeshingEvidence, MeshingFailure, MeshingJoinIdentity,
    MeshingPartitionDescriptor, MeshingPartitionIdentity, MeshingProgress, MeshingRequest,
    MeshingStageEvidence, MeshingStageIdentity, MeshingStageManifest, MeshingStageResultIdentity,
    MeshingValidationIdentity, MeshingWorkloadRequest, MeshingWorkloadResult, MetricFieldRequest,
    SolverMeshAdaptationLineage, SolverMeshArtifact, SolverMeshTopology, SolverMeshTransferMap,
    SolverMeshValidation, StableDigest,
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
    "analysis.mesh.solver-transfer-map/v2",
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
        self.validate_canonical_with_cancellation(&crate::NeverCancelled)
    }

    fn canonical_digest(&self) -> Result<StableDigest, MeshingContractError> {
        self.validate_canonical()?;
        artifact_identity_digest(self)
    }
}

impl SolverMeshArtifact {
    pub fn canonical_encode_with_cancellation(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, MeshingContractError> {
        self.validate_canonical_with_cancellation(cancellation)?;
        super::encode_contract_with_cancellation(Self::DOMAIN, self, Self::LIMITS, cancellation)
    }

    /// Validates the original artifact and sealed identity under the supplied execution signal.
    /// Decoding and already-owned validation share this owner; neither trusts claimed hashes.
    pub fn validate_canonical_with_cancellation(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), MeshingContractError> {
        let digest = self.admitted_identity_digest_with_cancellation(cancellation)?;
        self.canonical_digest
            .validate_nonzero("artifact.canonical_digest")?;
        if self.canonical_digest != digest {
            return Err(MeshingContractError::invalid(
                "artifact.canonical_digest",
                "does not match the canonical artifact payload",
            ));
        }
        Ok(())
    }

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
        artifact.validate_canonical_with_cancellation(cancellation)?;
        Ok(artifact)
    }

    /// Recomputes the sealed identity digest from the artifact's current content without the
    /// semantic payload validation. A caller that already admitted an artifact with this digest
    /// through `validate_canonical*` uses it to authenticate that another value has the same
    /// content; it never admits an artifact on its own.
    pub fn content_digest_with_cancellation(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<StableDigest, MeshingContractError> {
        if let Some(identity) =
            self.topology
                .admitted(self.schema_version, &self.geometry, &self.resolved_request)
        {
            return Ok(identity.digest());
        }
        artifact_identity_digest_with_cancellation(self, cancellation)
    }

    /// Validated payload identity: the topology's admitted identity when it was derived under
    /// this exact header, otherwise full payload validation and a fresh digest, which is then
    /// recorded on the topology. The cheap header checks always run.
    fn admitted_identity_digest_with_cancellation(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<StableDigest, MeshingContractError> {
        if let Some(identity) =
            self.topology
                .admitted(self.schema_version, &self.geometry, &self.resolved_request)
        {
            // An admitted identity skips the walk, not the execution's cancellation contract.
            crate::contracts::canonical::artifact::validation_checkpoint(cancellation)?;
            self.validate_header()?;
            return Ok(identity.digest());
        }
        self.validate_payload_with_cancellation(cancellation)?;
        let digest = artifact_identity_digest_with_cancellation(self, cancellation)?;
        self.topology.admit(
            self.schema_version,
            &self.geometry,
            &self.resolved_request,
            digest,
        );
        Ok(digest)
    }

    /// Canonical encoded size of this validated artifact, derived once per admitted identity.
    pub fn canonical_encoded_len(&self) -> Result<u64, MeshingContractError> {
        self.validate_canonical()?;
        let Some(identity) =
            self.topology
                .admitted(self.schema_version, &self.geometry, &self.resolved_request)
        else {
            return Ok(self.canonical_encode()?.len() as u64);
        };
        if let Some(length) = identity.encoded_len().get() {
            return Ok(*length);
        }
        let length = super::encode_contract(Self::DOMAIN, self, Self::LIMITS)?.len() as u64;
        let _ = identity.encoded_len().set(length);
        Ok(length)
    }

    pub fn seal_canonical_digest(&mut self) -> Result<StableDigest, MeshingContractError> {
        self.seal_canonical_digest_with_cancellation(&crate::NeverCancelled)
    }

    pub fn seal_canonical_digest_with_cancellation(
        &mut self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<StableDigest, MeshingContractError> {
        let digest = self.admitted_identity_digest_with_cancellation(cancellation)?;
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
    super::digest_with_cancellation(&encoded, cancellation)
}

fn artifact_identity_digest(
    artifact: &SolverMeshArtifact,
) -> Result<StableDigest, MeshingContractError> {
    artifact_identity_digest_with_cancellation(artifact, &crate::NeverCancelled)
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

impl<'a> From<&'a crate::SolverMeshProjection> for SolverMeshArtifactIdentity<'a> {
    fn from(projection: &'a crate::SolverMeshProjection) -> Self {
        SolverMeshArtifactIdentity {
            schema_version: crate::SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
            canonical_digest: StableDigest::ZERO,
            root_stage_manifest_digest: StableDigest::ZERO,
            geometry: &projection.geometry,
            resolved_request: &projection.resolved_request,
            topology: &projection.topology,
        }
    }
}

impl crate::SolverMeshProjection {
    /// Profile the actual borrowed artifact identity before consuming this
    /// projection. No temporary rich artifact, request or topology is cloned.
    pub fn artifact_sealing_workspace_bytes(
        &self,
        control: &dyn runmat_canonical_codec::CanonicalCodecControl,
    ) -> Result<u64, MeshingContractError> {
        let identity = SolverMeshArtifactIdentity::from(self);
        let encoding = runmat_canonical_codec::profile_canonical_encoding_with_control(
            &identity,
            <SolverMeshArtifact as CanonicalMeshingContract>::LIMITS.into(),
            control,
        )
        .map_err(|error| MeshingContractError::invalid(error.field, error.reason))?;
        Ok(encoding.resident_workspace_bytes_estimate.max(
            self.topology
                .validation_scratch_bytes(self.resolved_request.element_order)?,
        ))
    }
}
