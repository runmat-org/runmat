//! Cooperative projection codec. Execution and ordinary callers retain identical
//! canonical bytes, identities and independent topology validation.
use super::{CanonicalMeshingContract, MeshingCanonicalLimits};
use crate::{
    MeshingCancellationSignal, MeshingContractError, NeverCancelled, SolverMeshProjection,
    StableDigest,
};
#[cfg(test)]
mod tests;

impl CanonicalMeshingContract for SolverMeshProjection {
    const DOMAIN: &'static str = "analysis.mesh.solver-projection/v1";
    const LIMITS: MeshingCanonicalLimits = MeshingCanonicalLimits::ARTIFACT;

    fn validate_canonical(&self) -> Result<(), MeshingContractError> {
        self.validate()
    }
    fn canonical_encode(&self) -> Result<Vec<u8>, MeshingContractError> {
        self.canonical_encode_with_cancellation(&NeverCancelled)
    }
    fn canonical_decode(bytes: &[u8]) -> Result<Self, MeshingContractError> {
        Self::canonical_decode_with_cancellation(bytes, &NeverCancelled)
    }
    fn canonical_digest(&self) -> Result<StableDigest, MeshingContractError> {
        self.canonical_digest_with_cancellation(&NeverCancelled)
    }
}

impl SolverMeshProjection {
    pub fn canonical_encode_with_cancellation(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, MeshingContractError> {
        self.validate_with_cancellation(cancellation)?;
        super::encode_contract_with_cancellation(Self::DOMAIN, self, Self::LIMITS, cancellation)
    }

    pub fn canonical_decode_with_cancellation(
        bytes: &[u8],
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, MeshingContractError> {
        let projection: Self = super::decode_contract_with_cancellation(
            Self::DOMAIN,
            bytes,
            Self::LIMITS,
            cancellation,
        )?;
        projection.validate_with_cancellation(cancellation)?;
        Ok(projection)
    }

    /// Admit typed cold state and owner-defined topology scratch before their
    /// allocation. The original grammar, identity and topology checks still run.
    pub fn canonical_decode_with_resource_admission(
        bytes: &[u8],
        control: &dyn runmat_canonical_codec::CanonicalCodecControl,
        cancellation: &dyn MeshingCancellationSignal,
        admit: impl FnMut(u64) -> Result<(), MeshingContractError>,
    ) -> Result<Self, MeshingContractError> {
        super::metadata::decode_validated(
            bytes,
            control,
            admit,
            |projection: &Self| {
                projection
                    .topology
                    .validation_scratch_bytes(projection.resolved_request.element_order)
            },
            |projection| projection.validate_with_cancellation(cancellation),
        )
    }

    pub fn canonical_digest_with_cancellation(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<StableDigest, MeshingContractError> {
        let encoded = self.canonical_encode_with_cancellation(cancellation)?;
        super::digest_with_cancellation(&encoded, cancellation)
    }
}
