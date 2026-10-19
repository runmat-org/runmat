//! Owned manifest metadata capacities, excluding the caller's inline value and
//! encoded chunk payloads. Exhaustive field matches keep schema changes explicit.
use super::*;
impl MeshingStageManifest {
    pub fn owned_buffer_bytes(&self) -> Result<u64, MeshingContractError> {
        let Self {
            schema_version: _,
            stage: _,
            result_kind: _,
            logical_result_identity: _,
            disposition: _,
            prerequisite_manifest_digests,
            invariant_summary_digest: _,
            chunks,
            total_encoded_length: _,
        } = self;
        let prerequisites = prerequisite_manifest_digests
            .capacity()
            .checked_mul(std::mem::size_of::<StableDigest>());
        let descriptors = chunks
            .capacity()
            .checked_mul(std::mem::size_of::<MeshingChunkDescriptor>());
        for descriptor in chunks {
            let MeshingChunkDescriptor {
                ordinal: _,
                first_logical_entity_ordinal: _,
                digest: _,
                media_type: _,
                schema_version: _,
                encoded_length: _,
                decoded_length: _,
                logical_entity_count: _,
            } = descriptor;
        }
        prerequisites
            .and_then(|bytes| descriptors.and_then(|other| bytes.checked_add(other)))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                MeshingContractError::invalid(
                    "manifest storage",
                    "owned metadata capacity inventory overflow",
                )
            })
    }
}
