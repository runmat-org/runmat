//! Requested scratch owned alongside artifact validation, not by numerical consumers.
use super::{
    artifact_workspace::slots, ElementOrder, MeshingContractError, SolverMeshArtifact,
    SolverMeshTopology, StableDigest,
};
use runmat_canonical_codec::CanonicalLimits;

impl SolverMeshTopology {
    /// Requested temporary flat buffers for independent topology validation.
    /// Original topology/request storage, allocator bookkeeping and RSS are separate.
    pub fn validation_scratch_bytes(
        &self,
        order: ElementOrder,
    ) -> Result<u64, MeshingContractError> {
        validation_scratch_bytes(self, order)
    }
}

impl SolverMeshArtifact {
    /// Maximum requested scratch for encoding an already owned artifact, including
    /// independent topology validation and its borrowed identity projection.
    /// Validation scratch is released before identity/output encoding. Unlike cold
    /// decode admission, this path constructs no JSON value tree. Original typed
    /// buffers, allocator bookkeeping and other live owners are charged separately.
    pub fn canonical_encoding_resident_bytes_estimate(&self) -> Result<u64, MeshingContractError> {
        let encoding = runmat_canonical_codec::profile_canonical_encoding(
            self,
            CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
        )
        .map_err(|error| {
            MeshingContractError::invalid("mesh validation resources", error.to_string())
        })?
        .resident_workspace_bytes_estimate;
        Ok(
            validation_scratch_bytes(&self.topology, self.resolved_request.element_order)?
                .max(encoding),
        )
    }

    /// Additional requested heap for validation, excluding the borrowed artifact, stack,
    /// allocator bookkeeping, diagnostic error strings and RSS. Counts the actual flat-buffer
    /// reservations; sorting is in place and authored lookup inventories stay borrowed.
    /// Payload scratch is released before canonical identity encoding. The identity projection
    /// only zeroes two fixed-width digests and has this artifact's serialization widths.
    /// Profiling is not validation and can reject malformed serialization before admission.
    /// The estimate depends only on inventory sizes: both sealed digests are profiled at their
    /// widest encoding, so a stale or zeroed seal is reported by validation, after admission.
    pub fn validation_resident_bytes_estimate(&self) -> Result<u64, MeshingContractError> {
        // The identity digest uses the direct encoder, which holds the encoded bytes and its
        // ordering buffers but builds no value tree.
        let encoding = runmat_canonical_codec::profile_canonical_encoding(
            &DigestIndependentArtifact::from(self),
            CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
        )
        .map_err(|error| {
            MeshingContractError::invalid("mesh validation resources", error.to_string())
        })?
        .resident_workspace_bytes_estimate;
        Ok(
            validation_scratch_bytes(&self.topology, self.resolved_request.element_order)?
                .max(encoding),
        )
    }
}

/// The artifact's serialization shape with both sealed digests at their widest encoding.
/// Field names and order match `SolverMeshArtifact`.
#[derive(serde::Serialize)]
struct DigestIndependentArtifact<'a> {
    schema_version: u16,
    canonical_digest: StableDigest,
    root_stage_manifest_digest: StableDigest,
    geometry: &'a super::GeometryRevisionRef,
    resolved_request: &'a super::MeshingRequest,
    topology: &'a SolverMeshTopology,
}

impl<'a> From<&'a SolverMeshArtifact> for DigestIndependentArtifact<'a> {
    fn from(artifact: &'a SolverMeshArtifact) -> Self {
        const WIDEST: StableDigest = StableDigest::from_bytes([u8::MAX; 32]);
        Self {
            schema_version: artifact.schema_version,
            canonical_digest: WIDEST,
            root_stage_manifest_digest: WIDEST,
            geometry: &artifact.geometry,
            resolved_request: &artifact.resolved_request,
            topology: &artifact.topology,
        }
    }
}

/// Exact requested flat-buffer inventory for the maximum live validation phase.
/// Invalid connectivity/provenance/field lists are checked in place without allocating copies.
/// Counts are derived from supplied inventories, never their claimed canonical identities.
pub(super) fn validation_scratch_bytes(
    topology: &SolverMeshTopology,
    order: ElementOrder,
) -> Result<u64, MeshingContractError> {
    let nodes = topology.nodes.len();
    let elements = topology.volume_elements.len();
    let faces = topology.boundary_faces.len();
    let edges = topology.boundary_edges.len();
    let mut retained_ids = 0;
    for count in [nodes, elements, faces, edges] {
        add(&mut retained_ids, slots::<u64>(count)?)?;
    }
    let identity = slots::<StableDigest>(nodes.max(elements).max(faces).max(edges))?;
    let neighbors = slots::<(u64, u64)>(topology.neighbors.len())?;
    let order = match order {
        ElementOrder::Tet4 => slots::<u8>(nodes)?,
        ElementOrder::Tet10 => {
            let mut bytes = slots::<u8>(nodes)?;
            add(&mut bytes, slots::<u8>(nodes)?)?;
            add(
                &mut bytes,
                slots::<([u64; 2], u64)>(volume_edge_occurrences(elements)?)?,
            )?;
            bytes
        }
    };
    let classification = slots::<u8>(elements.max(faces))?;
    add(
        &mut retained_ids,
        identity.max(neighbors).max(order).max(classification),
    )?;
    Ok(retained_ids)
}

pub(super) fn volume_edge_occurrences(elements: usize) -> Result<usize, MeshingContractError> {
    elements.checked_mul(6).ok_or_else(overflow)
}
fn add(total: &mut u64, bytes: u64) -> Result<(), MeshingContractError> {
    *total = total.checked_add(bytes).ok_or_else(overflow)?;
    Ok(())
}
fn overflow() -> MeshingContractError {
    MeshingContractError::invalid(
        "mesh validation resources",
        "requested scratch byte inventory overflow",
    )
}

#[cfg(test)]
mod tests;
