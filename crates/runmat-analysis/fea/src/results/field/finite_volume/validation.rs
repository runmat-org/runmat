//! Native field-layout integrity and its own counted workspace; source admission is separate.
use super::*;
use runmat_canonical_codec::serialized_materialization_resident_bytes_estimate;
use std::mem::size_of;

impl FiniteVolumeFieldTopology {
    /// Requested additional heap for original layout validation, excluding borrowed layout,
    /// allocator overhead, diagnostics and RSS. Called before native publication constructs
    /// its layout; exact source counts are supplied by the original numerical topology owner.
    pub fn validation_workspace_bytes(
        nodes: usize,
        cells: usize,
        faces: usize,
    ) -> Result<u64, RecoveredFieldError> {
        let mut total = 0u64;
        for count in [nodes, cells, faces] {
            total = total
                .checked_add(slots(count)?)
                .ok_or_else(|| invalid("native validation workspace overflow"))?;
        }
        // Only serialized width is represented here, never a physical identity. All digest
        // bytes at their maximal encoding width and Some(neighbor) bound one original face.
        let digest = StableDigest::from_bytes([u8::MAX; 32]);
        let identity_encoding = serialized_materialization_resident_bytes_estimate(
            &([digest; 3], digest, Some(digest)),
            LIMITS,
        )
        .map_err(invalid)?;
        total
            .checked_add(identity_encoding)
            .ok_or_else(|| invalid("native validation workspace overflow"))
    }
}

/// Whether face identities are recomputed, or were computed from these faces by the caller.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum FaceIdentities {
    Recompute,
    Computed,
}

pub(super) fn validate(
    layout: &FiniteVolumeFieldTopology,
    identities_check: FaceIdentities,
) -> Result<(), RecoveredFieldError> {
    FiniteVolumeFieldTopology::validation_workspace_bytes(
        layout.node_identities.len(),
        layout.cell_identities.len(),
        layout.oriented_faces.len(),
    )?;
    if !layout.identity().has_valid_identity()
        || layout.projection_digest == Digest::from_bytes([0; 32])
        || layout.projection_digest == Digest::sha256([])
        || layout.oriented_faces.is_empty()
    {
        return Err(invalid("invalid native finite-volume layout"));
    }
    let nodes = identities(
        layout.node_identities.iter().copied(),
        layout.node_identities.len(),
    )?;
    let cells = identities(
        layout.cell_identities.iter().copied(),
        layout.cell_identities.len(),
    )?;
    let _faces = identities(
        layout.oriented_faces.iter().map(|face| face.identity),
        layout.oriented_faces.len(),
    )?;
    for face in &layout.oriented_faces {
        if !face.corner_identities.windows(2).all(|w| w[0] < w[1])
            || face
                .corner_identities
                .iter()
                .any(|node| nodes.binary_search(node).is_err())
            || cells.binary_search(&face.owner).is_err()
            || face.neighbor.is_some_and(|neighbor| {
                neighbor == face.owner || cells.binary_search(&neighbor).is_err()
            })
            || (identities_check == FaceIdentities::Recompute
                && face.identity
                    != face_identity(face.corner_identities, face.owner, face.neighbor)?)
        {
            return Err(invalid("invalid exact owner-oriented face inventory"));
        }
    }
    Ok(())
}

fn slots(count: usize) -> Result<u64, RecoveredFieldError> {
    count
        .checked_mul(size_of::<StableDigest>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid("native validation workspace exceeds addressable slots"))
}
fn identities(
    ids: impl Iterator<Item = StableDigest>,
    count: usize,
) -> Result<Vec<StableDigest>, RecoveredFieldError> {
    slots(count)?;
    let mut sorted = Vec::new();
    sorted.try_reserve_exact(count).map_err(invalid)?;
    for identity in ids {
        if sorted.len() == count {
            return Err(invalid("native identity count mismatch"));
        }
        sorted.push(identity);
    }
    sorted.sort_unstable();
    if sorted.is_empty()
        || sorted.first() == Some(&StableDigest::ZERO)
        || sorted.windows(2).any(|pair| pair[0] == pair[1])
    {
        return Err(invalid("invalid native finite-volume identity inventory"));
    }
    Ok(sorted)
}
