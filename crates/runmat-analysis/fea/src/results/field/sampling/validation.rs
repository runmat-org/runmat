use super::*;
use std::collections::BTreeSet;

impl SampledVolumeFieldTopology {
    /// Admit a stored layout: identities, corner inventory and digest are checked here, and the
    /// complete canonical corner-edge space is derived from the corners. `expected_digest` is
    /// the stored digest of a decoded layout; a derived layout computes its own.
    pub(super) fn admit(
        data: SamplingData,
        expected_digest: Option<Digest>,
    ) -> Result<Self, RecoveredFieldError> {
        let unique = |ids: &[StableDigest]| {
            !ids.is_empty()
                && !ids.contains(&StableDigest::ZERO)
                && ids.iter().collect::<BTreeSet<_>>().len() == ids.len()
        };
        let sampling_digest = digest("sampling-layout", &data)?;
        if data.schema_version != SCHEMA_VERSION
            || expected_digest.is_some_and(|expected| expected != sampling_digest)
            || !(FieldTopologyIdentity::SampledVolumeMesh {
                mesh_digest: data.mesh_digest,
                sampling_digest,
            })
            .has_valid_identity()
            || data.projection_digest == Digest::from_bytes([0; 32])
            || data.projection_digest == Digest::sha256([])
            || !unique(&data.node_identities)
            || !unique(&data.element_identities)
            || data.element_corner_nodes.len() != data.element_identities.len()
            || data
                .element_identities
                .len()
                .checked_mul(data.rule.points_per_element())
                .is_none()
        {
            return Err(invalid(
                "sampled volume topology identity or complete inventory is invalid",
            ));
        }
        let oriented_edges = derive_edges(&data)?;
        let value = Self {
            data,
            sampling_digest,
            oriented_edges,
        };
        value.entity_counts().validate()?;
        Ok(value)
    }
    /// Admission proved the layout; its stored parts are immutable afterwards, so this checks
    /// only that the derived edge space still matches the stored inventory sizes.
    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        if self.oriented_edges.is_empty()
            || self.data.element_corner_nodes.len() != self.data.element_identities.len()
        {
            return Err(invalid(
                "sampled volume topology identity or complete inventory is invalid",
            ));
        }
        self.entity_counts().validate()
    }
}

/// The canonical corner-edge space: every corner pair of every element, directed from the
/// lesser to the greater node identity, sorted by endpoint identities and unique.
fn derive_edges(data: &SamplingData) -> Result<Vec<SampledOrientedEdge>, RecoveredFieldError> {
    let nodes = &data.node_identities;
    // Rank nodes by identity so an edge key orders exactly as its identity pair.
    let mut by_identity: Vec<u32> = (0..nodes.len())
        .map(|index| u32::try_from(index).map_err(|_| invalid("sampled node count exceeds range")))
        .collect::<Result<_, _>>()?;
    by_identity.sort_unstable_by_key(|&index| nodes[index as usize]);
    let mut rank = vec![0u32; nodes.len()];
    for (position, &index) in by_identity.iter().enumerate() {
        rank[index as usize] = position as u32;
    }
    let mut keys = Vec::with_capacity(data.element_corner_nodes.len() * 6);
    for corners in &data.element_corner_nodes {
        if corners.iter().any(|&corner| corner as usize >= nodes.len())
            || corners.iter().collect::<BTreeSet<_>>().len() != 4
        {
            return Err(invalid("sampled element corner inventory is invalid"));
        }
        for a in 0..4 {
            for b in (a + 1)..4 {
                let (low, high) = {
                    let (x, y) = (rank[corners[a] as usize], rank[corners[b] as usize]);
                    (x.min(y), x.max(y))
                };
                keys.push((u64::from(low) << 32) | u64::from(high));
            }
        }
    }
    keys.sort_unstable();
    keys.dedup();
    if keys.is_empty() {
        return Err(invalid(
            "sampled edge inventory is not the complete canonical corner-edge space",
        ));
    }
    keys.into_iter()
        .map(|key| {
            let node_identities = [
                nodes[by_identity[(key >> 32) as usize] as usize],
                nodes[by_identity[(key & 0xffff_ffff) as usize] as usize],
            ];
            Ok(SampledOrientedEdge {
                identity: stable_identity("oriented-edge", &node_identities)?,
                node_identities,
            })
        })
        .collect()
}
