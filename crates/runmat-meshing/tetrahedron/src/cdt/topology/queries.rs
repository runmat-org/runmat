//! Queries over the canonical, admitted node and tetrahedron ordering.
use super::{DelaunayVolumeTetrahedron, DelaunayVolumeTopology, StableDigest};

/// Find the original oriented cell without scanning unrelated cells. Canonical
/// cells are ordered by their sorted vertex indices; nodes are identity ordered.
pub(in crate::cdt) fn find_oriented_tetrahedron(
    topology: &DelaunayVolumeTopology,
    identities: [StableDigest; 4],
) -> Option<(usize, &DelaunayVolumeTetrahedron)> {
    find_with_probe(topology, identities, &mut || {})
}

fn find_with_probe<'a>(
    topology: &'a DelaunayVolumeTopology,
    identities: [StableDigest; 4],
    probe: &mut impl FnMut(),
) -> Option<(usize, &'a DelaunayVolumeTetrahedron)> {
    let mut key = [0; 4];
    for (slot, identity) in key.iter_mut().zip(identities) {
        *slot = u32::try_from(
            topology
                .nodes
                .binary_search_by(|node| {
                    probe();
                    node.identity.cmp(&identity)
                })
                .ok()?,
        )
        .ok()?;
    }
    let index = canonical_tetrahedron_index(topology, key, probe)?;
    let cell = &topology.tetrahedra[index];
    (cell
        .vertex_indices
        .map(|vertex| topology.nodes[vertex as usize].identity)
        == identities)
        .then_some((index, cell))
}

/// Shared canonical cell lookup for insertion and refinement reads.
pub(in crate::cdt) fn canonical_tetrahedron_index(
    topology: &DelaunayVolumeTopology,
    mut vertices: [u32; 4],
    probe: &mut impl FnMut(),
) -> Option<usize> {
    vertices.sort_unstable();
    topology
        .tetrahedra
        .binary_search_by(|cell| {
            probe();
            let mut key = cell.vertex_indices;
            key.sort_unstable();
            key.cmp(&vertices)
        })
        .ok()
}

#[cfg(test)]
pub(in crate::cdt) mod tests;
