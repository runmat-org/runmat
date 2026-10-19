//! Exact connectivity comparison and local difference keys against one common
//! canonical base. Deduplication retains changed cells, not a full mesh per trial.
use super::{DelaunayVolumeTopology, StableDigest};
use std::cmp::Ordering;

pub(in crate::cdt::volume_sliver) fn keys(
    topology: &DelaunayVolumeTopology,
) -> impl Iterator<Item = [StableDigest; 4]> + '_ {
    topology.tetrahedra.iter().map(|tetrahedron| {
        let mut key = tetrahedron
            .vertex_indices
            .map(|index| topology.nodes[index as usize].identity);
        key.sort_unstable();
        key
    })
}

#[cfg(test)]
pub(in crate::cdt::volume_sliver) fn compare_connectivity(
    left: &DelaunayVolumeTopology,
    right: &DelaunayVolumeTopology,
) -> Ordering {
    keys(left).cmp(keys(right))
}

pub(in crate::cdt::volume_sliver) fn connectivity_delta(
    base: &DelaunayVolumeTopology,
    candidate: &DelaunayVolumeTopology,
) -> (Vec<[StableDigest; 4]>, Vec<[StableDigest; 4]>) {
    let mut base = keys(base).peekable();
    let mut candidate = keys(candidate).peekable();
    let mut removed = Vec::new();
    let mut created = Vec::new();
    // Canonical topology orders tetrahedra by these same stable corner keys.
    while let (Some(left), Some(right)) = (base.peek(), candidate.peek()) {
        match left.cmp(right) {
            Ordering::Less => removed.push(base.next().unwrap()),
            Ordering::Greater => created.push(candidate.next().unwrap()),
            Ordering::Equal => {
                base.next();
                candidate.next();
            }
        }
    }
    removed.extend(base);
    created.extend(candidate);
    (removed, created)
}

#[cfg(test)]
mod tests;
