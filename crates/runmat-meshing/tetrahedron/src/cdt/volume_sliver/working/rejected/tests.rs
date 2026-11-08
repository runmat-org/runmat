use super::*;
use crate::cdt::{
    topology::build_delaunay_volume_topology_with_regions, DelaunayTopologyOptions,
    DelaunayVolumeTopology,
};
use runmat_meshing_core::NeverCancelled;

fn identity(index: usize) -> StableDigest {
    StableDigest::from_bytes([index as u8 + 1; 32])
}

/// Five cells in a row, each touching the next at one node: cell `i` has nodes `3i` to `3i + 3`.
/// A node listed in `moved` is shifted a little along y.
fn strip(moved: &[usize]) -> DelaunayVolumeTopology {
    let nodes = (0..16)
        .map(|index| {
            let (cell, slot) = (index / 3, index % 3);
            let x = 2.0 * cell as f64;
            let mut point = match slot {
                0 => [x, 0.0, 0.0],
                1 => [x + 1.0, 1.0, 0.0],
                _ => [x + 1.0, 0.0, 1.0],
            };
            if moved.contains(&index) {
                point[1] += 0.25;
            }
            DelaunayVolumeNode::explicit(identity(index), point)
        })
        .collect();
    let cells = (0..5)
        .map(|cell| {
            let first = 3 * cell;
            ([first, first + 1, first + 2, first + 3], None)
        })
        .collect();
    build_delaunay_volume_topology_with_regions(
        nodes,
        cells,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn key(worst_shape_priority: u64) -> CompleteSearchKey {
    CompleteSearchKey::new(
        [3, 1, 2, 0].map(identity),
        QualityObjective::Shape,
        Some(worst_shape_priority),
        None,
    )
}

#[test]
fn a_rejected_search_stays_rejected_until_its_neighborhood_or_floor_changes() {
    let mut rejected = RejectedCompleteRelocations::default();
    rejected.record(&strip(&[]), key(7), BTreeSet::new());
    assert!(rejected.still_rejected(&strip(&[]), &key(7)));
    // Node 12 shares no cell with the source's one-ring (nodes 0 to 6).
    assert!(rejected.still_rejected(&strip(&[12]), &key(7)));
    // A different comparison floor is a different search.
    assert!(!rejected.still_rejected(&strip(&[]), &key(8)));
    // Node 8 lies in the cell around node 6, so the search reads it.
    assert!(!rejected.still_rejected(&strip(&[8]), &key(7)));
    // The stale record is gone.
    assert!(!rejected.still_rejected(&strip(&[]), &key(7)));
}

#[test]
fn nodes_a_candidate_changed_extend_the_neighborhood() {
    let mut rejected = RejectedCompleteRelocations::default();
    rejected.record(&strip(&[]), key(7), [identity(11)].into_iter().collect());
    assert!(!rejected.still_rejected(&strip(&[12]), &key(7)));
}
