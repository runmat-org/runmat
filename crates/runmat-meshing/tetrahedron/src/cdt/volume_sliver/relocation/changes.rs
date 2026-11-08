//! Nodes of the cells a complete candidate changed, found by one merge over both topologies.
use std::collections::BTreeSet;

use runmat_meshing_core::StableDigest;

use crate::cdt::DelaunayVolumeTopology;

/// Adds to `changed` every node of a cell that `candidate` has and `current` lacks, and of a cell
/// that `current` has and `candidate` lacks. Cells match by nodes (identity and coordinates) and
/// region.
///
/// Both topologies list nodes by identity and cells by sorted vertex indices. The candidate's
/// shared nodes map to current indices in increasing order, so its mapped cell keys keep that
/// order and one merge pairs equal cells. A cell the merge cannot pair counts as changed, so an
/// input out of this order only enlarges the result.
pub(super) fn record_changed_nodes(
    current: &DelaunayVolumeTopology,
    candidate: &DelaunayVolumeTopology,
    changed: &mut BTreeSet<StableDigest>,
) {
    let mut cursor = 0;
    let mapped = candidate
        .nodes
        .iter()
        .map(|node| {
            while cursor < current.nodes.len() && current.nodes[cursor].identity < node.identity {
                cursor += 1;
            }
            (cursor < current.nodes.len() && current.nodes[cursor] == *node)
                .then_some(cursor as u32)
        })
        .collect::<Vec<_>>();
    let sorted = |mut key: [u32; 4]| {
        key.sort_unstable();
        key
    };
    let mut add = |nodes: &[crate::cdt::DelaunayVolumeNode], vertices: [u32; 4]| {
        changed.extend(vertices.map(|vertex| nodes[vertex as usize].identity));
    };
    let mut next = 0;
    for cell in &candidate.tetrahedra {
        let vertices = cell.vertex_indices.map(|vertex| mapped[vertex as usize]);
        let [Some(a), Some(b), Some(c), Some(d)] = vertices else {
            add(&candidate.nodes, cell.vertex_indices);
            continue;
        };
        let key = sorted([a, b, c, d]);
        while next < current.tetrahedra.len()
            && sorted(current.tetrahedra[next].vertex_indices) < key
        {
            add(&current.nodes, current.tetrahedra[next].vertex_indices);
            next += 1;
        }
        if next < current.tetrahedra.len()
            && sorted(current.tetrahedra[next].vertex_indices) == key
            && current.tetrahedra[next].region_id == cell.region_id
        {
            next += 1;
        } else {
            add(&candidate.nodes, cell.vertex_indices);
        }
    }
    for cell in &current.tetrahedra[next..] {
        add(&current.nodes, cell.vertex_indices);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdt::{
        topology::build_delaunay_volume_topology_with_regions, DelaunayTopologyOptions,
        DelaunayVolumeNode,
    };
    use runmat_meshing_core::NeverCancelled;

    fn node(seed: u8, coordinates: [f64; 3]) -> DelaunayVolumeNode {
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([seed; 32]), coordinates)
    }

    fn topology(nodes: Vec<DelaunayVolumeNode>, cells: Vec<[u32; 4]>) -> DelaunayVolumeTopology {
        build_delaunay_volume_topology_with_regions(
            nodes,
            cells.into_iter().map(|cell| (cell, None)).collect(),
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap()
    }

    /// Two cells on a shared face, and the same pair after the apex of the second moves to a
    /// new node: only the moved cell's nodes, and the new node, are changed.
    #[test]
    fn a_moved_apex_changes_only_its_cell() {
        let base = vec![
            node(1, [0., 0., 0.]),
            node(2, [1., 0., 0.]),
            node(3, [0., 1., 0.]),
            node(4, [0., 0., 1.]),
            node(5, [0., 0., -1.]),
        ];
        let current = topology(base.clone(), vec![[0, 1, 2, 3], [0, 1, 2, 4]]);
        let mut moved = base[..4].to_vec();
        moved.push(node(6, [0.1, 0.1, -1.]));
        let candidate = topology(moved, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);
        let mut changed = BTreeSet::new();
        record_changed_nodes(&current, &candidate, &mut changed);
        let expected = [1, 2, 3, 5, 6]
            .map(|seed| StableDigest::from_bytes([seed; 32]))
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(changed, expected);
        changed.clear();
        record_changed_nodes(&current, &current, &mut changed);
        assert!(changed.is_empty());
    }
}
