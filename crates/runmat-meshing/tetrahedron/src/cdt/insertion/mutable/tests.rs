use std::collections::BTreeSet;

use runmat_meshing_core::{quality::predicate::PredicateSign, NeverCancelled, StableDigest};

use super::*;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use crate::cdt::{build_delaunay_volume_point_set, DelaunayPointSetOptions};

fn pseudo_random_nodes(count: u32) -> Vec<DelaunayVolumeNode> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1_u64 << 53) as f64)
    };
    (0..count)
        .map(|index| {
            let mut identity = [0_u8; 32];
            identity[..4].copy_from_slice(&(index + 1).to_be_bytes());
            DelaunayVolumeNode::explicit(
                StableDigest::from_bytes(identity),
                [next(), next(), next()],
            )
        })
        .collect()
}

/// The first cell whose circumsphere strictly contains another node, if any.
fn delaunay_violation(topology: &TopologyMutation) -> Option<u32> {
    for index in 0..topology.tetrahedron_slot_count() as u32 {
        let Some(cell) = topology.tetrahedron(index) else {
            continue;
        };
        let mut points = cell.vertex_indices.map(|vertex| topology.nodes[vertex as usize]);
        if orient_volume_nodes(points).unwrap() == PredicateSign::Negative {
            points.swap(0, 1);
        }
        for (query, node) in topology.nodes.iter().enumerate() {
            if cell.vertex_indices.contains(&(query as u32)) {
                continue;
            }
            if topology
                .insphere_nodes([points[0], points[1], points[2], points[3], *node])
                .unwrap()
                == PredicateSign::Positive
            {
                return Some(index);
            }
        }
    }
    None
}

/// Splits interior edge `edge` at its midpoint through the edge star and runs the link flips.
fn split_and_legalize(
    topology: &mut TopologyMutation,
    edge: [u32; 2],
    identity: StableDigest,
    flips: LinkFlips<'_>,
) -> u32 {
    let [x, y] = edge;
    let star = topology.vertex_stars[x as usize]
        .iter()
        .copied()
        .filter(|index| {
            topology
                .tetrahedron(*index)
                .unwrap()
                .vertex_indices
                .contains(&y)
        })
        .collect::<BTreeSet<_>>();
    let midpoint = std::array::from_fn(|axis| {
        0.5 * (topology.nodes[x as usize].coordinates_m.canonical_value()[axis]
            + topology.nodes[y as usize].coordinates_m.canonical_value()[axis])
    });
    let inserted = topology
        .append_node(DelaunayVolumeNode::explicit(identity, midpoint))
        .unwrap();
    let mut created = Vec::new();
    for index in &star {
        let cell = topology.tetrahedron(*index).unwrap();
        for replaced in [x, y] {
            created.push((
                cell.vertex_indices
                    .map(|vertex| if vertex == replaced { inserted } else { vertex }),
                cell.region_id.clone(),
            ));
        }
    }
    topology.replace_cavity(&star, created).unwrap();
    let mut work = InsertionWork::new(DelaunayInsertionOptions::default(), &NeverCancelled);
    let empty = BTreeSet::new();
    legalize_inserted_node(
        topology,
        inserted,
        &mut Protection::new(&empty),
        true,
        flips,
        &mut work,
    )
    .unwrap();
    inserted
}

fn interior_edges(topology: &TopologyMutation, limit: usize) -> Vec<[u32; 2]> {
    let mut edges = BTreeSet::new();
    for index in 0..topology.tetrahedron_slot_count() as u32 {
        let Some(cell) = topology.tetrahedron(index) else {
            continue;
        };
        if cell.neighbors.iter().any(Option::is_none) {
            continue;
        }
        for first in 0..4 {
            for second in first + 1..4 {
                let mut edge = [cell.vertex_indices[first], cell.vertex_indices[second]];
                edge.sort_unstable();
                edges.insert(edge);
            }
        }
    }
    edges
        .into_iter()
        .filter(|edge| {
            topology.vertex_stars[edge[0] as usize].iter().all(|index| {
                let cell = topology.tetrahedron(*index).unwrap();
                !cell.vertex_indices.contains(&edge[1])
                    || cell.neighbors.iter().all(Option::is_some)
            })
        })
        .take(limit)
        .collect()
}

#[test]
fn complete_link_flips_restore_delaunay_after_edge_star_splits() {
    let canonical = build_delaunay_volume_point_set(
        pseudo_random_nodes(48),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let base = TopologyMutation::from_canonical(canonical);
    assert_eq!(delaunay_violation(&base), None);
    let kept = BTreeSet::new();
    let mut two_to_three_stalls = 0;
    let edges = interior_edges(&base, 40);
    assert!(edges.len() >= 20, "fixture has {} interior edges", edges.len());
    for (rank, edge) in edges.into_iter().enumerate() {
        let identity = StableDigest::from_bytes([0xa0 + (rank % 32) as u8; 32]);
        let mut complete = base.clone();
        split_and_legalize(
            &mut complete,
            edge,
            identity,
            LinkFlips::Complete {
                kept_edges: &kept,
                facet_flips: false,
            },
        );
        assert_eq!(
            delaunay_violation(&complete),
            None,
            "complete link flips left an illegal cell after splitting edge {edge:?}"
        );
        let mut partial = base.clone();
        split_and_legalize(&mut partial, edge, identity, LinkFlips::TwoToThree);
        if delaunay_violation(&partial).is_some() {
            two_to_three_stalls += 1;
        }
    }
    // The fixture must exercise link faces that 2-3 flips alone cannot repair.
    assert!(two_to_three_stalls > 0);
}

#[test]
fn complete_link_flips_keep_listed_edges() {
    let canonical = build_delaunay_volume_point_set(
        pseudo_random_nodes(48),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let base = TopologyMutation::from_canonical(canonical);
    let edges = interior_edges(&base, 40);
    let all_edges = {
        let mut keys = BTreeSet::new();
        for index in 0..base.tetrahedron_slot_count() as u32 {
            let Some(cell) = base.tetrahedron(index) else {
                continue;
            };
            for first in 0..4 {
                for second in first + 1..4 {
                    let mut key = [
                        base.nodes[cell.vertex_indices[first] as usize].identity,
                        base.nodes[cell.vertex_indices[second] as usize].identity,
                    ];
                    key.sort_unstable();
                    keys.insert(key);
                }
            }
        }
        keys
    };
    for (rank, edge) in edges.into_iter().enumerate() {
        let mut topology = base.clone();
        split_and_legalize(
            &mut topology,
            edge,
            StableDigest::from_bytes([0xa0 + (rank % 32) as u8; 32]),
            LinkFlips::Complete {
                kept_edges: &all_edges,
                facet_flips: false,
            },
        );
        let split = {
            let mut key = edge.map(|vertex| base.nodes[vertex as usize].identity);
            key.sort_unstable();
            key
        };
        for key in all_edges.iter().filter(|key| **key != split) {
            let indices = key.map(|identity| topology.node_index(identity).unwrap());
            assert!(topology.has_edge(indices), "a 3-2 or 4-4 flip removed kept edge {key:?}");
        }
    }
}
