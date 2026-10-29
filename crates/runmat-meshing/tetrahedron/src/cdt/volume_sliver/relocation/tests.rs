use runmat_meshing_core::{NeverCancelled, StableDigest};
use runmat_meshing_size::metric::MetricTensor3;

use super::geometry::{pattern_search_vertex_star, relocated_star_minimum_jacobian};
use super::{
    grouped_relocation_mutations, rebuild_with_relocated_node, replacement_identity,
    volume_weighted_star_centroid, DelaunayVolumeSliverMutation,
};
use crate::cdt::{
    build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode,
    DelaunayVolumeSliverOptions,
};

#[test]
fn fixed_star_relocation_canonically_remaps_node_indices() {
    let identity = |value| StableDigest::from_bytes([value; 32]);
    let node = |value, point| DelaunayVolumeNode::explicit(identity(value), point);
    let topology = build_delaunay_volume_topology(
        vec![
            node(1, [0.0, 0.0, 0.0]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
        ],
        vec![[0, 1, 2, 3]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut replacement_identity = [3; 32];
    replacement_identity[31] = 4;
    let replacement = DelaunayVolumeNode::explicit(
        StableDigest::from_bytes(replacement_identity),
        [0.8, 0.0, 0.0],
    );

    let rebuilt = rebuild_with_relocated_node(
        &topology,
        1,
        replacement,
        DelaunayVolumeSliverOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
    .expect("same-connectivity relocation should remain admissible");

    assert_eq!(rebuilt.nodes.len(), 4);
    assert_eq!(rebuilt.nodes[2], replacement);
    assert_eq!(rebuilt.tetrahedra.len(), 1);
    assert!(rebuilt.tetrahedra[0].vertex_indices.contains(&2));
}

#[test]
fn cavity_centroid_is_weighted_by_tetrahedron_volume() {
    let identity = |value| StableDigest::from_bytes([value; 32]);
    let node = |value, point| DelaunayVolumeNode::explicit(identity(value), point);
    let topology = build_delaunay_volume_topology(
        vec![
            node(1, [0.0, 0.0, 0.0]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
            node(5, [0.1, 0.1, 0.1]),
        ],
        vec![[4, 0, 1, 2], [4, 0, 3, 1], [4, 0, 2, 3], [4, 1, 3, 2]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    let centroid = volume_weighted_star_centroid(&topology, 4).unwrap();
    for coordinate in centroid {
        assert!((coordinate - 0.25).abs() <= 1.0e-14);
    }
}

#[test]
fn pattern_search_improves_the_worst_incident_tetrahedron() {
    let identity = |value| StableDigest::from_bytes([value; 32]);
    let node = |value, point| DelaunayVolumeNode::explicit(identity(value), point);
    let topology = build_delaunay_volume_topology(
        vec![
            node(1, [0.0, 0.0, 0.0]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
            node(5, [0.02, 0.02, 0.02]),
        ],
        vec![[4, 0, 1, 2], [4, 0, 3, 1], [4, 0, 2, 3], [4, 1, 3, 2]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let source_index = 4;
    let star = &topology.incidence.vertex_stars[source_index];
    let metric = MetricTensor3::isotropic_length_m(1.0).unwrap();
    let start = topology.nodes[source_index].coordinates_m.canonical_value();
    let before = relocated_star_minimum_jacobian(&topology, star, source_index, start, metric);

    let point = pattern_search_vertex_star(
        &topology,
        star,
        source_index,
        start,
        metric,
        &NeverCancelled,
    )
    .unwrap()
    .expect("the off-center interior vertex should be movable");
    let after = relocated_star_minimum_jacobian(&topology, star, source_index, point, metric);

    assert!(after > before);
    assert!(after > 0.1);
}

#[test]
fn grouped_relocations_name_earlier_members_by_their_replacements() {
    let id = |byte: u8| StableDigest::from_bytes([byte; 32]);
    let node = |source: StableDigest| {
        DelaunayVolumeNode::explicit(replacement_identity(source), [0.0, 0.0, 0.0])
    };
    let (a, b, c, d) = (id(1), id(2), id(3), id(4));
    // Both cells were captured before the group moved; b's cell contains a.
    let mutations =
        grouped_relocation_mutations([(a, node(a), [a, b, c, d]), (b, node(b), [a, b, c, d])]);
    let tetrahedra = mutations
        .iter()
        .map(|mutation| match mutation {
            DelaunayVolumeSliverMutation::Relocated(relocation) => {
                relocation.source_tetrahedron_node_identities
            }
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    assert_eq!(tetrahedra[0], [a, b, c, d]);
    assert_eq!(tetrahedra[1], [replacement_identity(a), b, c, d]);
}
