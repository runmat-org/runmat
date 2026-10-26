use super::*;
use runmat_meshing_core::{NeverCancelled, StableDigest};
fn node(id: u8, coordinates: [f64; 3]) -> DelaunayVolumeNode {
    DelaunayVolumeNode::explicit(StableDigest::from_bytes([id; 32]), coordinates)
}
fn fixture() -> (TopologyMutation, BTreeSet<[StableDigest; 3]>) {
    let nodes = vec![
        node(1, [0., 0., 0.]),
        node(2, [1., 0., 0.]),
        node(3, [0., 1., 0.]),
        node(4, [0., 0., 1.]),
    ];
    let topology = crate::cdt::build_delaunay_volume_topology(
        nodes,
        vec![[0, 1, 2, 3]],
        Default::default(),
        &NeverCancelled,
    )
    .unwrap();
    let barriers = topology
        .incidence
        .boundary_facets
        .iter()
        .map(|f| {
            let mut ids = f
                .vertex_indices
                .map(|v| topology.nodes[v as usize].identity);
            ids.sort_unstable();
            ids
        })
        .collect();
    (TopologyMutation::from_canonical(topology), barriers)
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Explicit conforming insertion splits the complete containing cell, internal face or
//   edge support, preserves the exact outer boundary, retains every point and passes independent
//   canonical admission. A protected boundary split rejects without changing the original state.
//   This contract does not assert Delaunay empty-sphere quality or original CAD convergence.
#[test]
fn conforming_cell_face_and_edge_insertions_preserve_the_exact_outer_boundary() {
    let (mut topology, barriers) = fixture();
    for (id, p) in [
        (5, [0.25, 0.25, 0.25]),
        (6, [0.5, 0.125, 0.125]),
        (7, [0.125, 0.125, 0.125]),
    ] {
        insert_conforming_volume_node_into_mutation_with_barriers(
            &mut topology,
            node(id, p),
            &barriers,
            Default::default(),
            &NeverCancelled,
        )
        .unwrap();
        let snapshot = topology
            .validated_snapshot(Default::default(), &NeverCancelled)
            .unwrap();
        let actual = snapshot
            .incidence
            .boundary_facets
            .iter()
            .map(|f| {
                let mut ids = f
                    .vertex_indices
                    .map(|v| snapshot.nodes[v as usize].identity);
                ids.sort_unstable();
                ids
            })
            .collect();
        assert_eq!(barriers, actual);
        assert!(snapshot
            .incidence
            .vertex_stars
            .iter()
            .all(|star| !star.is_empty()));
        assert_eq!(snapshot.nodes.len(), id as usize);
    }
    let before = topology
        .validated_snapshot(Default::default(), &NeverCancelled)
        .unwrap();
    assert!(insert_conforming_volume_node_into_mutation_with_barriers(
        &mut topology,
        node(8, [0.25, 0.25, 0.]),
        &barriers,
        Default::default(),
        &NeverCancelled
    )
    .is_err());
    let after = topology
        .validated_snapshot(Default::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(
        serde_json::to_value(before).unwrap(),
        serde_json::to_value(after).unwrap()
    );
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Exact predicates establish a point outside a chosen tetrahedron but inside its
//   Euclidean circumsphere. Explicit conforming insertion retains that unaffected tetrahedron and
//   passes independent physical topology admission. Quality refill therefore does not silently
//   impose Euclidean conflict growth; callers requesting Delaunay insertion retain their separate
//   operator. This is not a whole-study accuracy or throughput claim.
#[test]
fn conforming_refill_keeps_an_unaffected_tetrahedron_inside_the_euclidean_conflict_sphere() {
    let (mut topology, barriers) = fixture();
    insert_conforming_volume_node_into_mutation_with_barriers(
        &mut topology,
        node(5, [0.2, 0.2, 0.2]),
        &barriers,
        Default::default(),
        &NeverCancelled,
    )
    .unwrap();
    let protected_cell = [1, 2, 3, 4];
    let index = topology.tetrahedron_index(protected_cell).unwrap();
    let vertices = topology.tetrahedron(index).unwrap().vertex_indices;
    let inserted = node(6, [0.65, 0.3, 0.01]);
    let mut work = InsertionWork::new(Default::default(), &NeverCancelled);
    assert!(!point_in_tetrahedron(&topology, vertices, inserted, &mut work).unwrap());
    assert_eq!(
        cavity::circumsphere_relation(&topology, vertices, inserted, &mut work).unwrap(),
        PredicateSign::Positive
    );
    insert_conforming_volume_node_into_mutation_with_barriers(
        &mut topology,
        inserted,
        &barriers,
        Default::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert!(
        topology.tetrahedron_index(protected_cell).is_some(),
        "Euclidean sphere growth must not erase a geometrically unaffected quality cell"
    );
    let result = topology
        .freeze(Default::default(), &NeverCancelled)
        .unwrap();
    assert_eq!(result.nodes.len(), 6);
    assert!(result
        .incidence
        .vertex_stars
        .iter()
        .all(|star| !star.is_empty()));
}
