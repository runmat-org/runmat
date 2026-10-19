use super::*;
use crate::cdt::{DelaunayInsertionOptions, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_meshing_core::NeverCancelled;

#[test]
fn reinsertion_keeps_cavity_boundary_vertices_without_shrinking_the_hull() {
    let topology = build_delaunay_volume_topology_with_regions(
        [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]
            .into_iter()
            .enumerate()
            .map(|(index, coordinates)| {
                DelaunayVolumeNode::explicit(
                    StableDigest::from_bytes([(index + 1) as u8; 32]),
                    coordinates,
                )
            })
            .collect(),
        vec![([0, 1, 2, 3], None)],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut cavity = BTreeSet::from([0]);
    let mut work = InsertionWork::new(DelaunayInsertionOptions::default(), &NeverCancelled);
    shrink_reinsertion_cavity(&topology, &mut cavity, 0, &mut work).unwrap();
    assert_eq!(cavity, BTreeSet::from([0]));
}

#[test]
fn expanded_reinsertion_cavity_preserves_an_interior_admitted_point() {
    let points = [
        [0., 0., 0.],
        [2., 0., 0.],
        [0., 2., 0.],
        [0., 0., 2.],
        [0.25, 0.25, 0.25],
    ];
    let topology = build_delaunay_volume_topology_with_regions(
        points
            .into_iter()
            .enumerate()
            .map(|(index, coordinates)| {
                DelaunayVolumeNode::explicit(
                    StableDigest::from_bytes([(index + 1) as u8; 32]),
                    coordinates,
                )
            })
            .collect(),
        vec![
            ([0, 1, 2, 4], None),
            ([0, 1, 3, 4], None),
            ([0, 2, 3, 4], None),
            ([1, 2, 3, 4], None),
        ],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut cavity = (0..topology.tetrahedra.len()).collect::<BTreeSet<_>>();
    let mut work = InsertionWork::new(DelaunayInsertionOptions::default(), &NeverCancelled);
    shrink_reinsertion_cavity(&topology, &mut cavity, 0, &mut work).unwrap();
    let retained = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !cavity.contains(index))
        .collect::<Vec<_>>();
    assert!(!retained.is_empty());
    assert!(retained
        .iter()
        .any(|(_, tetrahedron)| tetrahedron.vertex_indices.contains(&4)));
    let boundary = local_tetrahedron_boundary_faces(
        &cavity
            .iter()
            .map(|index| topology.tetrahedra[*index].vertex_indices)
            .collect::<Vec<_>>(),
    );
    assert!(boundary.iter().any(|face| face.contains(&4)));
    assert_eq!(cavity.len(), 3);
}
