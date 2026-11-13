use super::*;
use crate::cdt::insertion::validate_delaunay_volume_topology;
use crate::cdt::{DelaunayInsertionOptions, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_meshing_core::NeverCancelled;

#[test]
fn local_repair_removes_an_illegal_three_tetrahedron_edge_star() {
    let points = [
        [0., 0., -2.],
        [0., 0., 2.],
        [1., 0., 0.],
        [-0.5, 1., 0.],
        [-0.5, -1., 0.],
    ];
    let original = build_delaunay_volume_topology_with_regions(
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
            ([0, 1, 2, 3], None),
            ([0, 1, 3, 4], None),
            ([0, 1, 4, 2], None),
        ],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let options = DelaunayInsertionOptions::default();
    assert!(validate_delaunay_volume_topology(&original, options, &NeverCancelled).is_err());
    let mut work = InsertionWork::new(options, &NeverCancelled);
    let repaired = restore(original.clone(), &BTreeSet::new(), true, &mut work).unwrap();
    assert_eq!(repaired.nodes, original.nodes);
    assert_eq!(repaired.tetrahedra.len(), 2);
    let boundary = |topology: &DelaunayVolumeTopology| {
        topology
            .incidence
            .boundary_facets
            .iter()
            .map(|facet| facet.vertex_indices)
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(boundary(&repaired), boundary(&original));
    validate_delaunay_volume_topology(&repaired, options, &NeverCancelled).unwrap();
}

#[test]
fn neighbor_driven_reinsertion_repairs_an_illegal_four_tetrahedron_star() {
    let points = [
        [0., 0., -2.],
        [0., 0., 2.],
        [1., 0., 0.],
        [0., 1., 0.],
        [-1., 0., 0.],
        [0., -1., 0.],
    ];
    let original = build_delaunay_volume_topology_with_regions(
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
            ([0, 1, 2, 3], None),
            ([0, 1, 3, 4], None),
            ([0, 1, 4, 5], None),
            ([0, 1, 5, 2], None),
        ],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let options = DelaunayInsertionOptions::default();
    validate_delaunay_volume_topology(&original, options, &NeverCancelled).unwrap_err();
    let mut work = InsertionWork::new(options, &NeverCancelled);
    assert!(
        first_reducible_edge(&original, &BTreeSet::new(), true, true, &mut work)
            .unwrap()
            .is_none()
    );
    assert!(first_illegal_face(
        &original,
        &BTreeSet::new(),
        true,
        None,
        SphereRelationPolicy::PhysicalAdmission,
        &mut work
    )
    .unwrap()
    .is_none());
    let repaired = restore(original.clone(), &BTreeSet::new(), true, &mut work).unwrap();
    assert_eq!(repaired.nodes, original.nodes);
    let boundary = |topology: &DelaunayVolumeTopology| {
        topology
            .incidence
            .boundary_facets
            .iter()
            .map(|facet| facet.vertex_indices)
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(boundary(&repaired), boundary(&original));
    validate_delaunay_volume_topology(&repaired, options, &NeverCancelled).unwrap();
}

#[test]
fn local_repair_preserves_a_legal_cospherical_edge_star() {
    let points = [
        [0., 0., -5.],
        [0., 0., 5.],
        [3., 4., 0.],
        [-3., 4., 0.],
        [0., -5., 0.],
    ];
    let original = build_delaunay_volume_topology_with_regions(
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
            ([0, 1, 2, 3], None),
            ([0, 1, 3, 4], None),
            ([0, 1, 4, 2], None),
        ],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let options = DelaunayInsertionOptions::default();
    validate_delaunay_volume_topology(&original, options, &NeverCancelled).unwrap();
    let mut work = InsertionWork::new(options, &NeverCancelled);
    let repaired = restore(original.clone(), &BTreeSet::new(), true, &mut work).unwrap();
    assert_eq!(
        repaired, original,
        "repair must preserve an already valid cospherical star"
    );
}

mod fork_boundary;

#[test]
fn cavity_admission_rejects_stranding_an_original_interior_point() {
    let nodes = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0.25, 0.25, 0.25],
    ]
    .into_iter()
    .enumerate()
    .map(|(index, coordinates)| {
        DelaunayVolumeNode::explicit(
            StableDigest::from_bytes([(index + 1) as u8; 32]),
            coordinates,
        )
    })
    .collect();
    let original = build_delaunay_volume_topology_with_regions(
        nodes,
        vec![
            ([4, 1, 2, 3], None),
            ([0, 4, 2, 3], None),
            ([0, 1, 4, 3], None),
            ([0, 1, 2, 4], None),
        ],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let removed = [0, 1, 2, 3];
    let created = [[0, 1, 2, 3]];
    let old_boundary = local_tetrahedron_boundary_faces(
        &original
            .tetrahedra
            .iter()
            .map(|tet| tet.vertex_indices)
            .collect::<Vec<_>>(),
    );
    assert_eq!(local_tetrahedron_boundary_faces(&created), old_boundary);
    assert!(!prospective_tetrahedra_form_embedded_manifold(&original, &removed, &created).unwrap());
}
