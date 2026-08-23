use super::*;
use crate::cdt::validate_delaunay_facet_recovery;

#[test]
fn carving_preserves_two_constructively_recovered_steiner_facets() {
    let (coordinates, constraints) = constraints();
    let initial = build_delaunay_volume_point_set(
        constraints.volume_nodes(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let segments = recover_delaunay_segments(
        initial,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let recovery = recover_delaunay_facets(
        segments.clone(),
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let repeated = recover_delaunay_facets(
        segments.clone(),
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(recovery, repeated);
    assert_eq!(recovery.steiner_insertions.len(), 2);
    assert_eq!(
        recovery
            .steiner_insertions
            .iter()
            .map(|insertion| insertion.constraint_index)
            .collect::<Vec<_>>(),
        vec![0, 4]
    );
    assert!(recovery
        .steiner_insertions
        .iter()
        .all(|insertion| insertion.insertion_round == 0 && insertion.candidate_rank == 0));
    validate_delaunay_facet_recovery(
        &recovery,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    let mut forged_identity = recovery.clone();
    forged_identity.steiner_insertions[0].node_identity = StableDigest::from_bytes([0xfe; 32]);
    assert_eq!(
        validate_delaunay_facet_recovery(
            &forged_identity,
            &constraints,
            DelaunayFacetRecoveryOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayFacetRecoveryErrorKind::InvalidTopology
    );
    let mut missing_lineage = recovery.clone();
    missing_lineage.steiner_insertions.clear();
    assert_eq!(
        validate_delaunay_facet_recovery(
            &missing_lineage,
            &constraints,
            DelaunayFacetRecoveryOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayFacetRecoveryErrorKind::InvalidTopology
    );

    let bounded = recover_delaunay_facets(
        segments,
        &constraints,
        DelaunayFacetRecoveryOptions {
            maximum_cavity_steiner_nodes: 1,
            ..DelaunayFacetRecoveryOptions::default()
        },
        &NeverCancelled,
    )
    .unwrap_err();
    assert_eq!(bounded.kind, DelaunayFacetRecoveryErrorKind::ResourceLimit);

    let carving = carve_delaunay_volume(
        &recovery,
        &constraints,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert!(!carving.topology.tetrahedra.is_empty());
    assert!(carving
        .topology
        .incidence
        .unassigned_tetrahedron_indices
        .is_empty());
    assert_eq!(
        carving
            .topology
            .incidence
            .regions
            .iter()
            .map(|region| region.region_id.clone())
            .collect::<Vec<_>>(),
        vec![region("lower"), region("upper")]
    );
    validate_delaunay_carving(
        &recovery,
        &constraints,
        &carving,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    for (facet, interior, region_id) in [
        (
            [0, 1, 2],
            tetrahedron_centroid(coordinates, [0, 1, 2, 3]),
            "lower",
        ),
        (
            [9, 10, 11],
            tetrahedron_centroid(coordinates, [9, 10, 11, 12]),
            "upper",
        ),
    ] {
        let expected = sides_containing_point(coordinates, facet, interior, region_id);
        let actual = constraints
            .facets
            .iter()
            .find(|entry| {
                let mut left = entry.vertex_indices;
                let mut right = facet;
                left.sort_unstable();
                right.sort_unstable();
                left == right
            })
            .unwrap();
        assert_eq!(
            (&actual.positive_side, &actual.negative_side),
            (&expected.0, &expected.1)
        );
    }
}

fn constraints() -> ([[f64; 3]; 18], DelaunayConstraints) {
    let base = [
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [2.107, 1.69, 4.032],
        [0.829, 3.623, -2.904],
        [2.198, 2.453, 1.044],
        [3.133, 1.536, -0.125],
        [1.195, 1.667, 0.446],
        [2.171, 3.458, -0.456],
    ];
    let mut coordinates = [[0.0; 3]; 18];
    coordinates[..9].copy_from_slice(&base);
    coordinates[9..].copy_from_slice(&base.map(|point| [point[0], point[1], point[2] + 20.0]));
    let facet_vertices = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
        [9, 10, 11],
        [9, 10, 12],
        [9, 11, 12],
        [10, 11, 12],
    ];
    let mut facets = facet_vertices
        .into_iter()
        .enumerate()
        .map(|(index, vertex_indices)| {
            let (interior, region_id) = if index < 4 {
                (tetrahedron_centroid(coordinates, [0, 1, 2, 3]), "lower")
            } else {
                (tetrahedron_centroid(coordinates, [9, 10, 11, 12]), "upper")
            };
            let (positive_side, negative_side) =
                sides_containing_point(coordinates, vertex_indices, interior, region_id);
            DelaunayConstraintFacet {
                facet_id: StableDigest::from_bytes([(index + 90) as u8; 32]),
                chart_id: StableDigest::from_bytes([(index + 110) as u8; 32]),
                vertex_indices,
                source_face_id: entity(PersistentEntityKind::Face, &format!("face:{index}")),
                positive_side,
                negative_side,
                contact_ids: Vec::new(),
            }
        })
        .collect::<Vec<_>>();
    facets.sort_by_key(|facet| {
        let mut key = facet.vertex_indices;
        key.sort_unstable();
        (key, facet.facet_id)
    });
    let mut segment_vertices = facets
        .iter()
        .flat_map(|facet| {
            let vertices = facet.vertex_indices;
            [
                [vertices[0], vertices[1]],
                [vertices[1], vertices[2]],
                [vertices[2], vertices[0]],
            ]
            .map(|mut edge| {
                edge.sort_unstable();
                edge
            })
        })
        .collect::<Vec<_>>();
    segment_vertices.sort_unstable();
    segment_vertices.dedup();
    let constraints = DelaunayConstraints {
        nodes: coordinates
            .into_iter()
            .enumerate()
            .map(|(index, coordinates_m)| DelaunayConstraintNode {
                identity: StableDigest::from_bytes([(index + 1) as u8; 32]),
                source_vertex_id: None,
                coordinates_m,
            })
            .collect(),
        segments: segment_vertices
            .into_iter()
            .map(|vertex_indices| DelaunayConstraintSegment {
                vertex_indices,
                source_edge_id: None,
                source_edge_parameters: None,
            })
            .collect(),
        facets,
    };
    (coordinates, constraints)
}

fn tetrahedron_centroid(coordinates: [[f64; 3]; 18], nodes: [usize; 4]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        nodes
            .iter()
            .map(|node| coordinates[*node][axis])
            .sum::<f64>()
            / 4.0
    })
}
