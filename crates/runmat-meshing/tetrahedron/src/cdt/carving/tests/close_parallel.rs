use super::*;

#[test]
fn carving_preserves_a_close_parallel_face_volume() {
    let coordinates = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0e-8],
        [1.0, 0.0, 1.0e-8],
        [0.0, 1.0, 1.0e-8],
    ];
    let oriented_facets = [
        [0, 2, 1],
        [3, 4, 5],
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4],
        [2, 0, 3],
        [2, 3, 5],
    ];
    let interior = [0.25, 0.25, 0.5e-8];
    let mut facets = oriented_facets
        .into_iter()
        .enumerate()
        .map(|(index, vertex_indices)| {
            let (positive_side, negative_side) =
                sides_containing_point(coordinates, vertex_indices, interior, "thin-prism");
            DelaunayConstraintFacet {
                facet_id: StableDigest::from_bytes([(index + 20) as u8; 32]),
                chart_id: StableDigest::from_bytes([(index + 40) as u8; 32]),
                vertex_indices,
                source_face_id: entity(PersistentEntityKind::Face, &format!("prism:{index}")),
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
    let point_set = build_delaunay_volume_point_set(
        constraints.volume_nodes(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let segments = recover_delaunay_segments(
        point_set,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let recovery = recover_delaunay_facets(
        segments,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let carving = carve_delaunay_volume(
        &recovery,
        &constraints,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert!(!carving.topology.tetrahedra.is_empty());
    assert!(carving.removed_tetrahedra.is_empty());
    assert_eq!(carving.topology.incidence.regions.len(), 1);
    assert_eq!(
        carving.topology.incidence.regions[0].region_id,
        region("thin-prism")
    );
    validate_delaunay_carving(
        &recovery,
        &constraints,
        &carving,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}
