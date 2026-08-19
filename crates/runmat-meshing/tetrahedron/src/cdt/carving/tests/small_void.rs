use super::*;

#[test]
fn carving_preserves_a_small_nested_void() {
    let coordinates = [
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [2.0, 2.0, 2.0],
        [2.0001, 2.0, 2.0],
        [2.0, 2.0001, 2.0],
        [2.0, 2.0, 2.0001],
    ];
    let outer_facets = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let inner_facets = [[4, 5, 6], [4, 7, 5], [4, 6, 7], [5, 7, 6]];
    let solid = DelaunayConstraintFacetSide::Region(region("shell"));
    let mut facets = outer_facets
        .into_iter()
        .enumerate()
        .map(|(index, vertex_indices)| {
            let (positive_side, negative_side) = sides_for_point(
                coordinates,
                vertex_indices,
                [1.0, 1.0, 1.0],
                solid.clone(),
                DelaunayConstraintFacetSide::Exterior,
            );
            constraint_facet(index, vertex_indices, positive_side, negative_side)
        })
        .chain(
            inner_facets
                .into_iter()
                .enumerate()
                .map(|(inner_index, vertex_indices)| {
                    let opposite = (4..8)
                        .find(|vertex| !vertex_indices.contains(vertex))
                        .unwrap();
                    let (positive_side, negative_side) = sides_for_point(
                        coordinates,
                        vertex_indices,
                        coordinates[opposite as usize],
                        DelaunayConstraintFacetSide::Void,
                        solid.clone(),
                    );
                    constraint_facet(
                        inner_index + outer_facets.len(),
                        vertex_indices,
                        positive_side,
                        negative_side,
                    )
                }),
        )
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
    assert!(!carving.removed_tetrahedra.is_empty());
    assert_eq!(carving.topology.incidence.regions.len(), 1);
    assert!(carving.facets.iter().any(|facet| facet.borders_void));
    validate_delaunay_carving(
        &recovery,
        &constraints,
        &carving,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

fn constraint_facet(
    index: usize,
    vertex_indices: [u32; 3],
    positive_side: DelaunayConstraintFacetSide,
    negative_side: DelaunayConstraintFacetSide,
) -> DelaunayConstraintFacet {
    DelaunayConstraintFacet {
        facet_id: StableDigest::from_bytes([(index + 20) as u8; 32]),
        chart_id: StableDigest::from_bytes([(index + 40) as u8; 32]),
        vertex_indices,
        source_face_id: entity(PersistentEntityKind::Face, &format!("nested:{index}")),
        positive_side,
        negative_side,
        contact_ids: Vec::new(),
    }
}
