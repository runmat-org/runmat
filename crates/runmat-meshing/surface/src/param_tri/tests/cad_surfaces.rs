use super::*;

#[test]
fn discretizes_surfaces_with_cad_face_ownership() {
    let topology = single_triangle_topology();
    let cad_topology =
        build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

    let surface = discretize_cad_surfaces(
        &topology,
        &cad_evaluation,
        SurfaceDiscretizationOptions::default(),
    )
    .expect("cad-owned surface should discretize");

    assert_eq!(surface.elements.len(), 1);
    assert_eq!(
        surface.elements[0].cad_face_id,
        Some("cad_face_7".to_string())
    );
    assert_eq!(surface.elements[0].parametric_node_uv.len(), 3);
    assert_eq!(surface.elements[0].max_projection_error_m, 0.0);
}

#[test]
fn centroid_subdivision_preserves_cad_face_ownership_and_boundary_edges() {
    let topology = single_triangle_topology();
    let cad_topology =
        build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

    let surface = discretize_cad_surfaces(
        &topology,
        &cad_evaluation,
        SurfaceDiscretizationOptions {
            centroid_subdivision: true,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("cad-owned surface should subdivide");

    assert_eq!(surface.nodes.len(), 4);
    assert_eq!(surface.elements.len(), 3);
    assert!(surface
        .elements
        .iter()
        .all(|element| element.cad_face_id == Some("cad_face_7".to_string())));
    assert!(surface
        .elements
        .iter()
        .any(|element| { element.node_ids[0..2] == [0, 1] && element.source_edge_ids[0] == 0 }));
    assert_eq!(surface.nodes[3].coordinates_m, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
}

#[test]
fn curve_driven_cad_surface_uses_curve_boundary_nodes() {
    let topology = single_triangle_topology();
    let cad_topology =
        build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 0.25,
            min_segments_per_edge: 2,
            max_segments_per_edge: 2,
        },
    )
    .expect("curves should discretize");

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 2,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("cad-owned curve surface should discretize");

    assert_eq!(surface.elements.len(), 4);
    assert!(surface.nodes.len() > topology.vertices.len());
    let curve_validation = surface
        .curve_boundary_validation
        .as_ref()
        .expect("curve boundary validation evidence");
    assert_eq!(curve_validation.source_edge_count, topology.edges.len());
    assert_eq!(curve_validation.curve_node_count, curves.nodes.len());
    assert_eq!(curve_validation.curve_element_count, curves.elements.len());
    assert_eq!(curve_validation.max_endpoint_error_m, 0.0);
    let loop_coverage = surface
        .loop_coverage
        .as_ref()
        .expect("surface loop coverage evidence");
    assert_eq!(loop_coverage.source_face_count, topology.faces.len());
    assert_eq!(loop_coverage.recovered_face_count, topology.faces.len());
    assert_eq!(loop_coverage.boundary_loop_count, topology.faces.len());
    assert_eq!(
        loop_coverage.recovered_source_edge_count,
        topology.edges.len()
    );
    assert_eq!(loop_coverage.boundary_segment_count, curves.elements.len());
    assert_eq!(loop_coverage.max_loops_per_face, 1);
    assert!(surface
        .elements
        .iter()
        .all(|element| element.cad_face_id == Some("cad_face_7".to_string())));
    assert!(surface.elements.iter().any(|element| {
        element
            .source_edge_ids
            .iter()
            .any(|edge_id| *edge_id != INTERNAL_SOURCE_EDGE_ID)
    }));
    assert!(surface.elements.iter().any(|element| {
        element
            .source_edge_ids
            .iter()
            .any(|edge_id| *edge_id == INTERNAL_SOURCE_EDGE_ID)
    }));
}

#[test]
fn curve_driven_cad_surface_without_exact_samples_avoids_lattice_nodes() {
    let topology = single_triangle_topology();
    let cad_topology =
        build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 0.25,
            min_segments_per_edge: 4,
            max_segments_per_edge: 4,
        },
    )
    .expect("curves should discretize");

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 4,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("cad-owned curve surface should discretize");

    assert_eq!(surface.exact_cad_sample_node_count, 0);
    assert_eq!(surface.nodes.len(), topology.vertices.len() + 9);
    assert_eq!(surface.elements.len(), 10);
    assert_eq!(
        surface
            .loop_coverage
            .as_ref()
            .expect("surface loop coverage evidence")
            .boundary_segment_count,
        curves.elements.len()
    );
}

#[test]
fn curve_driven_cad_surface_preserves_single_triangle_loop_without_extra_fan_node() {
    let topology = single_triangle_topology();
    let cad_topology =
        build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 10.0,
            min_segments_per_edge: 1,
            max_segments_per_edge: 1,
        },
    )
    .expect("curves should discretize");

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions::default(),
    )
    .expect("cad-owned curve surface should discretize");

    assert_eq!(surface.nodes.len(), topology.vertices.len());
    assert_eq!(surface.elements.len(), 1);
    assert!(surface.curve_boundary_validation.is_some());
    assert_eq!(
        surface
            .loop_coverage
            .as_ref()
            .expect("surface loop coverage evidence")
            .boundary_segment_count,
        3
    );
    assert_eq!(surface.elements[0].node_ids, [0, 1, 2]);
    assert!(surface.elements[0]
        .source_edge_ids
        .iter()
        .all(|edge_id| *edge_id != INTERNAL_SOURCE_EDGE_ID));
}

#[test]
fn curve_boundary_fan_orients_elements_to_cad_frame() {
    let face = single_triangle_topology().faces[0].clone();
    let frame = planar_test_frame(face.face_id);
    let mut nodes = vec![
        SurfaceNode {
            node_id: 0,
            source_vertex_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        SurfaceNode {
            node_id: 1,
            source_vertex_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        SurfaceNode {
            node_id: 2,
            source_vertex_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
    ];
    let segments = [
        FaceCurveSegment {
            node_ids: [1, 0],
            source_edge_id: 0,
        },
        FaceCurveSegment {
            node_ids: [2, 1],
            source_edge_id: 1,
        },
        FaceCurveSegment {
            node_ids: [0, 2],
            source_edge_id: 2,
        },
    ];
    let mut elements = Vec::<SurfaceElement>::new();

    append_curve_fan_face_elements(&face, &frame, &segments, &mut nodes, &mut elements);

    assert_eq!(elements.len(), 3);
    assert!(elements.iter().all(|element| {
        let points = element
            .node_ids
            .map(|node_id| nodes[node_id as usize].coordinates_m);
        dot(
            cross(sub(points[1], points[0]), sub(points[2], points[0])),
            frame.unit_normal,
        ) > 0.0
    }));
    for source_edge_id in [0, 1, 2] {
        assert!(elements
            .iter()
            .any(|element| element.source_edge_ids.contains(&source_edge_id)));
    }
}
