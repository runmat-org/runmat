use super::*;

#[test]
fn curve_driven_cad_surface_uses_exact_face_domain_samples() {
    let topology = single_triangle_topology();
    let cad_topology =
        build_cad_topology(&geometry_with_face_domain_sample(), &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    assert_eq!(cad_evaluation.report.evaluator_rejected_sample_count, 1);
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 1.0,
            min_segments_per_edge: 1,
            max_segments_per_edge: 1,
        },
    )
    .expect("curves should discretize");

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 1,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("cad-owned curve surface should discretize");

    assert_eq!(surface.nodes.len(), topology.vertices.len() + 1);
    assert!(surface.elements.len() >= 2);
    assert_eq!(surface.exact_cad_sample_node_count, 1);
    assert_eq!(surface.rejected_exact_cad_sample_count, 0);
    assert!(surface
        .nodes
        .iter()
        .any(|node| node.coordinates_m == [0.25, 0.25, 0.0]));
    assert!(
        (surface
            .elements
            .iter()
            .map(|element| element.area_m2)
            .sum::<f64>()
            - topology.faces[0].area_m2)
            .abs()
            <= 1.0e-12
    );
    assert!(surface
        .elements
        .iter()
        .all(|element| element.cad_face_id == Some("face_a".to_string())));
}

#[test]
fn curve_driven_cad_surface_preserves_area_with_multiple_exact_samples() {
    let topology = single_triangle_topology();
    let cad_topology = build_cad_topology(&geometry_with_area_regressing_face_samples(), &topology)
        .expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 1.0,
            min_segments_per_edge: 1,
            max_segments_per_edge: 1,
        },
    )
    .expect("curves should discretize");

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 1,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("cad-owned curve surface should discretize");
    let recovered_area = surface
        .elements
        .iter()
        .filter(|element| element.source_face_id == 7)
        .map(|element| element.area_m2)
        .sum::<f64>();

    assert_eq!(surface.nodes.len(), topology.vertices.len() + 3);
    assert_eq!(surface.elements.len(), 7);
    assert_eq!(surface.exact_cad_sample_node_count, 3);
    assert_eq!(surface.rejected_exact_cad_sample_count, 0);
    assert!((recovered_area - topology.faces[0].area_m2).abs() <= 1.0e-12);
    assert!(surface
        .elements
        .iter()
        .all(|element| element.cad_face_id == Some("face_a".to_string())));
}

#[test]
fn curve_driven_cad_surface_splits_edge_hit_exact_samples_without_cracks() {
    let topology = single_triangle_topology();
    let cad_topology = build_cad_topology(&geometry_with_edge_hit_face_samples(), &topology)
        .expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 1.0,
            min_segments_per_edge: 1,
            max_segments_per_edge: 1,
        },
    )
    .expect("curves should discretize");

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 1,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("cad-owned curve surface should discretize");
    let recovered_area = surface
        .elements
        .iter()
        .filter(|element| element.source_face_id == 7)
        .map(|element| element.area_m2)
        .sum::<f64>();

    assert_eq!(surface.exact_cad_sample_node_count, 2);
    assert_eq!(surface.rejected_exact_cad_sample_count, 0);
    assert!((recovered_area - topology.faces[0].area_m2).abs() <= 1.0e-12);
    assert_local_surface_edges_are_recovered(&surface.elements);
}

#[test]
fn curve_driven_cad_surface_rejects_samples_outside_concave_trim_loop() {
    let mut topology = single_triangle_topology();
    topology.faces[0].area_m2 = 0.275;
    let cad_topology = build_cad_topology(&geometry_with_concave_trim_rejected_sample(), &topology)
        .expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let curves = concave_trim_curve_discretization();

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 2,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("concave trimmed surface should discretize");
    let recovered_area = surface
        .elements
        .iter()
        .filter(|element| element.source_face_id == 7)
        .map(|element| element.area_m2)
        .sum::<f64>();
    let trim_loop = [[0.0, 0.0], [0.5, 0.45], [1.0, 0.0], [0.0, 1.0]];

    assert_eq!(surface.exact_cad_sample_node_count, 0);
    assert_eq!(surface.rejected_exact_cad_sample_count, 1);
    assert!((recovered_area - topology.faces[0].area_m2).abs() <= 1.0e-12);
    assert!(!surface
        .nodes
        .iter()
        .any(|node| node.coordinates_m == [0.5, 0.2, 0.0]));
    assert!(surface.elements.iter().all(|element| {
        let centroid = triangle_centroid_2d(element.node_ids.map(|node_id| {
            let point = surface.nodes[node_id as usize].coordinates_m;
            [point[0], point[1]]
        }));
        point_in_polygon_2d(centroid, &trim_loop)
    }));
    assert_surface_edges_are_recovered(&surface.elements, &[[0, 3], [1, 3], [1, 2], [0, 2]]);
}
