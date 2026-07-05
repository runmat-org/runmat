use super::*;

#[test]
fn curve_driven_face_elements_triangulate_holed_loop_domain() {
    let face = SourceTopologyFace {
        face_id: 7,
        source_triangle_id: 11,
        node_ids: [0, 1, 2],
        edge_ids: [0, 1, 2],
        region_ids: vec!["face_a".to_string()],
        area_m2: 0.96,
        unit_normal: [0.0, 0.0, 1.0],
    };
    let frame = planar_test_frame(7);
    let mut nodes = square_with_square_hole_surface_nodes();
    let segment_loops = vec![
        vec![
            FaceCurveSegment {
                node_ids: [0, 1],
                source_edge_id: 0,
            },
            FaceCurveSegment {
                node_ids: [1, 2],
                source_edge_id: 1,
            },
            FaceCurveSegment {
                node_ids: [2, 3],
                source_edge_id: 2,
            },
            FaceCurveSegment {
                node_ids: [3, 0],
                source_edge_id: 3,
            },
        ],
        vec![
            FaceCurveSegment {
                node_ids: [4, 5],
                source_edge_id: 4,
            },
            FaceCurveSegment {
                node_ids: [5, 6],
                source_edge_id: 5,
            },
            FaceCurveSegment {
                node_ids: [6, 7],
                source_edge_id: 6,
            },
            FaceCurveSegment {
                node_ids: [7, 4],
                source_edge_id: 7,
            },
        ],
    ];
    let mut elements = Vec::<SurfaceElement>::new();

    let report =
        append_curve_driven_face_elements(&face, &frame, &segment_loops, &mut nodes, &mut elements);
    let recovered_area = elements.iter().map(|element| element.area_m2).sum::<f64>();
    let hole = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]];

    assert_eq!(report, ExactCadSampleSurfaceReport::default());
    assert!(!elements.is_empty());
    assert!(
        (recovered_area - face.area_m2).abs() <= 1.0e-12,
        "recovered_area={recovered_area} expected_area={} element_count={}",
        face.area_m2,
        elements.len()
    );
    assert!(elements.iter().all(|element| {
        let centroid = triangle_centroid_2d(element.node_ids.map(|node_id| {
            let point = nodes[node_id as usize].coordinates_m;
            [point[0], point[1]]
        }));
        !point_in_polygon_2d(centroid, &hole)
    }));
    assert_surface_edges_are_recovered(
        &elements,
        &[
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 3],
            [4, 5],
            [5, 6],
            [6, 7],
            [4, 7],
        ],
    );
}

#[test]
fn face_curve_segment_loops_extracts_multiple_closed_loops() {
    let segments = vec![
        FaceCurveSegment {
            node_ids: [0, 1],
            source_edge_id: 0,
        },
        FaceCurveSegment {
            node_ids: [1, 2],
            source_edge_id: 1,
        },
        FaceCurveSegment {
            node_ids: [2, 0],
            source_edge_id: 2,
        },
        FaceCurveSegment {
            node_ids: [3, 4],
            source_edge_id: 3,
        },
        FaceCurveSegment {
            node_ids: [4, 5],
            source_edge_id: 4,
        },
        FaceCurveSegment {
            node_ids: [5, 3],
            source_edge_id: 5,
        },
    ];

    let loops = face_curve_segment_loops(7, &segments).expect("multi-loop face should extract");

    assert_eq!(
        loops
            .iter()
            .map(|loop_segments| loop_segments.len())
            .collect::<Vec<_>>(),
        vec![3, 3]
    );
    assert_eq!(
        loops
            .iter()
            .map(|loop_segments| {
                loop_segments
                    .iter()
                    .map(|segment| segment.source_edge_id)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec![vec![0, 1, 2], vec![3, 4, 5]]
    );
}

#[test]
fn face_curve_segment_loops_order_shuffled_single_loop_deterministically() {
    let segments = vec![
        FaceCurveSegment {
            node_ids: [3, 0],
            source_edge_id: 13,
        },
        FaceCurveSegment {
            node_ids: [1, 2],
            source_edge_id: 11,
        },
        FaceCurveSegment {
            node_ids: [2, 3],
            source_edge_id: 12,
        },
        FaceCurveSegment {
            node_ids: [0, 1],
            source_edge_id: 10,
        },
    ];

    let loops = face_curve_segment_loops(7, &segments).expect("loop should be valid");

    assert_eq!(loops.len(), 1);
    assert_eq!(
        loops[0],
        vec![
            FaceCurveSegment {
                node_ids: [0, 1],
                source_edge_id: 10,
            },
            FaceCurveSegment {
                node_ids: [1, 2],
                source_edge_id: 11,
            },
            FaceCurveSegment {
                node_ids: [2, 3],
                source_edge_id: 12,
            },
            FaceCurveSegment {
                node_ids: [3, 0],
                source_edge_id: 13,
            },
        ]
    );
}

#[test]
fn rejects_open_face_curve_loop_before_triangulation() {
    let segments = vec![
        FaceCurveSegment {
            node_ids: [0, 1],
            source_edge_id: 0,
        },
        FaceCurveSegment {
            node_ids: [1, 2],
            source_edge_id: 1,
        },
    ];

    let err =
        face_curve_segment_loops(7, &segments).expect_err("open face loop should fail closed");

    assert_eq!(
        err,
        SurfaceDiscretizationError::InvalidFaceLoopTopology {
            face_id: 7,
            node_id: 0,
            incident_segment_count: 1,
        }
    );
}
