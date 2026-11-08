use super::pick::{project_pick_point, ray_intersects_triangle, PickRay};
use super::*;
use crate::core::ProjectionType;

#[test]
fn geometry_scene_defaults_to_grid_and_presentation_can_override_it() {
    let scene = GeometryScene::new("scene", 1, Vec::new());
    assert!(scene.show_grid);
    let presentation: GeometryScenePresentation =
        serde_json::from_value(serde_json::json!({ "showGrid": false })).unwrap();
    assert_eq!(presentation.show_grid, Some(false));
    assert_eq!(GeometryScenePresentation::default().show_grid, None);
}

#[test]
fn pick_index_returns_region_for_triangle() {
    let material = cad_default_material();
    let chunk = GeometrySceneChunk::indexed_triangles(
        "face_chunk",
        vec![
            vertex([-1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2],
        material,
    )
    .with_regions(vec![GeometrySceneRegion::new(
        "face_a",
        Some("Face A".to_string()),
        Some("cad-face".to_string()),
        vec![GeometrySceneTriangleRange::new(0, 1)],
    )]);
    let scene = GeometryScene::new("scene", 1, vec![chunk]);
    let index = GeometryScenePickIndex::build(&scene);
    let mut camera = Camera::new();
    camera.position = Vec3::new(0.0, 0.0, 5.0);
    camera.target = Vec3::ZERO;
    camera.up = Vec3::Y;
    camera.aspect_ratio = 1.0;
    camera.projection = ProjectionType::Perspective {
        fov: 45.0_f32.to_radians(),
        near: 0.1,
        far: 100.0,
    };
    camera.mark_dirty();

    let hit = index.pick(GeometryScenePickRequest {
        camera,
        surface_size: [800.0, 800.0],
        position: [400.0, 400.0],
        mode: GeometryScenePickMode::Face,
    });
    let hit = hit.expect("center ray should hit the triangle");
    assert_eq!(hit.region_id, Some("face_a".to_string()));
    assert_eq!(hit.vertex_indices, [0, 1, 2]);
    assert!((hit.barycentric_coordinates.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
}

#[test]
fn a_face_pick_takes_the_face_nearest_the_eye_in_every_depth_mode() {
    // Two faces under the cursor, 1 unit apart along the view: the pick is the front one.
    let face = |id: &str, z: f32| {
        GeometrySceneChunk::indexed_triangles(
            id,
            [[-1.0, -1.0, z], [1.0, -1.0, z], [0.0, 1.0, z]]
                .map(|p| vertex(p, [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]))
                .to_vec(),
            vec![0, 1, 2],
            cad_default_material(),
        )
        .with_regions(vec![GeometrySceneRegion::new(
            id,
            None,
            Some("cad-face".to_string()),
            vec![GeometrySceneTriangleRange::new(0, 1)],
        )])
    };
    let scene = GeometryScene::new("scene", 1, vec![face("back", -1.0), face("front", 0.0)]);
    let index = GeometryScenePickIndex::build(&scene);
    for depth_mode in [crate::core::DepthMode::Standard, crate::core::DepthMode::ReversedZ] {
        for parallel in [false, true] {
            let mut camera = Camera::new();
            camera.depth_mode = depth_mode;
            camera.parallel = parallel;
            camera.position = Vec3::new(0.0, 0.0, 5.0);
            camera.target = Vec3::ZERO;
            camera.up = Vec3::Y;
            camera.aspect_ratio = 1.0;
            camera.projection = ProjectionType::Perspective { fov: 45.0_f32.to_radians(), near: 0.1, far: 100.0 };
            camera.mark_dirty();
            let hit = index
                .pick(GeometryScenePickRequest { camera, surface_size: [800.0, 800.0], position: [400.0, 400.0], mode: GeometryScenePickMode::Face })
                .expect("the ray hits both faces");
            assert_eq!(hit.region_id.as_deref(), Some("front"), "{depth_mode:?}, parallel {parallel}");
        }
    }
}

#[test]
fn edge_pick_ignores_face_diagonals_and_prefers_visible_front_edge() {
    fn square(chunk_id: &str, z: f32) -> GeometrySceneChunk {
        GeometrySceneChunk::indexed_triangles(
            chunk_id,
            [
                [-1.0, -1.0, z],
                [1.0, -1.0, z],
                [1.0, 1.0, z],
                [0.0, 1.0, z],
                [-1.0, 1.0, z],
            ]
            .map(|position| vertex(position, [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]))
            .to_vec(),
            vec![0, 1, 2, 0, 2, 3, 0, 3, 4],
            cad_default_material(),
        )
        .with_regions(vec![GeometrySceneRegion::new(
            "face",
            None,
            None,
            vec![GeometrySceneTriangleRange::new(0, 3)],
        )])
    }
    let scene = GeometryScene::new(
        "stacked",
        1,
        vec![square("rear", -1.0), square("front", 0.0)],
    );
    let index = GeometryScenePickIndex::build(&scene);
    let mut camera = Camera::new();
    camera.position = Vec3::new(0.0, 0.0, 5.0);
    camera.target = Vec3::ZERO;
    camera.up = Vec3::Y;
    camera.aspect_ratio = 1.0;
    camera.projection = ProjectionType::Perspective {
        fov: 45.0_f32.to_radians(),
        near: 0.1,
        far: 100.0,
    };
    camera.mark_dirty();
    let top = project_pick_point(
        camera.view_proj_matrix(),
        Vec3::new(0.0, 1.0, 0.0),
        Vec2::splat(800.0),
    )
    .unwrap()
    .0;
    let edge = index
        .pick(GeometryScenePickRequest {
            camera: camera.clone(),
            surface_size: [800.0, 800.0],
            position: [top.x, top.y - 4.0],
            mode: GeometryScenePickMode::Edge,
        })
        .expect("visible top edge should be selectable just outside the face");
    assert_eq!(edge.chunk_id, "front");
    let Some(GeometrySceneTopologySelection::Edge {
        chunk_id, segments, ..
    }) = edge.topology_selection
    else {
        panic!("expected a selected edge");
    };
    assert_eq!(chunk_id, "front");
    assert_eq!(segments.len(), 2);
    assert!(segments
        .iter()
        .any(|segment| *segment == [2, 3] || *segment == [3, 2]));
    assert!(segments
        .iter()
        .any(|segment| *segment == [3, 4] || *segment == [4, 3]));
    assert!(
        index
            .pick(GeometryScenePickRequest {
                camera: camera.clone(),
                surface_size: [800.0, 800.0],
                position: [400.0, 400.0],
                mode: GeometryScenePickMode::Edge,
            })
            .is_none(),
        "the coplanar triangulation diagonal is not a feature edge"
    );
    let corner = project_pick_point(
        camera.view_proj_matrix(),
        Vec3::new(1.0, 1.0, 0.0),
        Vec2::splat(800.0),
    )
    .unwrap()
    .0;
    let point = index
        .pick(GeometryScenePickRequest {
            camera,
            surface_size: [800.0, 800.0],
            position: [corner.x + 4.0, corner.y - 4.0],
            mode: GeometryScenePickMode::Point,
        })
        .expect("visible front vertex should be selectable near the silhouette");
    assert_eq!(
        point.topology_selection,
        Some(GeometrySceneTopologySelection::Point {
            chunk_id: "front".into(),
            vertex_index: 2,
        })
    );
}

#[test]
fn presentation_selection_colors_indexed_triangle_region() {
    let material = cad_default_material();
    let chunk = GeometrySceneChunk::indexed_triangles(
        "face_chunk",
        vec![
            vertex([-1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([2.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2, 1, 3, 2],
        material,
    )
    .with_regions(vec![GeometrySceneRegion::new(
        "face_b",
        Some("Face B".to_string()),
        Some("cad-face".to_string()),
        vec![GeometrySceneTriangleRange::new(1, 1)],
    )]);

    let render_data = chunk.render_data_with_presentation(&GeometryScenePresentation {
        selected_region_ids: vec!["face_b".to_string()],
        ..Default::default()
    });

    assert_eq!(render_data.vertices.len(), 7);
    assert_eq!(
        render_data.indices.as_deref(),
        Some(&[0, 1, 2, 4, 5, 6][..])
    );
    assert_eq!(
        render_data.vertices[4].color,
        SELECTED_PAINT.face([0.5, 0.5, 0.5, 1.0])
    );
    assert_eq!(
        render_data.vertices[5].color,
        SELECTED_PAINT.face([0.5, 0.5, 0.5, 1.0])
    );
    assert_eq!(
        render_data.vertices[6].color,
        SELECTED_PAINT.face([0.5, 0.5, 0.5, 1.0])
    );
}

#[test]
fn wireframe_presentation_draws_only_explicit_edge_chunks() {
    let chunk = GeometrySceneChunk::indexed_triangles(
        "solver-surface",
        vec![
            vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2],
        cad_default_material(),
    );

    let render_data = chunk.render_data_with_presentation(&GeometryScenePresentation {
        display_mode: GeometrySceneDisplayMode::Wireframe,
        ..Default::default()
    });

    assert!(render_data.draw_calls.is_empty());
    let xray = chunk.render_data_with_presentation(&GeometryScenePresentation {
        display_mode: GeometrySceneDisplayMode::Edges,
        ..Default::default()
    });
    assert_eq!(xray.material.albedo.w, 0.38);
    let mut edge_data = chunk.render_data.clone();
    edge_data.pipeline_type = PipelineType::Lines;
    edge_data.indices = None;
    edge_data.vertices.truncate(2);
    edge_data.draw_calls = vec![DrawCall {
        vertex_offset: 0,
        vertex_count: 2,
        index_offset: None,
        index_count: None,
        instance_count: 1,
    }];
    let edge_chunk = GeometrySceneChunk::from_render_data("solver-mesh-edges", edge_data);
    let xray_edges = edge_chunk.render_data_with_presentation(&GeometryScenePresentation {
        display_mode: GeometrySceneDisplayMode::Edges,
        ..Default::default()
    });
    assert_eq!(xray_edges.pipeline_type, PipelineType::LinesNoDepth);
}

#[test]
fn cad_wireframe_excludes_coplanar_tessellation_diagonals() {
    let surface = GeometrySceneChunk::indexed_triangles(
        "cad-surface",
        vec![
            vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2, 1, 3, 2],
        cad_default_material(),
    )
    .with_regions(vec![GeometrySceneRegion::new(
        "cad-face",
        None,
        None,
        vec![GeometrySceneTriangleRange { start: 0, count: 2 }],
    )]);
    let edges = surface.wireframe_edges([0.58, 0.72, 0.88, 0.95]).unwrap();
    assert_eq!(edges.render_data.vertices.len(), 8); // four boundary edges
    let untagged = GeometrySceneChunk {
        regions: Vec::new(),
        ..surface.clone()
    };
    assert_eq!(
        untagged
            .wireframe_edges([0.58, 0.72, 0.88, 0.95])
            .unwrap()
            .render_data
            .vertices
            .len(),
        10
    );
    let shaded = GeometryScenePresentation::default();
    assert!(!surface
        .render_data_with_presentation(&shaded)
        .draw_calls
        .is_empty());
    assert!(!edges
        .render_data_with_presentation(&shaded)
        .draw_calls
        .is_empty());
    assert!(edges
        .render_data_with_presentation(&GeometryScenePresentation {
            edge_overlay_enabled: false,
            ..Default::default()
        })
        .draw_calls
        .is_empty());
    let wireframe = GeometryScenePresentation {
        display_mode: GeometrySceneDisplayMode::Wireframe,
        ..Default::default()
    };
    assert!(surface
        .render_data_with_presentation(&wireframe)
        .draw_calls
        .is_empty());
    assert!(!edges
        .render_data_with_presentation(&wireframe)
        .draw_calls
        .is_empty());
}

#[test]
fn cad_wireframe_keeps_a_boundary_between_coplanar_regions() {
    let surface = GeometrySceneChunk::indexed_triangles(
        "cad-surface",
        vec![
            vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2, 1, 3, 2],
        cad_default_material(),
    )
    .with_regions(vec![
        GeometrySceneRegion::new(
            "left",
            None,
            None,
            vec![GeometrySceneTriangleRange { start: 0, count: 1 }],
        ),
        GeometrySceneRegion::new(
            "right",
            None,
            None,
            vec![GeometrySceneTriangleRange { start: 1, count: 1 }],
        ),
    ]);
    let edges = surface.wireframe_edges([0.58, 0.72, 0.88, 0.95]).unwrap();
    assert_eq!(edges.render_data.vertices.len(), 10); // outer boundary plus shared CAD edge
}

#[test]
fn presentation_selection_colors_direct_triangle_region() {
    let material = cad_default_material();
    let base_color = [0.5, 0.5, 0.5, 1.0];
    let vertices = vec![
        vertex([-1.0, -1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
        vertex([1.0, -1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
        vertex([0.0, 1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
        vertex([1.0, -1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
        vertex([2.0, -1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
        vertex([0.0, 1.0, 0.0], base_color, [0.0, 0.0, 1.0]),
    ];
    let chunk = GeometrySceneChunk::from_render_data(
        "face_chunk",
        RenderData {
            pipeline_type: PipelineType::Triangles,
            vertices,
            indices: None,
            gpu_vertices: None,
            bounds: None,
            material,
            draw_calls: vec![DrawCall {
                vertex_offset: 0,
                vertex_count: 6,
                index_offset: None,
                index_count: None,
                instance_count: 1,
            }],
            image: None,
        },
    )
    .with_regions(vec![GeometrySceneRegion::new(
        "face_b",
        Some("Face B".to_string()),
        Some("cad-face".to_string()),
        vec![GeometrySceneTriangleRange::new(1, 1)],
    )]);

    let render_data = chunk.render_data_with_presentation(&GeometryScenePresentation {
        selected_region_id: Some("face_b".to_string()),
        ..Default::default()
    });

    assert!(render_data.indices.is_none());
    assert_eq!(render_data.vertices.len(), 6);
    assert_eq!(render_data.vertices[0].color, base_color);
    assert_eq!(render_data.vertices[1].color, base_color);
    assert_eq!(render_data.vertices[2].color, base_color);
    assert_eq!(
        render_data.vertices[3].color,
        SELECTED_PAINT.face(base_color)
    );
    assert_eq!(
        render_data.vertices[4].color,
        SELECTED_PAINT.face(base_color)
    );
    assert_eq!(
        render_data.vertices[5].color,
        SELECTED_PAINT.face(base_color)
    );
}

#[test]
fn presentation_resolves_hidden_and_isolated_owner_visibility() {
    let current_hidden = BTreeSet::from(["panel_b".to_string()]);
    let all_owner_ids = ["panel_a", "panel_b", "panel_c"];

    assert_eq!(
        GeometryScenePresentation::default()
            .resolved_hidden_owner_node_ids(all_owner_ids, &current_hidden),
        current_hidden
    );

    assert_eq!(
        GeometryScenePresentation {
            hidden_owner_node_ids: Some(vec![
                " panel_a ".to_string(),
                String::new(),
                "panel_a".to_string(),
            ]),
            ..Default::default()
        }
        .resolved_hidden_owner_node_ids(all_owner_ids, &BTreeSet::new()),
        BTreeSet::from(["panel_a".to_string()])
    );

    assert_eq!(
        GeometryScenePresentation {
            hidden_owner_node_ids: Some(vec!["panel_c".to_string()]),
            isolated_owner_node_ids: Some(vec!["panel_a".to_string()]),
            ..Default::default()
        }
        .resolved_hidden_owner_node_ids(all_owner_ids, &BTreeSet::new()),
        BTreeSet::from(["panel_b".to_string(), "panel_c".to_string()])
    );

    assert_eq!(
        GeometryScenePresentation {
            hidden_owner_node_ids: Some(Vec::new()),
            isolated_owner_node_ids: None,
            ..Default::default()
        }
        .resolved_hidden_owner_node_ids(all_owner_ids, &current_hidden),
        BTreeSet::new()
    );
}

#[test]
fn presentation_section_distinguishes_preserve_clear_and_plane() {
    let preserve: GeometryScenePresentation =
        serde_json::from_value(serde_json::json!({})).expect("preserve presentation");
    assert_eq!(preserve.section, None);

    let clear: GeometryScenePresentation =
        serde_json::from_value(serde_json::json!({ "section": null })).expect("clear presentation");
    assert_eq!(clear.section, Some(None));

    let sectioned: GeometryScenePresentation = serde_json::from_value(serde_json::json!({
        "section": {
            "plane": {
                "normal": [1.0, 0.0, 0.0],
                "origin": [0.0, 0.0, 0.0],
                "label": "midplane"
            }
        }
    }))
    .expect("section presentation");
    let section = sectioned
        .section
        .as_ref()
        .and_then(Option::as_ref)
        .expect("section");
    assert_eq!(section.plane.normal, [1.0, 0.0, 0.0]);
    assert_eq!(section.plane.label.as_deref(), Some("midplane"));
}

#[test]
fn presentation_accepts_bounded_view_presets() {
    let front: GeometryScenePresentation =
        serde_json::from_value(serde_json::json!({ "viewPreset": "front" }))
            .expect("front view presentation");
    assert_eq!(front.view_preset, Some(GeometrySceneViewPreset::Front));

    let isometric: GeometryScenePresentation =
        serde_json::from_value(serde_json::json!({ "viewPreset": "isometric" }))
            .expect("isometric view presentation");
    assert_eq!(
        isometric.view_preset,
        Some(GeometrySceneViewPreset::Isometric)
    );
}

#[test]
fn presentation_section_clips_triangle_render_data() {
    let material = cad_default_material();
    let chunk = GeometrySceneChunk::indexed_triangles(
        "face_chunk",
        vec![
            vertex([-1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2],
        material,
    );
    let render_data = chunk.render_data_with_presentation(&GeometryScenePresentation {
        section: Some(Some(GeometrySceneSection {
            plane: GeometrySceneSectionPlane {
                normal: [1.0, 0.0, 0.0],
                origin: Some([0.0, 0.0, 0.0]),
                offset: None,
                label: Some("midplane".to_string()),
            },
            second_plane: None,
            cap: true,
        })),
        ..Default::default()
    });

    assert_eq!(render_data.pipeline_type, PipelineType::Triangles);
    assert!(render_data.indices.is_none());
    assert_eq!(render_data.vertices.len(), 6);
    assert_eq!(
        render_data.draw_calls[0].vertex_count,
        render_data.vertices.len()
    );
    assert!(render_data
        .vertices
        .iter()
        .all(|vertex| vertex.position[0] >= -SECTION_DISTANCE_EPSILON));
}

#[test]
fn presentation_region_annotations_emit_typed_marker_and_symbol_nodes() {
    let material = cad_default_material();
    let chunk = GeometrySceneChunk::indexed_triangles(
        "face_chunk",
        vec![
            vertex([-1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2],
        material,
    )
    .with_regions(vec![GeometrySceneRegion::new(
        "loaded_face",
        Some("Loaded face".to_string()),
        Some("cad-face".to_string()),
        vec![GeometrySceneTriangleRange::new(0, 1)],
    )]);
    let scene = GeometryScene::new("scene", 1, vec![chunk]);
    let nodes = scene.nodes_with_presentation(&GeometryScenePresentation {
        region_annotations: vec![
            GeometrySceneRegionAnnotation {
                region_id: "loaded_face".to_string(),
                color: Some([0.9, 0.1, 0.1, 1.0]),
                kind: GeometrySceneAnnotationKind::Force,
                label: Some("load".to_string()),
                direction: Some([0.0, 0.0, 1.0]),
                size: Some(18.0),
            },
            GeometrySceneRegionAnnotation {
                region_id: "loaded_face".to_string(),
                color: Some([0.2, 0.3, 0.8, 1.0]),
                kind: GeometrySceneAnnotationKind::InitialCondition,
                label: Some("initial".to_string()),
                direction: None,
                size: Some(18.0),
            },
        ],
        ..Default::default()
    });

    let marker = nodes
        .iter()
        .find(|node| node.name == "FEA region markers")
        .and_then(|node| node.render_data.as_ref())
        .expect("marker annotation node");
    assert_eq!(marker.pipeline_type, PipelineType::Points);
    assert_eq!(marker.vertices.len(), 1);
    assert!((marker.vertices[0].position[1] - (-1.0 / 3.0)).abs() < 1.0e-6);
    assert_eq!(marker.vertices[0].normal[2], 18.0);

    let vector = nodes
        .iter()
        .find(|node| node.name == "FEA study symbols")
        .and_then(|node| node.render_data.as_ref())
        .expect("vector annotation node");
    assert_eq!(vector.pipeline_type, PipelineType::AnnotationStroke);
    assert_eq!(vector.vertices.len(), 10 * 6);
    assert!(vector.vertices[2].position[2] > vector.vertices[0].position[2]);
    assert!(vector
        .vertices
        .iter()
        .all(|vertex| vertex.position[2] >= 0.0));
}

#[test]
fn a_hidden_region_draws_no_annotations() {
    let chunk = GeometrySceneChunk::indexed_triangles(
        "face_chunk",
        vec![
            vertex([-1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2],
        cad_default_material(),
    )
    .with_regions(vec![GeometrySceneRegion::new(
        "loaded_face",
        Some("Loaded face".to_string()),
        Some("cad-face".to_string()),
        vec![GeometrySceneTriangleRange::new(0, 1)],
    )]);
    let scene = GeometryScene::new("scene", 1, vec![chunk]);
    let presentation = |hidden: Vec<String>| GeometryScenePresentation {
        region_annotations: vec![
            GeometrySceneRegionAnnotation {
                region_id: "loaded_face".to_string(),
                color: None,
                kind: GeometrySceneAnnotationKind::InitialCondition,
                label: Some("initial".to_string()),
                direction: None,
                size: Some(18.0),
            },
            GeometrySceneRegionAnnotation {
                region_id: "loaded_face".to_string(),
                color: None,
                kind: GeometrySceneAnnotationKind::Label,
                label: Some("feature.face".to_string()),
                direction: None,
                size: None,
            },
        ],
        hidden_region_ids: hidden,
        ..Default::default()
    };
    let markers = |p: &GeometryScenePresentation| {
        scene.nodes_with_presentation(p).iter().any(|node| node.name == "FEA region markers")
    };
    let shown = presentation(Vec::new());
    assert!(markers(&shown) && shown.shows_region("loaded_face"));
    let hidden = presentation(vec!["loaded_face".to_string()]);
    assert!(!markers(&hidden), "a hidden region keeps its marker");
    assert!(!hidden.shows_region("loaded_face"), "a hidden region keeps its label");
}

#[test]
fn face_annotations_use_one_support_and_a_distributed_load_group() {
    let chunk = GeometrySceneChunk::indexed_triangles(
        "rectangular_face",
        vec![
            vertex([-2.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([2.0, -1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([2.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([-2.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2, 0, 2, 3],
        cad_default_material(),
    )
    .with_regions(vec![GeometrySceneRegion::new(
        "face",
        None,
        None,
        vec![GeometrySceneTriangleRange::new(0, 2)],
    )]);
    let scene = GeometryScene::new("face", 1, vec![chunk]);
    let annotations = [
        GeometrySceneAnnotationKind::Traction,
        GeometrySceneAnnotationKind::FixedSupport,
    ]
    .into_iter()
    .map(|kind| GeometrySceneRegionAnnotation {
        region_id: "face".into(),
        color: None,
        kind,
        label: None,
        direction: Some([0.0, 0.0, 1.0]),
        size: None,
    })
    .collect();
    let nodes = scene.nodes_with_presentation(&GeometryScenePresentation {
        region_annotations: annotations,
        ..Default::default()
    });
    let symbols = nodes
        .iter()
        .find(|node| node.name == "FEA study symbols")
        .and_then(|node| node.render_data.as_ref())
        .unwrap();
    assert_eq!(symbols.pipeline_type, PipelineType::AnnotationStroke);
    // Three filled-head load arrows (ten segments each) and one six-segment support.
    assert_eq!(symbols.vertices.len(), (3 * 10 + 6) * 6);
    assert!(symbols.vertices[0].position != symbols.vertices[60].position);
    assert!(symbols.vertices[60].color[3] > 0.0);
    assert_eq!(symbols.vertices[180].color[3], 0.0);
    // The far endpoint reverses its tangent, so its ribbon side signs reverse too.
    assert_eq!(symbols.vertices[0].tex_coords[0], -1.0);
    assert_eq!(symbols.vertices[2].tex_coords[0], -1.0);
}

#[test]
fn directional_annotation_without_a_direction_uses_a_marker() {
    let chunk = GeometrySceneChunk::indexed_triangles(
        "face",
        vec![
            vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2],
        cad_default_material(),
    )
    .with_regions(vec![GeometrySceneRegion::new(
        "face",
        None,
        None,
        vec![GeometrySceneTriangleRange::new(0, 1)],
    )]);
    let scene = GeometryScene::new("face", 1, vec![chunk]);
    let nodes = scene.nodes_with_presentation(&GeometryScenePresentation {
        region_annotations: vec![GeometrySceneRegionAnnotation {
            region_id: "face".into(),
            color: None,
            kind: GeometrySceneAnnotationKind::HeatFlux,
            label: None,
            direction: None,
            size: None,
        }],
        ..Default::default()
    });
    assert!(nodes.iter().any(|node| node.name == "FEA region markers"));
    assert!(!nodes.iter().any(|node| node.name == "FEA study symbols"));
}

#[test]
fn topology_selection_colors_face_and_draws_edge_and_point() {
    let wire = serde_json::to_value(GeometrySceneTopologySelection::Edge {
        chunk_id: "picked_chunk".into(),
        vertex_indices: [0, 1],
        segments: Vec::new(),
    })
    .unwrap();
    assert_eq!(
        wire,
        serde_json::json!({
            "kind": "edge", "chunkId": "picked_chunk", "vertexIndices": [0, 1]
        })
    );
    let incoming: GeometryScenePresentation = serde_json::from_value(serde_json::json!({
        "selectedTopology": [
            { "kind": "edge", "chunkId": "picked_chunk", "vertexIndices": [0, 1] },
            { "kind": "point", "chunkId": "picked_chunk", "vertexIndex": 2 }
        ]
    }))
    .unwrap();
    assert_eq!(incoming.selected_topology.len(), 2);
    let chunk = GeometrySceneChunk::indexed_triangles(
        "picked_chunk",
        vec![
            vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2],
        cad_default_material(),
    );
    let scene = GeometryScene::new("picked", 1, vec![chunk]);
    let nodes = scene.nodes_with_presentation(&GeometryScenePresentation {
        selected_topology: vec![
            GeometrySceneTopologySelection::Face {
                chunk_id: "picked_chunk".into(),
                triangle_index: 0,
            },
            GeometrySceneTopologySelection::Edge {
                chunk_id: "picked_chunk".into(),
                vertex_indices: [0, 1],
                segments: vec![[0, 1], [1, 2]],
            },
            GeometrySceneTopologySelection::Point {
                chunk_id: "picked_chunk".into(),
                vertex_index: 2,
            },
        ],
        ..Default::default()
    });
    let face = nodes[0].render_data.as_ref().unwrap();
    assert_eq!(face.indices.as_ref().unwrap()[0], 3);
    assert_eq!(
        face.vertices[3].color,
        SELECTED_PAINT.face(face.vertices[0].color)
    );
    let topology = nodes
        .iter()
        .find(|node| node.name == "Selected geometry topology")
        .and_then(|node| node.render_data.as_ref())
        .unwrap();
    assert_eq!(topology.pipeline_type, PipelineType::AnnotationStroke);
    assert_eq!(topology.vertices.len(), 5 * 6);
    assert_eq!(topology.vertices[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(topology.vertices[0].tex_coords[1], 1.25);
    assert_eq!(topology.vertices[6].tex_coords[1], 1.0);
    assert_eq!(topology.vertices[12].position, [1.0, 0.0, 0.0]);
    assert_eq!(topology.vertices[24].position, [0.0, 1.0, 0.0]);
    assert_eq!(topology.vertices[24].tex_coords[1], 2.0);
    assert_eq!(topology.vertices[24].normal, [-1.0, -1.0, 0.0]);
}
#[test]
fn small_cad_face_picking_is_invariant_to_coordinate_scale() {
    for scale in [1.0e-9, 1.0e-3, 1.0, 1.0e9] {
        let ray = PickRay {
            origin: Vec3::new(0.25, 0.25, 1.0) * scale,
            direction: -Vec3::Z,
        };
        let hit = ray_intersects_triangle(&ray, Vec3::ZERO, Vec3::X * scale, Vec3::Y * scale)
            .expect("valid face must remain selectable");
        assert!((hit / scale - 1.0).abs() < 1.0e-6);
    }
}

#[test]
fn scale_independent_picking_still_rejects_parallel_and_behind_faces() {
    for direction in [Vec3::X, Vec3::Z, Vec3::ZERO] {
        let ray = PickRay {
            origin: Vec3::new(0.25, 0.25, 1.0),
            direction,
        };
        assert!(ray_intersects_triangle(&ray, Vec3::ZERO, Vec3::X, Vec3::Y).is_none());
    }
}

#[test]
fn selection_tints_the_face_and_draws_its_boundary_edges_in_the_selection_color() {
    let grey = [0.66, 0.72, 0.8, 1.0];
    let face = GeometrySceneChunk::indexed_triangles(
        "faces",
        vec![
            vertex([0.0, 0.0, 0.0], grey, [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], grey, [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], grey, [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2],
        cad_default_material(),
    )
    .with_regions(vec![GeometrySceneRegion::new(
        "face:0:0",
        None,
        None,
        vec![GeometrySceneTriangleRange::new(0, 1)],
    )]);
    let black = [0.07, 0.08, 0.1, 1.0];
    // Three segments sharing vertices; the first two bound the selected face.
    let mut edges = GeometrySceneChunk::from_render_data(
        "lines",
        RenderData {
            pipeline_type: PipelineType::Lines,
            vertices: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 0.0]]
                .map(|p| vertex(p, black, [0.0, 0.0, 1.0]))
                .to_vec(),
            indices: Some(vec![0, 1, 1, 2, 2, 3]),
            gpu_vertices: None,
            bounds: None,
            material: Material::default(),
            draw_calls: vec![DrawCall {
                vertex_offset: 0,
                vertex_count: 4,
                index_offset: Some(0),
                index_count: Some(6),
                instance_count: 1,
            }],
            image: None,
        },
    );
    edges.segment_regions = vec![GeometrySceneRegion::new(
        "face:0:0",
        None,
        None,
        vec![GeometrySceneTriangleRange::new(0, 2)],
    )];
    let selected = GeometryScenePresentation {
        selected_region_ids: vec!["face:0:0".to_string()],
        ..Default::default()
    };

    let shaded = face.render_data_with_presentation(&selected);
    let tinted = shaded.vertices[shaded.indices.as_ref().unwrap()[0] as usize].color;
    // Part way to the selection tint: lighter than the selection color, at the face's alpha.
    assert_eq!(tinted, SELECTED_PAINT.face(grey));
    assert!(tinted[0] > grey[0] && tinted[0] < GEOMETRY_SELECTED_REGION_COLOR[0]);
    assert_eq!(tinted[3], 1.0);

    let lines = edges.render_data_with_presentation(&selected);
    let indices = lines.indices.clone().unwrap();
    let color = |at: usize| lines.vertices[indices[at] as usize].color;
    for at in 0..4 {
        assert_eq!(color(at), GEOMETRY_SELECTED_REGION_COLOR, "boundary corner {at}");
    }
    // The third segment keeps its colour although it shares vertex 2 with the second.
    assert_eq!(color(4), black);
    assert_eq!(color(5), black);

    let hovered = edges.render_data_with_presentation(&GeometryScenePresentation {
        hovered_region_id: Some("face:0:0".to_string()),
        ..Default::default()
    });
    let hi = hovered.indices.clone().unwrap();
    assert_eq!(hovered.vertices[hi[0] as usize].color, GEOMETRY_HOVER_EDGE_COLOR);

    // Selected edges draw heavier than the rest, hovered ones a little less.
    let stroke = super::edge_stroke::widen(lines, Some(1.25));
    let widths: Vec<f32> = stroke.vertices.iter().map(|v| v.tex_coords[1]).collect();
    assert_eq!(&widths[..12], &[2.5; 12], "the two boundary segments");
    assert_eq!(&widths[12..], &[1.25; 6], "the third segment keeps the edge width");
    let hover_stroke = super::edge_stroke::widen(hovered, Some(1.25));
    assert_eq!(hover_stroke.vertices[0].tex_coords[1], 2.0);

    // On a near-black material (carbon, black nylon) the tint strengthens so the selection reads
    // orange, and hover stays lighter than selection.
    let carbon = [0.05, 0.05, 0.06, 1.0];
    let on_carbon = SELECTED_PAINT.face(carbon);
    assert!(on_carbon[0] > 0.65 && on_carbon[0] > on_carbon[2] * 2.0, "{on_carbon:?}");
    assert!(HOVER_PAINT.face(carbon)[0] < on_carbon[0]);
    // Light materials keep the plain 40 % tint.
    assert!((super::paint::tint_on(grey, 0.4) - 0.4).abs() < 1e-6);
}

#[test]
fn region_colors_fill_faces_keep_edges_and_sit_under_selection() {
    let grey = [0.66, 0.72, 0.8, 0.9];
    let face = GeometrySceneChunk::indexed_triangles(
        "faces",
        vec![
            vertex([0.0, 0.0, 0.0], grey, [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], grey, [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], grey, [0.0, 0.0, 1.0]),
            vertex([1.0, 1.0, 0.0], grey, [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2, 1, 3, 2],
        cad_default_material(),
    )
    .with_regions(vec![
        GeometrySceneRegion::new("face:0:0", None, None, vec![GeometrySceneTriangleRange::new(0, 1)]),
        GeometrySceneRegion::new("face:0:1", None, None, vec![GeometrySceneTriangleRange::new(1, 1)]),
    ]);
    let green = [0.2, 0.7, 0.3, 1.0];
    let colored = GeometryScenePresentation {
        region_colors: vec![GeometrySceneRegionColor { region_id: "face:0:0".into(), color: green }],
        ..Default::default()
    };
    let out = face.render_data_with_presentation(&colored);
    let idx = out.indices.clone().unwrap();
    // The face takes the color at its own alpha; the other face keeps its shade.
    assert_eq!(out.vertices[idx[0] as usize].color, [0.2, 0.7, 0.3, 0.9]);
    assert_eq!(out.vertices[idx[3] as usize].color, grey);
    // Selection tints over the analysis color.
    let both = face.render_data_with_presentation(&GeometryScenePresentation {
        selected_region_ids: vec!["face:0:0".into()],
        ..colored.clone()
    });
    let bi = both.indices.clone().unwrap();
    assert_eq!(both.vertices[bi[0] as usize].color, SELECTED_PAINT.face([0.2, 0.7, 0.3, 0.9]));

    let black = [0.07, 0.08, 0.1, 1.0];
    let mut edges = GeometrySceneChunk::from_render_data(
        "lines",
        RenderData {
            pipeline_type: PipelineType::Lines,
            vertices: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]].map(|p| vertex(p, black, [0.0, 0.0, 1.0])).to_vec(),
            indices: Some(vec![0, 1]),
            gpu_vertices: None,
            bounds: None,
            material: Material::default(),
            draw_calls: vec![DrawCall { vertex_offset: 0, vertex_count: 2, index_offset: Some(0), index_count: Some(2), instance_count: 1 }],
            image: None,
        },
    );
    edges.segment_regions = vec![GeometrySceneRegion::new("face:0:0", None, None, vec![GeometrySceneTriangleRange::new(0, 1)])];
    let lines = edges.render_data_with_presentation(&colored);
    assert!(lines.vertices.iter().all(|v| v.color == black), "edges keep their color");
    // The presentation round-trips through its camelCase JSON form.
    let json = serde_json::to_value(&colored).unwrap();
    assert_eq!(json["regionColors"][0]["regionId"], "face:0:0");
    let back: GeometryScenePresentation = serde_json::from_value(json).unwrap();
    assert_eq!(back, colored);
}

#[test]
fn saved_line_segment_regions_round_trip_and_reject_overruns() {
    let lines: crate::event::GeometryLines = serde_json::from_value(serde_json::json!({
        "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        "colors": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "indices": [0, 1, 1, 2],
        "meshId": null, "label": "Edges", "visible": true, "wireframeOnly": true,
        "segmentRegions": [{ "regionId": "face:0:1", "segmentRanges": [{ "start": 1, "count": 1 }] }]
    }))
    .unwrap();
    let chunk = lines.clone().restore("lines".into()).unwrap();
    assert_eq!(chunk.segment_regions.len(), 1);
    assert_eq!(chunk.segment_regions[0].triangle_ranges, vec![GeometrySceneTriangleRange::new(1, 1)]);
    let mut bad = lines;
    bad.segment_regions[0].segment_ranges[0].count = 2;
    assert!(bad.restore("lines".into()).is_err());
}
