//! Section cuts with caps on and off and a second plane; hidden regions in drawing and picking.
use super::*;
use crate::core::ProjectionType;

/// A closed unit cube centred on the origin, outward-wound, one region per face.
fn cube() -> GeometrySceneChunk {
    let p = |x: f32, y: f32, z: f32| vertex([x, y, z], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]);
    let mut vertices = Vec::new();
    for x in [-1.0, 1.0] {
        for y in [-1.0, 1.0] {
            for z in [-1.0, 1.0] {
                vertices.push(p(x, y, z));
            }
        }
    }
    // Corner index = 4*(x>0) + 2*(y>0) + (z>0); faces wound outward.
    let quads: [[u32; 4]; 6] = [
        [0, 1, 3, 2], // -x
        [4, 6, 7, 5], // +x
        [0, 4, 5, 1], // -y
        [2, 3, 7, 6], // +y
        [0, 2, 6, 4], // -z
        [1, 5, 7, 3], // +z
    ];
    let mut indices = Vec::new();
    for [a, b, c, d] in quads {
        indices.extend([a, b, c, a, c, d]);
    }
    let regions = (0..6u32)
        .map(|face| {
            GeometrySceneRegion::new(
                format!("face_{face}"),
                None,
                None,
                vec![GeometrySceneTriangleRange::new(face * 2, 2)],
            )
        })
        .collect();
    GeometrySceneChunk::indexed_triangles("cube", vertices, indices, cad_default_material())
        .with_regions(regions)
        .with_owner_node_ids(vec!["body:1".to_string()])
}

fn plane(normal: [f32; 3]) -> GeometrySceneSectionPlane {
    GeometrySceneSectionPlane {
        normal,
        origin: Some([0.0, 0.0, 0.0]),
        offset: None,
        label: None,
    }
}

fn cut(section: GeometrySceneSection) -> RenderData {
    cube().render_data_with_presentation(&GeometryScenePresentation {
        section: Some(Some(section)),
        ..Default::default()
    })
}

fn area(render_data: &RenderData) -> f32 {
    render_data
        .vertices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [t[0], t[1], t[2]].map(|v| Vec3::from_array(v.position));
            (b - a).cross(c - a).length() / 2.0
        })
        .sum()
}

#[test]
fn a_capped_cut_closes_the_cube_and_an_open_cut_leaves_it_hollow() {
    // Half a cube: 5 half faces (1 + 4 × 0.5 = 3 faces of area 4), and the cap is one more face.
    let capped = cut(GeometrySceneSection {
        plane: plane([1.0, 0.0, 0.0]),
        second_plane: None,
        cap: true,
    });
    let open = cut(GeometrySceneSection {
        plane: plane([1.0, 0.0, 0.0]),
        second_plane: None,
        cap: false,
    });
    assert!(
        (area(&open) - 12.0).abs() < 1.0e-3,
        "open cut area {}",
        area(&open)
    );
    assert!(
        (area(&capped) - 16.0).abs() < 1.0e-3,
        "capped cut area {}",
        area(&capped)
    );
    assert!(open.vertices.iter().all(|v| v.position[0] >= -1.0e-5));
}

#[test]
fn a_second_plane_cuts_away_one_corner_and_caps_both_cut_faces() {
    // Keeps x ≥ 0 or y ≥ 0: three quarters of the cube. Its surface is three quarters of the
    // cube's 24, and each plane's cap covers the removed quarter's side on it (2 each).
    let open = cut(GeometrySceneSection {
        plane: plane([1.0, 0.0, 0.0]),
        second_plane: Some(plane([0.0, 1.0, 0.0])),
        cap: false,
    });
    assert!(
        (area(&open) - 18.0).abs() < 1.0e-3,
        "open corner cut area {}",
        area(&open)
    );
    assert!(open.vertices.chunks_exact(3).all(|t| t
        .iter()
        .all(|v| v.position[0] >= -1.0e-5 || v.position[1] >= -1.0e-5)));
    let capped = cut(GeometrySceneSection {
        plane: plane([1.0, 0.0, 0.0]),
        second_plane: Some(plane([0.0, 1.0, 0.0])),
        cap: true,
    });
    assert!(
        (area(&capped) - 22.0).abs() < 1.0e-3,
        "capped corner cut area {}",
        area(&capped)
    );
}

#[test]
fn section_presentation_reads_cap_and_second_plane_and_defaults_cap_on() {
    let section: GeometrySceneSection =
        serde_json::from_value(serde_json::json!({ "plane": { "normal": [1, 0, 0] } })).unwrap();
    assert!(section.cap);
    assert!(section.second_plane.is_none());
    let section: GeometrySceneSection = serde_json::from_value(serde_json::json!({
        "plane": { "normal": [1, 0, 0] }, "secondPlane": { "normal": [0, 1, 0] }, "cap": false
    }))
    .unwrap();
    assert!(!section.cap);
    assert_eq!(
        section.second_plane.map(|p| p.normal),
        Some([0.0, 1.0, 0.0])
    );
}

#[test]
fn hidden_regions_draw_nothing_and_picks_pass_through_them() {
    let presentation = GeometryScenePresentation {
        hidden_region_ids: vec!["face_5".to_string()],
        ..Default::default()
    };
    let drawn = cube().render_data_with_presentation(&presentation);
    assert!(
        (area(&drawn) - 20.0).abs() < 1.0e-3,
        "five faces drawn, area {}",
        area(&drawn)
    );

    let scene = GeometryScene::new("scene", 1, vec![cube()]);
    let mut index = GeometryScenePickIndex::build(&scene);
    let mut camera = Camera::new();
    camera.position = Vec3::new(0.0, 0.0, 6.0);
    camera.target = Vec3::ZERO;
    camera.up = Vec3::Y;
    camera.aspect_ratio = 1.0;
    camera.projection = ProjectionType::Perspective {
        fov: 45.0_f32.to_radians(),
        near: 0.1,
        far: 100.0,
    };
    camera.mark_dirty();
    let request = || GeometryScenePickRequest {
        camera: camera.clone(),
        surface_size: [800.0, 800.0],
        position: [400.0, 400.0],
        mode: GeometryScenePickMode::Face,
    };

    let top = index.pick(request()).expect("the +z face faces the eye");
    assert_eq!(top.region_id.as_deref(), Some("face_5"));
    assert_eq!(top.owner_node_ids, vec!["body:1".to_string()]);

    index.set_hidden(&presentation);
    let through = index
        .pick(request())
        .expect("the far face behind the hidden one");
    assert_eq!(through.region_id.as_deref(), Some("face_4"));

    index.set_hidden(&GeometryScenePresentation {
        hidden_owner_node_ids: Some(vec!["body:1".to_string()]),
        ..Default::default()
    });
    assert!(
        index.pick(request()).is_none(),
        "a hidden body is not picked"
    );
}

/// The cap's vertex colors: the cap faces away from the kept side (normal -x for a +x plane).
fn cap_colors(render_data: &RenderData) -> Vec<[f32; 4]> {
    render_data
        .vertices
        .iter()
        .filter(|v| v.normal[0] < -0.9)
        .map(|v| v.color)
        .collect()
}

#[test]
fn a_cap_takes_the_body_color_not_a_setup_tint() {
    let section = Some(GeometrySceneSection {
        plane: plane([1.0, 0.0, 0.0]),
        second_plane: None,
        cap: true,
    });
    let plain = cube().render_data_with_presentation(&GeometryScenePresentation {
        section: Some(section.clone()),
        ..Default::default()
    });
    // A load tint on the -y face, which the cut crosses.
    let tinted = cube().render_data_with_presentation(&GeometryScenePresentation {
        section: Some(section),
        region_highlights: vec![GeometrySceneRegionHighlight {
            region_id: "face_2".to_string(),
            color: Some([0.9, 0.1, 0.5, 1.0]),
            role: None,
            label: None,
        }],
        ..Default::default()
    });
    assert!(!cap_colors(&plain).is_empty());
    assert_eq!(cap_colors(&tinted), cap_colors(&plain));
}

#[test]
fn a_cap_over_a_result_field_is_body_grey() {
    // A result field: white material, the field's colors on the vertices.
    let mut field = cube();
    field.material.albedo = Vec4::ONE;
    field.render_data.material.albedo = Vec4::ONE;
    for v in &mut field.render_data.vertices {
        v.color = [0.95, 0.15, 0.1, 1.0];
    }
    let render_data = field.render_data_with_presentation(&GeometryScenePresentation {
        section: Some(Some(GeometrySceneSection {
            plane: plane([1.0, 0.0, 0.0]),
            second_plane: None,
            cap: true,
        })),
        ..Default::default()
    });
    // The body grey, leaned toward the cut tint as every cap is.
    let body = cad_default_material().albedo;
    let expected = body.lerp(
        Vec4::from_array(super::section_cap::SECTION_CAP_TINT),
        super::section_cap::SECTION_CAP_TINT_SHARE,
    );
    let caps = cap_colors(&render_data);
    assert!(!caps.is_empty());
    for [r, g, b, _] in caps {
        assert!(
            (Vec3::new(r, g, b) - expected.truncate()).length() < 1e-4,
            "a cap keeps no field color: {r} {g} {b}"
        );
    }
}

/// The cube as a CAD import draws it: each face its own chunk of one mesh (body), with that
/// face's feature edges in a line chunk beside it.
fn cube_by_face() -> GeometryScene {
    let whole = cube();
    let indices = whole.indices.clone().unwrap();
    let chunks = (0..6usize)
        .map(|face| {
            let mut chunk = GeometrySceneChunk::indexed_triangles(
                format!("mesh_1:chunk_{face}"),
                whole.vertices.clone(),
                indices[face * 6..face * 6 + 6].to_vec(),
                cad_default_material(),
            )
            .with_mesh_id("mesh_1")
            .with_regions(vec![GeometrySceneRegion::new(
                format!("face_{face}"),
                None,
                None,
                vec![GeometrySceneTriangleRange::new(0, 2)],
            )]);
            chunk.chunk_id = format!("mesh_1:chunk_{face}");
            let edges = chunk.wireframe_edges([0.1, 0.1, 0.1, 1.0]);
            std::iter::once(chunk).chain(edges)
        })
        .flatten()
        .collect::<Vec<_>>();
    // The edge chunks belong to the same body: a cap reads only its surface triangles.
    assert_eq!(chunks.len(), 12);
    assert!(chunks
        .iter()
        .all(|chunk| chunk.mesh_id.as_deref() == Some("mesh_1")));
    GeometryScene::new("cube-by-face", 1, chunks)
}

fn cap_nodes(presentation: &GeometryScenePresentation) -> Vec<RenderData> {
    cube_by_face()
        .nodes_with_presentation(presentation)
        .into_iter()
        .filter(|node| node.name == "Section cap")
        .filter_map(|node| node.render_data)
        .collect()
}

#[test]
fn a_body_drawn_one_face_per_chunk_is_capped_across_its_chunks_in_the_body_color() {
    let section = |cap| {
        Some(Some(GeometrySceneSection {
            plane: plane([0.0, 0.0, -1.0]),
            second_plane: None,
            cap,
        }))
    };
    let plain = GeometryScenePresentation {
        section: section(true),
        ..Default::default()
    };
    let caps = cap_nodes(&plain);
    assert_eq!(caps.len(), 1, "one cap for the one body");
    // The whole 2 x 2 cut face, facing the removed side (+z).
    assert!((area(&caps[0]) - 4.0).abs() < 1.0e-3, "{}", area(&caps[0]));
    assert!(caps[0].vertices.iter().all(|v| v.normal[2] > 0.9));
    // A load tint on a face the cut crosses (-y) leaves the cap in the body color.
    let tinted = GeometryScenePresentation {
        section: section(true),
        region_highlights: vec![GeometrySceneRegionHighlight {
            region_id: "face_2".to_string(),
            color: Some([0.9, 0.1, 0.5, 1.0]),
            role: None,
            label: None,
        }],
        ..Default::default()
    };
    let colors = |caps: &[RenderData]| caps[0].vertices.iter().map(|v| v.color).collect::<Vec<_>>();
    assert_eq!(colors(&cap_nodes(&tinted)), colors(&caps));
    // Open, or with no section, there is no cap.
    assert!(cap_nodes(&GeometryScenePresentation {
        section: section(false),
        ..Default::default()
    })
    .is_empty());
    assert!(cap_nodes(&GeometryScenePresentation::default()).is_empty());
}

#[test]
fn a_section_views_kept_symbols_hide_behind_material_and_removed_ones_go() {
    // Supports on the cube's bottom face (kept, below a cut at z = 0.5) and top face (removed).
    let support = |region: &str| GeometrySceneRegionAnnotation {
        region_id: region.into(),
        color: None,
        kind: GeometrySceneAnnotationKind::FixedSupport,
        label: None,
        direction: None,
        size: None,
    };
    let scene = GeometryScene::new("cube", 1, vec![cube()]);
    let symbols = |section: Option<GeometrySceneSection>, regions: &[&str]| {
        scene
            .nodes_with_presentation(&GeometryScenePresentation {
                region_annotations: regions.iter().map(|r| support(r)).collect(),
                section: Some(section),
                ..Default::default()
            })
            .into_iter()
            .find(|node| node.name == "FEA study symbols")
            .and_then(|node| node.render_data)
    };
    let top_cut = GeometrySceneSection {
        plane: GeometrySceneSectionPlane {
            normal: [0.0, 0.0, -1.0],
            origin: Some([0.0, 0.0, 0.5]),
            offset: None,
            label: None,
        },
        second_plane: None,
        cap: true,
    };
    // Unsectioned, a support draws over the model so one on a far face still shows.
    let whole = symbols(None, &["face_4"]).unwrap();
    assert_eq!(whole.pipeline_type, PipelineType::AnnotationStroke);
    // Sectioned, the kept one is depth-tested like the faces, so material in front hides it.
    let kept = symbols(Some(top_cut.clone()), &["face_4"]).unwrap();
    assert_eq!(kept.pipeline_type, PipelineType::AnnotationStrokeDepth);
    assert_eq!(kept.vertices.len(), whole.vertices.len());
    assert!(!PipelineType::AnnotationStrokeDepth.is_depth_free_overlay());
    // The one on the removed top face goes with it.
    assert!(symbols(Some(top_cut), &["face_5"]).is_none());
}
