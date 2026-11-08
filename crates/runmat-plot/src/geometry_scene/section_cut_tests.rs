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
        .map(|face| GeometrySceneRegion::new(format!("face_{face}"), None, None, vec![GeometrySceneTriangleRange::new(face * 2, 2)]))
        .collect();
    GeometrySceneChunk::indexed_triangles("cube", vertices, indices, cad_default_material())
        .with_regions(regions)
        .with_owner_node_ids(vec!["body:1".to_string()])
}

fn plane(normal: [f32; 3]) -> GeometrySceneSectionPlane {
    GeometrySceneSectionPlane { normal, origin: Some([0.0, 0.0, 0.0]), offset: None, label: None }
}

fn cut(section: GeometrySceneSection) -> RenderData {
    cube().render_data_with_presentation(&GeometryScenePresentation { section: Some(Some(section)), ..Default::default() })
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
    let capped = cut(GeometrySceneSection { plane: plane([1.0, 0.0, 0.0]), second_plane: None, cap: true });
    let open = cut(GeometrySceneSection { plane: plane([1.0, 0.0, 0.0]), second_plane: None, cap: false });
    assert!((area(&open) - 12.0).abs() < 1.0e-3, "open cut area {}", area(&open));
    assert!((area(&capped) - 16.0).abs() < 1.0e-3, "capped cut area {}", area(&capped));
    assert!(open.vertices.iter().all(|v| v.position[0] >= -1.0e-5));
}

#[test]
fn a_second_plane_cuts_away_one_corner_and_caps_both_cut_faces() {
    // Keeps x ≥ 0 or y ≥ 0: three quarters of the cube. Its surface is three quarters of the
    // cube's 24, and each plane's cap covers the removed quarter's side on it (2 each).
    let open = cut(GeometrySceneSection { plane: plane([1.0, 0.0, 0.0]), second_plane: Some(plane([0.0, 1.0, 0.0])), cap: false });
    assert!((area(&open) - 18.0).abs() < 1.0e-3, "open corner cut area {}", area(&open));
    assert!(open.vertices.chunks_exact(3).all(|t| t.iter().all(|v| v.position[0] >= -1.0e-5 || v.position[1] >= -1.0e-5)));
    let capped = cut(GeometrySceneSection { plane: plane([1.0, 0.0, 0.0]), second_plane: Some(plane([0.0, 1.0, 0.0])), cap: true });
    assert!((area(&capped) - 22.0).abs() < 1.0e-3, "capped corner cut area {}", area(&capped));
}

#[test]
fn section_presentation_reads_cap_and_second_plane_and_defaults_cap_on() {
    let section: GeometrySceneSection = serde_json::from_value(serde_json::json!({ "plane": { "normal": [1, 0, 0] } })).unwrap();
    assert!(section.cap);
    assert!(section.second_plane.is_none());
    let section: GeometrySceneSection = serde_json::from_value(serde_json::json!({
        "plane": { "normal": [1, 0, 0] }, "secondPlane": { "normal": [0, 1, 0] }, "cap": false
    }))
    .unwrap();
    assert!(!section.cap);
    assert_eq!(section.second_plane.map(|p| p.normal), Some([0.0, 1.0, 0.0]));
}

#[test]
fn hidden_regions_draw_nothing_and_picks_pass_through_them() {
    let presentation = GeometryScenePresentation { hidden_region_ids: vec!["face_5".to_string()], ..Default::default() };
    let drawn = cube().render_data_with_presentation(&presentation);
    assert!((area(&drawn) - 20.0).abs() < 1.0e-3, "five faces drawn, area {}", area(&drawn));

    let scene = GeometryScene::new("scene", 1, vec![cube()]);
    let mut index = GeometryScenePickIndex::build(&scene);
    let mut camera = Camera::new();
    camera.position = Vec3::new(0.0, 0.0, 6.0);
    camera.target = Vec3::ZERO;
    camera.up = Vec3::Y;
    camera.aspect_ratio = 1.0;
    camera.projection = ProjectionType::Perspective { fov: 45.0_f32.to_radians(), near: 0.1, far: 100.0 };
    camera.mark_dirty();
    let request = || GeometryScenePickRequest { camera: camera.clone(), surface_size: [800.0, 800.0], position: [400.0, 400.0], mode: GeometryScenePickMode::Face };

    let top = index.pick(request()).expect("the +z face faces the eye");
    assert_eq!(top.region_id.as_deref(), Some("face_5"));
    assert_eq!(top.owner_node_ids, vec!["body:1".to_string()]);

    index.set_hidden(&presentation);
    let through = index.pick(request()).expect("the far face behind the hidden one");
    assert_eq!(through.region_id.as_deref(), Some("face_4"));

    index.set_hidden(&GeometryScenePresentation { hidden_owner_node_ids: Some(vec!["body:1".to_string()]), ..Default::default() });
    assert!(index.pick(request()).is_none(), "a hidden body is not picked");
}
