//! Section caps on the path a .fea file's viewport uses: a STEP part imported through OCCT, built
//! into a preview scene, then cut by the renderer's section with a setup tint on a face.
use super::*;
use runmat_plot::geometry_scene::{
    GeometryScenePresentation, GeometrySceneRegionHighlight, GeometrySceneSection,
    GeometrySceneSectionPlane,
};

fn bracket() -> GeometryAsset {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/simulation/examples/geometry/bracket.step"
    );
    let bytes = std::fs::read(path).unwrap();
    geometry_load(path, &bytes).unwrap()
}

fn top_section(origin: [f32; 3], cap: bool) -> GeometryScenePresentation {
    GeometryScenePresentation {
        section: Some(Some(GeometrySceneSection {
            plane: GeometrySceneSectionPlane {
                normal: [0.0, 0.0, -1.0],
                origin: Some(origin),
                offset: None,
                label: Some("top".to_string()),
            },
            second_plane: None,
            cap,
        })),
        ..GeometryScenePresentation::default()
    }
}

/// The section cap vertices the scene draws (one node per body), with each vertex's color.
fn cap_vertices(
    scene: &runmat_plot::GeometryScene,
    presentation: &GeometryScenePresentation,
) -> Vec<([f32; 3], [f32; 4])> {
    scene
        .nodes_with_presentation(presentation)
        .into_iter()
        .filter(|node| node.name == "Section cap")
        .filter_map(|node| node.render_data)
        .flat_map(|data| data.vertices.into_iter().map(|v| (v.position, v.color)))
        .collect()
}

#[test]
fn a_cut_step_part_is_capped_in_its_body_color_with_a_load_tint_on_a_cut_face() {
    let asset = bracket();
    let scene =
        geometry_preview_scene(&asset, "bracket", GeometryPreviewSceneOptions::default()).unwrap();
    // A CAD import draws each face as its own chunk: no chunk on its own is closed.
    assert!(scene.chunks.iter().filter(|c| !c.wireframe_only).count() > 1);
    let (min, max) = (scene.bounds.min, scene.bounds.max);
    let centre = ((min + max) / 2.0).to_array();
    // The Desktop's Top section through the middle: it cuts the hook's wall and ribs.
    let plain = cap_vertices(&scene, &top_section(centre, true));
    assert!(plain.len() >= 3, "the cut is capped");
    assert!(plain.iter().all(|(p, _)| (p[2] - centre[2]).abs() < 1.0e-6));
    assert!(cap_vertices(&scene, &top_section(centre, false)).is_empty());
    // The hook seat's load tint (the face the cut crosses) leaves the cap in the body color.
    let seat: Vec<String> = asset
        .regions
        .iter()
        .filter(|region| region.name == "Hook seat")
        .map(|region| region.region_id.clone())
        .collect();
    assert!(!seat.is_empty());
    let mut tinted = top_section(centre, true);
    tinted.region_highlights = seat
        .into_iter()
        .map(|region_id| GeometrySceneRegionHighlight {
            region_id,
            color: Some([0.78, 0.16, 0.45, 0.78]),
            role: None,
            label: None,
        })
        .collect();
    let colors = |vertices: &[([f32; 3], [f32; 4])]| {
        let mut c: Vec<String> = vertices.iter().map(|(_, c)| format!("{c:?}")).collect();
        c.sort();
        c
    };
    assert_eq!(colors(&cap_vertices(&scene, &tinted)), colors(&plain));
    // The cut's whole section is filled: the hook wall and its ribs, over 50 mm².
    let area: f32 = plain
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [t[0].0, t[1].0, t[2].0].map(glam::Vec3::from_array);
            (b - a).cross(c - a).length() / 2.0
        })
        .sum();
    assert!(area > 5.0e-5, "cap area {area} m²");
}
