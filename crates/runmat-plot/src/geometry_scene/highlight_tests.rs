//! Role highlights (a study's supports, loads and thermal regions) tint a face by their alpha, so
//! the lit shading still reads under them.
use super::*;

fn square(grey: [f32; 4]) -> GeometrySceneChunk {
    GeometrySceneChunk::indexed_triangles(
        "faces",
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
        .map(|p| vertex(p, grey, [0.0, 0.0, 1.0]))
        .to_vec(),
        vec![0, 1, 2, 1, 3, 2],
        cad_default_material(),
    )
    .with_regions(vec![
        GeometrySceneRegion::new(
            "face:0:0",
            None,
            None,
            vec![GeometrySceneTriangleRange::new(0, 1)],
        ),
        GeometrySceneRegion::new(
            "face:0:1",
            None,
            None,
            vec![GeometrySceneTriangleRange::new(1, 1)],
        ),
    ])
}

fn highlighted(color: [f32; 4]) -> GeometryScenePresentation {
    GeometryScenePresentation {
        region_highlights: vec![GeometrySceneRegionHighlight {
            region_id: "face:0:0".into(),
            color: Some(color),
            role: Some("boundary".into()),
            label: None,
        }],
        ..Default::default()
    }
}

#[test]
fn a_highlight_tints_the_face_by_its_alpha_and_keeps_the_face_opaque() {
    let grey = [0.66, 0.72, 0.8, 1.0];
    let blue = [0.2, 0.45, 0.95, 0.76];
    let out = square(grey).render_data_with_presentation(&highlighted(blue));
    let idx = out.indices.clone().unwrap();
    let tinted = out.vertices[idx[0] as usize].color;
    for axis in 0..3 {
        let expected = grey[axis] + (blue[axis] - grey[axis]) * 0.76;
        assert!((tinted[axis] - expected).abs() < 1e-6, "{tinted:?}");
    }
    assert_eq!(tinted[3], 1.0, "the tint never makes the face translucent");
    assert_eq!(
        out.vertices[idx[3] as usize].color, grey,
        "the other face keeps its color"
    );
    assert!(out.material.albedo.w >= 0.98 && out.material.alpha_mode != AlphaMode::Blend);

    // Alpha 1 fills; a translucent display mode keeps its own alpha under the highlight.
    let full = square(grey).render_data_with_presentation(&highlighted([0.2, 0.45, 0.95, 1.0]));
    let fi = full.indices.clone().unwrap();
    let filled = full.vertices[fi[0] as usize].color;
    for (got, want) in filled.iter().zip([0.2, 0.45, 0.95, 1.0]) {
        assert!((got - want).abs() < 1e-6, "{filled:?}");
    }
    let glass = [0.66, 0.72, 0.8, 0.4];
    let see = square(glass).render_data_with_presentation(&highlighted(blue));
    let si = see.indices.clone().unwrap();
    assert_eq!(see.vertices[si[0] as usize].color[3], 0.4);
}

#[test]
fn an_instanced_highlight_carries_its_alpha_as_the_tint_amount() {
    let chunk = square([0.66, 0.72, 0.8, 1.0]).with_instances(
        vec![GeometrySceneInstance {
            transform: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            region_id: "face:0:0".into(),
            label: None,
            tag: Some("occurrence".into()),
        }],
        None,
    );
    let paints = chunk.instance_paints(&highlighted([0.2, 0.45, 0.95, 0.76]));
    assert_eq!(paints[0].tint, [0.2, 0.45, 0.95, 0.76]);
}
