//! Measure's dimensions drawn in the model: the presentation carries them, the scene draws each
//! as a line with end dots over the model.

use super::*;

fn plate() -> GeometryScene {
    let chunk = GeometrySceneChunk::indexed_triangles(
        "plate",
        vec![
            vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.04, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 0.03, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2],
        cad_default_material(),
    );
    GeometryScene::new("scene", 1, vec![chunk])
}

fn dimension_node(nodes: &[SceneNode]) -> Option<&SceneNode> {
    nodes.iter().find(|node| node.name == "Measure dimensions")
}

#[test]
fn a_presentation_reads_dimensions_from_the_hosts_json() {
    let presentation: GeometryScenePresentation = serde_json::from_value(serde_json::json!({
        "dimensions": [{ "from": [0.0, 0.0, 0.0], "to": [0.04, 0.0, 0.0], "label": "40.000 mm" }]
    }))
    .expect("presentation with a dimension");
    assert_eq!(
        presentation.dimensions,
        vec![GeometrySceneDimension {
            from: [0.0, 0.0, 0.0],
            to: [0.04, 0.0, 0.0],
            label: "40.000 mm".into(),
            color: None,
            extension_lines: Vec::new(),
        }]
    );
    // An empty list stays out of the saved presentation.
    let saved = serde_json::to_value(GeometryScenePresentation::default()).unwrap();
    assert!(saved.get("dimensions").is_none());
}

#[test]
fn a_dimension_draws_its_line_and_end_dots_over_the_model() {
    let color = [0.1, 0.4, 0.9, 1.0];
    let nodes = plate().nodes_with_presentation(&GeometryScenePresentation {
        dimensions: vec![GeometrySceneDimension {
            from: [0.0, 0.0, 0.0],
            to: [0.04, 0.0, 0.0],
            label: "40.000 mm".into(),
            color: Some(color),
            extension_lines: Vec::new(),
        }],
        ..Default::default()
    });
    let node = dimension_node(&nodes).expect("a dimension node");
    let data = node.render_data.as_ref().unwrap();
    // Over the model, as the study symbols draw.
    assert_eq!(data.pipeline_type, PipelineType::AnnotationStroke);
    let positions: Vec<[f32; 3]> = data.vertices.iter().map(|v| v.position).collect();
    for end in [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]] {
        assert!(positions.contains(&end), "no vertex at the end {end:?}");
    }
    assert!(
        positions
            .iter()
            .all(|p| p[1] == 0.0 && p[2] == 0.0 && (0.0..=0.04).contains(&p[0])),
        "the line leaves the measured span: {positions:?}"
    );
    // Halo strokes are dark; the line and dots take the dimension's color.
    assert!(data.vertices.iter().any(|v| v.color[..3] == color[..3]));
    // Two strokes (halo, line) of six vertices and two dots of six.
    assert_eq!(data.vertices.len(), 24);
}

#[test]
fn no_dimension_draws_nothing_and_a_broken_one_is_skipped() {
    assert!(dimension_node(
        &plate().nodes_with_presentation(&GeometryScenePresentation::default())
    )
    .is_none());
    let nodes = plate().nodes_with_presentation(&GeometryScenePresentation {
        dimensions: vec![GeometrySceneDimension {
            from: [f32::NAN, 0.0, 0.0],
            to: [0.04, 0.0, 0.0],
            label: "?".into(),
            color: None,
            extension_lines: Vec::new(),
        }],
        ..Default::default()
    });
    assert!(dimension_node(&nodes).is_none());
}

#[test]
fn touching_items_measure_zero_and_show_their_dot() {
    let nodes = plate().nodes_with_presentation(&GeometryScenePresentation {
        dimensions: vec![GeometrySceneDimension {
            from: [0.01, 0.01, 0.0],
            to: [0.01, 0.01, 0.0],
            label: "0.000 mm".into(),
            color: None,
            extension_lines: Vec::new(),
        }],
        ..Default::default()
    });
    let data = dimension_node(&nodes)
        .and_then(|n| n.render_data.as_ref())
        .expect("dots");
    assert_eq!(data.vertices.len(), 12);
    assert!(data
        .vertices
        .iter()
        .all(|v| v.color[..3] == GEOMETRY_DIMENSION_COLOR[..3]));
}

#[test]
fn extension_lines_draw_thin_from_a_gap_at_the_item_past_the_dimension_line() {
    // A normal distance standing off the first face: the dimension line rises at x = 0.04 and
    // an extension line in the face's plane runs from the face's edge (x = 0.02) to its foot.
    let presentation: GeometryScenePresentation = serde_json::from_value(serde_json::json!({
        "dimensions": [{
            "from": [0.04, 0.0, 0.0],
            "to": [0.04, 0.0, 0.02],
            "label": "20.000 mm",
            "extensionLines": [[[0.02, 0.0, 0.0], [0.04, 0.0, 0.0]]]
        }]
    }))
    .expect("presentation with an extension line");
    assert_eq!(
        presentation.dimensions[0].extension_lines,
        vec![[[0.02, 0.0, 0.0], [0.04, 0.0, 0.0]]]
    );
    let saved = serde_json::to_value(&presentation.dimensions[0]).unwrap();
    assert!(saved["extensionLines"].is_array(), "{saved}");
    let nodes = plate().nodes_with_presentation(&presentation);
    let data = dimension_node(&nodes)
        .and_then(|n| n.render_data.as_ref())
        .expect("dimension node");
    assert_eq!(data.pipeline_type, PipelineType::AnnotationStroke);
    // Halo + line + two dots (24), then one extension stroke (6).
    assert_eq!(data.vertices.len(), 30);
    let extension = &data.vertices[24..];
    // The thin annotation style, narrower than the dimension line's.
    assert!(extension.iter().all(|v| v.tex_coords[1] == 0.0));
    assert!(data.vertices[6..12].iter().all(|v| v.tex_coords[1] == 1.0));
    let xs: Vec<f32> = extension.iter().map(|v| v.position[0]).collect();
    let (lo, hi) = xs
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(*x), hi.max(*x)));
    // A gap at the face's edge, and a short run past the dimension line.
    assert!(lo > 0.02 && lo < 0.025, "{xs:?}");
    assert!(hi > 0.04 && hi < 0.045, "{xs:?}");
    assert!(extension
        .iter()
        .all(|v| v.position[1] == 0.0 && v.position[2] == 0.0));
}
