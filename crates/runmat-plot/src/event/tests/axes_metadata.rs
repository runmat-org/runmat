//! Axes metadata, annotations and null numeric values in scene JSON.

use super::*;

#[test]
fn axes_metadata_deserializes_without_axes_style() {
    let json = r#"{
        "legendEnabled": true,
        "colormap": "Parula",
        "titleStyle": {"visible": true},
        "xLabelStyle": {"visible": true},
        "yLabelStyle": {"visible": true},
        "zLabelStyle": {"visible": true},
        "legendStyle": {"visible": true}
    }"#;
    let serialized: SerializedAxesMetadata = serde_json::from_str(json).unwrap();
    let metadata = AxesMetadata::from(serialized);
    assert!(metadata.axes_style.color.is_none());
    assert!(metadata.axes_style.font_size.is_none());
    assert!(metadata.axes_style.font_weight.is_none());
    assert!(metadata.axes_style.font_angle.is_none());
    assert!(metadata.axes_style.interpreter.is_none());
    assert!(metadata.axes_style.visible);
}

#[test]
fn figure_scene_roundtrip_preserves_axes_local_annotation_metadata() {
    let mut figure = Figure::new().with_subplot_grid(1, 2);
    figure.set_sg_title("All Panels");
    figure.set_sg_title_style(TextStyle {
        font_weight: Some("bold".into()),
        font_size: Some(20.0),
        ..Default::default()
    });
    figure.set_active_axes_index(0);
    figure.set_axes_title(0, "Left");
    figure.set_axes_subtitle(0, "Left details");
    figure.set_axes_xlabel(0, "LX");
    figure.set_axes_ylabel(0, "LY");
    figure.set_axes_legend_enabled(0, false);
    figure.set_axes_title(1, "Right");
    figure.set_axes_xlabel(1, "RX");
    figure.set_axes_ylabel(1, "RY");
    figure.set_axes_legend_enabled(1, true);
    figure.set_axes_legend_style(
        1,
        LegendStyle {
            location: Some("northeast".into()),
            font_weight: Some("bold".into()),
            orientation: Some("horizontal".into()),
            ..Default::default()
        },
    );
    if let Some(meta) = figure.axes_metadata.get_mut(0) {
        meta.title_style.font_weight = Some("bold".into());
        meta.title_style.font_angle = Some("italic".into());
        meta.subtitle_style.font_size = Some(12.0);
        meta.subtitle_style.font_weight = Some("normal".into());
    }
    figure.set_active_axes_index(1);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");

    assert_eq!(rebuilt.active_axes_index, 1);
    assert_eq!(rebuilt.sg_title.as_deref(), Some("All Panels"));
    assert_eq!(rebuilt.sg_title_style.font_weight.as_deref(), Some("bold"));
    assert_eq!(rebuilt.sg_title_style.font_size, Some(20.0));
    assert_eq!(
        rebuilt.axes_metadata(0).and_then(|m| m.title.as_deref()),
        Some("Left")
    );
    assert_eq!(
        rebuilt.axes_metadata(0).and_then(|m| m.subtitle.as_deref()),
        Some("Left details")
    );
    assert_eq!(
        rebuilt.axes_metadata(0).and_then(|m| m.x_label.as_deref()),
        Some("LX")
    );
    assert_eq!(
        rebuilt.axes_metadata(0).and_then(|m| m.y_label.as_deref()),
        Some("LY")
    );
    assert!(!rebuilt.axes_metadata(0).unwrap().legend_enabled);
    assert_eq!(
        rebuilt
            .axes_metadata(0)
            .unwrap()
            .title_style
            .font_weight
            .as_deref(),
        Some("bold")
    );
    assert_eq!(
        rebuilt
            .axes_metadata(0)
            .unwrap()
            .title_style
            .font_angle
            .as_deref(),
        Some("italic")
    );
    assert_eq!(
        rebuilt.axes_metadata(0).unwrap().subtitle_style.font_size,
        Some(12.0)
    );
    assert_eq!(
        rebuilt
            .axes_metadata(0)
            .unwrap()
            .subtitle_style
            .font_weight
            .as_deref(),
        Some("normal")
    );
    assert_eq!(
        rebuilt.axes_metadata(1).and_then(|m| m.title.as_deref()),
        Some("Right")
    );
    assert_eq!(
        rebuilt.axes_metadata(1).and_then(|m| m.x_label.as_deref()),
        Some("RX")
    );
    assert_eq!(
        rebuilt.axes_metadata(1).and_then(|m| m.y_label.as_deref()),
        Some("RY")
    );
    assert_eq!(
        rebuilt
            .axes_metadata(1)
            .unwrap()
            .legend_style
            .location
            .as_deref(),
        Some("northeast")
    );
    assert_eq!(
        rebuilt
            .axes_metadata(1)
            .unwrap()
            .legend_style
            .font_weight
            .as_deref(),
        Some("bold")
    );
    assert_eq!(
        rebuilt
            .axes_metadata(1)
            .unwrap()
            .legend_style
            .orientation
            .as_deref(),
        Some("horizontal")
    );
}

#[test]
fn figure_scene_roundtrip_preserves_axes_local_log_modes() {
    let mut figure = Figure::new().with_subplot_grid(1, 2);
    figure.set_axes_log_modes(0, true, false);
    figure.set_axes_log_modes(1, false, true);
    figure.set_active_axes_index(1);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");

    assert!(rebuilt.axes_metadata(0).unwrap().x_log);
    assert!(!rebuilt.axes_metadata(0).unwrap().y_log);
    assert!(!rebuilt.axes_metadata(1).unwrap().x_log);
    assert!(rebuilt.axes_metadata(1).unwrap().y_log);
    assert!(!rebuilt.x_log);
    assert!(rebuilt.y_log);
}

#[test]
fn figure_scene_roundtrip_preserves_zlabel_and_view_state() {
    let mut figure = Figure::new().with_subplot_grid(1, 2);
    figure.set_axes_zlabel(1, "Height");
    figure.set_axes_view(1, 45.0, 20.0);
    figure.set_active_axes_index(1);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");

    assert_eq!(
        rebuilt.axes_metadata(1).unwrap().z_label.as_deref(),
        Some("Height")
    );
    assert_eq!(
        rebuilt.axes_metadata(1).unwrap().view_azimuth_deg,
        Some(45.0)
    );
    assert_eq!(
        rebuilt.axes_metadata(1).unwrap().view_elevation_deg,
        Some(20.0)
    );
    assert_eq!(rebuilt.z_label.as_deref(), Some("Height"));
}

#[test]
fn figure_scene_roundtrip_preserves_pie_metadata() {
    let mut figure = Figure::new();
    let pie = crate::plots::PieChart::new(vec![1.0, 2.0], None)
        .unwrap()
        .with_slice_labels(vec!["A".into(), "B".into()])
        .with_explode(vec![false, true]);
    figure.add_pie_chart(pie);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let crate::plots::figure::PlotElement::Pie(pie) = rebuilt.plots().next().unwrap() else {
        panic!("expected pie")
    };
    assert_eq!(pie.slice_labels, vec!["A", "B"]);
    assert_eq!(pie.explode, vec![false, true]);
}

#[test]
fn scene_plot_deserialize_maps_null_numeric_values_to_nan() {
    let json = r#"{
      "schemaVersion": 1,
      "layout": { "axesRows": 1, "axesCols": 1, "axesIndices": [0] },
      "metadata": {
        "gridEnabled": true,
        "legendEnabled": false,
        "colorbarEnabled": false,
        "axisEqual": false,
        "backgroundRgba": [1,1,1,1],
        "legendEntries": []
      },
      "plots": [
        {
          "kind": "surface",
          "x": [0.0, null],
          "y": [0.0, 1.0],
          "z": [[0.0, null], [1.0, 2.0]],
          "colormap": "Parula",
          "shading_mode": "Smooth",
          "wireframe": false,
          "alpha": 1.0,
          "flatten_z": false,
          "color_limits": null,
          "axes_index": 0,
          "label": null,
          "visible": true
        }
      ]
    }"#;
    let scene: FigureScene = serde_json::from_str(json).expect("scene should deserialize");
    let ScenePlot::Surface { x, z, .. } = &scene.plots[0] else {
        panic!("expected surface plot");
    };
    assert!(x[1].is_nan());
    assert!(z[0][1].is_nan());
}

#[test]
fn scene_plot_deserialize_maps_null_scatter3_components_to_nan() {
    let json = r#"{
      "schemaVersion": 1,
      "layout": { "axesRows": 1, "axesCols": 1, "axesIndices": [0] },
      "metadata": {
        "gridEnabled": true,
        "legendEnabled": false,
        "colorbarEnabled": false,
        "axisEqual": false,
        "backgroundRgba": [1,1,1,1],
        "legendEntries": []
      },
      "plots": [
        {
          "kind": "scatter3",
          "points": [[0.0, 1.0, null], [1.0, null, 2.0]],
          "colors_rgba": [[0.2, 0.4, 0.6, 1.0], [0.1, 0.2, 0.3, 1.0]],
          "point_size": 6.0,
          "point_sizes": [3.0, null],
          "axes_index": 0,
          "label": null,
          "visible": true
        }
      ]
    }"#;
    let scene: FigureScene = serde_json::from_str(json).expect("scene should deserialize");
    let ScenePlot::Scatter3 {
        points,
        point_sizes,
        ..
    } = &scene.plots[0]
    else {
        panic!("expected scatter3 plot");
    };
    assert!(points[0][2].is_nan());
    assert!(points[1][1].is_nan());
    assert!(point_sizes.as_ref().unwrap()[1].is_nan());
}
