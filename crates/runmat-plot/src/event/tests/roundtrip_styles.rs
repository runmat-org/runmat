//! Figure scene round trips: contours, bars, histograms, error bars, quivers and native storage.

use super::*;

#[test]
fn figure_scene_roundtrip_preserves_contour_and_fill_plots() {
    let mut figure = Figure::new();
    let bounds = BoundingBox::new(Vec3::new(-1.0, -2.0, 0.0), Vec3::new(3.0, 4.0, 0.0));
    let vertices = vec![Vertex {
        position: [0.0, 0.0, 0.0],
        color: [1.0, 0.0, 0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
        tex_coords: [0.0, 0.0],
    }];
    let fill = ContourFillPlot::from_vertices(vertices.clone(), bounds).with_label("fill");
    let contour = ContourPlot::from_vertices(vertices, 0.0, bounds)
        .with_label("lines")
        .with_line_width(2.0);
    figure.add_contour_fill_plot(fill);
    figure.add_contour_plot(contour);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    assert!(matches!(
        rebuilt.plots().next(),
        Some(PlotElement::ContourFill(_))
    ));
    let Some(PlotElement::Contour(contour)) = rebuilt.plots().nth(1) else {
        panic!("expected contour")
    };
    assert_eq!(contour.line_width, 2.0);
}

#[test]
fn figure_scene_roundtrip_preserves_stem_style_surface() {
    let mut figure = Figure::new();
    let mut stem = StemPlot::new(vec![0.0, 1.0], vec![1.0, 2.0])
        .unwrap()
        .with_style(
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            2.0,
            crate::plots::line::LineStyle::Dashed,
            -1.0,
        )
        .with_baseline_style(Vec4::new(0.0, 0.0, 0.0, 1.0), false)
        .with_label("Impulse");
    stem.set_marker(Some(crate::plots::line::LineMarkerAppearance {
        kind: crate::plots::scatter::MarkerStyle::Square,
        size: 8.0,
        edge_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
        face_color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        filled: true,
    }));
    figure.add_stem_plot(stem);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::Stem(stem) = rebuilt.plots().next().unwrap() else {
        panic!("expected stem")
    };
    assert_eq!(stem.baseline, -1.0);
    assert_eq!(stem.line_width, 2.0);
    assert_eq!(stem.label.as_deref(), Some("Impulse"));
    assert!(!stem.baseline_visible);
    assert!(stem.marker.as_ref().map(|m| m.filled).unwrap_or(false));
    assert_eq!(stem.marker.as_ref().map(|m| m.size), Some(8.0));
}

#[test]
fn figure_scene_roundtrip_preserves_bar_plot() {
    let mut figure = Figure::new();
    let bar = BarChart::new(vec!["A".into(), "B".into()], vec![2.0, 3.5])
        .unwrap()
        .with_style(Vec4::new(0.2, 0.4, 0.8, 1.0), 0.95)
        .with_outline(Vec4::new(0.1, 0.1, 0.1, 1.0), 1.5)
        .with_label("Histogram")
        .with_stack_offsets(vec![1.0, 0.5]);
    figure.add_bar_chart(bar);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::Bar(bar) = rebuilt.plots().next().unwrap() else {
        panic!("expected bar")
    };
    assert_eq!(bar.labels, vec!["A", "B"]);
    assert_eq!(bar.values().unwrap_or(&[]), &[2.0, 3.5]);
    assert_eq!(bar.bar_width, 0.95);
    assert_eq!(bar.outline_width, 1.5);
    assert_eq!(bar.label.as_deref(), Some("Histogram"));
    assert_eq!(bar.stack_offsets().unwrap_or(&[]), &[1.0, 0.5]);
    assert!(bar.histogram_bin_edges().is_none());
}

#[test]
fn figure_scene_roundtrip_preserves_histogram_bin_edges() {
    let mut figure = Figure::new();
    let mut bar = BarChart::new(vec!["bin1".into(), "bin2".into()], vec![4.0, 5.0]).unwrap();
    bar.set_histogram_bin_edges(vec![0.0, 0.5, 1.0]);
    figure.add_bar_chart(bar);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::Bar(bar) = rebuilt.plots().next().unwrap() else {
        panic!("expected bar")
    };
    assert_eq!(bar.histogram_bin_edges().unwrap_or(&[]), &[0.0, 0.5, 1.0]);
}

#[test]
fn figure_scene_roundtrip_preserves_polar_histogram_state() {
    let mut figure = Figure::new();
    let mut bar = BarChart::new(vec!["bin1".into(), "bin2".into()], vec![4.0, 5.0]).unwrap();
    bar.set_histogram_bin_edges(vec![0.0, std::f64::consts::PI, std::f64::consts::TAU]);
    bar.set_polar_histogram(true);
    bar.set_polar_histogram_display_style(PolarHistogramDisplayStyle::Stairs);
    figure.add_bar_chart(bar);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::Bar(bar) = rebuilt.plots().next().unwrap() else {
        panic!("expected bar")
    };
    assert!(bar.is_polar_histogram());
    assert_eq!(
        bar.polar_histogram_display_style(),
        PolarHistogramDisplayStyle::Stairs
    );
    assert_eq!(
        bar.histogram_bin_edges().unwrap_or(&[]),
        &[0.0, std::f64::consts::PI, std::f64::consts::TAU]
    );
}

#[test]
fn figure_scene_roundtrip_preserves_errorbar_style_surface() {
    let mut figure = Figure::new();
    let mut error = ErrorBar::new_vertical(
        vec![0.0, 1.0],
        vec![1.0, 2.0],
        vec![0.1, 0.2],
        vec![0.2, 0.3],
    )
    .unwrap()
    .with_style(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        2.0,
        crate::plots::line::LineStyle::Dashed,
        10.0,
    )
    .with_label("Err");
    error.set_marker(Some(crate::plots::line::LineMarkerAppearance {
        kind: crate::plots::scatter::MarkerStyle::Triangle,
        size: 8.0,
        edge_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
        face_color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        filled: true,
    }));
    figure.add_errorbar(error);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::ErrorBar(error) = rebuilt.plots().next().unwrap() else {
        panic!("expected errorbar")
    };
    assert_eq!(error.line_width, 2.0);
    assert_eq!(error.cap_size, 10.0);
    assert_eq!(error.label.as_deref(), Some("Err"));
    assert_eq!(error.line_style, crate::plots::line::LineStyle::Dashed);
    assert!(error.marker.as_ref().map(|m| m.filled).unwrap_or(false));
}

#[test]
fn figure_scene_roundtrip_preserves_errorbar_both_direction() {
    let mut figure = Figure::new();
    let error = ErrorBar::new_both(
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![0.1, 0.2],
        vec![0.2, 0.3],
        vec![0.3, 0.4],
        vec![0.4, 0.5],
    )
    .unwrap();
    figure.add_errorbar(error);
    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::ErrorBar(error) = rebuilt.plots().next().unwrap() else {
        panic!("expected errorbar")
    };
    assert_eq!(
        error.orientation,
        crate::plots::errorbar::ErrorBarOrientation::Both
    );
    assert_eq!(error.x_neg, vec![0.1, 0.2]);
    assert_eq!(error.x_pos, vec![0.2, 0.3]);
}

#[test]
fn figure_scene_roundtrip_preserves_quiver_plot() {
    let mut figure = Figure::new();
    let quiver = QuiverPlot::new(
        vec![0.0, 1.0],
        vec![1.0, 2.0],
        vec![0.5, -0.5],
        vec![1.0, 0.25],
    )
    .unwrap()
    .with_style(Vec4::new(0.2, 0.3, 0.4, 1.0), 2.0, 1.5, 0.2)
    .with_label("Field");
    figure.add_quiver_plot(quiver);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::Quiver(quiver) = rebuilt.plots().next().unwrap() else {
        panic!("expected quiver")
    };
    assert_eq!(quiver.u, vec![0.5, -0.5]);
    assert_eq!(quiver.v, vec![1.0, 0.25]);
    assert_eq!(quiver.line_width, 2.0);
    assert_eq!(quiver.scale, 1.5);
    assert_eq!(quiver.head_size, 0.2);
    assert_eq!(quiver.label.as_deref(), Some("Field"));
}

#[test]
fn figure_scene_roundtrip_preserves_quiver3_plot() {
    let mut figure = Figure::new();
    let quiver = QuiverPlot::new3d(
        vec![0.0, 1.0],
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![0.5, -0.5],
        vec![1.0, 0.25],
        vec![2.0, -1.0],
    )
    .unwrap()
    .with_style(Vec4::new(0.2, 0.3, 0.4, 1.0), 2.0, 1.5, 0.2)
    .with_label("Field3");
    figure.add_quiver_plot(quiver);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::Quiver(quiver) = rebuilt.plots().next().unwrap() else {
        panic!("expected quiver")
    };
    assert_eq!(quiver.z.as_ref().unwrap(), &vec![3.0, 4.0]);
    assert_eq!(quiver.w.as_ref().unwrap(), &vec![2.0, -1.0]);
    assert_eq!(quiver.label.as_deref(), Some("Field3"));
}

#[test]
fn figure_scene_roundtrip_preserves_image_surface_mode_and_color_grid() {
    let mut figure = Figure::new();
    let surface = SurfacePlot::new(
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![vec![0.0, 0.0], vec![0.0, 0.0]],
    )
    .unwrap()
    .with_flatten_z(true)
    .with_image_mode(true)
    .with_color_grid(vec![
        vec![Vec4::new(1.0, 0.0, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0)],
        vec![Vec4::new(0.0, 0.0, 1.0, 1.0), Vec4::new(1.0, 1.0, 1.0, 1.0)],
    ]);
    figure.add_surface_plot(surface);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::Surface(surface) = rebuilt.plots().next().unwrap() else {
        panic!("expected surface")
    };
    assert!(surface.flatten_z);
    assert!(surface.image_mode);
    assert!(surface.color_grid.is_some());
    assert_eq!(
        surface.color_grid.as_ref().unwrap()[0][0],
        Vec4::new(1.0, 0.0, 0.0, 1.0)
    );
}

#[test]
fn figure_scene_roundtrip_preserves_native_surface_storage() {
    let wide = vec![
        9_007_199_254_740_993,
        9_007_199_254_740_994,
        9_007_199_254_740_995,
        9_007_199_254_740_996,
    ];
    let mut surface = SurfacePlot::new(
        vec![1.0, 2.0],
        vec![1.0, 2.0],
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    )
    .unwrap();
    surface.set_source_data(
        None,
        None,
        Some(
            NumericPlotData::new(NumericStorage::U64(wide.clone()), vec![2, 2])
                .expect("native surface data"),
        ),
    );
    let colors = vec![1_u8, 2, 3, 4];
    surface
        .set_color_source(
            NumericPlotData::new(NumericStorage::U8(colors.clone()), vec![2, 2])
                .expect("native surface colors"),
        )
        .expect("matching CData");
    let mut figure = Figure::new();
    figure.add_surface_plot(surface);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::Surface(surface) = rebuilt.plots().next().unwrap() else {
        panic!("expected surface")
    };
    let (_, _, source_z) = surface.source_data();
    assert_eq!(
        source_z.map(NumericPlotData::storage),
        Some(&NumericStorage::U64(wide))
    );
    assert_eq!(
        surface.color_source().map(NumericPlotData::storage),
        Some(&NumericStorage::U8(colors))
    );
}

#[test]
fn figure_scene_roundtrip_preserves_native_patch_storage() {
    let faces = vec![1_i16, 2, 3];
    let vertices = vec![0_u64, 1, 0, 0, 0, 1];
    let mut patch = PatchPlot::new(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![vec![0, 1, 2]],
    )
    .unwrap();
    patch
        .set_source_data(crate::plots::PatchSourceDataOwned {
            colors: Some(
                NumericPlotData::new(NumericStorage::U8(vec![1, 2, 3]), vec![3, 1])
                    .expect("native patch colors"),
            ),
            color_kind: crate::plots::PatchColorSourceKind::FaceVertex,
            faces: Some(
                NumericPlotData::new(NumericStorage::I16(faces.clone()), vec![1, 3])
                    .expect("native faces"),
            ),
            vertices: Some(
                NumericPlotData::new(NumericStorage::U64(vertices.clone()), vec![3, 2])
                    .expect("native vertices"),
            ),
            ..Default::default()
        })
        .expect("valid native patch data");
    let mut figure = Figure::new();
    figure.add_patch_plot(patch);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::Patch(patch) = rebuilt.plots().next().unwrap() else {
        panic!("expected patch")
    };
    let (_, _, _, source_c, source_faces, source_vertices) = patch.source_data();
    assert_eq!(
        source_c.map(NumericPlotData::storage),
        Some(&NumericStorage::U8(vec![1, 2, 3]))
    );
    assert_eq!(
        source_faces.map(NumericPlotData::storage),
        Some(&NumericStorage::I16(faces))
    );
    assert_eq!(
        source_vertices.map(NumericPlotData::storage),
        Some(&NumericStorage::U64(vertices))
    );
}

#[test]
fn legacy_surface_scene_rebuilds_canonical_f64_sources() {
    let snapshot = FigureSnapshot::capture(&Figure::new());
    let scene = FigureScene {
        schema_version: FigureScene::SCHEMA_VERSION,
        geometry_overlay: None,
        geometry_physical_context: None,
        geometry_lines: Vec::new(),
        geometry_orientation_triad: true,
        layout: snapshot.layout,
        metadata: snapshot.metadata,
        plots: vec![ScenePlot::Surface {
            x: vec![0.0, 1.0],
            y: vec![0.0, 1.0],
            z: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            x_source: None,
            y_source: None,
            z_source: None,
            c_source: None,
            x_grid: None,
            y_grid: None,
            colormap: "Parula".into(),
            shading_mode: "Smooth".into(),
            wireframe: false,
            alpha: 1.0,
            flatten_z: false,
            image_mode: false,
            cell_labels: None,
            color_mapping: Default::default(),
            color_grid_rgba: None,
            color_limits: None,
            axes_index: 0,
            label: None,
            visible: true,
        }],
    };

    let rebuilt = scene.into_figure().expect("legacy surface scene");
    let PlotElement::Surface(surface) = rebuilt.plots().next().unwrap() else {
        panic!("expected surface")
    };
    let (x, y, z) = surface.source_data();
    assert!(matches!(
        x.map(NumericPlotData::storage),
        Some(NumericStorage::F64(_))
    ));
    assert!(matches!(
        y.map(NumericPlotData::storage),
        Some(NumericStorage::F64(_))
    ));
    assert!(matches!(
        z.map(NumericPlotData::storage),
        Some(NumericStorage::F64(_))
    ));
}

#[test]
fn figure_scene_roundtrip_preserves_parametric_surface_coordinate_grids() {
    let mut figure = Figure::new();
    let surface = SurfacePlot::from_coordinate_grids(
        vec![vec![0.0, 0.5], vec![0.2, 0.8]],
        vec![vec![0.0, 0.1], vec![0.7, 1.0]],
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    )
    .unwrap();
    figure.add_surface_plot(surface);

    let scene = FigureScene::capture(&figure);
    let ScenePlot::Surface {
        x_grid: Some(scene_x_grid),
        y_grid: Some(scene_y_grid),
        ..
    } = &scene.plots[0]
    else {
        panic!("expected exported coordinate grids");
    };
    assert_eq!(scene_x_grid[1][0], 0.2);
    assert_eq!(scene_y_grid[0][1], 0.1);

    let rebuilt = scene.into_figure().expect("scene restore should succeed");
    let PlotElement::Surface(surface) = rebuilt.plots().next().unwrap() else {
        panic!("expected surface")
    };
    assert_eq!(surface.x_grid.as_ref().unwrap()[1][1], 0.8);
    assert_eq!(surface.y_grid.as_ref().unwrap()[1][0], 0.7);
}

#[test]
fn figure_scene_roundtrip_preserves_area_lower_curve() {
    let mut figure = Figure::new();
    let area = AreaPlot::new(vec![1.0, 2.0], vec![2.0, 3.0])
        .unwrap()
        .with_lower_curve(vec![0.5, 1.0])
        .with_label("Stacked");
    figure.add_area_plot(area);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let PlotElement::Area(area) = rebuilt.plots().next().unwrap() else {
        panic!("expected area")
    };
    assert_eq!(area.lower_y, Some(vec![0.5, 1.0]));
    assert_eq!(area.label.as_deref(), Some("Stacked"));
}

#[test]
fn figure_scene_roundtrip_preserves_axes_local_limits_and_colormap_state() {
    let mut figure = Figure::new().with_subplot_grid(1, 2);
    figure.set_axes_limits(1, Some((1.0, 2.0)), Some((3.0, 4.0)));
    figure.set_axes_z_limits(1, Some((5.0, 6.0)));
    figure.set_axes_grid_enabled(1, false);
    figure.set_axes_minor_grid_enabled(1, true);
    figure.set_axes_hidden_line_removal(1, false);
    figure.set_axes_box_enabled(1, false);
    figure.set_axes_axis_equal(1, true);
    figure.set_axes_data_aspect_ratio(1, [1.0, 2.0, 3.0], "manual");
    figure.set_axes_kind(1, AxesKind::Polar);
    figure.set_axes_colorbar_enabled(1, true);
    figure.set_axes_colormap(1, ColorMap::Hot);
    figure.set_axes_color_limits(1, Some((0.0, 10.0)));
    figure.set_axes_tick_label_rotations(1, Some(30.0), Some(-45.0));
    figure.set_axes_style(
        1,
        TextStyle {
            font_size: Some(14.0),
            ..Default::default()
        },
    );
    figure.set_active_axes_index(1);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");
    let meta = rebuilt.axes_metadata(1).unwrap();
    assert_eq!(meta.x_limits, Some((1.0, 2.0)));
    assert_eq!(meta.y_limits, Some((3.0, 4.0)));
    assert_eq!(meta.z_limits, Some((5.0, 6.0)));
    assert!(!meta.grid_enabled);
    assert!(meta.minor_grid_enabled);
    assert!(meta.minor_grid_explicit);
    assert!(!meta.hidden_line_removal);
    assert!(!meta.box_enabled);
    assert!(!meta.axis_equal);
    assert_eq!(meta.data_aspect_ratio, [1.0, 2.0, 3.0]);
    assert_eq!(meta.data_aspect_ratio_mode, "manual");
    assert_eq!(meta.axes_kind, AxesKind::Polar);
    assert!(meta.colorbar_enabled);
    assert_eq!(format!("{:?}", meta.colormap), "Hot");
    assert_eq!(meta.color_limits, Some((0.0, 10.0)));
    assert_eq!(meta.x_tick_label_rotation, Some(30.0));
    assert_eq!(meta.y_tick_label_rotation, Some(-45.0));
    assert_eq!(meta.axes_style.font_size, Some(14.0));
}

#[test]
fn figure_scene_roundtrip_preserves_listed_colormap_rows() {
    let mut figure = Figure::new();
    let listed = ColorMap::from_rgb_rows(vec![[0.0, 0.25, 1.0], [1.0, 0.5, 0.0], [0.2, 0.3, 0.4]])
        .expect("listed colormap");
    figure.set_axes_colormap(0, listed);

    let scene = FigureScene::capture(&figure);
    let token = scene.metadata.colormap.as_deref().expect("colormap token");
    assert!(token.starts_with("listed:"));

    let rebuilt = scene.into_figure().expect("scene restore should succeed");
    let meta = rebuilt.axes_metadata(0).expect("axes metadata");
    let ColorMap::Listed(colors) = &meta.colormap else {
        panic!("expected listed colormap");
    };
    assert_eq!(
        colors.as_ref(),
        &[[0.0, 0.25, 1.0], [1.0, 0.5, 0.0], [0.2, 0.3, 0.4]]
    );
}
