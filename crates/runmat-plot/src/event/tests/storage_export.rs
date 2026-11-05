//! Scene numeric storage, readback failures and async export of every plot element.

use super::*;

#[test]
fn numeric_storage_reads_null_as_nan() {
    let data: SceneNumericData =
        serde_json::from_str(r#"{"storage":{"dtype":"f64","values":[1.0,null]},"shape":[1,2]}"#)
            .expect("null decodes as NaN");
    let SceneNumericStorage::F64(values) = data.storage else {
        panic!("expected f64 storage")
    };
    assert_eq!(values[0], 1.0);
    assert!(values[1].is_nan());
}

#[test]
fn typed_scene_readback_keeps_the_original_device_failure() {
    let cause = crate::plots::PlotDataExportError::Readback(PlotReadbackError::Access(
        runmat_accelerate_api::WgpuAccessError::WrongDevice,
    ));
    let error = SceneExportFailure::readback_typed(cause.clone());
    assert_eq!(error.error.kind, SceneExportErrorKind::Readback);
    assert_eq!(error.error.message, cause.to_string());
    assert_eq!(
        error
            .source()
            .and_then(|source| source.downcast_ref::<crate::plots::PlotDataExportError>()),
        Some(&cause)
    );
    assert!(error
        .source()
        .and_then(Error::source)
        .and_then(|source| source.downcast_ref::<PlotReadbackError>())
        .is_some());
}

#[test]
fn public_scene_export_error_remains_a_two_field_value() {
    let error = SceneExportError {
        kind: SceneExportErrorKind::Readback,
        message: "device readback failed".into(),
    };
    let failure = SceneExportFailure::from(error.clone());
    assert_eq!(failure.error, error);
    assert!(failure.source().is_none());
}

#[test]
fn async_scene_export_covers_every_plot_element_variant() {
    let bounds = BoundingBox::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
    let cases: Vec<(&str, PlotElement)> = vec![
        (
            "line",
            PlotElement::Line(LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap()),
        ),
        (
            "scatter",
            PlotElement::Scatter(ScatterPlot::new(vec![0.0, 1.0], vec![2.0, 3.0]).unwrap()),
        ),
        (
            "bar",
            PlotElement::Bar(BarChart::new(vec!["A".into(), "B".into()], vec![1.0, 2.0]).unwrap()),
        ),
        (
            "errorbar",
            PlotElement::ErrorBar(Box::new(
                ErrorBar::new_vertical(
                    vec![0.0, 1.0],
                    vec![1.0, 2.0],
                    vec![0.1, 0.2],
                    vec![0.3, 0.4],
                )
                .unwrap(),
            )),
        ),
        (
            "stairs",
            PlotElement::Stairs(StairsPlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap()),
        ),
        (
            "stem",
            PlotElement::Stem(StemPlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap()),
        ),
        (
            "area",
            PlotElement::Area(AreaPlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap()),
        ),
        (
            "quiver",
            PlotElement::Quiver(
                QuiverPlot::new(vec![0.0], vec![0.0], vec![1.0], vec![1.0]).unwrap(),
            ),
        ),
        (
            "pie",
            PlotElement::Pie(PieChart::new(vec![1.0, 2.0], None).unwrap()),
        ),
        (
            "surface",
            PlotElement::Surface(
                SurfacePlot::new(
                    vec![0.0, 1.0],
                    vec![0.0, 1.0],
                    vec![vec![0.0, 1.0], vec![1.0, 2.0]],
                )
                .unwrap(),
            ),
        ),
        (
            "mesh",
            PlotElement::Mesh(Box::new(
                MeshPlot::new(
                    vec![
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec3::new(1.0, 0.0, 0.0),
                        Vec3::new(0.0, 1.0, 0.0),
                    ],
                    vec![[0, 1, 2]],
                )
                .unwrap(),
            )),
        ),
        (
            "patch",
            PlotElement::Patch(
                PatchPlot::new(
                    vec![
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec3::new(1.0, 0.0, 0.0),
                        Vec3::new(0.0, 1.0, 0.0),
                    ],
                    vec![vec![0, 1, 2]],
                )
                .unwrap(),
            ),
        ),
        (
            "line3",
            PlotElement::Line3(
                Line3Plot::new(vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, 3.0]).unwrap(),
            ),
        ),
        (
            "scatter3",
            PlotElement::Scatter3(Scatter3Plot::new(vec![Vec3::new(0.0, 0.0, 0.0)]).unwrap()),
        ),
        (
            "contour",
            PlotElement::Contour(ContourPlot::from_vertices(
                vec![
                    Vertex::new(Vec3::new(0.0, 0.0, 0.0), Vec4::ONE),
                    Vertex::new(Vec3::new(1.0, 1.0, 0.0), Vec4::ONE),
                ],
                0.0,
                bounds,
            )),
        ),
        (
            "contour_fill",
            PlotElement::ContourFill(ContourFillPlot::from_vertices(
                vec![
                    Vertex::new(Vec3::new(0.0, 0.0, 0.0), Vec4::ONE),
                    Vertex::new(Vec3::new(1.0, 0.0, 0.0), Vec4::ONE),
                    Vertex::new(Vec3::new(0.0, 1.0, 0.0), Vec4::ONE),
                ],
                bounds,
            )),
        ),
        (
            "reference_line",
            PlotElement::ReferenceLine(
                ReferenceLine::new(ReferenceLineOrientation::Vertical, 0.5).unwrap(),
            ),
        ),
    ];

    for (name, plot) in cases {
        let scene_plot = futures::executor::block_on(ScenePlot::from_plot_for_export(&plot, 0))
            .unwrap_or_else(|err| panic!("{name} export failed: {err}"));
        scene_plot
            .validate_exportable()
            .unwrap_or_else(|err| panic!("{name} validation failed: {err}"));
    }
}

#[test]
fn capture_snapshot_reflects_layout_and_metadata() {
    let mut figure = Figure::new()
        .with_title("Demo")
        .with_sg_title("Overview")
        .with_labels("X", "Y")
        .with_grid(false)
        .with_subplot_grid(1, 2);
    figure.set_name("Window Name");
    figure.set_number_title(false);
    figure.set_visible(false);
    figure.set_background_color(Vec4::new(0.0, 0.0, 0.0, 1.0));
    let line = LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    figure.add_line_plot_on_axes(line, 1);

    let snapshot = FigureSnapshot::capture(&figure);
    assert_eq!(snapshot.layout.axes_rows, 1);
    assert_eq!(snapshot.layout.axes_cols, 2);
    assert_eq!(snapshot.metadata.title.as_deref(), Some("Demo"));
    assert_eq!(snapshot.metadata.name.as_deref(), Some("Window Name"));
    assert!(!snapshot.metadata.number_title);
    assert!(!snapshot.metadata.visible);
    assert_eq!(snapshot.metadata.sg_title.as_deref(), Some("Overview"));
    assert_eq!(snapshot.metadata.background_rgba, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(snapshot.metadata.legend_entries.len(), 0);
    assert_eq!(snapshot.plots.len(), 1);
    assert_eq!(snapshot.plots[0].axes_index, 1);
    assert!(!snapshot.metadata.grid_enabled);
}

#[test]
fn surface_scene_validation_uses_surface_plot_orientation() {
    let scene_plot = ScenePlot::Surface {
        x: vec![0.0, 1.0, 2.0],
        y: vec![10.0, 20.0],
        z: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        x_source: None,
        y_source: None,
        z_source: None,
        c_source: None,
        x_grid: None,
        y_grid: None,
        colormap: "Parula".to_string(),
        shading_mode: "Smooth".to_string(),
        wireframe: false,
        alpha: 1.0,
        flatten_z: false,
        image_mode: false,
        cell_labels: None,
        color_mapping: Default::default(),
        color_grid_rgba: Some(vec![
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            vec![[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]],
            vec![[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]],
        ]),
        color_limits: None,
        axes_index: 0,
        label: None,
        visible: true,
    };
    scene_plot.validate_exportable().unwrap();

    let transposed = ScenePlot::Surface {
        x: vec![0.0, 1.0, 2.0],
        y: vec![10.0, 20.0],
        z: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        x_source: None,
        y_source: None,
        z_source: None,
        c_source: None,
        x_grid: None,
        y_grid: None,
        colormap: "Parula".to_string(),
        shading_mode: "Smooth".to_string(),
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
    };
    let err = transposed.validate_exportable().unwrap_err();
    assert!(err
        .to_string()
        .contains("row count (2) must match x length (3)"));
}

#[test]
fn image_mode_surface_scene_roundtrip_preserves_color_grid() {
    let snapshot = FigureSnapshot::capture(&Figure::new());
    let scene = FigureScene {
        schema_version: FigureScene::SCHEMA_VERSION,
        geometry_overlay: None,
        geometry_physical_context: None,
        geometry_lines: Vec::new(),
        layout: snapshot.layout,
        metadata: snapshot.metadata,
        plots: vec![ScenePlot::Surface {
            x: vec![0.0, 1.0],
            y: vec![10.0, 20.0, 30.0],
            z: vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]],
            x_source: None,
            y_source: None,
            z_source: None,
            c_source: None,
            x_grid: None,
            y_grid: None,
            colormap: "Parula".to_string(),
            shading_mode: "None".to_string(),
            wireframe: false,
            alpha: 1.0,
            flatten_z: true,
            image_mode: true,
            cell_labels: None,
            color_mapping: Default::default(),
            color_grid_rgba: Some(vec![
                vec![
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ],
                vec![
                    [1.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                ],
            ]),
            color_limits: None,
            axes_index: 0,
            label: None,
            visible: true,
        }],
    };

    let rebuilt = scene.into_figure().expect("image surface scene restores");
    let Some(PlotElement::Surface(surface)) = rebuilt.plots().next() else {
        panic!("expected surface plot");
    };
    assert!(surface.image_mode);
    assert!(surface.flatten_z);
    let grid = surface.color_grid.as_ref().expect("color grid");
    assert_eq!(grid.len(), 2);
    assert_eq!(grid[0].len(), 3);
    assert_eq!(grid[1][2], Vec4::new(0.0, 1.0, 1.0, 1.0));
}

#[test]
fn sg_title_style_omitted_when_sg_title_absent() {
    let figure = Figure::new().with_title("Only regular title");
    let snapshot = FigureSnapshot::capture(&figure);
    assert!(snapshot.metadata.sg_title.is_none());
    assert!(
        snapshot.metadata.sg_title_style.is_none(),
        "sgTitleStyle must be None when sgTitle is absent"
    );
    let json = serde_json::to_string(&snapshot.metadata).unwrap();
    assert!(
        !json.contains("sgTitleStyle"),
        "sgTitleStyle must not appear in serialized JSON when sgTitle is absent"
    );
}
