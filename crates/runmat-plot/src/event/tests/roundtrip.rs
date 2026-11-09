//! Figure scene round trips: lines, patches, meshes, surfaces and scatter plots.

use super::*;

#[test]
fn figure_scene_roundtrip_reconstructs_supported_plots() {
    let mut figure = Figure::new().with_title("Replay").with_subplot_grid(1, 2);
    figure.set_name("Roundtrip");
    figure.set_number_title(false);
    figure.set_visible(false);
    figure.set_position([100.0, 100.0, 1000.0, 700.0]);
    let mut line = LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
    line.label = Some("line".to_string());
    figure.add_line_plot_on_axes(line, 0);
    let mut scatter = ScatterPlot::new(vec![0.0, 1.0, 2.0], vec![2.0, 3.0, 4.0]).unwrap();
    scatter.label = Some("scatter".to_string());
    figure.add_scatter_plot_on_axes(scatter, 1);

    let scene = FigureScene::capture(&figure);
    let rebuilt = scene.into_figure().expect("scene restore should succeed");
    assert_eq!(rebuilt.axes_grid(), (1, 2));
    assert_eq!(rebuilt.plots().count(), 2);
    assert_eq!(rebuilt.title.as_deref(), Some("Replay"));
    assert_eq!(rebuilt.name.as_deref(), Some("Roundtrip"));
    assert!(!rebuilt.number_title);
    assert!(!rebuilt.visible);
    assert_eq!(rebuilt.position, [100.0, 100.0, 1000.0, 700.0]);
}

#[test]
fn figure_scene_roundtrip_reconstructs_patch() {
    let mut figure = Figure::new();
    let mut patch = PatchPlot::new(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![vec![0, 1, 2]],
    )
    .unwrap();
    patch.set_label(Some("tri".into()));
    patch.set_force_3d(true);
    figure.add_patch_plot(patch);

    let scene = FigureScene::capture(&figure);
    assert_eq!(scene.schema_version, FigureScene::SCHEMA_VERSION);
    assert!(matches!(scene.plots.first(), Some(ScenePlot::Patch { .. })));
    let rebuilt = scene.into_figure().expect("patch scene restore");
    let Some(PlotElement::Patch(patch)) = rebuilt.plots().next() else {
        panic!("expected patch plot");
    };
    assert_eq!(patch.faces(), &[vec![0, 1, 2]]);
    assert_eq!(patch.label(), Some("tri"));
    assert!(patch.force_3d());
}

#[test]
fn figure_scene_rejects_invalid_schema_versions() {
    let mut scene = FigureScene::capture(&Figure::new());
    scene.schema_version = 0;
    let err = scene.clone().into_figure().expect_err("schema 0 must fail");
    assert!(err.contains("unsupported figure scene schema version 0"));

    scene.schema_version = FigureScene::SCHEMA_VERSION + 1;
    let err = scene.into_figure().expect_err("future schema must fail");
    assert!(err.contains(&format!(
        "unsupported figure scene schema version {}",
        FigureScene::SCHEMA_VERSION + 1
    )));
}

#[test]
fn figure_scene_rejects_invalid_position_metadata() {
    let mut scene = FigureScene::capture(&Figure::new());
    scene.metadata.position = Some([0.0, 0.0, 0.0, 400.0]);
    let err = scene
        .clone()
        .into_figure()
        .expect_err("zero-width position must fail");
    assert!(err.contains("Position width and height must be positive"));

    scene.metadata.position = Some([0.0, f64::NAN, 300.0, 400.0]);
    let err = scene
        .into_figure()
        .expect_err("non-finite position must fail");
    assert!(err.contains("Position values must be finite"));
}

#[test]
fn figure_scene_rejects_patch_in_older_schema() {
    let mut figure = Figure::new();
    figure.add_patch_plot(
        PatchPlot::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2]],
        )
        .unwrap(),
    );

    let mut scene = FigureScene::capture(&figure);
    assert!(matches!(scene.plots.first(), Some(ScenePlot::Patch { .. })));
    scene.schema_version = 1;

    let err = scene
        .into_figure()
        .expect_err("older patch schema must fail");
    assert!(err.contains("patch plots require figure scene schema version 2"));
}

#[test]
fn figure_scene_roundtrip_preserves_mesh_plot() {
    let mut figure = Figure::new();
    let mut mesh = MeshPlot::new(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![[0, 1, 2]],
    )
    .unwrap();
    mesh.set_mesh_id(Some("mesh_1".to_string()));
    mesh.set_label(Some("mesh tri".to_string()));
    mesh.set_face_alpha(0.7);
    mesh.set_edge_width(0.25);
    mesh.set_edge_mode(MeshEdgeMode::Feature);
    mesh.set_feature_edge_groups(Some(vec![3]))
        .expect("feature group should be accepted");
    mesh.set_vertex_colors(Some(vec![
        Vec4::new(0.3, 0.4, 0.5, 1.0),
        Vec4::new(0.3, 0.4, 0.5, 1.0),
        Vec4::new(0.3, 0.4, 0.5, 1.0),
    ]))
    .expect("vertex colors should be accepted");
    mesh.set_triangle_colors(Some(vec![Vec4::new(0.3, 0.4, 0.5, 1.0)]))
        .expect("triangle color should be accepted");
    mesh.set_regions(vec![MeshRegion::new(
        "region_default",
        Some("Default Region".to_string()),
        Some("mesh_default".to_string()),
        vec![MeshTriangleRange::new(0, 1)],
    )]);
    mesh.set_highlighted_region_id(Some("region_default".to_string()));
    figure.add_mesh_plot(mesh);

    let scene = FigureScene::capture(&figure);
    assert_eq!(scene.schema_version, FigureScene::SCHEMA_VERSION);
    assert!(matches!(scene.plots.first(), Some(ScenePlot::Mesh { .. })));
    let rebuilt = scene.into_figure().expect("mesh scene restore");
    let Some(PlotElement::Mesh(mesh)) = rebuilt.plots().next() else {
        panic!("expected mesh plot");
    };
    assert_eq!(mesh.mesh_id(), Some("mesh_1"));
    assert_eq!(mesh.triangles(), &[[0, 1, 2]]);
    assert_eq!(mesh.label(), Some("mesh tri"));
    assert!((mesh.face_alpha() - 0.7).abs() < f32::EPSILON);
    assert!((mesh.edge_width() - 0.25).abs() < f32::EPSILON);
    assert_eq!(mesh.edge_mode(), MeshEdgeMode::Feature);
    assert_eq!(mesh.feature_edge_groups().unwrap(), &[3]);
    assert_eq!(
        mesh.vertex_colors()
            .and_then(|colors| colors.first().copied()),
        Some(Vec4::new(0.3, 0.4, 0.5, 1.0))
    );
    assert_eq!(
        mesh.triangle_colors()
            .and_then(|colors| colors.first().copied()),
        Some(Vec4::new(0.3, 0.4, 0.5, 1.0))
    );
    assert_eq!(mesh.regions().len(), 1);
    assert_eq!(mesh.regions()[0].region_id, "region_default");
    assert_eq!(mesh.highlighted_region_id(), Some("region_default"));
}

#[test]
fn figure_scene_roundtrip_preserves_mesh_fea_overlays() {
    let mut figure = Figure::new();
    let mut mesh = MeshPlot::new(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![[0, 1, 2]],
    )
    .unwrap();
    mesh.set_scalar_field(Some(MeshScalarField {
        field_id: "fea.structural.von_mises".to_string(),
        label: Some("Von Mises".to_string()),
        location: MeshFieldLocation::Vertex,
        values: vec![0.0, 0.5, 1.0],
        color_limits: Some([0.0, 1.0]),
        colormap: "viridis".to_string(),
        alpha: 0.8,
    }))
    .unwrap();
    mesh.set_vector_field(Some(MeshVectorField {
        field_id: "fea.em.flux_density".to_string(),
        label: Some("Flux density".to_string()),
        location: MeshFieldLocation::Triangle,
        vectors: vec![Vec3::new(0.0, 0.0, 1.0)],
        scale: 0.25,
        stride: 1,
        color: Vec4::new(0.9, 0.7, 0.2, 1.0),
    }))
    .unwrap();
    mesh.set_deformation(Some(MeshDeformation {
        field_id: "fea.structural.displacement".to_string(),
        label: Some("Displacement".to_string()),
        displacements: vec![Vec3::ZERO, Vec3::Z, Vec3::ZERO],
        scale: 0.5,
    }))
    .unwrap();
    figure.add_mesh_plot(mesh);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("mesh scene restore");
    let Some(PlotElement::Mesh(mesh)) = rebuilt.plots().next() else {
        panic!("expected mesh plot");
    };
    assert_eq!(
        mesh.scalar_field().map(|field| field.field_id.as_str()),
        Some("fea.structural.von_mises")
    );
    assert_eq!(
        mesh.vector_field().map(|field| field.field_id.as_str()),
        Some("fea.em.flux_density")
    );
    assert_eq!(
        mesh.deformation().map(|field| field.field_id.as_str()),
        Some("fea.structural.displacement")
    );
}

#[test]
fn figure_scene_rejects_mesh_in_older_schema() {
    let mut figure = Figure::new();
    figure.add_mesh_plot(
        MeshPlot::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap(),
    );

    let mut scene = FigureScene::capture(&figure);
    assert!(matches!(scene.plots.first(), Some(ScenePlot::Mesh { .. })));
    scene.schema_version = 2;

    let err = scene
        .into_figure()
        .expect_err("older mesh schema must fail");
    assert!(err.contains("mesh plots require figure scene schema version 3"));
}

#[test]
fn line_markers_survive_figure_scene_roundtrip() {
    let mut figure = Figure::new();
    let mut line = LinePlot::new(vec![0.0, 1.0, 2.0], vec![1.0, 3.0, 2.0]).unwrap();
    line.set_line_style(crate::plots::line::LineStyle::None);
    line.set_marker(Some(crate::plots::line::LineMarkerAppearance {
        kind: MarkerStyle::Square,
        size: 7.0,
        edge_color: Vec4::new(0.1, 0.2, 0.3, 1.0),
        face_color: Vec4::new(0.4, 0.5, 0.6, 1.0),
        filled: true,
    }));
    figure.add_line_plot(line);

    let scene = FigureScene::capture(&figure);
    let json = serde_json::to_string(&scene).unwrap();
    let rebuilt: FigureScene = serde_json::from_str(&json).unwrap();
    let rebuilt = rebuilt.into_figure().expect("figure");
    let PlotElement::Line(line) = rebuilt.plots().next().unwrap() else {
        panic!("expected line")
    };
    let marker = line.marker.as_ref().expect("marker kept");
    assert_eq!(marker.kind, MarkerStyle::Square);
    assert_eq!(marker.size, 7.0);
    assert!(marker.filled);
    assert_eq!(marker.face_color, Vec4::new(0.4, 0.5, 0.6, 1.0));
}

#[test]
fn hidden_lines_stay_out_of_the_legend_after_scene_roundtrip() {
    let mut figure = Figure::new();
    figure.add_line_plot(
        LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0])
            .unwrap()
            .with_label("shown"),
    );
    let mut hidden = LinePlot::new(vec![0.0, 1.0], vec![1.0, 0.0])
        .unwrap()
        .with_label("[1 0]");
    hidden.handle_visibility = "off".to_string();
    figure.add_line_plot(hidden);

    let json = serde_json::to_string(&FigureScene::capture(&figure)).unwrap();
    let rebuilt: FigureScene = serde_json::from_str(&json).unwrap();
    let labels: Vec<String> = rebuilt
        .into_figure()
        .expect("figure")
        .legend_entries()
        .into_iter()
        .map(|e| e.label)
        .collect();
    assert_eq!(labels, vec!["shown".to_string()]);
}

#[test]
fn figure_scene_rejects_unknown_reference_line_orientation() {
    let mut scene = FigureScene::capture(&Figure::new());
    scene.plots.push(ScenePlot::ReferenceLine {
        orientation: "VERTICAL".into(),
        value: 2.0,
        color_rgba: [0.1, 0.2, 0.3, 1.0],
        line_width: 1.0,
        line_style: "Solid".into(),
        label: None,
        display_name: None,
        label_orientation: "horizontal".into(),
        axes_index: 0,
        visible: true,
        color_auto: false,
    });

    let rebuilt = scene.clone().into_figure().expect("valid orientation");
    let PlotElement::ReferenceLine(line) = rebuilt.plots().next().unwrap() else {
        panic!("expected reference line")
    };
    assert!(matches!(
        line.orientation,
        ReferenceLineOrientation::Vertical
    ));

    let ScenePlot::ReferenceLine { orientation, .. } = &mut scene.plots[0] else {
        panic!("expected reference line scene plot")
    };
    *orientation = "diagonal".into();

    let err = scene
        .into_figure()
        .expect_err("unknown orientation must fail");
    assert!(err.contains("unknown reference line orientation 'diagonal'"));
}

#[test]
fn figure_scene_roundtrip_reconstructs_surface_and_scatter3() {
    let mut figure = Figure::new().with_title("Replay3D").with_subplot_grid(1, 2);
    let mut surface = SurfacePlot::new(
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![vec![0.0, 1.0], vec![1.0, 2.0]],
    )
    .expect("surface data should be valid");
    surface.label = Some("surface".to_string());
    figure.add_surface_plot_on_axes(surface, 0);

    let mut scatter3 = Scatter3Plot::new(vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(2.0, 3.0, 4.0),
    ])
    .expect("scatter3 data should be valid");
    scatter3.label = Some("scatter3".to_string());
    figure.add_scatter3_plot_on_axes(scatter3, 1);

    let scene = FigureScene::capture(&figure);
    let rebuilt = scene.into_figure().expect("scene restore should succeed");
    assert_eq!(rebuilt.axes_grid(), (1, 2));
    assert_eq!(rebuilt.plots().count(), 2);
    assert_eq!(rebuilt.title.as_deref(), Some("Replay3D"));
    assert!(matches!(
        rebuilt.plots().next(),
        Some(PlotElement::Surface(_))
    ));
    assert!(matches!(
        rebuilt.plots().nth(1),
        Some(PlotElement::Scatter3(_))
    ));
}

#[test]
fn scatter3_scene_roundtrip_preserves_wide_integer_source_storage() {
    let wide = 9_007_199_254_740_993_u64;
    let shape = vec![1, 2];
    let scatter3 = Scatter3Plot::from_numeric_data(
        NumericPlotData::new(NumericStorage::U64(vec![1, wide]), shape.clone()).unwrap(),
        NumericPlotData::new(NumericStorage::I16(vec![-2, 3]), shape.clone()).unwrap(),
        NumericPlotData::new(NumericStorage::F32(vec![4.0, 5.0]), shape).unwrap(),
    )
    .unwrap();
    let mut figure = Figure::new();
    figure.add_scatter3_plot(scatter3);

    let json = serde_json::to_string(&FigureScene::capture(&figure)).unwrap();
    assert!(json.contains(&wide.to_string()));
    let rebuilt = serde_json::from_str::<FigureScene>(&json)
        .unwrap()
        .into_figure()
        .unwrap();
    let Some(PlotElement::Scatter3(scatter3)) = rebuilt.plots().next() else {
        panic!("expected scatter3")
    };
    let (x, y, z) = scatter3.source_data().expect("typed source data");
    assert_eq!(x.storage(), &NumericStorage::U64(vec![1, wide]));
    assert_eq!(y.storage(), &NumericStorage::I16(vec![-2, 3]));
    assert_eq!(z.storage(), &NumericStorage::F32(vec![4.0, 5.0]));
}

#[test]
fn figure_scene_roundtrip_preserves_line3_plot() {
    let mut figure = Figure::new();
    let line3 = Line3Plot::new(vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, 3.0])
        .unwrap()
        .with_label("Trajectory");
    figure.add_line3_plot(line3);

    let rebuilt = FigureScene::capture(&figure)
        .into_figure()
        .expect("scene restore should succeed");

    let PlotElement::Line3(line3) = rebuilt.plots().next().unwrap() else {
        panic!("expected line3")
    };
    let (x, _, z) = line3.host_xyz_f64().unwrap().unwrap();
    assert_eq!(x, vec![0.0, 1.0]);
    assert_eq!(z, vec![2.0, 3.0]);
    assert_eq!(line3.label.as_deref(), Some("Trajectory"));
}

#[test]
fn figure_scene_json_roundtrip_preserves_wide_integer_line_sources() {
    let wide = 9_007_199_254_740_993_u64;
    let x = NumericPlotData::new(NumericStorage::U64(vec![wide, wide + 1]), vec![1, 2]).unwrap();
    let y = NumericPlotData::new(NumericStorage::I64(vec![-(wide as i64), 7]), vec![1, 2]).unwrap();
    let mut figure = Figure::new();
    figure.add_line_plot(LinePlot::from_numeric_data(x, y).unwrap());

    let json = serde_json::to_string(&FigureScene::capture(&figure)).unwrap();
    let rebuilt = serde_json::from_str::<FigureScene>(&json)
        .unwrap()
        .into_figure()
        .unwrap();
    let PlotElement::Line(line) = rebuilt.plots().next().unwrap() else {
        panic!("expected line")
    };
    let (x, y) = line.source_data();
    assert_eq!(
        x.unwrap().storage(),
        &NumericStorage::U64(vec![wide, wide + 1])
    );
    assert_eq!(
        y.unwrap().storage(),
        &NumericStorage::I64(vec![-(wide as i64), 7])
    );
}

#[test]
fn figure_scene_json_roundtrip_preserves_nonfinite_native_float_sources() {
    let double_bits = [
        0x3ff8_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0042,
    ];
    let single_bits = [
        0x3fc0_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0042,
    ];
    let x = NumericPlotData::new(
        NumericStorage::F64(double_bits.map(f64::from_bits).to_vec()),
        vec![1, double_bits.len()],
    )
    .unwrap();
    let y = NumericPlotData::new(
        NumericStorage::F32(single_bits.map(f32::from_bits).to_vec()),
        vec![1, single_bits.len()],
    )
    .unwrap();
    let mut figure = Figure::new();
    figure.add_line_plot(LinePlot::from_numeric_data(x, y).unwrap());

    let scene = FigureScene::capture(&figure);
    assert_eq!(scene.schema_version, FigureScene::SCHEMA_VERSION);
    let encoded = serde_json::to_string(&scene).unwrap();
    assert!(encoded.contains(r#""f64_bits":"7ff8000000000042""#));
    assert!(encoded.contains(r#""f32_bits":"7fc00042""#));
    let rebuilt = serde_json::from_str::<FigureScene>(&encoded)
        .unwrap()
        .into_figure()
        .unwrap();
    let Some(PlotElement::Line(line)) = rebuilt.plots().next() else {
        panic!("expected line")
    };
    let (x, y) = line.source_data();
    let Some(NumericStorage::F64(x_values)) = x.map(NumericPlotData::storage) else {
        panic!("expected native double source")
    };
    let Some(NumericStorage::F32(y_values)) = y.map(NumericPlotData::storage) else {
        panic!("expected native single source")
    };
    assert_eq!(x.unwrap().shape(), &[1, double_bits.len()]);
    assert_eq!(y.unwrap().shape(), &[1, single_bits.len()]);
    assert_eq!(
        x_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        double_bits
    );
    assert_eq!(
        y_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        single_bits
    );
    // Canonical bit tags stay strict. Legacy null-gap admission has its
    // own test; null is not a malformed tag under that import contract.
    assert!(serde_json::from_str::<SceneNumericStorage>(
        r#"{"dtype":"f64","values":[{"f64_bits":"not-bits"}]}"#
    )
    .is_err());

    let mut legacy_figure = Figure::new()
        .with_title("Legacy title")
        .with_labels("Time", "Value")
        .with_legend(true);
    legacy_figure.add_line_plot(
        LinePlot::from_numeric_data(
            NumericPlotData::new(NumericStorage::F64(vec![1.0, -0.0]), vec![1, 2]).unwrap(),
            NumericPlotData::new(NumericStorage::F32(vec![2.0, 3.0]), vec![1, 2]).unwrap(),
        )
        .unwrap(),
    );
    let mut legacy = FigureScene::capture(&legacy_figure);
    legacy.schema_version = 1;
    legacy.metadata.axes_metadata = None;
    let finite_legacy = serde_json::to_string(&legacy).unwrap();
    let rebuilt = serde_json::from_str::<FigureScene>(&finite_legacy)
        .unwrap()
        .into_figure()
        .expect("finite legacy scene remains readable");
    assert!(rebuilt.legend_enabled);
    assert!(rebuilt.axes_metadata(0).unwrap().legend_enabled);
    assert_eq!(rebuilt.title.as_deref(), Some("Legacy title"));
    assert_eq!(rebuilt.x_label.as_deref(), Some("Time"));
    assert_eq!(rebuilt.y_label.as_deref(), Some("Value"));
}

#[test]
fn figure_scene_json_roundtrip_preserves_wide_integer_stairs_and_stem_sources() {
    let wide = 9_007_199_254_740_993_u64;
    let source = || {
        (
            NumericPlotData::new(NumericStorage::U64(vec![wide, wide + 1]), vec![1, 2]).unwrap(),
            NumericPlotData::new(NumericStorage::I16(vec![-2, 3]), vec![1, 2]).unwrap(),
        )
    };
    let (stairs_x, stairs_y) = source();
    let (stem_x, stem_y) = source();
    let mut figure = Figure::new();
    figure.add_stairs_plot(StairsPlot::new_with_source(stairs_x, stairs_y).unwrap());
    figure.add_stem_plot(StemPlot::new_with_source(stem_x, stem_y).unwrap());

    let json = serde_json::to_string(&FigureScene::capture(&figure)).unwrap();
    assert!(json.contains(&wide.to_string()));
    let rebuilt = serde_json::from_str::<FigureScene>(&json)
        .unwrap()
        .into_figure()
        .unwrap();
    let plots = rebuilt.plots().collect::<Vec<_>>();
    let PlotElement::Stairs(stairs) = plots[0] else {
        panic!("expected stairs")
    };
    let PlotElement::Stem(stem) = plots[1] else {
        panic!("expected stem")
    };
    assert_eq!(
        stairs.x_source.as_ref().unwrap().storage(),
        &NumericStorage::U64(vec![wide, wide + 1])
    );
    assert_eq!(
        stairs.y_source.as_ref().unwrap().storage(),
        &NumericStorage::I16(vec![-2, 3])
    );
    assert_eq!(
        stem.x_source.as_ref().unwrap().storage(),
        &NumericStorage::U64(vec![wide, wide + 1])
    );
    assert_eq!(
        stem.y_source.as_ref().unwrap().storage(),
        &NumericStorage::I16(vec![-2, 3])
    );
}

#[test]
fn figure_scene_json_roundtrip_preserves_wide_integer_quiver_sources() {
    let wide = 9_007_199_254_740_993_u64;
    let i16_data =
        |values: Vec<i16>| NumericPlotData::new(NumericStorage::I16(values), vec![1, 2]).unwrap();
    let u = NumericPlotData::new(NumericStorage::U64(vec![wide, wide + 2]), vec![1, 2]).unwrap();
    let quiver = QuiverPlot::from_numeric_data(
        i16_data(vec![0, 1]),
        i16_data(vec![2, 3]),
        u,
        i16_data(vec![4, 5]),
    )
    .unwrap();
    let mut figure = Figure::new();
    figure.add_quiver_plot(quiver);

    let json = serde_json::to_string(&FigureScene::capture(&figure)).unwrap();
    let rebuilt = serde_json::from_str::<FigureScene>(&json)
        .unwrap()
        .into_figure()
        .unwrap();
    let PlotElement::Quiver(quiver) = rebuilt.plots().next().unwrap() else {
        panic!("expected quiver")
    };
    let (_, _, _, u, _, _) = quiver.source_data().expect("quiver source data");
    assert_eq!(u.storage(), &NumericStorage::U64(vec![wide, wide + 2]));
}
