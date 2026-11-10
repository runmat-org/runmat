//! A saved field result imported as a figure keeps its geometry scene.
//!
//! The browser opens a captured run's figures through `importFigureScene`, which restores a
//! figure with `FigureScene::into_figure`. A field result is saved from a geometry scene: its
//! legend and color bar (the overlay), outline lines and unit context travel beside the meshes.
//! These tests save such a scene, decode it from its JSON bytes, restore the figure and check
//! that the figure still carries all three, and that exporting the figure again writes them.

use glam::Vec4;
use runmat_plot::geometry_scene::{GeometryPhysicalContext, GeometrySceneScalarLegend};
use runmat_plot::plots::{Figure, LinePlot};
use runmat_plot::{
    cad_default_material, geometry_scene_vertex, FigureScene, GeometryScene, GeometrySceneChunk,
    GeometrySceneCompleteness, GeometrySceneOverlay,
};
use runmat_types::physical::{CoordinateUnits, LengthUnit, LengthUnitEvidence};

const MINIMUM_K: f64 = 301.066;
const MAXIMUM_K: f64 = 303.869;

fn field_chunk() -> GeometrySceneChunk {
    let corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)];
    let colors = [
        [0.2, 0.1, 0.6, 1.0],
        [0.2, 0.5, 0.8, 1.0],
        [0.9, 0.8, 0.2, 1.0],
        [0.9, 0.3, 0.1, 1.0],
    ];
    let vertices = corners
        .iter()
        .zip(colors)
        .map(|(&(x, y), color)| geometry_scene_vertex([x, y, 0.0], color, [0.0, 0.0, 1.0]))
        .collect();
    let mut material = cad_default_material();
    material.albedo = Vec4::ONE;
    GeometrySceneChunk::indexed_triangles("field", vertices, vec![0, 1, 2, 0, 2, 3], material)
}

fn temperature_overlay() -> GeometrySceneOverlay {
    GeometrySceneOverlay {
        source_name: None,
        status: GeometrySceneCompleteness::Complete,
        quality_label: None,
        format: None,
        source_label: None,
        allow_create_fea_study: false,
        byte_count: None,
        mesh_count: 1,
        vertex_count: 4,
        triangle_count: 2,
        progress_percent: None,
        scalar_legend: Some(GeometrySceneScalarLegend {
            label: "Temperature".into(),
            unit: "K".into(),
            minimum: MINIMUM_K,
            maximum: MAXIMUM_K,
            limit: None,
            note: None,
        }),
        region_count: 0,
        mapped_region_count: 0,
        assembly_nodes: vec![],
        regions: vec![],
        warnings: vec![],
    }
}

fn field_result() -> GeometryScene {
    let field = field_chunk();
    let outline = field
        .wireframe_edges([0.1, 0.1, 0.1, 1.0])
        .expect("outline");
    let mut scene = GeometryScene::new("battery-temperature", 1, vec![field, outline])
        .with_title("Temperature")
        .with_overlay(temperature_overlay())
        .with_physical_context(GeometryPhysicalContext {
            coordinate_units: CoordinateUnits::new(LengthUnit::Millimeter),
            source_unit_evidence: LengthUnitEvidence::explicit_fallback(
                LengthUnit::Millimeter,
                "battery module fixture",
            ),
        });
    scene.show_grid = false;
    scene
}

/// Save a scene and decode it from its bytes, as a captured run's figure file is read.
fn saved_bytes(scene: &FigureScene) -> FigureScene {
    let bytes = serde_json::to_vec(scene).expect("encode");
    serde_json::from_slice(&bytes).expect("decode")
}

fn assert_temperature_scene(scene: &GeometryScene) {
    let legend = scene
        .overlay
        .as_ref()
        .and_then(|overlay| overlay.scalar_legend.as_ref())
        .expect("the scalar legend survives");
    assert_eq!(legend.label, "Temperature");
    assert_eq!(legend.unit, "K");
    assert_eq!(legend.minimum, MINIMUM_K);
    assert_eq!(legend.maximum, MAXIMUM_K);
    assert!(
        scene.physical_context.is_some(),
        "the unit context survives"
    );
    assert!(
        scene
            .chunks
            .iter()
            .any(|chunk| chunk.render_data.pipeline_type == runmat_plot::core::PipelineType::Lines),
        "the outline lines survive"
    );
    assert_eq!(scene.title.as_deref(), Some("Temperature"));
    assert!(!scene.show_grid);
}

fn imported_figure() -> Figure {
    saved_bytes(&FigureScene::from_geometry_scene(&field_result()))
        .into_figure()
        .expect("import")
}

#[test]
fn an_imported_field_result_keeps_its_legend_colormap_range_lines_and_units() {
    let figure = imported_figure();
    let source = figure
        .geometry_source()
        .expect("the figure keeps the geometry scene it was saved from");
    assert_temperature_scene(source);
    // The figure's own plots still list the field for the figure's details.
    assert_eq!(figure.plots().count(), 1);
}

#[test]
fn exporting_an_imported_field_result_again_keeps_its_legend() {
    let figure = imported_figure();
    let again = saved_bytes(&FigureScene::capture(&figure));
    assert!(again.is_geometry_export());
    let restored = again.into_figure().expect("second import");
    assert_temperature_scene(restored.geometry_source().expect("geometry scene"));
}

#[test]
fn each_import_draws_as_its_own_scene() {
    let first = imported_figure();
    let second = imported_figure();
    assert_ne!(
        first.geometry_source().unwrap().scene_id,
        second.geometry_source().unwrap().scene_id,
        "two imported figures must not share a renderer cache key"
    );
}

#[test]
fn a_plain_figure_scene_imports_without_a_geometry_scene() {
    let mut figure = Figure::new();
    figure.add_line_plot(LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]).expect("line"));
    let scene = saved_bytes(&FigureScene::capture(&figure));
    assert!(!scene.is_geometry_export());
    assert!(scene.into_figure().unwrap().geometry_source().is_none());
}

#[test]
fn plotting_into_an_imported_result_draws_the_figure_plots() {
    let mut figure = imported_figure();
    figure.add_line_plot(LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]).expect("line"));
    assert!(
        figure.geometry_source().is_none(),
        "a figure with added plots no longer shows the saved scene"
    );
}
