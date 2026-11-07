//! Plot renderer tests.

use super::*;
use crate::core::DrawCall;

#[test]
fn pale_scene_lines_darken_on_light_background_only() {
    let line = |color: [f32; 4]| Vertex::new(Vec3::ZERO, Vec4::from_array(color));
    let mut render_data = crate::core::RenderData {
        pipeline_type: crate::core::PipelineType::Lines,
        vertices: vec![
            line([0.92, 0.92, 1.0, 0.35]),
            line([0.58, 0.72, 0.88, 0.95]),
        ],
        indices: None,
        gpu_vertices: None,
        bounds: None,
        material: crate::core::Material::default(),
        draw_calls: Vec::new(),
        image: None,
    };
    darken_pale_lines_for_light_background(&mut render_data);
    let outline = Vec4::from_array(render_data.vertices[0].color);
    assert!((relative_luminance(outline) - 0.38).abs() < 1e-3);
    assert!(outline.z > outline.x, "keeps its blue tint");
    assert_eq!(outline.w, 0.5);
    // Mid-tone wireframe edges already contrast with white and stay as saved.
    assert_eq!(render_data.vertices[1].color, [0.58, 0.72, 0.88, 0.95]);
}

#[test]
fn xray_opens_only_for_scenes_whose_every_surface_is_see_through() {
    use crate::geometry_scene::{cad_default_material, vertex, GeometrySceneChunk};
    let surface = |id: &str, alpha: f32| {
        let mut material = cad_default_material();
        material.albedo.w = alpha;
        GeometrySceneChunk::indexed_triangles(
            id,
            vec![
                vertex([0.0, 0.0, 0.0], [0.8, 0.8, 0.8, alpha], [0.0, 0.0, 1.0]),
                vertex([1.0, 0.0, 0.0], [0.8, 0.8, 0.8, alpha], [0.0, 0.0, 1.0]),
                vertex([0.0, 1.0, 0.0], [0.8, 0.8, 0.8, alpha], [0.0, 0.0, 1.0]),
            ],
            vec![0, 1, 2],
            material,
        )
    };
    let scene = |chunks| GeometryScene::new("scene", 1, chunks);
    // Glass among opaque parts stays blended on its own; the rest stays solid.
    assert!(!scene_authored_as_xray(&scene(vec![
        surface("skin", 1.0),
        surface("glass", 0.35)
    ])));
    assert!(scene_authored_as_xray(&scene(vec![
        surface("a", 0.38),
        surface("b", 0.38)
    ])));
    assert!(!scene_authored_as_xray(&scene(Vec::new())));
}

#[test]
fn log_grid_values_follow_decades_inside_limits() {
    assert_eq!(
        PlotRenderer::log_grid_values(10.0, 1000.0, false),
        vec![10.0, 100.0, 1000.0]
    );
    assert_eq!(
        PlotRenderer::log_grid_values(15.0, 45.0, true),
        vec![20.0, 30.0, 40.0]
    );
    assert!(PlotRenderer::log_grid_values(-1.0, 100.0, false).is_empty());
}
use crate::plots::{figure::PlotElement, Figure, PatchPlot};

#[test]
fn indexed_renderer_respects_cleared_draw_calls_and_index_ranges() {
    assert!(indexed_draw_ranges(&[], 6).is_empty());
    let calls = [DrawCall {
        vertex_offset: 0,
        vertex_count: 4,
        index_offset: Some(2),
        index_count: Some(3),
        instance_count: 2,
    }];
    assert_eq!(indexed_draw_ranges(&calls, 6), vec![(2..5, 0..2)]);
}

fn patch_element(vertices: Vec<Vec3>) -> PlotElement {
    PlotElement::Patch(PatchPlot::new(vertices, vec![vec![0, 1, 2]]).unwrap())
}

#[test]
fn patch_3d_detection_preserves_small_scene_depth() {
    let patch = patch_element(vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0e-4, 0.0, 0.0),
        Vec3::new(0.0, 1.0e-4, 1.0e-8),
    ]);

    assert!(patch.is_3d_content());
}

#[test]
fn patch_3d_detection_ignores_large_scene_relative_noise() {
    let patch = patch_element(vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0e6, 0.0, 0.0),
        Vec3::new(0.0, 1.0e6, 0.5),
    ]);

    assert!(!patch.is_3d_content());
}

#[test]
fn applies_z_limits_to_3d_display_bounds() {
    let mut figure = Figure::new();
    figure.set_axes_z_limits(0, Some((-2.0, 3.0)));
    let bounds = BoundingBox::new(Vec3::new(-1.0, -1.0, -10.0), Vec3::new(1.0, 1.0, 10.0));

    let limited = PlotRenderer::apply_3d_display_limits_to_bounds(bounds, Some(&figure), 0);

    assert_eq!(limited.min.z, -2.0);
    assert_eq!(limited.max.z, 3.0);
    assert_eq!(limited.min.x, -1.0);
    assert_eq!(limited.max.x, 1.0);
}

#[test]
fn applies_data_aspect_ratio_to_3d_display_bounds() {
    let mut figure = Figure::new();
    figure.set_axes_data_aspect_ratio(0, [1.0, 2.0, 4.0], "manual");
    let bounds = BoundingBox::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 1.0));

    let display = PlotRenderer::apply_3d_display_limits_to_bounds(bounds, Some(&figure), 0);

    // A fixed data aspect keeps the limits; the axis scale carries the aspect.
    assert_eq!(display.min, Vec3::ZERO);
    assert_eq!(display.max, Vec3::new(1.0, 2.0, 1.0));
}

#[test]
fn data_aspect_scales_one_data_unit_per_axis_without_moving_limits() {
    // daspect([1 2 4]) over a 1 x 2 x 1 box: units = max(1/1, 2/2, 1/4) = 1, so one data
    // unit is 1, 2 and 4 box units long along x, y and z.
    let scale = data_aspect_axis_scale(Vec3::new(1.0, 2.0, 1.0), [1.0, 2.0, 4.0]);
    assert!((scale - Vec3::new(1.0, 0.5, 0.25)).abs().max_element() < 1.0e-6);
    // axis equal on a flat 2320 x 1320 x 220 mm pack: one shared scale, no cube.
    let scale = data_aspect_axis_scale(Vec3::new(2320.0, 1320.0, 220.0), [1.0; 3]);
    assert!((scale - Vec3::splat(1.0 / 2320.0)).abs().max_element() < 1.0e-9);
}

#[test]
fn degenerate_z_bounds_remain_finite_for_3d_display_bounds() {
    let figure = Figure::new();
    let bounds = BoundingBox::new(Vec3::new(-2.0, -2.0, 0.0), Vec3::new(2.0, 2.0, 0.0));

    // Figure metadata applies its default tick-aligned axis limits. The
    // metadata-free fallback instead supplies the minimum camera span.
    for (figure, z_min, z_max) in [(Some(&figure), -0.6, 0.6), (None, -0.05, 0.05)] {
        let display = PlotRenderer::apply_3d_display_limits_to_bounds(bounds, figure, 0);
        assert!(PlotRenderer::bounds_are_finite(display));
        assert_eq!(display.min.z, z_min);
        assert_eq!(display.max.z, z_max);
        assert_eq!(display.min.x, bounds.min.x);
        assert_eq!(display.max.x, bounds.max.x);
        assert_eq!(display.min.y, bounds.min.y);
        assert_eq!(display.max.y, bounds.max.y);
    }
}

#[test]
fn geometry_scene_view_presets_map_to_camera_presets() {
    assert_eq!(
        geometry_scene_view_preset_to_camera(GeometrySceneViewPreset::Isometric),
        CameraViewPreset::Perspective
    );
    assert_eq!(
        geometry_scene_view_preset_to_camera(GeometrySceneViewPreset::Front),
        CameraViewPreset::Front
    );
    assert_eq!(
        geometry_scene_view_preset_to_camera(GeometrySceneViewPreset::Top),
        CameraViewPreset::Top
    );
}

#[test]
fn positive_log_bounds_preserve_valid_positive_range() {
    assert_eq!(
        crate::plots::figure::positive_log_bounds(1.0, 100.0),
        (1.0, 100.0)
    );
}

#[test]
fn positive_log_bounds_recover_from_nonpositive_lower_bound() {
    let (lo, hi) = crate::plots::figure::positive_log_bounds(-10.0, 100.0);

    assert!(lo > 0.0);
    assert_eq!(hi, 100.0);
    assert!(lo < hi);
}

#[test]
fn axes_view_contract_tracks_subplot_limits() {
    let mut base = Figure::new();
    base.set_subplot_grid(2, 1);
    let mut limited = base.clone();
    limited.set_axes_limits(0, Some((0.0, 30.0)), None);
    limited.set_axes_limits(1, Some((200.0, 450.0)), None);

    let base_contract = PlotRenderer::axes_view_contract_for_figure(&base);
    let limited_contract = PlotRenderer::axes_view_contract_for_figure(&limited);

    assert_ne!(base_contract, limited_contract);
    assert_eq!(limited_contract.axes[0].x_limits, Some((0.0, 30.0)));
    assert_eq!(limited_contract.axes[1].x_limits, Some((200.0, 450.0)));
}

#[test]
fn figure_level_minor_grid_survives_default_axes_metadata() {
    let mut figure = Figure::new();
    figure.minor_grid_enabled = true;

    assert!(!figure.axes_metadata(0).unwrap().minor_grid_enabled);
    assert!(PlotRenderer::minor_grid_for_axes(
        Some(&figure),
        figure.minor_grid_enabled,
        0
    ));
}

#[test]
fn explicit_axes_minor_grid_false_overrides_figure_level_minor_grid() {
    let mut figure = Figure::new();
    figure.minor_grid_enabled = true;
    figure.set_axes_minor_grid_enabled(0, false);

    let meta = figure.axes_metadata(0).unwrap();
    assert!(!meta.minor_grid_enabled);
    assert!(meta.minor_grid_explicit);
    assert!(!PlotRenderer::minor_grid_for_axes(
        Some(&figure),
        figure.minor_grid_enabled,
        0
    ));
}
