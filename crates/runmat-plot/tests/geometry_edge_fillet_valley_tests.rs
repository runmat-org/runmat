//! An edge where a concave fillet meets a wall draws whole in shaded-with-edges. The faces meet
//! tangent there, so the depth across the line changes at one rate up to the edge and the fillet
//! then curves toward the viewer; neither face is in front of the edge. A pocket's inner corner
//! fillet on the bracket's "Solver mesh" poster drew this edge dashed.
use glam::{Vec3, Vec4};
use runmat_plot::{
    core::{Camera, ProjectionType},
    core::{Material, PipelineType, Vertex},
    export::native_surface::{create_headless_context, is_headless_gpu_unavailable_error},
    geometry_scene::GeometryScenePresentation,
    styling::{PlotThemeConfig, ThemeVariant},
    GeometryScene, GeometrySceneChunk,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 200;

/// A wall on y = 0 above z = `radius`, and a concave quarter-round fillet of `radius` (axis along
/// x at y = -radius, z = radius) running from the wall down to a plate on z = 0, which continues
/// to y = -1. A dark edge runs along the fillet-to-wall tangent line (y = 0, z = radius), lifted
/// `lift` along the shared normal (-y) as the Desktop lifts body edges.
fn scene(radius: f32, segments: usize, lift: f32) -> GeometryScene {
    let light = Vec4::new(0.85, 0.87, 0.9, 1.0);
    // The profile in (y, z) from the wall's top down the fillet to the plate's front.
    let mut profile = vec![(0.0_f32, 1.0_f32)];
    for i in 0..=segments {
        let a = std::f32::consts::FRAC_PI_2 * i as f32 / segments as f32;
        profile.push((-radius + radius * a.cos(), radius - radius * a.sin()));
    }
    profile.push((-1.0, 0.0));
    let mut vertices = Vec::new();
    for &(y, z) in &profile {
        vertices.push(Vertex::new(Vec3::new(-1.0, y, z), light));
        vertices.push(Vertex::new(Vec3::new(1.0, y, z), light));
    }
    let mut indices = Vec::new();
    for k in 0..profile.len() as u32 - 1 {
        let (a, b, c, d) = (2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2);
        indices.extend_from_slice(&[a, c, b, a, d, c]);
    }
    let faces =
        GeometrySceneChunk::indexed_triangles("faces", vertices, indices, Material::default());
    let dark = Vec4::new(0.02, 0.02, 0.02, 1.0);
    let up = Vec3::new(0.0, -1.0, 0.0) * lift;
    let mut data = faces.render_data.clone();
    data.pipeline_type = PipelineType::Lines;
    data.vertices = [-0.6, 0.6]
        .iter()
        .map(|&x| Vertex::new(Vec3::new(x, 0.0, radius) + up, dark))
        .collect();
    data.indices = Some(vec![0, 1]);
    data.draw_calls[0].vertex_count = 2;
    data.draw_calls[0].index_count = Some(2);
    let edges = GeometrySceneChunk::from_render_data("body:edges", data);
    GeometryScene::new("edge-fillet-valley", 1, vec![faces, edges])
}

/// For each column in the middle third, whether it holds a dark pixel.
fn covered_columns(rgba: &[u8]) -> (usize, usize) {
    let (mut covered, mut total) = (0, 0);
    for x in WIDTH * 2 / 5..WIDTH * 3 / 5 {
        total += 1;
        let dark = (0..HEIGHT).any(|y| {
            let i = ((y * WIDTH + x) * 4) as usize;
            rgba[i] < 120 && rgba[i + 1] < 120 && rgba[i + 2] < 120
        });
        covered += usize::from(dark);
    }
    (covered, total)
}

/// The tangent line seen from in front and `elevation` degrees above, 1.25 px edges at 2×, from
/// `distance` away.
async fn render(
    elevation: f32,
    distance: f32,
    radius: f32,
    segments: usize,
    lift: f32,
) -> Option<Vec<u8>> {
    let mut context = match create_headless_context(WIDTH, HEIGHT).await {
        Ok(context) => context,
        Err(error) if is_headless_gpu_unavailable_error(&error) => return None,
        Err(error) => panic!("{error}"),
    };
    context.set_pixels_per_point(2.0);
    context.set_theme_config(PlotThemeConfig {
        variant: ThemeVariant::ClassicLight,
        ..Default::default()
    });
    let presentation = GeometryScenePresentation {
        edge_width_px: Some(1.25),
        ..Default::default()
    };
    let angle = elevation.to_radians();
    let target = Vec3::new(0.0, 0.0, radius);
    let mut camera = Camera::new();
    camera.position = target + Vec3::new(0.3, -distance * angle.cos(), distance * angle.sin());
    camera.target = target;
    camera.up = Vec3::Z;
    camera.axis_scale = Vec3::ONE;
    camera.aspect_ratio = WIDTH as f32 / HEIGHT as f32;
    camera.projection = ProjectionType::Perspective {
        fov: 2.0 * (1.2 / distance).atan(),
        near: distance * 0.02,
        far: distance * 4.0,
    };
    camera.mark_dirty();
    let rgba = context
        .render_geometry_scene_to_rgba_with_presentation(
            &scene(radius, segments, lift),
            Some(&camera),
            Some(&presentation),
        )
        .await
        .unwrap();
    if let Ok(directory) = std::env::var("RUNMAT_EDGE_STROKE_IMAGES") {
        let png = runmat_plot::export::cpu_surface::encode_png_bytes(WIDTH, HEIGHT, &rgba).unwrap();
        std::fs::write(
            format!("{directory}/edge-fillet-valley-{elevation}-{distance}-{radius}-{segments}-{lift}.png"),
            png,
        )
        .unwrap();
    }
    Some(rgba)
}

#[tokio::test]
async fn an_edge_where_a_fillet_meets_a_wall_draws_whole() {
    if std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to exercise the real renderer");
        return;
    }
    let mut failures = Vec::new();
    // From 20 degrees above (the fillet faces the viewer) to 82 (the wall above it seen nearly
    // edge-on, as the rib wall beside the pocket's inner fillet is on the poster).
    for elevation in [20.0_f32, 50.0, 70.0, 78.0, 82.0] {
        for distance in [5.0_f32, 50.0] {
            // A small fillet (a few px across) and a large one (tens of px); smooth, and faceted
            // in three as a 3 mm solver mesh draws a small fillet.
            for (radius, segments) in [(0.02_f32, 24), (0.2, 24), (0.2, 3), (0.05, 3)] {
                // No lift (the CLI poster), and the Desktop's lifts of up to 0.1 mm (`liftFor`).
                for lift in [0.0_f32, 3.0e-5, 1.0e-4] {
                    let Some(rgba) = render(elevation, distance, radius, segments, lift).await
                    else {
                        eprintln!("no headless GPU adapter");
                        return;
                    };
                    let (covered, total) = covered_columns(&rgba);
                    if covered < total {
                        failures.push(format!(
                            "{elevation} deg, {distance} away, radius {radius} in {segments}, lift {lift}: {covered}/{total}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "the fillet-to-wall edge has gaps: {failures:?}"
    );
}
