//! An edge along a valley crease (a boss's base where it meets its plate) draws whole across its
//! length in shaded-with-edges: both faces beside the line meet it at the crease, so neither
//! hides it.
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

/// A plate face on z = 0 in front of y = 0, a wall face on y = 0 rising from it, and a dark edge
/// along their crease, lifted `lift` along the faces' bisector as the Desktop lifts body edges.
fn scene(lift: f32) -> GeometryScene {
    let light = Vec4::new(0.85, 0.87, 0.9, 1.0);
    let plate = [
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
    ];
    let wall = [
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(-1.0, 0.0, 1.0),
    ];
    let vertices: Vec<Vertex> = plate
        .iter()
        .chain(wall.iter())
        .map(|&p| Vertex::new(p, light))
        .collect();
    let faces = GeometrySceneChunk::indexed_triangles(
        "faces",
        vertices,
        vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
        Material::default(),
    );
    let dark = Vec4::new(0.02, 0.02, 0.02, 1.0);
    let up = Vec3::new(0.0, -1.0, 1.0).normalize() * lift;
    let mut data = faces.render_data.clone();
    data.pipeline_type = PipelineType::Lines;
    data.vertices = [-0.6, 0.6]
        .iter()
        .map(|&x| Vertex::new(Vec3::new(x, 0.0, 0.0) + up, dark))
        .collect();
    data.indices = Some(vec![0, 1]);
    data.draw_calls[0].vertex_count = 2;
    data.draw_calls[0].index_count = Some(2);
    let edges = GeometrySceneChunk::from_render_data("body:edges", data);
    GeometryScene::new("edge-valley", 1, vec![faces, edges])
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

/// The crease seen from in front and `elevation` degrees above, 1.25 px edges at 2×, from
/// `distance` away.
async fn render(elevation: f32, distance: f32, lift: f32) -> Option<Vec<u8>> {
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
    let mut camera = Camera::new();
    camera.position = Vec3::new(0.3, -distance * angle.cos(), distance * angle.sin());
    camera.target = Vec3::ZERO;
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
            &scene(lift),
            Some(&camera),
            Some(&presentation),
        )
        .await
        .unwrap();
    if let Ok(directory) = std::env::var("RUNMAT_EDGE_STROKE_IMAGES") {
        let png = runmat_plot::export::cpu_surface::encode_png_bytes(WIDTH, HEIGHT, &rgba).unwrap();
        std::fs::write(
            format!("{directory}/edge-valley-{elevation}-{distance}-{lift}.png"),
            png,
        )
        .unwrap();
    }
    Some(rgba)
}

#[tokio::test]
async fn an_edge_along_a_valley_crease_draws_whole() {
    if std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to exercise the real renderer");
        return;
    }
    let mut failures = Vec::new();
    for elevation in [20.0_f32, 35.0, 50.0] {
        for distance in [5.0_f32, 50.0] {
            // The Desktop lifts body edges off their faces by up to 0.1 mm (`liftFor`).
            for lift in [3.0e-5_f32, 1.0e-4] {
                let Some(rgba) = render(elevation, distance, lift).await else {
                    eprintln!("no headless GPU adapter");
                    return;
                };
                let (covered, total) = covered_columns(&rgba);
                eprintln!("elevation {elevation} distance {distance} lift {lift}: {covered}/{total} columns");
                if covered < total {
                    failures.push(format!(
                        "{elevation} deg, {distance} away, lift {lift}: {covered}/{total}"
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "the crease edge has gaps: {failures:?}"
    );
}
