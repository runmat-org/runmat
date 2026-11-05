//! Geometry edges keep a set CSS-pixel weight on the real renderer.
use glam::{Vec3, Vec4};
use runmat_plot::{
    core::{Material, PipelineType, Vertex},
    export::native_surface::{create_headless_context, is_headless_gpu_unavailable_error},
    core::{Camera, ProjectionType},
    geometry_scene::GeometryScenePresentation,
    styling::{PlotThemeConfig, ThemeVariant},
    GeometryScene, GeometrySceneChunk,
};

const WIDTH: u32 = 240;
const HEIGHT: u32 = 160;

/// A light square face with a dark edge chunk along its border.
fn scene() -> GeometryScene {
    let corners = [
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, 1.0, 0.0),
    ];
    let light = Vec4::new(0.85, 0.87, 0.9, 1.0);
    let face = GeometrySceneChunk::indexed_triangles(
        "faces",
        corners.iter().map(|&p| Vertex::new(p, light)).collect(),
        vec![0, 1, 2, 0, 2, 3],
        Material::default(),
    );
    let dark = Vec4::new(0.02, 0.02, 0.02, 1.0);
    let mut data = face.render_data.clone();
    data.pipeline_type = PipelineType::Lines;
    data.vertices = corners.iter().map(|&p| Vertex::new(p, dark)).collect();
    data.indices = Some(vec![0, 1, 1, 2, 2, 3, 3, 0]);
    data.draw_calls[0].vertex_count = 4;
    data.draw_calls[0].index_count = Some(8);
    let edges = GeometrySceneChunk::from_render_data("body:edges", data);
    GeometryScene::new("edge-weight", 1, vec![face, edges])
}

/// The drawn width of the square's vertical edges: per row through the middle of the view, the
/// widest run of dark pixels, and the median of those (rows on grid lines or tick labels are
/// outliers either way).
fn edge_thickness(rgba: &[u8]) -> usize {
    let dark = |x: u32, y: u32| {
        let i = ((y * WIDTH + x) * 4) as usize;
        rgba[i] < 140 && rgba[i + 1] < 140 && rgba[i + 2] < 140
    };
    let mut widest: Vec<usize> = ((HEIGHT / 3)..(2 * HEIGHT / 3))
        .map(|y| {
            let (mut run, mut best) = (0, 0);
            for x in 0..WIDTH {
                run = if dark(x, y) { run + 1 } else { 0 };
                best = best.max(run);
            }
            best
        })
        .collect();
    widest.sort_unstable();
    widest[widest.len() / 2]
}

async fn render(edge_width_px: Option<f32>, pixels_per_point: f32) -> Option<Vec<u8>> {
    let mut context = match create_headless_context(WIDTH, HEIGHT).await {
        Ok(context) => context,
        Err(error) if is_headless_gpu_unavailable_error(&error) => return None,
        Err(error) => panic!("{error}"),
    };
    context.set_pixels_per_point(pixels_per_point);
    context.set_theme_config(PlotThemeConfig {
        variant: ThemeVariant::ClassicLight,
        ..Default::default()
    });
    let presentation = GeometryScenePresentation {
        edge_width_px,
        ..Default::default()
    };
    // Straight down onto the square, orthographic, so its edges are rows and columns.
    let mut camera = Camera::new();
    camera.position = Vec3::new(0.0, 0.0, 5.0);
    camera.target = Vec3::ZERO;
    camera.up = Vec3::Y;
    camera.axis_scale = Vec3::ONE;
    camera.aspect_ratio = WIDTH as f32 / HEIGHT as f32;
    camera.projection = ProjectionType::Orthographic {
        left: -1.5,
        right: 1.5,
        bottom: -1.0,
        top: 1.0,
        near: 0.1,
        far: 10.0,
    };
    camera.mark_dirty();
    let rgba = context
        .render_geometry_scene_to_rgba_with_presentation(&scene(), Some(&camera), Some(&presentation))
        .await
        .unwrap();
    if let Ok(directory) = std::env::var("RUNMAT_EDGE_STROKE_IMAGES") {
        let png = runmat_plot::export::cpu_surface::encode_png_bytes(WIDTH, HEIGHT, &rgba).unwrap();
        let name = format!("{directory}/edges-{:?}-{pixels_per_point}x.png", edge_width_px);
        std::fs::write(name, png).unwrap();
    }
    Some(rgba)
}

#[tokio::test]
async fn edges_draw_at_their_css_pixel_width_on_the_gpu() {
    if std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to exercise the real renderer");
        return;
    }
    let Some(hairline) = render(None, 1.0).await else {
        eprintln!("no headless GPU adapter");
        return;
    };
    let hairline = edge_thickness(&hairline);
    let wide = edge_thickness(&render(Some(3.0), 1.0).await.unwrap());
    let wide_2x = edge_thickness(&render(Some(3.0), 2.0).await.unwrap());
    assert!(
        (1..=2).contains(&hairline),
        "GPU hairline edges are one device pixel: {hairline}"
    );
    assert!(
        (3..=5).contains(&wide),
        "a 3 px edge at 1x is about 3 device px: {wide}"
    );
    assert!(
        (6..=8).contains(&wide_2x),
        "a 3 px edge at 2x is about 6 device px: {wide_2x}"
    );
}
