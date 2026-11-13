//! An edge just under a surface stays hidden across its whole stroke width on the camera
//! viewport path (the Desktop's and the headless export's), as a fuselage's frames and
//! stringers stay under its skin at a grazing view.
use glam::{Vec3, Vec4};
use runmat_plot::{
    core::{Camera, ProjectionType},
    core::{Material, PipelineType, Vertex},
    export::native_surface::{create_headless_context, is_headless_gpu_unavailable_error},
    geometry_scene::GeometryScenePresentation,
    styling::{PlotThemeConfig, ThemeVariant},
    GeometryScene, GeometrySceneChunk,
};

const WIDTH: u32 = 240;
const HEIGHT: u32 = 160;

/// A light square face on z = 0 and a dark edge along x at height `edge_z` over its middle.
fn scene(edge_z: f32) -> GeometryScene {
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
    data.vertices = [-0.6, 0.6]
        .iter()
        .map(|&x| Vertex::new(Vec3::new(x, 0.0, edge_z), dark))
        .collect();
    data.indices = Some(vec![0, 1]);
    data.draw_calls[0].vertex_count = 2;
    data.draw_calls[0].index_count = Some(2);
    let edges = GeometrySceneChunk::from_render_data("body:edges", data);
    GeometryScene::new("edge-depth", 1, vec![face, edges])
}

/// Dark pixels in the middle third of the view, where the line lies.
fn dark_pixels(rgba: &[u8]) -> usize {
    let mut n = 0;
    for y in HEIGHT / 3..2 * HEIGHT / 3 {
        for x in WIDTH / 3..2 * WIDTH / 3 {
            let i = ((y * WIDTH + x) * 4) as usize;
            n += usize::from(rgba[i] < 120 && rgba[i + 1] < 120 && rgba[i + 2] < 120);
        }
    }
    n
}

/// The scene seen 5° above the face's plane, nearly parallel, 3 px edges at 2×: one device pixel
/// across the line spans 0.036 of depth on the face, so the stroke's six pixels reach 0.1 in
/// front of and behind the line's own depth.
async fn render(edge_z: f32) -> Option<Vec<u8>> {
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
        edge_width_px: Some(3.0),
        ..Default::default()
    };
    let angle = 5.0_f32.to_radians();
    let mut camera = Camera::new();
    camera.position = Vec3::new(0.0, -5.0 * angle.cos(), 5.0 * angle.sin());
    camera.target = Vec3::ZERO;
    camera.up = Vec3::Z;
    camera.axis_scale = Vec3::ONE;
    camera.aspect_ratio = WIDTH as f32 / HEIGHT as f32;
    // A narrow perspective (the Desktop's 3D view; an orthographic camera draws as a 2D plot):
    // 0.5 high at the face's centre.
    camera.projection = ProjectionType::Perspective {
        fov: 2.0 * 0.05_f32.atan(),
        near: 0.1,
        far: 10.0,
    };
    camera.mark_dirty();
    let rgba = context
        .render_geometry_scene_to_rgba_with_presentation(
            &scene(edge_z),
            Some(&camera),
            Some(&presentation),
        )
        .await
        .unwrap();
    if let Ok(directory) = std::env::var("RUNMAT_EDGE_STROKE_IMAGES") {
        let png = runmat_plot::export::cpu_surface::encode_png_bytes(WIDTH, HEIGHT, &rgba).unwrap();
        std::fs::write(format!("{directory}/edge-depth-{edge_z}.png"), png).unwrap();
    }
    Some(rgba)
}

#[tokio::test]
async fn an_edge_under_a_surface_stays_hidden_across_its_stroke_at_a_grazing_view() {
    if std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to exercise the real renderer");
        return;
    }
    let Some(under) = render(-0.005).await else {
        eprintln!("no headless GPU adapter");
        return;
    };
    let over = render(0.005).await.unwrap();
    let (under, over) = (dark_pixels(&under), dark_pixels(&over));
    eprintln!("dark pixels: edge under the face {under}, over it {over}");
    assert!(
        over > 100,
        "the edge over the face draws: {over} dark pixels"
    );
    assert!(
        under < 10,
        "the edge 5 mm under the face stays hidden: {under} dark pixels"
    );
}
