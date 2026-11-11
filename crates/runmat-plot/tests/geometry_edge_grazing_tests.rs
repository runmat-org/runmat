//! A fuselage's frames and stringers stay under its skin where the skin curves away at a
//! grazing angle near its silhouette, at an overview zoom (about 1.5 cm per device pixel), while
//! the same edges laid on the skin's outer surface still draw.
use glam::{Vec3, Vec4};
use runmat_plot::{
    core::{Camera, ProjectionType},
    core::{Material, PipelineType, Vertex},
    export::native_surface::{create_headless_context, is_headless_gpu_unavailable_error},
    geometry_scene::GeometryScenePresentation,
    styling::{PlotThemeConfig, ThemeVariant},
    GeometryScene, GeometrySceneChunk,
};

const WIDTH: u32 = 480;
const HEIGHT: u32 = 320;
/// The skin's outer radius; the airliner's barrel is 1980 mm with a 3 mm skin.
const SKIN_R: f32 = 1.98;
/// Segments around the barrel: a 0.6 mm chord sag on the skin, under the 1 mm display chord.
const AROUND: usize = 128;
const SKIN: Vec4 = Vec4::new(0.85, 0.87, 0.9, 1.0);
/// The skin in red, to find its outline.
const RED: Vec4 = Vec4::new(0.9, 0.1, 0.1, 1.0);

/// A barrel skin along x (open, longer than the view) and edges at `edge_r`: frame rings at every
/// 0.5 m station and a line at every 15° stringer. With no `edge_r`, the skin alone.
fn scene(edge_r: Option<f32>, skin: Vec4) -> GeometryScene {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for i in 0..AROUND {
        let a = i as f32 / AROUND as f32 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        for x in [-30.0, 30.0] {
            let mut v = Vertex::new(Vec3::new(x, SKIN_R * c, SKIN_R * s), skin);
            v.normal = [0.0, c, s];
            vertices.push(v);
        }
        let (a0, b0) = (2 * i as u32, 2 * i as u32 + 1);
        let (a1, b1) = (
            2 * ((i + 1) % AROUND) as u32,
            2 * ((i + 1) % AROUND) as u32 + 1,
        );
        indices.extend_from_slice(&[a0, a1, b1, a0, b1, b0]);
    }
    let face =
        GeometrySceneChunk::indexed_triangles("faces", vertices, indices, Material::default());
    let Some(edge_r) = edge_r else {
        return GeometryScene::new("edge-grazing", 1, vec![face]);
    };

    let dark = Vec4::new(0.02, 0.02, 0.02, 1.0);
    let mut lines = Vec::new();
    let mut line_indices = Vec::new();
    let mut segment = |a: Vec3, b: Vec3| {
        line_indices.extend_from_slice(&[lines.len() as u32, lines.len() as u32 + 1]);
        lines.push(Vertex::new(a, dark));
        lines.push(Vertex::new(b, dark));
    };
    let ring = |x: f32, r: f32, i: usize| {
        let a = i as f32 / AROUND as f32 * std::f32::consts::TAU;
        Vec3::new(x, r * a.cos(), r * a.sin())
    };
    // Each frame's outer edges at `edge_r` and its inner flange's 147 mm further in.
    for station in -16..=16 {
        for (x, r) in [
            (0.0, edge_r),
            (0.04, edge_r),
            (0.0, edge_r - 0.147),
            (0.04, edge_r - 0.147),
        ] {
            let x = station as f32 * 0.5 + x;
            for i in 0..AROUND {
                segment(ring(x, r, i), ring(x, r, i + 1));
            }
        }
    }
    for k in 0..24 {
        let a = k as f32 * 15f32.to_radians();
        segment(
            Vec3::new(-8.0, edge_r * a.cos(), edge_r * a.sin()),
            Vec3::new(8.0, edge_r * a.cos(), edge_r * a.sin()),
        );
    }
    let mut data = face.render_data.clone();
    data.pipeline_type = PipelineType::Lines;
    data.draw_calls[0].vertex_count = lines.len();
    data.draw_calls[0].index_count = Some(line_indices.len());
    data.vertices = lines;
    data.indices = Some(line_indices);
    let edges = GeometrySceneChunk::from_render_data("body:edges", data);
    GeometryScene::new("edge-grazing", 1, vec![face, edges])
}

/// Pixels the edges change: those that differ from the skin drawn alone by more than a quarter
/// of the range in any channel. (A fixed darkness threshold also counts the skin's own shading,
/// which falls to about 100 of 255 on the barrel's lower side.)
fn edge_pixels(rgba: &[u8], skin_only: &[u8]) -> Vec<bool> {
    rgba.chunks_exact(4)
        .zip(skin_only.chunks_exact(4))
        .map(|(p, q)| (0..3).any(|k| p[k].abs_diff(q[k]) > 64))
        .collect()
}

/// Pixels at least `margin` pixels inside the skin's outline, from the skin drawn alone in red.
/// Along the outline an edge 3 mm under the skin lies within 0.2 px of it and meets the
/// background, where the probe keeps edges so that silhouettes stay drawn; this test is about
/// the skin's face.
fn inside_skin(red_skin: &[u8], margin: i32) -> Vec<bool> {
    let skin: Vec<bool> = red_skin
        .chunks_exact(4)
        .map(|p| p[0] as i32 > p[1] as i32 + 60)
        .collect();
    let (w, h) = (WIDTH as i32, HEIGHT as i32);
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (-margin..=margin).all(|dy| {
                (-margin..=margin).all(|dx| {
                    let (u, v) = (x + dx, y + dy);
                    u < 0 || v < 0 || u >= w || v >= h || skin[(v * w + u) as usize]
                })
            })
        })
        .collect()
}

/// The barrel seen from `elevation_deg` above (negative: below) its axis and `azimuth_deg` off
/// it, in the Desktop's parallel 3D view, 60 m away and 4.8 m high (1.5 cm per device pixel),
/// with the Desktop's 1.25 px edges at 2×.
async fn render(
    edge_r: Option<f32>,
    skin: Vec4,
    azimuth_deg: f32,
    elevation_deg: f32,
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
    let (az, el) = (azimuth_deg.to_radians(), elevation_deg.to_radians());
    let direction = Vec3::new(az.cos() * el.cos(), az.sin() * el.cos(), el.sin());
    let mut camera = Camera::new();
    camera.position = direction * 60.0;
    camera.target = Vec3::ZERO;
    camera.up = Vec3::Z;
    camera.axis_scale = Vec3::ONE;
    camera.aspect_ratio = WIDTH as f32 / HEIGHT as f32;
    camera.parallel = true;
    camera.projection = ProjectionType::Perspective {
        fov: 2.0 * (2.4_f32 / 60.0).atan(),
        near: 1.0,
        far: 200.0,
    };
    camera.mark_dirty();
    let rgba = context
        .render_geometry_scene_to_rgba_with_presentation(
            &scene(edge_r, skin),
            Some(&camera),
            Some(&presentation),
        )
        .await
        .unwrap();
    if let Ok(directory) = std::env::var("RUNMAT_EDGE_STROKE_IMAGES") {
        let png = runmat_plot::export::cpu_surface::encode_png_bytes(WIDTH, HEIGHT, &rgba).unwrap();
        let edges = match edge_r {
            Some(r) => format!("r{r}"),
            None if skin == RED => "red".to_string(),
            None => "none".to_string(),
        };
        let name =
            format!("{directory}/edge-grazing-{edges}-az{azimuth_deg}-el{elevation_deg}.png");
        std::fs::write(name, png).unwrap();
    }
    Some(rgba)
}

#[tokio::test]
async fn frames_and_stringers_stay_under_a_curved_skin_at_a_grazing_view() {
    if std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to exercise the real renderer");
        return;
    }
    // From the side and above, from below, and nearly along the axis, so the crown, the belly
    // and the sides each pass the silhouette at a grazing angle.
    for (azimuth, elevation) in [(60.0, 20.0), (60.0, -20.0), (25.0, 5.0)] {
        // 3 mm under the skin (on its inside face), as the airliner's frames and stringers are.
        let Some(skin_only) = render(None, SKIN, azimuth, elevation).await else {
            eprintln!("no headless GPU adapter");
            return;
        };
        let inside = inside_skin(&render(None, RED, azimuth, elevation).await.unwrap(), 3);
        let under = render(Some(SKIN_R - 0.003), SKIN, azimuth, elevation)
            .await
            .unwrap();
        // 3 mm over it: these edges are on the outer surface and must draw.
        let over = render(Some(SKIN_R + 0.003), SKIN, azimuth, elevation)
            .await
            .unwrap();
        let under = edge_pixels(&under, &skin_only);
        let outline = under
            .iter()
            .zip(&inside)
            .filter(|(e, i)| **e && !**i)
            .count();
        let under = under
            .iter()
            .zip(&inside)
            .filter(|(e, i)| **e && **i)
            .count();
        let over = edge_pixels(&over, &skin_only)
            .iter()
            .filter(|e| **e)
            .count();
        eprintln!(
            "az {azimuth} el {elevation}: edge pixels under the skin {under} (and {outline} on its outline), over it {over}"
        );
        assert!(
            over > 5000,
            "edges on the skin draw at az {azimuth} el {elevation}: {over} edge pixels"
        );
        assert!(
            under < 40,
            "edges 3 mm under the skin stay hidden at az {azimuth} el {elevation}: {under} edge pixels"
        );
    }
}
