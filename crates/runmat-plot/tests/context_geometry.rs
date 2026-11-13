//! Translucent context geometry around an opaque result in one geometry scene.
//!
//! A result (opaque) sits among context geometry (translucent, from `into_context`). Each scene
//! is saved and decoded through the persisted figure-scene form first, the way snapshot posters
//! and the live viewer load it. Set `RUNMAT_PLOT_CONTEXT_OUT=<dir>` to keep the PNGs.

use glam::{Vec3, Vec4};
use runmat_plot::core::{AlphaMode, Camera};
use runmat_plot::export::image::{ImageExportSettings, ImageExporter};
use runmat_plot::geometry_scene::GeometrySceneScalarLegend;
use runmat_plot::{
    cad_default_material, geometry_scene_vertex, FigureScene, GeometryScene, GeometrySceneChunk,
    GeometrySceneCompleteness, GeometrySceneOverlay,
};

const SIZE: u32 = 200;
const RESULT: [f32; 4] = [0.9, 0.1, 0.1, 1.0];
const GHOST: [f32; 4] = [0.7, 0.8, 0.9, 0.3];
const GHOST_EDGE: [f32; 4] = [0.6, 0.7, 0.9, 0.4];

/// A square in the plane z = `z`, facing the camera on +z.
fn square(id: &str, z: f32, half: f32, color: [f32; 4]) -> GeometrySceneChunk {
    let corners = [(-half, -half), (half, -half), (half, half), (-half, half)];
    let vertices = corners
        .iter()
        .map(|&(x, y)| geometry_scene_vertex([x, y, z], color, [0.0, 0.0, 1.0]))
        .collect();
    let mut material = cad_default_material();
    material.albedo = Vec4::from_array(color);
    GeometrySceneChunk::indexed_triangles(id, vertices, vec![0, 1, 2, 0, 2, 3], material)
}

fn ghost(id: &str, z: f32, half: f32) -> GeometrySceneChunk {
    square(id, z, half, [0.66, 0.72, 0.80, 1.0]).into_context(GHOST, GHOST_EDGE)
}

/// Save and decode the scene the way snapshot posters do.
fn saved(chunks: Vec<GeometrySceneChunk>) -> GeometryScene {
    let mut scene = GeometryScene::new("context", 1, chunks);
    scene.show_grid = false;
    FigureScene::from_geometry_scene(&scene)
        .into_geometry_scene("context", 1)
        .unwrap()
}

async fn render(name: &str, scene: &GeometryScene) -> Vec<u8> {
    render_at(name, scene, SIZE, SIZE).await
}

async fn render_at(name: &str, scene: &GeometryScene, width: u32, height: u32) -> Vec<u8> {
    let exporter = ImageExporter::with_settings(ImageExportSettings {
        width,
        height,
        ..Default::default()
    })
    .await
    .unwrap();
    let mut camera = Camera::new();
    camera.update_aspect_ratio(width as f32 / height as f32);
    camera.target = Vec3::ZERO;
    camera.up = Vec3::Y;
    camera.position = Vec3::new(0.0, 0.0, 6.0);
    camera.fit_bounds(Vec3::new(-1.0, -1.0, -2.0), Vec3::new(1.0, 1.0, 2.0));
    let png = exporter
        .render_geometry_scene_png_bytes_with_camera(scene, &camera)
        .await
        .unwrap();
    if let Ok(dir) = std::env::var("RUNMAT_PLOT_CONTEXT_OUT") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.png")), &png).unwrap();
    }
    image::load_from_memory(&png).unwrap().to_rgba8().into_raw()
}

/// The color at the image center.
fn centre(rgba: &[u8]) -> [i32; 3] {
    let index = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
    [0, 1, 2].map(|channel| rgba[index + channel] as i32)
}

#[test]
fn context_chunks_are_blended_and_survive_the_saved_scene() {
    let scene = saved(vec![
        square("result", 0.0, 0.5, RESULT),
        ghost("context", 1.0, 0.8),
    ]);
    let surfaces: Vec<_> = scene
        .chunks
        .iter()
        .filter(|chunk| chunk.indices.as_ref().is_some_and(|i| !i.is_empty()))
        .filter(|chunk| {
            chunk.render_data.pipeline_type == runmat_plot::core::PipelineType::Triangles
        })
        .collect();
    assert_eq!(surfaces.len(), 2);
    let blended =
        |chunk: &&&GeometrySceneChunk| chunk.render_data.material.alpha_mode == AlphaMode::Blend;
    assert_eq!(
        surfaces.iter().filter(blended).count(),
        1,
        "only the context blends"
    );
    let context = surfaces.iter().find(blended).unwrap();
    assert!((context.render_data.material.albedo.w - GHOST[3]).abs() < 1e-3);
}

#[tokio::test]
async fn context_in_front_tints_the_result_without_hiding_it() {
    let alone = centre(
        &render(
            "result-alone",
            &saved(vec![square("result", 0.0, 0.5, RESULT)]),
        )
        .await,
    );
    let ghost_only = centre(&render("ghost-alone", &saved(vec![ghost("context", 1.0, 0.8)])).await);
    let both = centre(
        &render(
            "ghost-in-front",
            &saved(vec![
                square("result", 0.0, 0.5, RESULT),
                ghost("context", 1.0, 0.8),
            ]),
        )
        .await,
    );
    // The result shows through: far redder than the ghost over the background.
    assert!(
        both[0] - both[1] > (alone[0] - alone[1]) / 2,
        "result hidden behind context: alone {alone:?}, ghost {ghost_only:?}, both {both:?}"
    );
    // And the ghost tints it: the frame is not the bare result.
    assert!(
        (both[1] - alone[1]).abs() + (both[2] - alone[2]).abs() > 20,
        "context not composited over the result: alone {alone:?}, both {both:?}"
    );
}

#[tokio::test]
async fn context_behind_the_result_leaves_it_untouched() {
    let alone = centre(
        &render(
            "result-alone-2",
            &saved(vec![square("result", 0.0, 0.5, RESULT)]),
        )
        .await,
    );
    let behind = centre(
        &render(
            "ghost-behind",
            &saved(vec![
                ghost("context", -1.0, 0.8),
                square("result", 0.0, 0.5, RESULT),
            ]),
        )
        .await,
    );
    let difference: i32 = (0..3).map(|c| (behind[c] - alone[c]).abs()).sum();
    assert!(
        difference <= 6,
        "context behind tinted the result: alone {alone:?}, behind {behind:?}"
    );
}

#[tokio::test]
async fn stacked_context_layers_all_composite() {
    let one = centre(&render("ghost-one", &saved(vec![ghost("near", 1.0, 0.8)])).await);
    // Whatever order the two layers draw in, the nearer must not hide the farther.
    let two = centre(
        &render(
            "ghost-two",
            &saved(vec![ghost("near", 1.0, 0.8), ghost("far", 0.5, 0.8)]),
        )
        .await,
    );
    let gain: i32 = (0..3).map(|c| two[c] - one[c]).sum();
    assert!(
        gain > 15,
        "the farther context layer was hidden: one {one:?}, two {two:?}"
    );
}

#[tokio::test]
async fn context_touching_a_result_face_does_not_tint_it() {
    // Context in the result's own plane (a barrier pressed against a cell face): the
    // result wins the tie everywhere instead of z-fighting with it.
    let alone = centre(
        &render(
            "result-alone-3",
            &saved(vec![square("result", 0.0, 0.5, RESULT)]),
        )
        .await,
    );
    let touching = centre(
        &render(
            "ghost-coplanar",
            &saved(vec![
                square("result", 0.0, 0.5, RESULT),
                ghost("context", 0.0, 0.8),
            ]),
        )
        .await,
    );
    let difference: i32 = (0..3).map(|c| (touching[c] - alone[c]).abs()).sum();
    assert!(
        difference <= 6,
        "coplanar context tinted the result: alone {alone:?}, touching {touching:?}"
    );
}

#[test]
fn merged_context_draws_every_piece_as_one_chunk() {
    let merged =
        GeometrySceneChunk::merged("context", vec![ghost("a", 1.0, 0.8), ghost("b", 0.5, 0.4)])
            .unwrap();
    assert_eq!(merged.vertices.len(), 8);
    let indices = merged.indices.as_ref().unwrap();
    assert_eq!(indices.len(), 12);
    assert_eq!(&indices[6..], &[4, 5, 6, 4, 6, 7]);
    assert_eq!(merged.render_data.material.alpha_mode, AlphaMode::Blend);
    assert_eq!(merged.triangle_count(), 4);
    assert!((merged.bounds.max.z - 1.0).abs() < 1e-6 && (merged.bounds.min.z - 0.5).abs() < 1e-6);
    // Lines and triangles never merge.
    let edges = square("e", 0.0, 1.0, RESULT)
        .wireframe_edges(GHOST_EDGE)
        .unwrap();
    assert!(GeometrySceneChunk::merged("mixed", vec![ghost("a", 1.0, 0.8), edges]).is_none());
}

fn result_overlay() -> GeometrySceneOverlay {
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
            minimum: 301.0,
            maximum: 304.0,
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

/// Pixels in the top band, left of the legend, that differ from the background.
fn title_band_ink(rgba: &[u8]) -> usize {
    let background = &rgba[((SIZE * 4) * (SIZE - 2) + 8) as usize..][..3];
    (0..SIZE / 8)
        .flat_map(|y| (0..SIZE / 2).map(move |x| ((y * SIZE + x) * 4) as usize))
        .filter(|&i| {
            (0..3)
                .map(|c| (rgba[i + c] as i32 - background[c] as i32).abs())
                .sum::<i32>()
                > 60
        })
        .count()
}

#[tokio::test]
async fn a_result_scene_draws_its_caption_as_title() {
    let mut captioned = saved(vec![square("result", 0.0, 0.5, RESULT)]);
    captioned.overlay = Some(result_overlay());
    let mut bare = captioned.clone();
    captioned.title = Some("Steady thermal".into());
    let with = title_band_ink(&render("captioned", &captioned).await);
    let without = title_band_ink(&render("uncaptioned", &bare).await);
    assert!(
        with > without + 20,
        "caption not drawn: {with} vs {without} ink pixels"
    );
    // A CAD preview's title is its file name: still not drawn without a result legend.
    bare.title = Some("pack.step".into());
    if let Some(overlay) = bare.overlay.as_mut() {
        overlay.scalar_legend = None;
    }
    let cad = title_band_ink(&render("cad-title", &bare).await);
    assert!(
        cad <= without + 5,
        "CAD source title drawn: {cad} vs {without}"
    );
}

/// Rows (from the top) holding caption ink: pixels across the middle third of the width that
/// differ from the background, above the result.
fn caption_rows(rgba: &[u8], width: u32, height: u32) -> Vec<u32> {
    let background = &rgba[(((height - 2) * width + 4) * 4) as usize..][..3];
    (0..height / 6)
        .filter(|&y| {
            (width / 3..2 * width / 3).any(|x| {
                let i = ((y * width + x) * 4) as usize;
                (0..3)
                    .map(|c| (rgba[i + c] as i32 - background[c] as i32).abs())
                    .sum::<i32>()
                    > 90
            })
        })
        .collect()
}

#[tokio::test]
async fn a_result_caption_is_laid_out_inside_the_frame() {
    let (width, height) = (640, 400);
    let mut scene = saved(vec![square("result", 0.0, 0.5, RESULT)]);
    scene.overlay = Some(result_overlay());
    scene.title = Some("Steady thermal on the cells".into());
    let rows = caption_rows(
        &render_at("caption-inside", &scene, width, height).await,
        width,
        height,
    );
    assert!(rows.len() >= 6, "caption not drawn: rows {rows:?}");
    // The whole caption sits below the top edge: its glyphs are not cut off.
    assert!(
        rows[0] >= 2,
        "caption clipped at the top edge: rows {rows:?}"
    );
}

/// A grey square below the world origin, so the origin triad (at z = 0, nearer the camera on
/// +z) would be drawn over it.
fn grey_scene(result: bool, show_grid: bool, show_triad: bool) -> GeometryScene {
    let mut scene = GeometryScene::new(
        "triad",
        1,
        vec![square("plate", -0.5, 0.5, [0.45, 0.45, 0.45, 1.0])],
    );
    scene.show_grid = show_grid;
    scene.show_orientation_triad = show_triad;
    if result {
        scene.overlay = Some(result_overlay());
    }
    FigureScene::from_geometry_scene(&scene)
        .into_geometry_scene("triad", 1)
        .unwrap()
}

/// Pixels whose color is dominated by red or green, the triad's x and y axis colors, in a
/// region given as fractions of the frame (x0, y0, x1, y1).
fn triad_ink(rgba: &[u8], region: [f32; 4]) -> usize {
    let at = |fraction: f32| (fraction * SIZE as f32) as u32;
    (at(region[1])..at(region[3]))
        .flat_map(|y| (at(region[0])..at(region[2])).map(move |x| ((y * SIZE + x) * 4) as usize))
        .filter(|&i| {
            let [r, g, b] = [0, 1, 2].map(|c| rgba[i + c] as i32);
            (r > g + 70 && r > b + 70) || (g > r + 70 && g > b + 70)
        })
        .count()
}

#[tokio::test]
async fn result_scenes_never_draw_the_world_triad() {
    // Away from the corner triad: the right part of the frame below the legend.
    let middle = [0.35, 0.45, 1.0, 1.0];
    let cad = triad_ink(
        &render("triad-cad", &grey_scene(false, true, false)).await,
        middle,
    );
    let result = triad_ink(
        &render("triad-result", &grey_scene(true, true, false)).await,
        middle,
    );
    assert!(
        cad > 20,
        "a CAD preview draws its world triad: {cad} pixels"
    );
    assert!(
        result <= 2,
        "a result scene drew the world triad: {result} pixels"
    );
}

#[tokio::test]
async fn the_corner_triad_and_grid_are_scene_options() {
    let corner = [0.0, 0.6, 0.4, 1.0];
    let on = triad_ink(
        &render("corner-on", &grey_scene(true, false, true)).await,
        corner,
    );
    let off = triad_ink(
        &render("corner-off", &grey_scene(true, false, false)).await,
        corner,
    );
    assert!(on > 20, "corner triad missing: {on} pixels");
    assert!(off <= 2, "corner triad drawn while off: {off} pixels");
    // The options survive the saved scene.
    let saved = grey_scene(true, false, false);
    assert!(!saved.show_grid && !saved.show_orientation_triad);
    assert!(grey_scene(true, true, true).show_orientation_triad);
}

/// With every item hidden the scene has no chunks; it still loads, and draws its grid and
/// corner triad.
#[tokio::test]
async fn an_empty_scene_keeps_its_grid_and_triad() {
    let mut scene = GeometryScene::new("empty", 1, Vec::new());
    scene.show_grid = true;
    scene.show_orientation_triad = true;
    let scene = FigureScene::from_geometry_scene(&scene)
        .into_geometry_scene("empty", 1)
        .expect("an empty scene loads");
    assert!(scene.chunks.is_empty() && scene.show_grid && scene.show_orientation_triad);
    let rgba = render("empty", &scene).await;
    assert!(
        triad_ink(&rgba, [0.0, 0.6, 0.4, 1.0]) > 20,
        "corner triad missing"
    );
}
