//! Translucent context geometry around an opaque result in one geometry scene.
//!
//! A result (opaque) sits among context geometry (translucent, from `into_context`). Each scene
//! is saved and decoded through the persisted figure-scene form first, the way snapshot posters
//! and the live viewer load it. Set `RUNMAT_PLOT_CONTEXT_OUT=<dir>` to keep the PNGs.

use glam::{Vec3, Vec4};
use runmat_plot::core::{AlphaMode, Camera};
use runmat_plot::export::image::{ImageExportSettings, ImageExporter};
use runmat_plot::{
    cad_default_material, geometry_scene_vertex, FigureScene, GeometryScene, GeometrySceneChunk,
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
    let exporter = ImageExporter::with_settings(ImageExportSettings {
        width: SIZE,
        height: SIZE,
        ..Default::default()
    })
    .await
    .unwrap();
    let mut camera = Camera::new();
    camera.update_aspect_ratio(1.0);
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

/// The colour at the image centre.
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
        |chunk: &&GeometrySceneChunk| chunk.render_data.material.alpha_mode == AlphaMode::Blend;
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
