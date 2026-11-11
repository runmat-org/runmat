//! Free helpers: view preset mapping, luminance, x-ray detection and pale-line darkening.

use super::*;

pub(super) fn geometry_scene_view_preset_to_camera(
    preset: GeometrySceneViewPreset,
) -> CameraViewPreset {
    match preset {
        GeometrySceneViewPreset::Perspective | GeometrySceneViewPreset::Isometric => {
            CameraViewPreset::Perspective
        }
        GeometrySceneViewPreset::Front => CameraViewPreset::Front,
        GeometrySceneViewPreset::Back => CameraViewPreset::Back,
        GeometrySceneViewPreset::Left => CameraViewPreset::Left,
        GeometrySceneViewPreset::Right => CameraViewPreset::Right,
        GeometrySceneViewPreset::Top => CameraViewPreset::Top,
        GeometrySceneViewPreset::Bottom => CameraViewPreset::Bottom,
    }
}

pub(super) fn relative_luminance(color: Vec4) -> f32 {
    0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z
}

/// A scene opens in x-ray when every one of its surfaces is see-through (a scene authored as
/// x-ray). One translucent part among opaque ones, such as glass in an assembly or a command
/// preview, draws blended on its own and leaves the rest of the model solid.
pub(super) fn scene_authored_as_xray(geometry_scene: &GeometryScene) -> bool {
    let mut surfaces = geometry_scene
        .chunks
        .iter()
        .filter(|chunk| chunk.render_data.pipeline_type == crate::core::PipelineType::Triangles)
        .peekable();
    surfaces.peek().is_some()
        && surfaces.all(|chunk| {
            chunk.material.albedo.w < 0.95
                || matches!(chunk.material.alpha_mode, crate::core::AlphaMode::Blend)
        })
}

/// Scene line colors are stored with the scene, independent of theme. Near-white lines (such
/// as the undeformed outline) read well on a dark background but vanish on a light one, so on
/// a light background they are darkened to a slate of the same hue and made a little more opaque.
pub(super) fn darken_pale_lines_for_light_background(render_data: &mut crate::core::RenderData) {
    const PALE: f32 = 0.8;
    const TARGET: f32 = 0.38;
    // GPU-resident vertices can't be recolored here; scene lines are CPU data.
    if render_data.gpu_vertices.is_some()
        || !matches!(
            render_data.pipeline_type,
            crate::core::PipelineType::Lines
                | crate::core::PipelineType::LinesNoDepth
                | crate::core::PipelineType::EdgeStroke
                | crate::core::PipelineType::EdgeStrokeNoDepth
        )
    {
        return;
    }
    for vertex in &mut render_data.vertices {
        let color = Vec4::from_array(vertex.color);
        let luminance = relative_luminance(color);
        if luminance <= PALE || color.w <= 0.0 {
            continue;
        }
        let scale = TARGET / luminance;
        vertex.color = [
            color.x * scale,
            color.y * scale,
            color.z * scale,
            color.w.max(0.5),
        ];
    }
}
