//! Saved geometry image export through the shared plot renderer.
use super::super::web;
use super::{
    engine_error, engine_error_with_source, surface_plot_camera_to_core_camera,
    PlottingBackendError,
};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::plotting::clone_geometry_scene;
use crate::BuiltinResult;

#[cfg(feature = "plot-core")]
pub async fn render_geometry_scene_snapshot(
    handle: u32,
    width: u32,
    height: u32,
    view: Option<String>,
) -> BuiltinResult<Vec<u8>> {
    render_geometry_scene_snapshot_inner(handle, width, height, view, None).await
}

/// Apply the same read-only observation used by native hosts before camera fitting.
#[cfg(feature = "plot-core")]
pub async fn render_geometry_scene_observation_snapshot(
    handle: u32,
    width: u32,
    height: u32,
    view: Option<String>,
    visibility: Option<serde_json::Value>,
    section: Option<serde_json::Value>,
    selected_region_ids: Vec<String>,
) -> BuiltinResult<Vec<u8>> {
    let source = clone_geometry_scene(handle)
        .ok_or_else(|| engine_error(format!("geometry scene handle {handle} does not exist")))?;
    let scene = runmat_plot::geometry_observation::observation_scene(
        &source,
        visibility,
        section,
        selected_region_ids,
    )
    .map_err(engine_error)?;
    render_geometry_observation_inner(scene, width, height, view, None).await
}

#[cfg(feature = "plot-core")]
pub fn measure_geometry_scene_observation(
    handle: u32,
    kind: &str,
    region_ids: &[String],
) -> BuiltinResult<serde_json::Value> {
    let scene = clone_geometry_scene(handle)
        .ok_or_else(|| engine_error(format!("geometry scene handle {handle} does not exist")))?;
    runmat_plot::geometry_observation::measure_scene(&scene, kind, region_ids).map_err(engine_error)
}

/// Render saved geometry with the same camera used by interactive playback.
#[cfg(feature = "plot-core")]
pub async fn render_geometry_scene_snapshot_with_camera_state(
    handle: u32,
    width: u32,
    height: u32,
    camera: web::PlotSurfaceCameraState,
) -> BuiltinResult<Vec<u8>> {
    render_geometry_scene_snapshot_with_camera_state_scaled(
        handle, width, height, camera, 1.0, None,
    )
    .await
}

/// As [`render_geometry_scene_snapshot_with_camera_state`], with titles, legends and other text
/// drawn at `pixels_per_point` device pixels per layout point, so a poster shown scaled down
/// keeps readable labels. `theme` draws the image in that plot theme (background, grid, text,
/// legend and triad) instead of the host's current one.
#[cfg(feature = "plot-core")]
pub async fn render_geometry_scene_snapshot_with_camera_state_scaled(
    handle: u32,
    width: u32,
    height: u32,
    camera: web::PlotSurfaceCameraState,
    pixels_per_point: f32,
    theme: Option<runmat_plot::styling::PlotThemeConfig>,
) -> BuiltinResult<Vec<u8>> {
    const SNAPSHOT_CONTEXT: &str = "renderGeometrySceneImage";
    let scene = clone_geometry_scene(handle).ok_or_else(|| {
        map_control_flow_with_builtin(
            engine_error(format!("geometry scene handle {handle} does not exist")),
            SNAPSHOT_CONTEXT,
        )
    })?;
    render_geometry_observation_scaled(
        scene,
        width,
        height,
        None,
        Some(camera),
        pixels_per_point,
        theme,
    )
    .await
}

#[cfg(feature = "plot-core")]
async fn render_geometry_scene_snapshot_inner(
    handle: u32,
    width: u32,
    height: u32,
    view: Option<String>,
    camera_state: Option<web::PlotSurfaceCameraState>,
) -> BuiltinResult<Vec<u8>> {
    const SNAPSHOT_CONTEXT: &str = "renderGeometrySceneImage";
    log::debug!(
        "runmat-runtime: render_geometry_scene_snapshot.start handle={} width={} height={} view={}",
        handle,
        width,
        height,
        view.as_deref().unwrap_or("")
    );
    let scene = clone_geometry_scene(handle).ok_or_else(|| {
        map_control_flow_with_builtin(
            engine_error(format!("geometry scene handle {handle} does not exist")),
            SNAPSHOT_CONTEXT,
        )
    })?;

    render_geometry_observation_inner(scene, width, height, view, camera_state).await
}

/// Render a request-local observation without changing a registered interactive scene.
#[cfg(feature = "plot-core")]
pub async fn render_geometry_observation_snapshot(
    scene: runmat_plot::GeometryScene,
    width: u32,
    height: u32,
    view: Option<String>,
) -> BuiltinResult<Vec<u8>> {
    render_geometry_observation_inner(scene, width, height, view, None).await
}

#[cfg(feature = "plot-core")]
async fn render_geometry_observation_inner(
    scene: runmat_plot::GeometryScene,
    width: u32,
    height: u32,
    view: Option<String>,
    camera_state: Option<web::PlotSurfaceCameraState>,
) -> BuiltinResult<Vec<u8>> {
    render_geometry_observation_scaled(scene, width, height, view, camera_state, 1.0, None).await
}

#[cfg(feature = "plot-core")]
async fn render_geometry_observation_scaled(
    scene: runmat_plot::GeometryScene,
    width: u32,
    height: u32,
    view: Option<String>,
    camera_state: Option<web::PlotSurfaceCameraState>,
    pixels_per_point: f32,
    theme: Option<runmat_plot::styling::PlotThemeConfig>,
) -> BuiltinResult<Vec<u8>> {
    const SNAPSHOT_CONTEXT: &str = "renderGeometrySceneImage";
    use runmat_plot::export::image::{ImageExportSettings, ImageExporter};
    let mut settings = ImageExportSettings {
        pixels_per_point,
        ..ImageExportSettings::default()
    };
    if width > 0 {
        settings.width = width;
    }
    if height > 0 {
        settings.height = height;
    }
    let camera = match camera_state {
        Some(state) => {
            let axis = state
                .axes
                .get(state.active_axes)
                .ok_or_else(|| engine_error("Saved geometry camera has no active axes."))?;
            surface_plot_camera_to_core_camera(axis)
        }
        None => geometry_scene_snapshot_camera(
            scene.bounds,
            settings.width,
            settings.height,
            parse_camera_view_preset(view.as_deref()),
        ),
    };
    let mut exporter = ImageExporter::with_settings(settings)
        .await
        .map_err(|err| {
            map_control_flow_with_builtin(
                engine_error_with_source(
                    "Geometry export initialization failed.",
                    PlottingBackendError::ImageExportInit(err),
                ),
                SNAPSHOT_CONTEXT,
            )
        })?;
    exporter.set_theme_config(theme.unwrap_or_else(web::current_plot_theme_config));

    let bytes = exporter
        .render_geometry_scene_png_bytes_with_camera(&scene, &camera)
        .await
        .map_err(|err| {
            log::warn!(
                "runmat-runtime: render_geometry_scene_snapshot.failed scene={} error={}",
                scene.scene_id,
                err
            );
            map_control_flow_with_builtin(
                engine_error_with_source(
                    format!("Geometry export failed: {err}"),
                    PlottingBackendError::ImageExport(err),
                ),
                SNAPSHOT_CONTEXT,
            )
        })?;
    log::debug!(
        "runmat-runtime: render_geometry_scene_snapshot.ok scene={} bytes={}",
        scene.scene_id,
        bytes.len()
    );
    Ok(bytes)
}

#[cfg(feature = "plot-core")]
fn parse_camera_view_preset(value: Option<&str>) -> runmat_plot::core::CameraViewPreset {
    use runmat_plot::core::CameraViewPreset;

    let Some(value) = value else {
        return CameraViewPreset::Perspective;
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "perspective" | "iso" | "isometric" => CameraViewPreset::Perspective,
        "top" | "xy" => CameraViewPreset::Top,
        "bottom" => CameraViewPreset::Bottom,
        "front" | "xz" => CameraViewPreset::Front,
        "back" => CameraViewPreset::Back,
        "left" | "yz" => CameraViewPreset::Left,
        "right" => CameraViewPreset::Right,
        _ => CameraViewPreset::Perspective,
    }
}

#[cfg(feature = "plot-core")]
fn geometry_scene_snapshot_camera(
    bounds: runmat_plot::core::BoundingBox,
    width: u32,
    height: u32,
    preset: runmat_plot::core::CameraViewPreset,
) -> runmat_plot::core::Camera {
    use glam::Vec3;
    use runmat_plot::core::{BoundingBox, Camera, CameraViewPreset};

    let mut camera = Camera::new();
    camera.update_aspect_ratio(width.max(1) as f32 / height.max(1) as f32);
    let bounds = if bounds.min.is_finite()
        && bounds.max.is_finite()
        && (bounds.max - bounds.min).length() > 1.0e-6
    {
        bounds
    } else {
        BoundingBox {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        }
    };
    let (direction, up) = match preset {
        CameraViewPreset::Perspective => (Vec3::new(1.0, -1.0, 1.0).normalize(), Vec3::Z),
        CameraViewPreset::Top => (Vec3::Z, Vec3::Y),
        CameraViewPreset::Bottom => (-Vec3::Z, Vec3::Y),
        CameraViewPreset::Front => (-Vec3::Y, Vec3::Z),
        CameraViewPreset::Back => (Vec3::Y, Vec3::Z),
        CameraViewPreset::Left => (-Vec3::X, Vec3::Z),
        CameraViewPreset::Right => (Vec3::X, Vec3::Z),
    };
    let center = (bounds.min + bounds.max) * 0.5;
    camera.up = up;
    camera.target = center;
    camera.position = center + direction;
    camera.fit_bounds(bounds.min, bounds.max);
    camera
}
