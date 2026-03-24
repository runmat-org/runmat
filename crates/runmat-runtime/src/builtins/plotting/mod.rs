//! Plotting builtins backed by the runmat-plot renderer.

#[path = "core/common.rs"]
pub(crate) mod common;
#[path = "core/context.rs"]
pub mod context;
#[path = "core/engine.rs"]
pub(crate) mod engine;
#[path = "core/gpu_helpers.rs"]
pub(crate) mod gpu_helpers;
#[path = "core/perf.rs"]
pub(crate) mod perf;
#[path = "core/point.rs"]
pub(crate) mod point;
#[path = "core/properties.rs"]
pub(crate) mod properties;
#[path = "core/state.rs"]
pub(crate) mod state;
#[path = "core/style.rs"]
pub(crate) mod style;
#[path = "core/web.rs"]
pub mod web;

#[path = "type_resolvers.rs"]
pub(crate) mod type_resolvers;

#[path = "ops/bar.rs"]
pub(crate) mod bar;
#[path = "ops/caxis.rs"]
pub(crate) mod caxis;
#[path = "ops/clf.rs"]
pub(crate) mod clf;
#[path = "ops/clim.rs"]
pub(crate) mod clim;
#[path = "ops/close.rs"]
pub(crate) mod close;
#[path = "ops/cmds.rs"]
pub(crate) mod cmds;
#[path = "ops/contour.rs"]
pub(crate) mod contour;
#[path = "ops/contourf.rs"]
pub(crate) mod contourf;
#[path = "ops/drawnow.rs"]
pub(crate) mod drawnow;
#[path = "ops/figure.rs"]
pub(crate) mod figure;
#[path = "ops/gca.rs"]
pub(crate) mod gca;
#[path = "ops/gcf.rs"]
pub(crate) mod gcf;
#[path = "ops/get.rs"]
pub(crate) mod get;
#[path = "ops/hist.rs"]
pub mod hist;
#[path = "ops/hold.rs"]
pub(crate) mod hold;
#[path = "ops/legend.rs"]
pub(crate) mod legend;
#[path = "ops/mesh.rs"]
pub(crate) mod mesh;
#[path = "ops/meshc.rs"]
pub(crate) mod meshc;
#[path = "ops/common/mod.rs"]
pub(crate) mod op_common;
#[path = "ops/plot.rs"]
pub(crate) mod plot;
#[path = "ops/scatter.rs"]
pub(crate) mod scatter;
#[path = "ops/scatter3.rs"]
pub(crate) mod scatter3;
#[path = "ops/set.rs"]
pub(crate) mod set;
#[path = "ops/stairs.rs"]
pub(crate) mod stairs;
#[path = "ops/subplot.rs"]
pub(crate) mod subplot;
#[path = "ops/surf.rs"]
pub(crate) mod surf;
#[path = "ops/surfc.rs"]
pub(crate) mod surfc;
#[path = "ops/title.rs"]
pub(crate) mod title;
#[path = "ops/xlabel.rs"]
pub(crate) mod xlabel;
#[path = "ops/xlim.rs"]
pub(crate) mod xlim;
#[path = "ops/ylabel.rs"]
pub(crate) mod ylabel;
#[path = "ops/ylim.rs"]
pub(crate) mod ylim;
#[path = "ops/zlim.rs"]
pub(crate) mod zlim;

pub use perf::{set_scatter_target_points, set_surface_vertex_budget};
pub use state::{
    clear_figure, clone_figure, close_figure, configure_subplot, current_axes_state,
    current_figure_handle, figure_handles, import_figure, install_figure_observer,
    new_figure_handle, reset_hold_state_for_run, reset_recent_figures, select_figure, set_hold,
    take_recent_figures, FigureAxesState, FigureError, FigureEventKind, FigureEventView,
    FigureHandle, HoldMode,
};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use web::present_figure_on_surface as web_present_figure_on_surface;
pub use web::{
    bind_surface_to_figure, detach_surface, fit_surface_extents, get_surface_camera_state,
    install_surface, present_surface, render_current_scene, reset_surface_camera, resize_surface,
    set_plot_theme_config, set_surface_camera_state, web_renderer_ready, PlotCameraProjection,
    PlotCameraState, PlotSurfaceCameraState,
};

#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
pub use web::handle_plot_surface_event;

pub(crate) fn plotting_error(builtin: &str, message: impl Into<String>) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(builtin)
        .build()
}

pub(crate) fn plotting_error_with_source(
    builtin: &str,
    message: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(builtin)
        .with_source(source)
        .build()
}

#[cfg(feature = "plot-core")]
pub fn export_figure_scene(handle: FigureHandle) -> crate::BuiltinResult<Option<Vec<u8>>> {
    let Some(figure) = clone_figure(handle) else {
        return Ok(None);
    };
    let scene = runmat_plot::event::FigureScene::capture(&figure);
    crate::replay::export_figure_scene_payload(&scene).map(Some)
}

#[cfg(feature = "plot-core")]
pub fn import_figure_scene(bytes: &[u8]) -> crate::BuiltinResult<Option<FigureHandle>> {
    let scene = crate::replay::import_figure_scene_payload(bytes)?;
    let figure = scene.into_figure().map_err(|err| {
        crate::replay_error_with_source(
            crate::ReplayErrorKind::ImportRejected,
            "invalid figure scene content",
            std::io::Error::new(std::io::ErrorKind::InvalidData, err),
        )
    })?;
    let handle = import_figure(figure);
    register_imported_figure(handle.as_u32());
    Ok(Some(handle))
}

#[cfg(feature = "plot-core")]
pub async fn import_figure_scene_async(bytes: &[u8]) -> crate::BuiltinResult<Option<FigureHandle>> {
    let scene = crate::replay::import_figure_scene_payload_async(bytes).await?;
    let figure = scene.into_figure().map_err(|err| {
        crate::replay_error_with_source(
            crate::ReplayErrorKind::ImportRejected,
            "invalid figure scene content",
            std::io::Error::new(std::io::ErrorKind::InvalidData, err),
        )
    })?;
    let handle = import_figure(figure);
    register_imported_figure(handle.as_u32());
    Ok(Some(handle))
}

#[cfg(feature = "plot-core")]
pub async fn import_figure_scene_from_path_async(
    path: &str,
) -> crate::BuiltinResult<Option<FigureHandle>> {
    let bytes = runmat_filesystem::read_async(path).await.map_err(|err| {
        crate::replay_error_with_source(
            crate::ReplayErrorKind::ImportRejected,
            format!("failed to read figure scene payload '{path}'"),
            err,
        )
    })?;
    import_figure_scene_async(&bytes).await
}

pub fn present_figure_on_surface(surface_id: u32, handle: u32) -> crate::BuiltinResult<()> {
    web_present_figure_on_surface(surface_id, handle)?;
    if take_imported_figure(handle) {
        let _ = reset_surface_camera(surface_id);
    }
    Ok(())
}

type ImportedFigureRegistry = Mutex<HashMap<u32, ()>>;

fn imported_figure_registry() -> &'static ImportedFigureRegistry {
    static REGISTRY: OnceLock<ImportedFigureRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_imported_figure(handle: u32) {
    if let Ok(mut map) = imported_figure_registry().lock() {
        map.insert(handle, ());
    }
}

fn take_imported_figure(handle: u32) -> bool {
    imported_figure_registry()
        .lock()
        .ok()
        .and_then(|mut map| map.remove(&handle))
        .is_some()
}

#[cfg(feature = "plot-core")]
pub use engine::{
    render_figure_png_bytes, render_figure_png_bytes_with_axes_cameras,
    render_figure_png_bytes_with_camera, render_figure_rgba_bytes,
    render_figure_rgba_bytes_with_axes_cameras, render_figure_rgba_bytes_with_camera,
    render_figure_snapshot,
};

pub mod ops {
    pub use super::hist;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::state;
    use std::sync::Once;

    pub(crate) fn ensure_plot_test_env() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            state::disable_rendering_for_tests();
        });
    }

    pub(crate) fn lock_plot_registry() -> state::PlotTestLockGuard {
        state::lock_plot_test_registry()
    }
}
