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

#[path = "ops/area.rs"]
pub(crate) mod area;
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
#[path = "ops/contour3.rs"]
pub(crate) mod contour3;
#[path = "ops/contourf.rs"]
pub(crate) mod contourf;
#[path = "ops/drawnow.rs"]
pub(crate) mod drawnow;
#[path = "ops/errorbar.rs"]
pub(crate) mod errorbar;
#[path = "ops/figure.rs"]
pub(crate) mod figure;
#[path = "ops/fill3.rs"]
pub(crate) mod fill3;
#[path = "ops/gca.rs"]
pub(crate) mod gca;
#[path = "ops/gcf.rs"]
pub(crate) mod gcf;
#[path = "ops/get.rs"]
pub(crate) mod get;
#[path = "ops/heatmap.rs"]
pub(crate) mod heatmap;
#[path = "ops/hist.rs"]
pub mod hist;
#[path = "ops/histogram.rs"]
pub(crate) mod histogram;
#[path = "ops/hold.rs"]
pub(crate) mod hold;
#[path = "ops/image.rs"]
pub(crate) mod image;
#[path = "ops/imagesc.rs"]
pub(crate) mod imagesc;
#[path = "ops/imshow.rs"]
pub(crate) mod imshow;
#[path = "ops/isgraphics.rs"]
pub(crate) mod isgraphics;
#[path = "ops/ishandle.rs"]
pub(crate) mod ishandle;
#[path = "ops/legend.rs"]
pub(crate) mod legend;
#[path = "ops/loglog.rs"]
pub(crate) mod loglog;
#[path = "ops/mesh.rs"]
pub(crate) mod mesh;
#[path = "ops/meshc.rs"]
pub(crate) mod meshc;
#[path = "ops/common/mod.rs"]
pub(crate) mod op_common;
#[path = "ops/patch.rs"]
pub(crate) mod patch;
#[path = "ops/pie.rs"]
pub(crate) mod pie;
#[path = "ops/plot.rs"]
pub(crate) mod plot;
#[path = "ops/plot3.rs"]
pub(crate) mod plot3;
#[path = "ops/polarplot.rs"]
pub(crate) mod polarplot;
#[path = "ops/print.rs"]
pub(crate) mod print;
#[path = "ops/quiver.rs"]
pub(crate) mod quiver;
#[path = "ops/scatter.rs"]
pub(crate) mod scatter;
#[path = "ops/scatter3.rs"]
pub(crate) mod scatter3;
#[path = "ops/scatterplot.rs"]
pub(crate) mod scatterplot;
#[path = "ops/semilogx.rs"]
pub(crate) mod semilogx;
#[path = "ops/semilogy.rs"]
pub(crate) mod semilogy;
#[path = "ops/set.rs"]
pub(crate) mod set;
#[path = "ops/sgtitle.rs"]
pub(crate) mod sgtitle;
#[path = "ops/stairs.rs"]
pub(crate) mod stairs;
#[path = "ops/stem.rs"]
pub(crate) mod stem;
#[path = "ops/subplot.rs"]
pub(crate) mod subplot;
#[path = "ops/suptitle.rs"]
pub(crate) mod suptitle;
#[path = "ops/surf.rs"]
pub(crate) mod surf;
#[path = "ops/surfc.rs"]
pub(crate) mod surfc;
#[path = "ops/text.rs"]
pub(crate) mod text;
#[path = "ops/title.rs"]
pub(crate) mod title;
#[path = "ops/view.rs"]
pub(crate) mod view;
#[path = "ops/xlabel.rs"]
pub(crate) mod xlabel;
#[path = "ops/xlim.rs"]
pub(crate) mod xlim;
#[path = "ops/xline.rs"]
pub(crate) mod xline;
#[path = "ops/ylabel.rs"]
pub(crate) mod ylabel;
#[path = "ops/ylim.rs"]
pub(crate) mod ylim;
#[path = "ops/yline.rs"]
pub(crate) mod yline;
#[path = "ops/zlabel.rs"]
pub(crate) mod zlabel;
#[path = "ops/zlim.rs"]
pub(crate) mod zlim;

pub use perf::{
    set_scatter_target_points, set_scene_export_budget_bytes, set_surface_vertex_budget,
};
pub use properties::resolve_plot_handle;
pub use state::{
    clear_figure, clone_figure, close_figure, configure_subplot, current_axes_state,
    current_figure_handle, figure_handles, import_figure, install_figure_observer,
    new_figure_handle, record_recent_figure, reset_hold_state_for_run, reset_plot_state,
    reset_recent_figures, select_axes_for_figure, select_figure, set_hold, take_recent_figures,
    FigureAxesState, FigureError, FigureEventKind, FigureEventView, FigureHandle, HoldMode,
};
#[cfg(all(feature = "plot-core", target_arch = "wasm32"))]
use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(feature = "plot-core")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use web::present_figure_on_surface as web_present_figure_on_surface;
use web::present_geometry_scene_on_surface as web_present_geometry_scene_on_surface;
pub use web::{
    bind_surface_to_figure, clear_closed_figure_surfaces, detach_surface, fit_surface_extents,
    get_surface_camera_state, install_surface, invalidate_surface_revisions,
    pick_geometry_scene_region, present_surface, render_current_scene, reset_surface_camera,
    resize_surface, set_geometry_scene_presentation, set_plot_theme_config,
    set_surface_camera_state, web_renderer_ready, PlotCameraProjection, PlotCameraState,
    PlotSurfaceCameraState,
};

#[doc(hidden)]
pub use state::PlotTestLockGuard;

#[doc(hidden)]
pub fn lock_plot_test_context() -> PlotTestLockGuard {
    state::lock_plot_test_registry()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePlottingMode {
    Auto,
    Interactive,
    Static,
}

#[cfg(feature = "gui")]
pub fn set_runtime_plotting_mode(mode: RuntimePlottingMode) {
    let mapped = match mode {
        RuntimePlottingMode::Auto => engine::native::RuntimePlottingMode::Auto,
        RuntimePlottingMode::Interactive => engine::native::RuntimePlottingMode::Interactive,
        RuntimePlottingMode::Static => engine::native::RuntimePlottingMode::Static,
    };
    engine::native::set_runtime_plotting_mode(mapped);
}

#[cfg(not(feature = "gui"))]
pub fn set_runtime_plotting_mode(_mode: RuntimePlottingMode) {}

#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
pub use web::{handle_plot_surface_event, take_surface_host_actions, PlotSurfaceHostAction};

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
pub async fn export_figure_scene(handle: FigureHandle) -> crate::BuiltinResult<Option<Vec<u8>>> {
    export_figure_scene_with_policy(
        handle,
        runmat_plot::event::resolve_scene_export_policy(Some(perf::scene_export_budget_bytes())),
    )
    .await
}

#[cfg(feature = "plot-core")]
pub async fn export_figure_scene_with_policy(
    handle: FigureHandle,
    policy: runmat_plot::event::SceneExportPolicy,
) -> crate::BuiltinResult<Option<Vec<u8>>> {
    let Some(figure) = clone_figure(handle) else {
        return Ok(None);
    };
    let scene = runmat_plot::event::FigureScene::capture_for_export(&figure, policy)
        .await
        .map_err(|err| {
            crate::replay_error_with_source(
                crate::ReplayErrorKind::ExportRejected,
                "invalid figure scene content",
                err,
            )
        })?;
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
pub fn import_geometry_scene_payload(bytes: &[u8]) -> crate::BuiltinResult<Option<u32>> {
    let scene = crate::replay::import_figure_scene_payload(bytes)?;
    let hash = geometry_scene_payload_hash(bytes);
    let scene_id = format!("geometry-scene-payload:{hash:016x}");
    let scene = scene.into_geometry_scene(scene_id, hash).map_err(|err| {
        crate::replay_error_with_source(
            crate::ReplayErrorKind::ImportRejected,
            "invalid geometry scene content",
            std::io::Error::new(std::io::ErrorKind::InvalidData, err),
        )
    })?;
    import_geometry_scene(scene).map(Some)
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

#[cfg(feature = "plot-core")]
pub fn import_geometry_scene(scene: runmat_plot::GeometryScene) -> crate::BuiltinResult<u32> {
    let handle = NEXT_GEOMETRY_SCENE_HANDLE.fetch_add(1, Ordering::Relaxed) as u32;
    insert_geometry_scene(handle, scene)?;
    register_imported_geometry_scene(handle);
    Ok(handle)
}

#[cfg(feature = "plot-core")]
pub fn clone_geometry_scene(handle: u32) -> Option<runmat_plot::GeometryScene> {
    get_geometry_scene(handle)
}

#[cfg(feature = "plot-core")]
pub fn append_geometry_scene_chunks(
    handle: u32,
    chunks: Vec<runmat_plot::GeometrySceneChunk>,
    overlay: Option<runmat_plot::GeometrySceneOverlay>,
) -> crate::BuiltinResult<()> {
    with_geometry_scene_mut(handle, |scene| {
        if !chunks.is_empty() {
            scene.append_chunks(chunks);
        }
        if let Some(overlay) = overlay {
            let merged = merge_geometry_scene_overlay(scene, overlay);
            scene.set_overlay(merged);
        }
    })
}

#[cfg(feature = "plot-core")]
pub fn close_geometry_scene(handle: u32) -> bool {
    remove_geometry_scene(handle)
}

#[cfg(feature = "plot-core")]
pub fn export_geometry_scene(handle: u32) -> crate::BuiltinResult<Option<Vec<u8>>> {
    let Some(scene) = clone_geometry_scene(handle) else {
        return Ok(None);
    };
    let scene = runmat_plot::event::FigureScene::from_geometry_scene(&scene);
    crate::replay::export_figure_scene_payload(&scene).map(Some)
}

#[cfg(feature = "plot-core")]
pub fn present_geometry_scene_on_surface(surface_id: u32, handle: u32) -> crate::BuiltinResult<()> {
    let Some(scene) = clone_geometry_scene(handle) else {
        return Err(crate::build_runtime_error(format!(
            "geometry scene handle {handle} does not exist"
        ))
        .with_builtin("plot")
        .build());
    };
    web_present_geometry_scene_on_surface(surface_id, handle, scene)?;
    if take_imported_geometry_scene(handle) {
        let _ = reset_surface_camera(surface_id);
    }
    Ok(())
}

pub fn present_figure_on_surface(surface_id: u32, handle: u32) -> crate::BuiltinResult<()> {
    web_present_figure_on_surface(surface_id, handle)?;
    if take_imported_figure(handle) {
        let _ = reset_surface_camera(surface_id);
    }
    Ok(())
}

#[cfg(feature = "plot-core")]
pub fn import_runtime_figure(figure: runmat_plot::plots::Figure) -> u32 {
    let handle = state::import_figure(figure);
    register_imported_figure(handle.as_u32());
    handle.as_u32()
}

type ImportedFigureRegistry = Mutex<HashMap<u32, ()>>;
#[cfg(all(feature = "plot-core", not(target_arch = "wasm32")))]
type GeometrySceneRegistry = Mutex<HashMap<u32, runmat_plot::GeometryScene>>;
#[cfg(feature = "plot-core")]
type ImportedGeometrySceneRegistry = Mutex<HashMap<u32, ()>>;

#[cfg(feature = "plot-core")]
static NEXT_GEOMETRY_SCENE_HANDLE: AtomicU64 = AtomicU64::new(1);

#[cfg(all(feature = "plot-core", target_arch = "wasm32"))]
thread_local! {
    static GEOMETRY_SCENE_REGISTRY: RefCell<HashMap<u32, runmat_plot::GeometryScene>> =
        RefCell::new(HashMap::new());
}

#[cfg(all(feature = "plot-core", not(target_arch = "wasm32")))]
fn geometry_scene_registry() -> &'static GeometrySceneRegistry {
    static REGISTRY: OnceLock<GeometrySceneRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(all(feature = "plot-core", not(target_arch = "wasm32")))]
fn insert_geometry_scene(
    handle: u32,
    scene: runmat_plot::GeometryScene,
) -> crate::BuiltinResult<()> {
    let mut guard = geometry_scene_registry().lock().map_err(|_| {
        crate::build_runtime_error("geometry scene registry lock poisoned")
            .with_builtin("plot")
            .build()
    })?;
    guard.insert(handle, scene);
    Ok(())
}

#[cfg(all(feature = "plot-core", target_arch = "wasm32"))]
fn insert_geometry_scene(
    handle: u32,
    scene: runmat_plot::GeometryScene,
) -> crate::BuiltinResult<()> {
    GEOMETRY_SCENE_REGISTRY.with(|registry| {
        registry.borrow_mut().insert(handle, scene);
    });
    Ok(())
}

#[cfg(all(feature = "plot-core", not(target_arch = "wasm32")))]
fn get_geometry_scene(handle: u32) -> Option<runmat_plot::GeometryScene> {
    geometry_scene_registry().lock().ok()?.get(&handle).cloned()
}

#[cfg(all(feature = "plot-core", target_arch = "wasm32"))]
fn get_geometry_scene(handle: u32) -> Option<runmat_plot::GeometryScene> {
    GEOMETRY_SCENE_REGISTRY.with(|registry| registry.borrow().get(&handle).cloned())
}

#[cfg(all(feature = "plot-core", not(target_arch = "wasm32")))]
fn with_geometry_scene_mut(
    handle: u32,
    update: impl FnOnce(&mut runmat_plot::GeometryScene),
) -> crate::BuiltinResult<()> {
    let mut guard = geometry_scene_registry().lock().map_err(|_| {
        crate::build_runtime_error("geometry scene registry lock poisoned")
            .with_builtin("plot")
            .build()
    })?;
    let scene = guard.get_mut(&handle).ok_or_else(|| {
        crate::build_runtime_error(format!("geometry scene handle {handle} does not exist"))
            .with_builtin("plot")
            .build()
    })?;
    update(scene);
    Ok(())
}

#[cfg(all(feature = "plot-core", target_arch = "wasm32"))]
fn with_geometry_scene_mut(
    handle: u32,
    update: impl FnOnce(&mut runmat_plot::GeometryScene),
) -> crate::BuiltinResult<()> {
    GEOMETRY_SCENE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let scene = registry.get_mut(&handle).ok_or_else(|| {
            crate::build_runtime_error(format!("geometry scene handle {handle} does not exist"))
                .with_builtin("plot")
                .build()
        })?;
        update(scene);
        Ok(())
    })
}

#[cfg(all(feature = "plot-core", not(target_arch = "wasm32")))]
fn remove_geometry_scene(handle: u32) -> bool {
    geometry_scene_registry()
        .lock()
        .ok()
        .and_then(|mut guard| guard.remove(&handle))
        .is_some()
}

#[cfg(all(feature = "plot-core", target_arch = "wasm32"))]
fn remove_geometry_scene(handle: u32) -> bool {
    GEOMETRY_SCENE_REGISTRY.with(|registry| registry.borrow_mut().remove(&handle).is_some())
}

#[cfg(feature = "plot-core")]
fn geometry_scene_payload_hash(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(feature = "plot-core")]
fn merge_geometry_scene_overlay(
    scene: &runmat_plot::GeometryScene,
    incoming: runmat_plot::GeometrySceneOverlay,
) -> runmat_plot::GeometrySceneOverlay {
    let Some(mut current) = scene.overlay.clone() else {
        let mut overlay = incoming;
        overlay.vertex_count = scene.vertex_count();
        overlay.triangle_count = scene.triangle_count();
        return overlay;
    };

    current.status = incoming.status;
    current.quality_label = incoming.quality_label.or(current.quality_label);
    current.format = incoming.format.or(current.format);
    current.source_label = incoming.source_label.or(current.source_label);
    current.allow_create_fea_study =
        current.allow_create_fea_study || incoming.allow_create_fea_study;
    current.byte_count = incoming.byte_count.or(current.byte_count);
    current.mesh_count = current.mesh_count.max(incoming.mesh_count);
    current.vertex_count = scene.vertex_count();
    current.triangle_count = scene.triangle_count();
    current.progress_percent = incoming.progress_percent;

    if current.assembly_nodes.is_empty() {
        current.assembly_nodes = incoming.assembly_nodes;
    }

    merge_region_summaries(&mut current.regions, incoming.regions);
    current.region_count = current.regions.len();
    current.mapped_region_count = current.region_count;
    merge_warnings(&mut current.warnings, incoming.warnings);
    current
}

#[cfg(feature = "plot-core")]
fn merge_region_summaries(
    current: &mut Vec<runmat_plot::GeometrySceneRegionSummary>,
    incoming: Vec<runmat_plot::GeometrySceneRegionSummary>,
) {
    let mut positions = HashMap::<String, usize>::with_capacity(current.len() + incoming.len());
    for (index, region) in current.iter().enumerate() {
        positions.insert(region.region_id.clone(), index);
    }
    for region in incoming {
        if let Some(index) = positions.get(&region.region_id).copied() {
            current[index].triangle_count = current[index]
                .triangle_count
                .saturating_add(region.triangle_count);
        } else {
            positions.insert(region.region_id.clone(), current.len());
            current.push(region);
        }
    }
}

#[cfg(feature = "plot-core")]
fn merge_warnings(current: &mut Vec<String>, incoming: Vec<String>) {
    for warning in incoming {
        if !current.iter().any(|item| item == &warning) {
            current.push(warning);
        }
    }
}

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
fn imported_geometry_scene_registry() -> &'static ImportedGeometrySceneRegistry {
    static REGISTRY: OnceLock<ImportedGeometrySceneRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(feature = "plot-core")]
pub fn register_imported_geometry_scene(handle: u32) {
    if let Ok(mut map) = imported_geometry_scene_registry().lock() {
        map.insert(handle, ());
    }
}

#[cfg(feature = "plot-core")]
fn take_imported_geometry_scene(handle: u32) -> bool {
    imported_geometry_scene_registry()
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
    render_figure_snapshot, render_figure_snapshot_with_camera_state,
    render_geometry_scene_snapshot,
};

pub mod ops {
    pub use super::hist;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::state;

    pub(crate) fn ensure_plot_test_env() {
        state::disable_rendering_for_tests();
    }

    pub(crate) fn lock_plot_registry() -> state::PlotTestLockGuard {
        state::lock_plot_test_registry()
    }
}
