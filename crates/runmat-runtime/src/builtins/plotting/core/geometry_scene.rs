#[cfg(all(feature = "plot-core", target_arch = "wasm32"))]
use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(feature = "plot-core")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(all(feature = "plot-core", not(target_arch = "wasm32")))]
use std::sync::{Mutex, OnceLock};

use super::import_registry::{register_imported_geometry_scene, take_imported_geometry_scene};

#[cfg(all(feature = "plot-core", not(target_arch = "wasm32")))]
type GeometrySceneRegistry = Mutex<HashMap<u32, runmat_plot::GeometryScene>>;

#[cfg(feature = "plot-core")]
static NEXT_GEOMETRY_SCENE_HANDLE: AtomicU64 = AtomicU64::new(1);

#[cfg(all(feature = "plot-core", target_arch = "wasm32"))]
thread_local! {
    static GEOMETRY_SCENE_REGISTRY: RefCell<HashMap<u32, runmat_plot::GeometryScene>> =
        RefCell::new(HashMap::new());
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
    super::web::present_geometry_scene_on_surface(surface_id, handle, scene)?;
    if take_imported_geometry_scene(handle) {
        let _ = super::web::reset_surface_camera(surface_id);
    }
    Ok(())
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
