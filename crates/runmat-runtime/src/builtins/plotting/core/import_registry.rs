use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

type ImportedFigureRegistry = Mutex<HashMap<u32, ()>>;

fn imported_figure_registry() -> &'static ImportedFigureRegistry {
    static REGISTRY: OnceLock<ImportedFigureRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn register_imported_figure(handle: u32) {
    if let Ok(mut map) = imported_figure_registry().lock() {
        map.insert(handle, ());
    }
}

pub(super) fn take_imported_figure(handle: u32) -> bool {
    imported_figure_registry()
        .lock()
        .ok()
        .and_then(|mut map| map.remove(&handle))
        .is_some()
}

#[cfg(feature = "plot-core")]
type ImportedGeometrySceneRegistry = Mutex<HashMap<u32, ()>>;

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
pub(super) fn take_imported_geometry_scene(handle: u32) -> bool {
    imported_geometry_scene_registry()
        .lock()
        .ok()
        .and_then(|mut map| map.remove(&handle))
        .is_some()
}
