//! Scene playback without constructing an execution session or a project filesystem.
//! Exported example scenes must be self-contained; original result artifacts stay separate.
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = importGeometryScene)]
pub fn import_geometry_scene(scene: &[u8]) -> Result<Option<u32>, JsValue> {
    runmat_runtime::builtins::plotting::import_geometry_scene_payload(scene)
        .map_err(|error| crate::wire::errors::runtime_error_to_js(&error))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = importFigureScene)]
pub fn import_figure_scene(scene: &[u8]) -> Result<Option<u32>, JsValue> {
    runmat_runtime::builtins::plotting::import_figure_scene(scene)
        .map(|handle| handle.map(|value| value.as_u32()))
        .map_err(|error| crate::wire::errors::runtime_error_to_js(&error))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = closeGeometryScene)]
pub fn close_geometry_scene(handle: u32) -> bool {
    runmat_runtime::builtins::plotting::close_geometry_scene(handle)
}
