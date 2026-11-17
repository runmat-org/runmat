//! The browser's study-geometry host.
//!
//! A study whose geometry is a RunMat MCAD part or assembly reads as the exact STEP AP242 bytes
//! the design regenerates to. The runtime worker has no OCCT kernel, so the page's geometry host
//! installs an async function here once per runtime worker:
//!
//! ```ts
//! setStudyGeometryProvider(
//!   provider: ((rootPath: string, files: Record<string, string>) => Promise<Uint8Array>) | null,
//! ): void
//! ```
//!
//! `rootPath` is the design's path and a key of `files`; `files` holds the text of the design and
//! of every file it reads (parts, sub-assemblies, imported STEP files), keyed by slash-separated
//! paths. The function resolves to the bytes `study_geometry(rootPath, files, null)` writes on the
//! OCCT kernel and rejects with the reason a design does not regenerate; the check reports that
//! reason at the geometry's `file =`. `null` removes the provider.
use crate::wire::errors::js_error;
use wasm_bindgen::prelude::*;

/// Installs the page's study-geometry provider for this runtime worker, or removes it (`null`).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = setStudyGeometryProvider)]
pub fn set_study_geometry_provider(provider: JsValue) -> Result<(), JsValue> {
    runmat_runtime::geometry::design_source::install_js_study_geometry_provider(provider)
        .map_err(|message| js_error(&message))
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = setStudyGeometryProvider)]
pub fn set_study_geometry_provider(_provider: JsValue) -> Result<(), JsValue> {
    Err(js_error(
        "setStudyGeometryProvider is only available when targeting wasm32",
    ))
}
