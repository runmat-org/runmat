//! The browser language server's study-geometry host.
//!
//! A study whose geometry is a RunMat MCAD part or assembly is checked against the exact STEP
//! bytes the design regenerates to (`design_geometry_async`). The language server's worker has no
//! OCCT kernel, so the host installs the same async function it gives the runtime worker
//! (runmat-wasm `setStudyGeometryProvider`), through the runtime's one adapter:
//!
//! ```ts
//! setStudyGeometryProvider(
//!   provider: ((rootPath: string, files: Record<string, string>) => Promise<Uint8Array>) | null,
//! ): void
//! ```
//!
//! Without a provider, a study on a part or assembly reports at its `file =` that this browser
//! session can't regenerate RunMat MCAD geometry. `null` removes the provider.
use wasm_bindgen::prelude::*;

/// Installs the host's study-geometry provider for this language server, or removes it (`null`).
#[wasm_bindgen(js_name = setStudyGeometryProvider)]
pub fn set_study_geometry_provider(provider: JsValue) -> Result<(), JsValue> {
    runmat_runtime::geometry::design_source::install_js_study_geometry_provider(provider)
        .map_err(|message| JsValue::from_str(&message))
}
