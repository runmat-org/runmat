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

#[cfg(target_arch = "wasm32")]
mod host {
    use crate::wire::errors::js_value_to_string;
    use runmat_runtime::geometry::design_source::{StudyGeometryFuture, StudyGeometryProvider};
    use std::collections::BTreeMap;
    use wasm_bindgen::{JsCast, JsValue};

    pub(super) struct JsStudyGeometry(pub(super) js_sys::Function);

    impl StudyGeometryProvider for JsStudyGeometry {
        fn study_geometry(
            &self,
            root: &str,
            files: &BTreeMap<String, String>,
        ) -> StudyGeometryFuture {
            let returned = call(&self.0, root, files);
            Box::pin(async move {
                let value = returned?;
                let value = match value.dyn_into::<js_sys::Promise>() {
                    Ok(promise) => wasm_bindgen_futures::JsFuture::from(promise)
                        .await
                        .map_err(js_value_to_string)?,
                    Err(value) => value,
                };
                let bytes = value.dyn_into::<js_sys::Uint8Array>().map_err(|_| {
                    "the study-geometry provider must resolve to a Uint8Array".to_string()
                })?;
                Ok(bytes.to_vec())
            })
        }
    }

    fn call(
        provider: &js_sys::Function,
        root: &str,
        files: &BTreeMap<String, String>,
    ) -> Result<JsValue, String> {
        let object = js_sys::Object::new();
        for (path, text) in files {
            js_sys::Reflect::set(&object, &JsValue::from_str(path), &JsValue::from_str(text))
                .map_err(js_value_to_string)?;
        }
        provider
            .call2(&JsValue::NULL, &JsValue::from_str(root), &object)
            .map_err(js_value_to_string)
    }
}

/// Installs the page's study-geometry provider for this runtime worker, or removes it (`null`).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = setStudyGeometryProvider)]
pub fn set_study_geometry_provider(provider: JsValue) -> Result<(), JsValue> {
    use runmat_runtime::geometry::design_source::set_study_geometry_provider as install;
    if provider.is_null() || provider.is_undefined() {
        install(None);
        return Ok(());
    }
    let function = provider
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_error("setStudyGeometryProvider expects an async function or null"))?;
    install(Some(std::rc::Rc::new(host::JsStudyGeometry(function))));
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = setStudyGeometryProvider)]
pub fn set_study_geometry_provider(_provider: JsValue) -> Result<(), JsValue> {
    Err(js_error(
        "setStudyGeometryProvider is only available when targeting wasm32",
    ))
}
