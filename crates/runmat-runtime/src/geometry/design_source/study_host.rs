//! A host-supplied regeneration of RunMat MCAD designs into study geometry.
//!
//! A browser session has no in-process OCCT kernel. Its geometry host (an OCCT build running in
//! its own worker) installs a [`StudyGeometryProvider`] once per runtime worker; the async
//! design-geometry readers then send it the design's file closure and use the bytes it returns.

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use runmat_cad_wasm::runtime::Files;

/// The provider's pending result: the study geometry bytes, or the host's reason. Like the
/// filesystem provider's futures it need not be `Send`; a browser provider awaits a JavaScript
/// promise.
pub type StudyGeometryFuture = Pin<Box<dyn Future<Output = Result<Vec<u8>, String>>>>;

/// Regenerates a part or assembly into its exact study geometry.
pub trait StudyGeometryProvider {
    /// The exact STEP AP242 bytes of the design at `root`, regenerated once from `files`: the
    /// bytes `runmat_cad_wasm::exchange::study::study_geometry(root, files, None)` writes on the
    /// OCCT kernel, with the STEP files the design's parts import taken from `files`.
    ///
    /// `files` holds the design's text and every file it reads, keyed by the design compilers'
    /// slash-separated paths; `root` is one of its keys. The returned future owns its inputs.
    fn study_geometry(&self, root: &str, files: &Files) -> StudyGeometryFuture;
}

thread_local! {
    static PROVIDER: RefCell<Option<Rc<dyn StudyGeometryProvider>>> = const { RefCell::new(None) };
}

/// Installs (or, with `None`, removes) this thread's provider and returns the previous one.
/// A browser runtime worker runs on one thread, so this is per worker.
pub fn set_study_geometry_provider(
    provider: Option<Rc<dyn StudyGeometryProvider>>,
) -> Option<Rc<dyn StudyGeometryProvider>> {
    PROVIDER.with(|slot| slot.replace(provider))
}

/// This thread's installed provider.
pub fn installed_study_geometry_provider() -> Option<Rc<dyn StudyGeometryProvider>> {
    PROVIDER.with(|slot| slot.borrow().clone())
}

/// Installs a JavaScript study-geometry provider for this thread, or removes it (`null` or
/// `undefined`). Every browser worker that checks or runs studies (the runtime worker and the
/// language server's worker) installs the host's function through this one adapter:
///
/// ```ts
/// (rootPath: string, files: Record<string, string>) => Promise<Uint8Array>
/// ```
///
/// The function resolves to the bytes [`StudyGeometryProvider::study_geometry`] describes and
/// rejects with the reason the design does not regenerate.
#[cfg(target_arch = "wasm32")]
pub fn install_js_study_geometry_provider(provider: wasm_bindgen::JsValue) -> Result<(), String> {
    use wasm_bindgen::JsCast;
    if provider.is_null() || provider.is_undefined() {
        set_study_geometry_provider(None);
        return Ok(());
    }
    let function = provider
        .dyn_into::<js_sys::Function>()
        .map_err(|_| "setStudyGeometryProvider expects an async function or null".to_string())?;
    set_study_geometry_provider(Some(Rc::new(js::JsStudyGeometry(function))));
    Ok(())
}

#[cfg(target_arch = "wasm32")]
mod js {
    use super::{StudyGeometryFuture, StudyGeometryProvider};
    use runmat_cad_wasm::runtime::Files;
    use wasm_bindgen::{JsCast, JsValue};

    pub(super) struct JsStudyGeometry(pub(super) js_sys::Function);

    fn js_value_to_string(value: JsValue) -> String {
        value
            .as_string()
            .or_else(|| {
                js_sys::Reflect::get(&value, &JsValue::from_str("message"))
                    .ok()
                    .and_then(|message| message.as_string())
            })
            .unwrap_or_else(|| format!("{value:?}"))
    }

    impl StudyGeometryProvider for JsStudyGeometry {
        fn study_geometry(&self, root: &str, files: &Files) -> StudyGeometryFuture {
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

    fn call(provider: &js_sys::Function, root: &str, files: &Files) -> Result<JsValue, String> {
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
