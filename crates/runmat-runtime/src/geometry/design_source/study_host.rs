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
