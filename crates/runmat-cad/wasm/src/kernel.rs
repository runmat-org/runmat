//! The geometry kernel the CAD runtime regenerates parts on.
//!
//! The product build installs OCCT once at startup ([`install`]): the browser and Desktop
//! worker run this runtime inside the OCCT geometry host (`runmat-geometry-wasm-host`, built
//! for emscripten), and native callers install `OcctKernel` the same way. Without an installed
//! kernel the runtime uses the grid reference kernel (`FakeKernel`), which stays as the fast
//! test double for this crate's and its callers' tests. Every compile result names the kernel
//! that made it (`kernel` in the JSON), so a product surface can check it never runs on the
//! test double.

use runmat_cad::evaluate::GeometryAnalyzer;
use runmat_cad::regen::kernel::ExactQueries;
use runmat_cad::regen::kernel::{CadKernel, Tessellator};
use runmat_cad::regen::FakeKernel;
use std::sync::OnceLock;

/// What the runtime needs of a kernel: operations, display tessellation, analyses and the exact
/// queries (hidden-line projection for drawings, STEP export).
pub trait DesignKernel: CadKernel + Tessellator + GeometryAnalyzer + ExactQueries {}
impl<T: CadKernel + Tessellator + GeometryAnalyzer + ExactQueries> DesignKernel for T {}

/// Makes a fresh kernel for one regeneration.
pub type KernelFactory = fn() -> Box<dyn DesignKernel>;

struct Installed {
    name: &'static str,
    make: KernelFactory,
}

static INSTALLED: OnceLock<Installed> = OnceLock::new();

/// Installs the kernel every later regeneration uses, named `name` in compile results (for
/// example `"occt"`). The first installation wins; it returns `false` when a kernel was already
/// installed.
pub fn install(name: &'static str, make: KernelFactory) -> bool {
    INSTALLED.set(Installed { name, make }).is_ok()
}

/// The installed kernel's name, or `"reference"` for the grid test double.
pub fn name() -> &'static str {
    INSTALLED.get().map_or("reference", |k| k.name)
}

/// A fresh kernel: the installed one, or the grid test double.
pub fn make() -> Box<dyn DesignKernel> {
    match INSTALLED.get() {
        Some(k) => (k.make)(),
        None => Box::new(FakeKernel::new()),
    }
}
