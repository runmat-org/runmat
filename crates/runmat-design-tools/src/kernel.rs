//! The kernel the design tools regenerate on. With the `occt` feature, [`install`] makes OCCT
//! the runtime's kernel (once per process), the same kernel Desktop and the browser run, so
//! `design.check`, `inspect`, `render` and `export` give Desktop's answers. Without it the
//! runtime keeps the grid test double (the tools' own tests).

/// Installs OCCT as the design kernel when the `occt` feature is on; otherwise does nothing.
/// Safe to call before every tool call.
pub fn install() {
    #[cfg(feature = "occt")]
    {
        runmat_cad_wasm::kernel::install("occt", || Box::new(runmat_geometry_io::OcctKernel::new()));
    }
}

/// The name of the kernel the tools regenerate on (`occt`, or `reference` for the test double).
pub fn name() -> &'static str {
    install();
    runmat_cad_wasm::kernel::name()
}

pub use runmat_cad_wasm::kernel::DesignKernel;

/// A fresh kernel for one regeneration: OCCT with the `occt` feature, else the test double.
pub fn make() -> Box<dyn DesignKernel> {
    install();
    runmat_cad_wasm::kernel::make()
}
