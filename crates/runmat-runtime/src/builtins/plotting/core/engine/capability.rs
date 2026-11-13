//! Compile-time plotting export capability admission.

use thiserror::Error;

/// Typed cause returned before an export operation touches a renderer,
/// filesystem provider, or resident operand.
#[derive(Debug, Error)]
#[error("PNG figure export requires a runtime built with plot-core support")]
pub(crate) struct PngExportUnavailable;

/// Admit PNG export without consulting invocation or ambient host state.
pub(crate) fn ensure_png_export_available() -> Result<(), PngExportUnavailable> {
    #[cfg(feature = "plot-core")]
    {
        Ok(())
    }
    #[cfg(not(feature = "plot-core"))]
    {
        Err(PngExportUnavailable)
    }
}
