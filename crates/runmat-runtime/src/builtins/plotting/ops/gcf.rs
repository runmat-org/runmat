//! MATLAB-compatible `gcf` builtin.

use runmat_macros::runtime_builtin;

use super::state::current_figure_handle;

#[runtime_builtin(
    name = "gcf",
    category = "plotting",
    summary = "Return the handle of the current figure.",
    keywords = "gcf,figure,plotting"
)]
pub fn gcf_builtin() -> Result<f64, String> {
    Ok(current_figure_handle().as_u32() as f64)
}
