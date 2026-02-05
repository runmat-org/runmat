//! MATLAB-compatible `gcf` builtin.

use runmat_macros::runtime_builtin;

use super::state::current_figure_handle;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

#[runtime_builtin(
    name = "gcf",
    category = "plotting",
    summary = "Return the handle of the current figure.",
    keywords = "gcf,figure,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::gcf"
)]
pub fn gcf_builtin() -> crate::BuiltinResult<f64> {
    Ok(current_figure_handle().as_u32() as f64)
}
