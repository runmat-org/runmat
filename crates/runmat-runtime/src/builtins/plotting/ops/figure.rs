//! MATLAB-compatible `figure` builtin for selecting/creating plotting windows.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::handles::parse_optional_figure_handle;
use super::state::{new_figure_handle, select_figure};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

#[runtime_builtin(
    name = "figure",
    category = "plotting",
    summary = "Create or select a plotting figure.",
    keywords = "figure,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::figure"
)]
pub fn figure_builtin(rest: Vec<Value>) -> crate::BuiltinResult<f64> {
    let handle = if rest.is_empty() {
        new_figure_handle()
    } else {
        match parse_optional_figure_handle(&rest[0], "figure")? {
            Some(handle) => {
                select_figure(handle);
                handle
            }
            None => new_figure_handle(),
        }
    };
    Ok(handle.as_u32() as f64)
}
