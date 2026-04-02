use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::clim::clim_builtin;
use crate::builtins::plotting::type_resolvers::get_type;

#[runtime_builtin(
    name = "caxis",
    category = "plotting",
    summary = "Query or set color limits.",
    keywords = "caxis,plotting,color",
    suppress_auto_output = true,
    type_resolver(get_type),
    builtin_path = "crate::builtins::plotting::caxis"
)]
pub fn caxis_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    clim_builtin(args)
}
