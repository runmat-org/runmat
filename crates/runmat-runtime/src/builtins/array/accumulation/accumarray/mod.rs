mod callback;
mod data;
mod error;
mod execute;
mod output;
mod provider;
mod shape;
mod sparse;
mod subscripts;
mod typed_output;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "accumarray",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::array::accumulation::accumarray"
)]
pub(crate) async fn accumarray_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    subscripts: Value,
    data: Value,
    options: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let request = provider::materialize(runtime, subscripts, data, options).await?;
    execute::run(request).await
}
