//! Public refline binding over the shared reference-line graphics service.

use crate::BuiltinResult;
use runmat_builtins::ReferenceLineOwner;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "refline",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::refline"
)]
pub(crate) async fn refline_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::reference_lines::run_reference_lines(runtime, args, ReferenceLineOwner::Refline).await
}

#[cfg(test)]
mod tests;
