use runmat_macros::runtime_builtin;
use runmat_value::Value;

fn active_context(operation: &str) -> crate::BuiltinResult<crate::context::RuntimeContext> {
    crate::context::legacy::active().ok_or_else(|| {
        crate::build_runtime_error(format!("{operation}: no active runtime context"))
            .with_builtin(operation)
            .with_identifier("RunMat:parallel:RuntimeContextUnavailable")
            .build()
    })
}

#[runtime_builtin(
    name = "parpool",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::pool"
)]
async fn parpool_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::parallel::pool::ensure(&active_context("parpool")?, &arguments)
}

#[runtime_builtin(
    name = "gcp",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::pool"
)]
async fn gcp_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::parallel::pool::current(&active_context("gcp")?, &arguments)
}
