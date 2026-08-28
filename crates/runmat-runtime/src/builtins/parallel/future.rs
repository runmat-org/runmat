use runmat_macros::runtime_builtin;
use runmat_value::Value;

fn lowering_error(name: &str) -> crate::RuntimeError {
    crate::build_runtime_error(format!(
        "{name}: this call must be lowered by an execution-capable RunMat executor"
    ))
    .with_builtin(name)
    .with_identifier("RunMat:parallel:LoweringRequired")
    .build()
}

#[runtime_builtin(
    name = "parfeval",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::future"
)]
async fn parfeval_builtin(_arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    Err(lowering_error("parfeval"))
}

#[runtime_builtin(
    name = "parfevalOnAll",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::future"
)]
async fn parfeval_on_all_builtin(_arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    Err(lowering_error("parfevalOnAll"))
}

#[runtime_builtin(
    name = "fetchOutputs",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::future"
)]
async fn fetch_outputs_builtin(_future: Value) -> crate::BuiltinResult<Value> {
    Err(lowering_error("fetchOutputs"))
}

#[runtime_builtin(
    name = "fetchNext",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::future"
)]
async fn fetch_next_builtin(_futures: Value, _rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    Err(lowering_error("fetchNext"))
}
