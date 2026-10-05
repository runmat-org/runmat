use runmat_value::Value;

#[runmat_macros::runtime_builtin(
    name = "rethrow",
    binding_variant = "default",
    builtin_path = "crate::builtins::diagnostics::rethrow"
)]
pub async fn rethrow_builtin_registered(e: Value) -> crate::BuiltinResult<Value> {
    match e {
        Value::MException(_) | Value::String(_) => crate::rethrow_builtin(e).await,
        other => Err(
            crate::build_runtime_error(format!("RunMat:error: {other:?}"))
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
        ),
    }
}
