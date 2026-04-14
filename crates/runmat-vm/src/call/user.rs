use crate::call::builtins::prepare_builtin_args;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

pub async fn try_builtin_fallback_single(
    name: &str,
    args: &[Value],
) -> Result<Option<Value>, RuntimeError> {
    let prepared = prepare_builtin_args(name, args).await?;
    match runmat_runtime::call_builtin_async(name, &prepared).await {
        Ok(result) => Ok(Some(result)),
        Err(_) => Ok(None),
    }
}

pub async fn try_builtin_fallback_multi(
    name: &str,
    args: &[Value],
    out_count: usize,
) -> Result<Option<Value>, RuntimeError> {
    let prepared = prepare_builtin_args(name, args).await?;
    match runmat_runtime::call_builtin_async(name, &prepared).await {
        Ok(result) => Ok(Some(crate::call::builtins::single_result_output_list(result, out_count))),
        Err(_) => Ok(None),
    }
}
