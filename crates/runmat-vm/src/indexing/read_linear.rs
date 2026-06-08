use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

pub async fn generic_index(base: &Value, indices: &[f64]) -> Result<Value, RuntimeError> {
    runmat_runtime::perform_indexing(base, indices).await
}
