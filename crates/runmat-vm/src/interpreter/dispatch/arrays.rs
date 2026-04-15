use crate::ops::arrays as array_ops;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;
use std::future::Future;

pub fn create_matrix(stack: &mut Vec<Value>, rows: usize, cols: usize) -> Result<(), RuntimeError> {
    array_ops::create_matrix(stack, rows, cols)
}

pub async fn create_matrix_dynamic<F, Fut>(
    stack: &mut Vec<Value>,
    num_rows: usize,
    create_from_values: F,
) -> Result<(), RuntimeError>
where
    F: FnMut(Vec<Vec<Value>>) -> Fut,
    Fut: Future<Output = Result<Value, RuntimeError>>,
{
    array_ops::create_matrix_dynamic(stack, num_rows, create_from_values).await
}

pub async fn create_range<F, Fut>(
    stack: &mut Vec<Value>,
    has_step: bool,
    call_colon: F,
) -> Result<(), RuntimeError>
where
    F: FnMut(Vec<Value>) -> Fut,
    Fut: Future<Output = Result<Value, RuntimeError>>,
{
    array_ops::create_range(stack, has_step, call_colon).await
}

pub fn pack_to_row(stack: &mut Vec<Value>, count: usize) -> Result<(), RuntimeError> {
    array_ops::pack_to_row(stack, count)
}

pub fn pack_to_col(stack: &mut Vec<Value>, count: usize) -> Result<(), RuntimeError> {
    array_ops::pack_to_col(stack, count)
}

pub fn unpack(stack: &mut Vec<Value>, out_count: usize) -> Result<(), RuntimeError> {
    array_ops::unpack(stack, out_count)
}
