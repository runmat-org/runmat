use std::future::Future;

use runmat_value::Value;

use crate::object::indexing::{ObjectSubscript, ObjectSubscriptPath};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub async fn execute_owned_subsref<Read, ReadFuture>(
    base: Value,
    path: ObjectSubscriptPath,
    read_first: Read,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError>
where
    Read: FnOnce(Value, ObjectSubscript) -> ReadFuture,
    ReadFuture: Future<Output = Result<Value, RuntimeError>>,
{
    let mut steps = path.into_steps().into_iter();
    let first = steps.next().ok_or_else(empty_path_error)?;
    let value = read_first(base, first).await?;
    let remaining: Vec<_> = steps.collect();
    if remaining.is_empty() {
        Ok(value)
    } else {
        super::read_subscript_path(
            value,
            ObjectSubscriptPath::new(remaining)?,
            caller_function_name,
        )
        .await
    }
}

pub async fn execute_owned_subsasgn<Read, ReadFuture, Write, WriteFuture>(
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    read_first: Read,
    write_first: Write,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError>
where
    Read: FnOnce(Value, ObjectSubscript) -> ReadFuture,
    ReadFuture: Future<Output = Result<Value, RuntimeError>>,
    Write: FnOnce(Value, ObjectSubscript, Vec<Value>) -> WriteFuture,
    WriteFuture: Future<Output = Result<Value, RuntimeError>>,
{
    let mut steps = path.into_steps().into_iter();
    let first = steps.next().ok_or_else(empty_path_error)?;
    let remaining: Vec<_> = steps.collect();
    if remaining.is_empty() {
        return write_first(base, first, values).await;
    }
    let child = read_first(base.clone(), first.clone()).await?;
    let updated = super::assign_subscript_path(
        child,
        ObjectSubscriptPath::new(remaining)?,
        values,
        caller_function_name,
    )
    .await?;
    write_first(base, first, vec![updated]).await
}

fn empty_path_error() -> RuntimeError {
    semantic_error(
        "InvalidObjectSubscriptPath",
        "object subscript path is empty",
    )
}
