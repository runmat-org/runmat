use std::future::Future;

use runmat_value::Value;

use crate::execution::RootedValue;
use crate::object::indexing::{ObjectSubscript, ObjectSubscriptPath};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub fn execute_owned_subsref<'name, Read, ReadFuture>(
    base: Value,
    path: ObjectSubscriptPath,
    read_first: Read,
    caller_function_name: Option<&'name str>,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'name, Read, ReadFuture>
where
    Read: FnOnce(Value, ObjectSubscript) -> ReadFuture,
    ReadFuture: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let runtime = crate::context::legacy::active();
    async move {
        execute_owned_subsref_with_context(
            runtime.as_ref(),
            base,
            path,
            read_first,
            caller_function_name,
        )
        .await
    }
}

pub fn execute_owned_subsref_in_context<'runtime, 'name, Read, ReadFuture>(
    runtime: &'runtime crate::context::RuntimeContext,
    base: Value,
    path: ObjectSubscriptPath,
    read_first: Read,
    caller_function_name: Option<&'name str>,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'runtime, 'name, Read, ReadFuture>
where
    Read: FnOnce(Value, ObjectSubscript) -> ReadFuture,
    ReadFuture: Future<Output = Result<RootedValue, RuntimeError>>,
{
    execute_owned_subsref_with_context(Some(runtime), base, path, read_first, caller_function_name)
}

pub fn execute_owned_subsref_with_context<'runtime, 'name, Read, ReadFuture>(
    runtime: Option<&'runtime crate::context::RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    read_first: Read,
    caller_function_name: Option<&'name str>,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'runtime, 'name, Read, ReadFuture>
where
    Read: FnOnce(Value, ObjectSubscript) -> ReadFuture,
    ReadFuture: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let work =
        execute_owned_subsref_in_scope(runtime, base, path, read_first, caller_function_name);
    super::super::context::scope(runtime, work)
}

async fn execute_owned_subsref_in_scope<Read, ReadFuture>(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    read_first: Read,
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError>
where
    Read: FnOnce(Value, ObjectSubscript) -> ReadFuture,
    ReadFuture: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let mut steps = path.into_steps_in_context(runtime)?.into_iter();
    let first = steps.next().ok_or_else(empty_path_error)?;
    let value = read_first(base, first).await?;
    if steps.len() == 0 {
        Ok(value)
    } else {
        let suffix = ObjectSubscriptPath::from_remaining_in_context(runtime, steps)?;
        match runtime {
            Some(runtime) => {
                super::read_subscript_path_in_context(
                    runtime,
                    value.value().clone(),
                    suffix,
                    caller_function_name,
                )
                .await
            }
            None => {
                super::read_subscript_path(value.value().clone(), suffix, caller_function_name)
                    .await
            }
        }
    }
}

pub fn execute_owned_subsasgn<'name, Read, ReadFuture, Write, WriteFuture>(
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    read_first: Read,
    write_first: Write,
    caller_function_name: Option<&'name str>,
) -> impl Future<Output = Result<RootedValue, RuntimeError>>
       + use<'name, Read, ReadFuture, Write, WriteFuture>
where
    Read: FnOnce(Value, ObjectSubscript) -> ReadFuture,
    ReadFuture: Future<Output = Result<RootedValue, RuntimeError>>,
    Write: FnOnce(Value, ObjectSubscript, Vec<Value>) -> WriteFuture,
    WriteFuture: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let runtime = crate::context::legacy::active();
    async move {
        execute_owned_subsasgn_with_context(
            runtime.as_ref(),
            base,
            path,
            values,
            read_first,
            write_first,
            caller_function_name,
        )
        .await
    }
}

pub fn execute_owned_subsasgn_in_context<'runtime, 'name, Read, ReadFuture, Write, WriteFuture>(
    runtime: &'runtime crate::context::RuntimeContext,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    read_first: Read,
    write_first: Write,
    caller_function_name: Option<&'name str>,
) -> impl Future<Output = Result<RootedValue, RuntimeError>>
       + use<'runtime, 'name, Read, ReadFuture, Write, WriteFuture>
where
    Read: FnOnce(Value, ObjectSubscript) -> ReadFuture,
    ReadFuture: Future<Output = Result<RootedValue, RuntimeError>>,
    Write: FnOnce(Value, ObjectSubscript, Vec<Value>) -> WriteFuture,
    WriteFuture: Future<Output = Result<RootedValue, RuntimeError>>,
{
    execute_owned_subsasgn_with_context(
        Some(runtime),
        base,
        path,
        values,
        read_first,
        write_first,
        caller_function_name,
    )
}

pub fn execute_owned_subsasgn_with_context<'runtime, 'name, Read, ReadFuture, Write, WriteFuture>(
    runtime: Option<&'runtime crate::context::RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    read_first: Read,
    write_first: Write,
    caller_function_name: Option<&'name str>,
) -> impl Future<Output = Result<RootedValue, RuntimeError>>
       + use<'runtime, 'name, Read, ReadFuture, Write, WriteFuture>
where
    Read: FnOnce(Value, ObjectSubscript) -> ReadFuture,
    ReadFuture: Future<Output = Result<RootedValue, RuntimeError>>,
    Write: FnOnce(Value, ObjectSubscript, Vec<Value>) -> WriteFuture,
    WriteFuture: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let work = execute_owned_subsasgn_in_scope(
        runtime,
        base,
        path,
        values,
        read_first,
        write_first,
        caller_function_name,
    );
    super::super::context::scope(runtime, work)
}

async fn execute_owned_subsasgn_in_scope<Read, ReadFuture, Write, WriteFuture>(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    read_first: Read,
    write_first: Write,
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError>
where
    Read: FnOnce(Value, ObjectSubscript) -> ReadFuture,
    ReadFuture: Future<Output = Result<RootedValue, RuntimeError>>,
    Write: FnOnce(Value, ObjectSubscript, Vec<Value>) -> WriteFuture,
    WriteFuture: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let mut steps = path.into_steps_in_context(runtime)?.into_iter();
    let first = steps.next().ok_or_else(empty_path_error)?;
    if steps.len() == 0 {
        return write_first(base, first, values).await;
    }
    let child = read_first(base.clone(), first.clone()).await?;
    let suffix = ObjectSubscriptPath::from_remaining_in_context(runtime, steps)?;
    let updated = match runtime {
        Some(runtime) => {
            super::assign_subscript_path_in_context(
                runtime,
                child.value().clone(),
                suffix,
                values,
                caller_function_name,
            )
            .await?
        }
        None => {
            super::assign_subscript_path(
                child.value().clone(),
                suffix,
                values,
                caller_function_name,
            )
            .await?
        }
    };
    write_first(base, first, vec![updated.value().clone()]).await
}

fn empty_path_error() -> RuntimeError {
    semantic_error(
        "InvalidObjectSubscriptPath",
        "object subscript path is empty",
    )
}
