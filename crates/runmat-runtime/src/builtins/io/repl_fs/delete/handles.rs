use super::*;

pub(super) async fn delete_handles(
    context: Option<&crate::context::RuntimeContext>,
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let mut mutated_last: Option<Value> = None;
    let mut total = 0usize;
    for value in values {
        total += process_handle_value(context, value, &mut mutated_last).await?;
    }
    if total == 1 {
        Ok(mutated_last.unwrap_or(Value::Num(0.0)))
    } else {
        Ok(Value::Num(0.0))
    }
}

async fn process_handle_value(
    context: Option<&crate::context::RuntimeContext>,
    value: &Value,
    mutated_last: &mut Option<Value>,
) -> BuiltinResult<usize> {
    match value {
        Value::HandleObject(handle) => {
            let mut method_dispatched = false;
            if let Some((delete_method, _owner)) = crate::class_registry::lookup_method(
                &handle.class_name,
                &runmat_types::MethodName::from("delete"),
            ) {
                if let Some(result) =
                    crate::user_functions::try_call_semantic_function_by_name_in_context(
                        context,
                        &delete_method.function_name,
                        &[Value::HandleObject(handle.clone())],
                        0,
                    )
                    .await
                {
                    result?;
                    method_dispatched = true;
                } else if runmat_builtins::builtin_function_by_name(&delete_method.function_name)
                    .is_some()
                    && delete_method.function_name != "delete"
                {
                    let arguments = [Value::HandleObject(handle.clone())];
                    match context {
                        Some(runtime) => {
                            crate::call_builtin_async_with_outputs_in_context(
                                runtime,
                                &delete_method.function_name,
                                &arguments,
                                0,
                            )
                            .await?;
                        }
                        None => {
                            crate::call_builtin_async_with_outputs(
                                &delete_method.function_name,
                                &arguments,
                                0,
                            )
                            .await?;
                        }
                    }
                    method_dispatched = true;
                } else {
                    return Err(delete_error_with(
                        &DELETE_ERROR_INVALID_HANDLE,
                        format!(
                            "delete: unresolved handle delete method '{}'",
                            delete_method.function_name
                        ),
                    ));
                }
            }
            if method_dispatched && !crate::is_handle_valid(handle) {
                let mut invalid = handle.clone();
                invalid.valid = false;
                *mutated_last = Some(Value::HandleObject(invalid));
                return Ok(1);
            }
            if !crate::set_handle_valid(handle, false) {
                return Err(delete_error_with(
                    &DELETE_ERROR_INVALID_HANDLE,
                    format!(
                        "delete: failed to invalidate handle object '{}'",
                        handle.class_name
                    ),
                ));
            }
            let mut invalid = handle.clone();
            invalid.valid = false;
            *mutated_last = Some(Value::HandleObject(invalid));
            Ok(1)
        }
        Value::Listener(listener) => {
            crate::invalidate_listener_registration(listener.id);
            let mut invalid = listener.clone();
            invalid.valid = false;
            invalid.enabled = false;
            *mutated_last = Some(Value::Listener(invalid));
            Ok(1)
        }
        Value::Cell(cell) => {
            let mut total = 0usize;
            for handle in cell.data() {
                let inner = handle;
                total += Box::pin(process_handle_value(context, inner, mutated_last)).await?;
            }
            Ok(total)
        }
        other => Err(delete_error_with(
            &DELETE_ERROR_INVALID_HANDLE,
            format!("delete: unsupported handle input {other:?}"),
        )),
    }
}

pub(super) fn is_handle_input(value: &Value) -> bool {
    match value {
        Value::HandleObject(_) | Value::Listener(_) => true,
        Value::Cell(cell) => cell.data().iter().all(|ptr| is_handle_input(ptr)),
        _ => false,
    }
}

pub(super) fn contains_handle_input(value: &Value) -> bool {
    match value {
        Value::HandleObject(_) | Value::Listener(_) => true,
        Value::Cell(cell) => cell.data().iter().any(|ptr| contains_handle_input(ptr)),
        _ => false,
    }
}
