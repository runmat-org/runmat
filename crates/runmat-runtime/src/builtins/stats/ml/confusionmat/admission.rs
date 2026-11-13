use super::*;
pub(super) async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("confusionmat: {err}")))
}

pub(super) async fn gather_values(
    values: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    if !values.iter().any(crate::dispatcher::value_contains_gpu) {
        return Ok(values);
    }
    values
        .try_map_preserving_roots_async(values.len(), runtime, async |index| {
            gather(values[index].clone()).await
        })
        .await
}

pub(super) fn ensure_confusionmat_extensions(
    group: &Value,
    grouphat: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if is_typed_integer_value(group) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONFUSIONMAT_INTEGER_GROUP_EXTENSION,
            NAME,
        )?;
    }
    if is_typed_integer_value(grouphat) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONFUSIONMAT_INTEGER_GROUPHAT_EXTENSION,
            NAME,
        )?;
    }
    for pair in rest.chunks_exact(2) {
        if scalar_text(&pair[0], "option name").is_ok_and(|name| canonical(&name) == "order")
            && is_typed_integer_value(&pair[1])
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CONFUSIONMAT_INTEGER_ORDER_EXTENSION,
                NAME,
            )?;
        }
    }
    if matches!(group, Value::GpuTensor(_))
        || matches!(grouphat, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONFUSIONMAT_RESIDENT_INPUT_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

pub(super) fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}
