use super::*;

pub(super) fn is_graphics_receiver_handle(handle: &runmat_value::HandleRef) -> bool {
    runmat_gc::gc_with_value(&handle.target, |target| match target {
        Value::Object(object) => object.graphics_receiver().is_some(),
        _ => false,
    })
    .unwrap_or(false)
}

pub(super) async fn load_graphics_member(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: &str,
) -> Result<RootedValue, RuntimeError> {
    let arguments = [base, Value::string(field.to_string())];
    match context {
        Some(context) => crate::call_builtin_async_in_context(context, "get", &arguments).await,
        None => crate::call_builtin_async("get", &arguments).await,
    }
}

pub(super) async fn store_graphics_member(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: &str,
    rhs: Value,
) -> Result<RootedValue, RuntimeError> {
    let arguments = [base.clone(), Value::string(field.to_string()), rhs];
    match context {
        Some(context) => crate::call_builtin_async_in_context(context, "set", &arguments).await?,
        None => crate::call_builtin_async("set", &arguments).await?,
    };
    RootedValue::new(base).map_err(|error| error.into_runtime_error())
}

pub(super) fn is_invalid_graphics_handle_error(err: &RuntimeError) -> bool {
    let text = err.to_string().to_ascii_lowercase();
    text.contains("unsupported or invalid plotting handle")
        || text.contains("invalid plotting handle")
        || text.contains("invalid figure handle")
        || text.contains("invalid axes handle")
}

pub(super) fn is_possible_graphics_handle_value(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite() && *v >= 0.0,
        Value::Int(i) => i.try_to_u64().is_some(),
        _ => false,
    }
}
