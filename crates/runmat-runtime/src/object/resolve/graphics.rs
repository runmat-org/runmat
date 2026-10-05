use super::*;

pub(super) async fn load_graphics_member(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: &str,
) -> Result<Value, RuntimeError> {
    let arguments = [base, Value::String(field.to_string())];
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
) -> Result<Value, RuntimeError> {
    let arguments = [base.clone(), Value::String(field.to_string()), rhs];
    match context {
        Some(context) => crate::call_builtin_async_in_context(context, "set", &arguments).await?,
        None => crate::call_builtin_async("set", &arguments).await?,
    };
    Ok(base)
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
