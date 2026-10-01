use super::*;
use runmat_builtins::{
    WBLINV_EXPLICIT_GPU_EXTENSION as EXPLICIT_GPU_EXTENSION,
    WBLINV_INTEGER_A_EXTENSION as INTEGER_A_EXTENSION,
    WBLINV_INTEGER_B_EXTENSION as INTEGER_B_EXTENSION,
    WBLINV_INTEGER_P_EXTENSION as INTEGER_P_EXTENSION,
    WBLINV_LOGICAL_INPUT_EXTENSION as LOGICAL_INPUT_EXTENSION,
};

fn ensure_extensions(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if super::is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_P_EXTENSION, "wblinv")?;
    }
    if rest.first().is_some_and(super::is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_A_EXTENSION, "wblinv")?;
    }
    if rest.get(1).is_some_and(super::is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_B_EXTENSION, "wblinv")?;
    }
    if std::iter::once(value)
        .chain(rest.iter())
        .any(super::is_logical_value)
    {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_INPUT_EXTENSION, "wblinv")?;
    }
    if std::iter::once(value).chain(rest.iter()).any(|value| {
        matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(&EXPLICIT_GPU_EXTENSION, "wblinv")?;
    }
    Ok(())
}

#[runtime_builtin(
    name = "wblinv",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::distributions::wblinv"
)]
pub(crate) async fn wblinv_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_extensions(&value, &rest)?;
    if !matches!(rest.len(), 0 | 2) {
        return Err(super::normal_error(
            "wblinv",
            "wblinv: expected one or three numeric arguments",
        ));
    }
    let inputs = std::iter::once(&value)
        .chain(rest.iter())
        .collect::<Vec<_>>();
    numeric_inputs("wblinv", &inputs, NumericShapePolicy::Broadcast)?;
    let output = super::NormalOutputPlan::inspect("wblinv", &value, &rest)?;
    let (a, b) = match rest.as_slice() {
        [] => (super::scalar_tensor(1.0), super::scalar_tensor(1.0)),
        [a, b] => (
            super::normal_value_to_tensor("wblinv", "a", a.clone()).await?,
            super::normal_value_to_tensor("wblinv", "b", b.clone()).await?,
        ),
        _ => {
            return Err(super::normal_error(
                "wblinv",
                "wblinv: expected one or three numeric arguments",
            ))
        }
    };
    let p = super::normal_value_to_tensor("wblinv", "p", value).await?;
    let (mut values, shape) = super::broadcast_tensors("wblinv", &[&p, &a, &b])?;
    let p = values.remove(0);
    let a = values.remove(0);
    let b = values.remove(0);
    let data = p
        .iter()
        .zip(a.iter())
        .zip(b.iter())
        .map(|((p, a), b)| super::wblinv_scalar(*p, *a, *b))
        .collect();
    output.finish("wblinv", shape, data)
}
