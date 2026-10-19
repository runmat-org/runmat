use super::super::common::{check_cancellation, ode_error, OdeInput};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

pub(super) async fn eval_rhs(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    input: &OdeInput,
) -> BuiltinResult<Vec<f64>> {
    check_cancellation(name, input.runtime.as_ref())?;
    let y_arg = if input.scalar_state {
        Value::Num(y[0])
    } else {
        Value::Tensor(
            Tensor::new(y.to_vec(), input.y_shape.clone())
                .map_err(|e| ode_error(name, format!("{name}: {e}")))?,
        )
    };
    let value = super::super::callback::derivative(function, t, y_arg).await?;
    check_cancellation(name, input.runtime.as_ref())?;
    if crate::builtins::common::validation::value_contains_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            input.extensions.integer_callback,
            name,
        )?;
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await?
        {
            return Err(ode_error(
                name,
                format!(
                    "{name}: integer derivative values must be exactly representable as double"
                ),
            ));
        }
    }
    if matches!(value.value(), Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value.value(), Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            input.extensions.logical_numeric,
            name,
        )?;
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            input.extensions.resident_input,
            name,
        )?;
    }
    let rhs = value
        .with_owned_async(|value| super::super::values::real_vector(name, value))
        .await?;
    if rhs.len() != y.len() {
        return Err(ode_error(
            name,
            format!(
                "{name}: derivative output length {} does not match state length {}",
                rhs.len(),
                y.len()
            ),
        ));
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(ode_error(
            name,
            format!("{name}: function value must be finite"),
        ));
    }
    Ok(rhs)
}
