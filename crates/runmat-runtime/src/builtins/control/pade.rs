//! Continuous-time pure-delay Padé coefficients.
//!
//! The model-delay overload needs delay metadata that the current ordinary
//! `tf`/`ss` owner does not carry; it must not be inferred from coefficients.

use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::tensor::scalar_dimension_from_value_typed;
use crate::builtins::control::tf_model::control_error;
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

mod coefficients;
mod output;
#[cfg(test)]
mod tests;

const NAME: &str = "pade";

fn invalid(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:pade:InvalidArgument", message)
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:pade:UnsupportedForm", message)
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:pade:NumericalFailure", message)
}

fn admission(runtime: Option<&RuntimeContext>, error: HostAllocationError) -> RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    crate::builtin::builtin_control_error(cancellation.as_ref(), NAME, error)
}

fn checkpoint(runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
    HostAllocationReceipt::policy(runtime)
        .map(|_| ())
        .map_err(|error| admission(runtime, error))
}

#[runtime_builtin(
    name = "pade",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::pade"
)]
async fn pade_builtin(
    runtime: Option<RuntimeContext>,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() != 1 {
        return Err(invalid("pade expects a delay and one approximation order"));
    }
    let (delay, dtype) = scalar_delay(&first)?;
    let order = scalar_order(&rest[0])?;
    // The one/zero-output scalar overload is a plot, not a coefficient
    // return. Validate the inputs first, as with other control leaves.
    if crate::output_count::current_output_count() != Some(2) {
        return Err(unsupported(
            "the pure-delay coefficient form requires exactly two outputs",
        ));
    }
    checkpoint(runtime.as_ref())?;
    output::coefficients(delay, order, dtype, runtime.as_ref())
}

fn scalar_delay(value: &Value) -> BuiltinResult<(f64, NumericDType)> {
    let (delay, dtype) = match value {
        Value::Num(value) => (*value, NumericDType::F64),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            if let Some(values) = tensor.as_f32_slice() {
                (f64::from(values[0]), NumericDType::F32)
            } else if let Some(values) = tensor.as_f64_slice() {
                (values[0], NumericDType::F64)
            } else {
                return Err(invalid("delay must be a real single or double scalar"));
            }
        }
        Value::Object(_) => {
            return Err(unsupported(
                "model-delay approximation requires retained model delay metadata",
            ));
        }
        _ => return Err(invalid("delay must be a real single or double scalar")),
    };
    if !delay.is_finite() || delay < 0.0 {
        return Err(invalid("delay must be finite and nonnegative"));
    }
    Ok((delay, dtype))
}

fn scalar_order(value: &Value) -> BuiltinResult<usize> {
    let decoded = match value {
        Value::Num(_) | Value::Int(_) => scalar_dimension_from_value_typed(value, false),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            let scalar = if let Some(storage) = tensor.integer_storage() {
                Value::Int(storage.value_at(0).expect("one integer element"))
            } else if let Some(values) = tensor.as_f32_slice() {
                Value::Num(f64::from(values[0]))
            } else if let Some(values) = tensor.as_f64_slice() {
                Value::Num(values[0])
            } else {
                return Err(invalid("order must be a positive integer scalar"));
            };
            scalar_dimension_from_value_typed(&scalar, false)
        }
        _ => return Err(invalid("order must be a positive integer scalar")),
    };
    let order = decoded
        .map_err(|error| {
            invalid(format!(
                "order must be a positive platform integer: {error}"
            ))
        })?
        .ok_or_else(|| invalid("order must be a positive integer scalar"))?;
    order
        .checked_add(1)
        .and_then(|_| order.checked_mul(2))
        .ok_or_else(|| invalid("order is too large"))?;
    Ok(order)
}
