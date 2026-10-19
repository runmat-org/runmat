use runmat_value::Value;

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::common::ode_error;

/// Decode the ODE solver's real-valued time/state/derivative vector. The
/// caller owns class and compatibility admission for its semantic operand.
/// A host value stays owned; only an actual resident handle is downloaded.
pub(super) async fn real_vector(name: &str, value: Value) -> BuiltinResult<Vec<f64>> {
    let value = match value {
        Value::GpuTensor(handle) => {
            crate::dispatcher::gather_if_needed_async(&Value::GpuTensor(handle)).await?
        }
        value => value,
    };
    let values = match value {
        Value::Num(number) => vec![number],
        Value::Int(integer) => vec![integer.to_f64()],
        Value::Bool(boolean) => vec![if boolean { 1.0 } else { 0.0 }],
        Value::Tensor(tensor) => tensor::tensor_values_f64(&tensor),
        Value::LogicalArray(array) => array
            .data
            .into_iter()
            .map(|element| if element != 0 { 1.0 } else { 0.0 })
            .collect(),
        other => {
            return Err(ode_error(
                name,
                format!("{name}: expected a real numeric vector, got {other:?}"),
            ));
        }
    };
    Ok(values)
}
