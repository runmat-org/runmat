mod algorithm;
use algorithm::{magic_doubly_even, magic_odd, magic_singly_even};
// Magic square construction. Contracts and inference live in the typed catalog.
use crate::build_runtime_error;
use crate::builtins::common::tensor;
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};
fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("magic").build()
}
#[runtime_builtin(
    name = "magic",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::magic"
)]
async fn magic_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        let error = &runmat_builtins::MAGIC_ERROR_TOO_MANY_OUTPUTS;
        return Err(build_runtime_error(error.message)
            .with_builtin("magic")
            .with_identifier(error.identifier.expect("output error identifier"))
            .build());
    }
    let n = parse_order(args).await?;
    let tensor = magic_tensor(n)?;
    Ok(tensor::tensor_into_value(tensor))
}

async fn parse_order(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<usize> {
    if args.len() != 1 {
        return Err(builtin_error("magic: requires exactly one input argument"));
    }
    parse_order_value(&args[0]).await
}

async fn parse_order_value(value: &Value) -> crate::BuiltinResult<usize> {
    if let Value::GpuTensor(handle) = value {
        if runmat_accelerate_api::handle_is_explicit(handle) {
            return Err(builtin_error("magic: gpuArray input is not supported"));
        }
        let host = crate::gather_if_needed_async(value)
            .await
            .map_err(|err| builtin_error(format!("magic: {err}")))?;
        return parse_order_host(&host).await;
    }
    parse_order_host(value).await
}

async fn parse_order_host(value: &Value) -> crate::BuiltinResult<usize> {
    if let Value::Int(value) = value {
        return parse_integer_order(value);
    }
    if let Value::Tensor(tensor) = value {
        if let Some(storage) = tensor.integer_storage() {
            return parse_integer_order(
                &storage
                    .value_at(0)
                    .ok_or_else(|| builtin_error("magic: input must contain a numeric value"))?,
            );
        }
        let raw = *tensor
            .materialize_f64()
            .first()
            .ok_or_else(|| builtin_error("magic: input must contain a numeric value"))?;
        return parse_floating_order(raw);
    }
    if let Value::Complex(re, _) = value {
        return parse_floating_order(*re);
    }
    if let Value::ComplexTensor(tensor) = value {
        if let Some(storage) = tensor.integer_storage() {
            return parse_integer_order(
                &storage
                    .real()
                    .value_at(0)
                    .ok_or_else(|| builtin_error("magic: input must contain a numeric value"))?,
            );
        }
        let raw = tensor
            .materialize_f64()
            .first()
            .map(|value| value.0)
            .ok_or_else(|| builtin_error("magic: input must contain a numeric value"))?;
        return parse_floating_order(raw);
    }
    if let Value::LogicalArray(array) = value {
        let raw = array
            .data
            .first()
            .map(|value| f64::from(*value != 0))
            .ok_or_else(|| builtin_error("magic: input must contain a numeric value"))?;
        return parse_floating_order(raw);
    }
    if let Value::CharArray(array) = value {
        let raw = array
            .data()
            .first()
            .map(|value| f64::from(value.raw()))
            .ok_or_else(|| builtin_error("magic: input must contain a numeric value"))?;
        return parse_floating_order(raw);
    }
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| builtin_error(format!("magic: {err}")))?
    else {
        return Err(builtin_error("magic: input must contain a numeric value"));
    };
    parse_floating_order(raw)
}

fn parse_floating_order(raw: f64) -> crate::BuiltinResult<usize> {
    if !raw.is_finite() {
        return Err(builtin_error("magic: dimension must be finite"));
    }
    let floored = raw.floor();
    if floored < 0.0 {
        return Err(builtin_error("magic: dimension must be non-negative"));
    }
    if floored > usize::MAX as f64 || (usize::BITS == 64 && floored == usize::MAX as f64) {
        return Err(builtin_error(
            "magic: dimension is too large for this platform",
        ));
    }
    Ok(floored as usize)
}

fn parse_integer_order(value: &IntValue) -> crate::BuiltinResult<usize> {
    let n = value
        .try_to_usize()
        .ok_or_else(|| builtin_error("magic: dimension is outside the supported range"))?;
    Ok(n)
}

fn magic_tensor(n: usize) -> Result<Tensor, crate::RuntimeError> {
    if n == 0 {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|err| builtin_error(format!("magic: {err}")));
    }

    if n == 2 {
        return Tensor::new(vec![1.0, 4.0, 3.0, 2.0], vec![2, 2])
            .map_err(|err| builtin_error(format!("magic: {err}")));
    }

    let data = if n % 2 == 1 {
        magic_odd(n)
    } else if n.is_multiple_of(4) {
        magic_doubly_even(n)
    } else {
        magic_singly_even(n)
    }
    .map_err(builtin_error)?;

    let data: Vec<f64> = data.into_iter().map(|v| v as f64).collect();
    Tensor::new(data, vec![n, n]).map_err(|err| builtin_error(format!("magic: {err}")))
}

#[cfg(test)]
mod tests;
