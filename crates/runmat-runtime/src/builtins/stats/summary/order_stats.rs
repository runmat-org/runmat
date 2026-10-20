//! Quantiles, percentiles, and ranks.

use std::cmp::Ordering;

use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

fn is_real_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

fn ensure_quantile_integer_extensions(
    name: &str,
    input: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    data_extension: &BuiltinExtensionDescriptor,
    probability_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if is_real_typed_integer_value(input) {
        crate::compatibility::ensure_builtin_extension_enabled(data_extension, name)?;
    }
    if rest.first().is_some_and(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(probability_extension, name)?;
    }
    Ok(())
}

fn order_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

async fn value_to_tensor(name: &str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| order_error(name, format!("{name}: {err}")))?;
    tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| order_error(name, format!("{name}: {err}")))
}

mod ordered;
use ordered::*;

mod cooperative_sort;
#[cfg(feature = "plot-core")]
pub(crate) use cooperative_sort::stable_sort_non_nan_f64;
#[cfg(feature = "plot-core")]
pub(crate) use cooperative_sort::CooperativeSortError;

mod quantiles;
use quantiles::{parse_quantile_args, quantile_tensor};
use runmat_builtins::QuantileRequestUnit;

mod ranks;
use ranks::tiedrank_tensor;

pub mod quantile;

pub mod prctile;

pub mod tiedrank;

#[cfg(test)]
mod tests;
