use runmat_builtins::NCHOOSEK_ERROR_INVALID_INPUT;
use runmat_value::Value;

use crate::BuiltinResult;

use super::{arguments, coefficient, combinations, error};

pub(super) async fn apply(first: Value, k: Value) -> BuiltinResult<Value> {
    if matches!(first, Value::GpuTensor(_)) || matches!(k, Value::GpuTensor(_)) {
        return Err(error::with_message(
            &NCHOOSEK_ERROR_INVALID_INPUT,
            "nchoosek: gpuArray inputs are not supported",
        ));
    }
    let selection = arguments::selection(&k)?;
    if let Some(population) = arguments::scalar_coefficient(&first) {
        let class = coefficient::resolve_class(population.class, selection.class)?;
        return coefficient::value(population.n, selection.value, class);
    }
    if arguments::is_numeric_scalar(&first) {
        return Err(error::with_message(
            &NCHOOSEK_ERROR_INVALID_INPUT,
            "nchoosek: scalar n must be a nonnegative integer",
        ));
    }
    combinations::value(first, selection.value)
}
