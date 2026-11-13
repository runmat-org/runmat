//! Legacy histogram counting with explicit bin endpoints.
//! Runtime-owned sink: admitted data/edges gather in the domain, returning host
//! outputs; original source buffers remain reusable after their fusion barrier.
mod admission;
mod compute;
mod edges;
mod error;
mod shape;
#[cfg(test)]
mod tests;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use admission::*;
pub use compute::{evaluate, HistcResult};
use edges::EdgeSets;
use error::*;
use runmat_builtins::{
    BuiltinErrorDescriptor, HISTC_ERROR_INTERNAL as ERROR_INTERNAL,
    HISTC_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT, HISTC_GPU_EXTENSION,
    HISTC_LOGICAL_EXTENSION, HISTC_WIDE_INTEGER_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};
use shape::*;
const NAME: &str = "histc";
const MAX_DIMENSION_ARGUMENT: usize = runmat_builtins::HISTC_MAX_DIMENSION;
#[runtime_builtin(
    name = "histc",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::hist::histc"
)]
async fn histc_builtin(
    x: Value,
    binranges: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        return Err(invalid_argument("histc: too many output arguments"));
    }
    let resident_inputs: Vec<_> = [&x, &binranges]
        .into_iter()
        .chain(rest.iter())
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let dim = match rest.len() {
        0 => None,
        1 => Some(
            tensor::dimension_from_value_async(&rest[0], NAME, false)
                .await
                .map_err(invalid_argument)?
                .ok_or_else(|| invalid_argument("histc: dim must be numeric"))
                .and_then(validate_dimension_argument)?,
        ),
        _ => {
            return Err(invalid_argument(
                "histc: expected x, binranges, and optional dim",
            ))
        }
    };
    ensure_histc_input_extensions(&x)?;
    ensure_histc_input_extensions(&binranges)?;
    let eval = evaluate(x, binranges, dim).await?;
    gpu_helpers::clear_input_residency(&resident_inputs, NAME)?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![Value::Tensor(eval.counts)]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![Value::Tensor(eval.counts), Value::Tensor(eval.indices)],
        ));
    }
    Ok(Value::Tensor(eval.counts))
}
