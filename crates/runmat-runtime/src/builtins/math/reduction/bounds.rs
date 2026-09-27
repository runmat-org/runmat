//! MATLAB-compatible `bounds` builtin.

use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runtime_builtin(
    name = "bounds",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::bounds"
)]
async fn bounds_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate_bounds(value, &rest).await?.into_value()
}

#[derive(Debug)]
struct BoundsEvaluation {
    smallest: Value,
    largest: Value,
}

impl BoundsEvaluation {
    fn into_value(self) -> BuiltinResult<Value> {
        if let Some(out_count) = crate::output_count::current_output_count() {
            if out_count == 0 {
                return Ok(Value::OutputList(Vec::new()));
            }
            return Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![self.smallest, self.largest],
            ));
        }
        Ok(Value::OutputList(vec![self.smallest, self.largest]))
    }
}

async fn evaluate_bounds(value: Value, rest: &[Value]) -> BuiltinResult<BoundsEvaluation> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BOUNDS_EXTRA_OUTPUTS_EXTENSION,
            "bounds",
        )?;
    }
    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(bounds_invalid_argument(
            "complex numbers with integer types are not supported",
        ));
    }
    let rest = normalize_bounds_args(rest)?;
    super::min::validate_reduction_arguments(&rest)
        .await
        .map_err(|error| map_bounds_reduction_error("min", error))?;
    let source = match &value {
        Value::GpuTensor(handle) => {
            gpu_helpers::select_resident_output_source([handle.clone()], "bounds")?
        }
        _ => None,
    };
    let value = gpu_helpers::gather_value_async(&value)
        .await
        .map_err(|err| bounds_internal(err.message()))?;
    // L07 owns extraction of the common extrema engine. These are retained
    // algorithm dependencies, not a second registry or a wrapper-built service.
    let empty = match &value {
        Value::Tensor(value) if value.is_empty() => Some(Value::Tensor(value.clone())),
        Value::ComplexTensor(value) if value.is_empty() => {
            Some(Value::ComplexTensor(value.clone()))
        }
        Value::LogicalArray(value) if value.data.is_empty() => Some(Value::Tensor(
            tensor::logical_to_tensor(value).map_err(bounds_internal)?,
        )),
        _ => None,
    };
    let evaluation = if let Some(empty) = empty {
        BoundsEvaluation {
            smallest: empty.clone(),
            largest: empty,
        }
    } else {
        let min_eval = super::min::evaluate(value.clone(), &rest)
            .await
            .map_err(|err| map_bounds_reduction_error("min", err))?;
        let max_eval = super::max::evaluate(value, &rest)
            .await
            .map_err(|err| map_bounds_reduction_error("max", err))?;
        BoundsEvaluation {
            smallest: min_eval.into_value(),
            largest: max_eval.into_value(),
        }
    };
    let Some(source) = source else {
        return Ok(evaluation);
    };
    let (smallest, largest) = super::resident_results::restore_pair(
        &source,
        evaluation.smallest,
        evaluation.largest,
        crate::output_count::current_output_count().unwrap_or(2),
        "bounds",
        &[&source],
    )
    .map_err(|error| bounds_internal(error.message()))?;
    Ok(BoundsEvaluation { smallest, largest })
}

fn normalize_bounds_args(rest: &[Value]) -> BuiltinResult<Vec<Value>> {
    if rest.len() > 2 {
        return Err(bounds_invalid_argument(
            "bounds accepts at most a dimension selector and nanflag",
        ));
    }
    let has_nanflag = rest.iter().any(is_nanflag);
    let mut out = Vec::with_capacity(rest.len() + 2);
    out.push(empty_placeholder());
    out.extend(rest.iter().map(|value| match keyword_of(value).as_deref() {
        Some("omitmissing") => Value::from("omitnan"),
        Some("includemissing") => Value::from("includenan"),
        _ => value.clone(),
    }));
    if !has_nanflag {
        out.push(Value::from("omitnan"));
    }
    Ok(out)
}

fn is_nanflag(value: &Value) -> bool {
    matches!(
        keyword_of(value).as_deref(),
        Some("omitnan" | "includenan" | "omitmissing" | "includemissing")
    )
}

fn map_bounds_reduction_error(label: &str, err: RuntimeError) -> RuntimeError {
    let detail = format!("{label} evaluation failed: {}", err.message());
    match err.identifier() {
        Some(identifier)
            if identifier.contains("InvalidArgument")
                || identifier.contains("InvalidInput")
                || identifier.contains("SizeMismatch") =>
        {
            bounds_invalid_argument(detail)
        }
        _ => bounds_internal(detail),
    }
}

fn empty_placeholder() -> Value {
    Value::Tensor(runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap())
}

fn bounds_error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin("bounds")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn bounds_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    bounds_error(&BOUNDS_ERROR_INVALID_ARGUMENT, detail)
}

fn bounds_internal(detail: impl AsRef<str>) -> RuntimeError {
    bounds_error(&BOUNDS_ERROR_INTERNAL, detail)
}

#[cfg(test)]
#[path = "bounds/tests.rs"]
mod tests;
