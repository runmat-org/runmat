//! MATLAB-compatible `maxk` and `mink` builtins.
//! Runtime-owned sink: exact host selection owns its gather and requested-output
//! restoration. A post-admission barrier must never erase explicit source intent.

use std::cmp::Ordering;

use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericScalar, Tensor, Value,
};

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runtime_builtin(
    name = "maxk",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::topk"
)]
async fn maxk_builtin(value: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate_topk(TopKKind::Max, value, rest.view())
        .await?
        .into_value(TopKKind::Max)
}

#[runtime_builtin(
    name = "mink",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::topk"
)]
async fn mink_builtin(value: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate_topk(TopKKind::Min, value, rest.view())
        .await?
        .into_value(TopKKind::Min)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopKKind {
    Max,
    Min,
}

impl TopKKind {
    fn name(self) -> &'static str {
        match self {
            TopKKind::Max => "maxk",
            TopKKind::Min => "mink",
        }
    }

    fn gpu_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            TopKKind::Max => &MAXK_GPU_INPUT_EXTENSION,
            TopKKind::Min => &MINK_GPU_INPUT_EXTENSION,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComparisonMethod {
    Auto,
    Abs,
    Real,
}

#[derive(Debug)]
pub struct TopKEvaluation {
    values: Value,
    indices: Value,
}

impl TopKEvaluation {
    fn into_value(self, kind: TopKKind) -> BuiltinResult<Value> {
        if let Some(out_count) = crate::output_count::current_output_count() {
            if out_count == 0 {
                return Ok(Value::output_list(Vec::new()));
            }
            if out_count == 1 {
                return Ok(Value::output_list(vec![self.values]));
            }
            if out_count == 2 {
                return Ok(Value::output_list(vec![self.values, self.indices]));
            }
            return Err(topk_invalid_argument(
                kind,
                "too many output arguments; maximum is 2",
            ));
        }
        Ok(self.values)
    }
}

#[derive(Clone, Debug)]
struct TopKArgs {
    k: usize,
    dim: Option<usize>,
    comparison: ComparisonMethod,
}

async fn evaluate_topk(
    kind: TopKKind,
    value: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<TopKEvaluation> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        return Err(topk_invalid_argument(
            kind,
            "too many output arguments; maximum is 2",
        ));
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&value, kind.name())?;
    for argument in rest {
        crate::builtins::common::validation::reject_typed_complex_integer(argument, kind.name())?;
    }
    let protected: Vec<_> = std::iter::once(&value)
        .chain(rest.iter())
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    if !protected.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(kind.gpu_extension(), kind.name())?;
    }
    let args = parse_topk_args(kind, rest).await?;
    let source = match &value {
        Value::GpuTensor(handle) => {
            gpu_helpers::select_resident_output_source([handle.clone()], kind.name())?
        }
        _ => None,
    };
    let input = gather_topk_input(kind, value).await?;
    let evaluation = match input {
        TopKInput::Real(tensor) => evaluate_real(kind, tensor, &args),
        TopKInput::Complex(tensor) => evaluate_complex(kind, tensor, &args),
        TopKInput::Logical(logical) => evaluate_logical(kind, logical, &args),
    }?;
    let evaluation = if let Some(source) = source {
        let (values, indices) = super::resident_results::restore_pair(
            &source,
            evaluation.values,
            evaluation.indices,
            crate::output_count::current_output_count().unwrap_or(1),
            kind.name(),
            &protected.iter().collect::<Vec<_>>(),
        )
        .map_err(|error| topk_internal(kind, error.message()))?;
        TopKEvaluation { values, indices }
    } else {
        evaluation
    };
    // Restoration's borrowed-input metadata guards re-mark inputs as resident.
    // Apply the completed sink barrier after those guards, leaving the distinct
    // requested output allocations resident and every source buffer reusable.
    gpu_helpers::clear_input_residency(&protected, kind.name())?;
    Ok(evaluation)
}

fn topk_error(
    kind: TopKKind,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}: {}",
        kind.name(),
        descriptor.message,
        detail.as_ref()
    ))
    .with_builtin(kind.name())
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn topk_invalid_argument(kind: TopKKind, detail: impl AsRef<str>) -> RuntimeError {
    topk_error(kind, &TOPK_ERROR_INVALID_ARGUMENT, detail)
}

fn topk_invalid_input(kind: TopKKind, detail: impl AsRef<str>) -> RuntimeError {
    topk_error(kind, &TOPK_ERROR_INVALID_INPUT, detail)
}

fn topk_internal(kind: TopKKind, detail: impl AsRef<str>) -> RuntimeError {
    topk_error(kind, &TOPK_ERROR_INTERNAL, detail)
}

#[path = "topk/arguments.rs"]
mod arguments;
use arguments::*;

#[path = "topk/input.rs"]
mod input;
use input::*;

#[path = "topk/host.rs"]
mod host;
use host::*;

#[path = "topk/ordering.rs"]
mod ordering;
use ordering::*;

#[path = "topk/shape.rs"]
mod shape;
use shape::*;

#[cfg(test)]
#[path = "topk/tests/mod.rs"]
mod tests;
