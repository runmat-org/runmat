//! MATLAB-compatible `filter` builtin with GPU-aware semantics for RunMat.
#[path = "filter/admission.rs"]
mod admission;
mod provider_output;
mod provider;
mod preflight;
mod host;
mod preparation;
#[path = "filter/inputs.rs"]
mod inputs;
#[path = "filter/initial_state.rs"]
mod initial_state;
#[cfg(test)]
#[path = "filter/tests.rs"]
pub(crate) mod tests;

use admission::FilterArgs;

use num_complex::Complex;
use runmat_accelerate_api::{GpuTensorHandle, ProviderIirFilterOptions};
use runmat_builtins::{
    BuiltinErrorDescriptor, FILTER_ERROR_ARG_COUNT, FILTER_ERROR_DENOMINATOR_A1_ZERO,
    FILTER_ERROR_EMPTY_DENOMINATOR, FILTER_ERROR_EMPTY_NUMERATOR, FILTER_ERROR_GATHER_FAILED,
    FILTER_ERROR_INTERNAL, FILTER_ERROR_INVALID_COEFFICIENTS, FILTER_ERROR_INVALID_DIMENSION,
    FILTER_ERROR_INVALID_INITIAL_STATE, FILTER_ERROR_INVALID_SIGNAL,
    FILTER_ERROR_PROVIDER_EXECUTION, FILTER_ERROR_PROVIDER_OUTPUT, FILTER_ERROR_PROVIDER_UPLOAD,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::common::SignalCooperation;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "filter";

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    filter_error_with_message(message, &FILTER_ERROR_INTERNAL)
}

fn filter_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    filter_error_with_message(error.message, error)
}

fn filter_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    filter_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn filter_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn filter_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

async fn parse_dimension_arg(value: &Value) -> BuiltinResult<usize> {
    tensor::dimension_from_value_async(value, BUILTIN_NAME, false)
        .await
        .map_err(|detail| filter_error_with_detail(&FILTER_ERROR_INVALID_DIMENSION, detail))?
        .ok_or_else(|| {
            filter_error_with_detail(
                &FILTER_ERROR_INVALID_DIMENSION,
                format!("received {value:?}"),
            )
        })
}

#[runtime_builtin(
    name = "filter",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::filter"
)]
async fn filter_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    b: Value,
    a: Value,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let eval = evaluate_with_runtime(runtime.as_ref(), b, a, x, rest.view()).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![eval.into_value()]));
        }
        let (output, final_state) = eval.into_pair();
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![output, final_state],
        ));
    }
    Ok(eval.into_value())
}

/// Evaluate the builtin once and expose both filter output and final state.
pub async fn evaluate(
    b: Value,
    a: Value,
    x: Value,
    rest: &[Value],
) -> BuiltinResult<FilterEvaluation> {
    evaluate_with_runtime(
        None,
        b,
        a,
        x,
        crate::call::FunctionArgumentView::from_slice(rest),
    )
    .await
}

async fn evaluate_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    b: Value,
    a: Value,
    x: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<FilterEvaluation> {
    let mut cooperation = SignalCooperation::new(runtime, BUILTIN_NAME);
    cooperation.completed(0).await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&b, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&a, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&x, BUILTIN_NAME)?;
    for value in rest {
        crate::builtins::common::validation::reject_typed_complex_integer(value, BUILTIN_NAME)?;
    }
    let mut args = FilterArgs::parse(b, a, x, rest, &mut cooperation).await?;
    if let Some(eval) = provider::try_filter_gpu(&args, &mut cooperation).await? {
        return Ok(eval);
    }
    args.signal.materialize(&mut cooperation).await?;
    args.initial.materialize(&mut cooperation).await?;
    host::filter_host(&args, &mut cooperation).await
}

#[derive(Debug, Clone)]
pub struct FilterEvaluation {
    output: Value,
    final_state: Value,
}

impl FilterEvaluation {
    pub fn into_value(self) -> Value {
        self.output
    }

    pub fn into_pair(self) -> (Value, Value) {
        (self.output, self.final_state)
    }

    pub fn final_state_value(&self) -> Value {
        self.final_state.clone()
    }
}
