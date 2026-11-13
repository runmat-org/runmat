//! Multinomial logistic regression compatibility helper.

mod admission;
mod allocation;
mod class_census;
mod cooperation;
mod data;
mod numeric;
mod options;
mod output;
mod solver;

use allocation::*;
use cooperation::FitCooperation;
use data::*;
use numeric::*;
use options::{prepare_values, OptionName, Options};
use output::*;
use solver::*;

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, MnrfitStatsetField,
    MNRFIT_ERROR_CONVERGENCE as ERROR_CONVERGENCE, MNRFIT_ERROR_INTERNAL as ERROR_INTERNAL,
    MNRFIT_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    MNRFIT_INTEGER_CONTROL_EXTENSION as INTEGER_CONTROL_EXTENSION,
    MNRFIT_INTEGER_PARAMETER_EXTENSION as INTEGER_PARAMETER_EXTENSION,
    MNRFIT_INTEGER_X_EXTENSION as INTEGER_X_EXTENSION,
    MNRFIT_INTEGER_Y_EXTENSION as INTEGER_Y_EXTENSION,
    MNRFIT_LEGACY_OPTIONS_EXTENSION as LEGACY_OPTIONS_EXTENSION,
    MNRFIT_LOGICAL_X_EXTENSION as LOGICAL_X_EXTENSION,
    MNRFIT_RESIDENT_INPUT_EXTENSION as RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "mnrfit";
const EPS: f64 = 1.0e-12;

fn error(message: impl Into<String>, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_INVALID_ARGUMENT)
}

fn convergence(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_CONVERGENCE)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_INTERNAL)
}

fn scalar_text(value: &Value) -> BuiltinResult<Option<std::borrow::Cow<'_, str>>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Some(std::borrow::Cow::Borrowed(text.as_str()))
        }
        Value::CharArray(chars) if chars.rows == 1 => Some(std::borrow::Cow::Owned(
            crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
                invalid("mnrfit: text value is not valid UTF-16")
            })?,
        )),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            array
                .data()
                .first()
                .and_then(|text| text.as_str().map(std::borrow::Cow::Borrowed))
        }
        _ => None,
    })
}

#[derive(Clone, Debug)]
struct FitResult {
    b: Value,
    dev: Value,
    stats: Value,
}

#[runtime_builtin(
    name = "mnrfit",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::ml::mnrfit"
)]
pub(crate) async fn mnrfit_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    x: Value,
    y: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    mnrfit_in_runtime(runtime.as_ref(), x, y, rest).await
}

#[cfg(test)]
pub(crate) async fn mnrfit_builtin(x: Value, y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    mnrfit_in_runtime(
        None,
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

fn check_cancellation(runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<()> {
    cooperation::FitCancellation::new(runtime).check()
}

async fn mnrfit_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    x: Value,
    y: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    check_cancellation(runtime)?;
    let mut cooperation = FitCooperation::new(runtime);
    let inspected = options::inspect(rest.view())?;
    ensure_extensions(&x, &y, rest.view(), &inspected)?;
    let names = options::validate(rest.view(), inspected)?;
    let admitted = admission::prepare(x, y, &mut cooperation).await?;
    let values = rest.into_iter().skip(1).step_by(2);
    let prepared =
        prepare_values(names.into_iter().zip(values).collect(), &mut cooperation).await?;
    prepared.preflight_host_weights(admitted.rows())?;
    let (predictor, response) = admitted.into_parts();
    let x = predictor.materialize(&mut cooperation).await?;
    cooperation.check()?;
    let response = response.materialize(&mut cooperation).await?;
    cooperation.check()?;
    let options = prepared.materialize(&mut cooperation).await?;
    cooperation.check()?;
    let result = fit(x, response, options, &mut cooperation).await?;
    cooperation.check()?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![result.b])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![result.b, result.dev, result.stats],
        )),
        None => Ok(result.b),
    }
}

fn ensure_extensions(
    x: &Value,
    y: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    inspected: &[options::OptionSlot],
) -> BuiltinResult<()> {
    if is_typed_integer(x) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_X_EXTENSION, NAME)?;
    }
    if is_typed_integer(y) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_Y_EXTENSION, NAME)?;
    }
    if is_logical(x) {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_X_EXTENSION, NAME)?;
    }
    for (name, pair) in options::identities(inspected).zip(rest.chunks_exact(2)) {
        let Some(name) = name else { continue };
        let value = &pair[1];
        match name {
            OptionName::IterationLimit => ensure_integer_option(value, &INTEGER_CONTROL_EXTENSION)?,
            OptionName::Weights | OptionName::Tolerance => {
                ensure_integer_option(value, &INTEGER_PARAMETER_EXTENSION)?
            }
            OptionName::MaxIter
            | OptionName::TolX
            | OptionName::Options
            | OptionName::Constant
            | OptionName::Display => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LEGACY_OPTIONS_EXTENSION,
                    NAME,
                )?;
                match name {
                    OptionName::MaxIter => {
                        ensure_integer_option(value, &INTEGER_CONTROL_EXTENSION)?
                    }
                    OptionName::TolX => ensure_integer_option(value, &INTEGER_PARAMETER_EXTENSION)?,
                    OptionName::Options => ensure_nested_option_integers(value)?,
                    _ => {}
                }
            }
            _ => {}
        }
    }
    if contains_explicit_gpu(x)
        || contains_explicit_gpu(y)
        || rest.iter().any(contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(&RESIDENT_INPUT_EXTENSION, NAME)?;
    }
    Ok(())
}

fn ensure_integer_option(
    value: &Value,
    extension: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if is_typed_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)?;
    }
    Ok(())
}

fn ensure_nested_option_integers(value: &Value) -> BuiltinResult<()> {
    let Value::Struct(st) = value else {
        return Ok(());
    };
    for field in MnrfitStatsetField::ALL {
        if let Some(value) = st.fields.get(field.public_name()) {
            let extension = if field.is_iteration_limit() {
                &INTEGER_CONTROL_EXTENSION
            } else {
                &INTEGER_PARAMETER_EXTENSION
            };
            ensure_integer_option(value, extension)?;
        }
    }
    Ok(())
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Struct(st) => st.fields.values().any(contains_explicit_gpu),
        Value::Cell(cell) => cell.data().iter().any(contains_explicit_gpu),
        _ => false,
    }
}

async fn gather_value(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await.map_err(|cause| {
        let mut error = invalid(format!("mnrfit: {cause}"));
        error.gpu_gather_retry = cause.gpu_gather_retry;
        error.span = cause.span;
        error.source = Some(Box::new(cause));
        error
    })
}

async fn fit(
    x: Tensor,
    response: ResponseOutcomes,
    options: Options,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<FitResult> {
    if x.rows() == 0 || x.cols() == 0 {
        return Err(invalid("mnrfit: X must be a nonempty numeric matrix"));
    }
    cooperation.check()?;
    let cancellation = cooperation.cancellation();
    let check = || cancellation.check();
    let data = prepare_data(&x, response, &options, &check, cooperation).await?;
    cooperation.completed(1).await?;
    let fit =
        fit_nominal_logit(&data, options.max_iter, options.tol_x, &check, cooperation).await?;
    let b = coefficients_value(
        &fit.beta,
        data.cols,
        data.outcomes.len(),
        &check,
        cooperation,
    )
    .await?;
    let stats = stats_value(&fit, &data, &check, cooperation).await?;
    Ok(FitResult {
        b,
        dev: Value::Num(fit.deviance),
        stats,
    })
}

#[cfg(test)]
#[path = "mnrfit/provider_tests.rs"]
mod provider_tests;

#[cfg(test)]
#[path = "mnrfit/tests.rs"]
mod tests;
