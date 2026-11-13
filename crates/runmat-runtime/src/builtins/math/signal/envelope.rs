//! MATLAB-compatible `envelope` builtin for signal upper/lower envelopes.

use runmat_accelerate_api::{
    ProviderEnvelopeMethod, ProviderEnvelopeRequest, ProviderEnvelopeResult,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, ENVELOPE_ERROR_ARG_COUNT, ENVELOPE_ERROR_INTERNAL,
    ENVELOPE_ERROR_INVALID_LENGTH, ENVELOPE_ERROR_INVALID_METHOD, ENVELOPE_ERROR_INVALID_SIGNAL,
    ENVELOPE_ERROR_TOO_MANY_OUTPUTS, ENVELOPE_INTEGER_CONTROL_EXTENSION,
    ENVELOPE_INTEGER_DATA_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::signal::common::{
    keyword, parse_nonnegative_integer, SignalCooperation,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[path = "envelope/admission.rs"]
mod admission;
use admission::{preflight_signal_metadata, signal_geometry};

const BUILTIN_NAME: &str = "envelope";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::envelope")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("signal-envelope"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("signal_envelope")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses provider-side resident analytic and RMS envelope kernels for real GPU tensors; peak mode gathers because extrema selection plus spline interpolation is sparse and irregular.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::envelope")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Envelope extraction is a windowed/FFT interpolation boundary and terminates fusion plans.",
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnvelopeMethod {
    Analytic,
    Rms,
    Peak,
}

#[derive(Clone, Debug)]
struct EnvelopeOptions {
    method: EnvelopeMethod,
    parameter: Option<usize>,
}

#[derive(Clone, Debug)]
struct SignalInput {
    data: Vec<f64>,
    shape: Vec<usize>,
    rows: usize,
    cols: usize,
    vector: bool,
    single_output: bool,
}

#[derive(Clone, Debug)]
struct EnvelopeEvaluation {
    input: SignalInput,
    upper: Vec<f64>,
    lower: Vec<f64>,
}

fn envelope_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    envelope_error_with_message(error.message, error)
}

fn envelope_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    envelope_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn envelope_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn is_typed_integer_value(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

fn provider_rejected_nonfinite_signal(
    error: &runmat_accelerate_api::ProviderExecutionError,
) -> bool {
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if source.is::<runmat_accelerate_api::EnvelopeSignalDomainError>() {
            return true;
        }
        let Some(next) = source.source() else {
            return false;
        };
        source = next;
    }
}

#[runtime_builtin(
    name = "envelope",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::envelope"
)]
async fn envelope_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    cooperation.completed(0).await?;
    if is_typed_integer_value(&x) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ENVELOPE_INTEGER_DATA_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if rest.first().is_some_and(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ENVELOPE_INTEGER_CONTROL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let options = parse_options(rest.view(), &mut cooperation).await?;
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        return Err(envelope_error(&ENVELOPE_ERROR_TOO_MANY_OUTPUTS));
    }
    preflight_signal_metadata(&x)?;
    #[cfg(not(feature = "plot-core"))]
    if crate::output_count::current_output_count() == Some(0)
        || (crate::output_context::requested_output_count() == Some(0)
            && crate::output_count::current_output_count().is_none())
    {
        if let Value::GpuTensor(handle) = &x {
            super::resident_output::owner(handle, || envelope_error(&ENVELOPE_ERROR_INTERNAL))?;
            return Err(crate::plotting_hooks::unavailable_presentation());
        }
    }
    let mut prepared_integer = if is_typed_integer_value(&x) {
        let receipt = cooperation
            .prepare_integer_boundary(&x, &ENVELOPE_INTEGER_DATA_EXTENSION, "signal")
            .await?;
        receipt
            .into_resident_payload()
            .map(|payload| {
                payload.into_value().map_err(|cause| {
                    let mut error =
                        envelope_error_with_detail(&ENVELOPE_ERROR_INVALID_SIGNAL, cause);
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error
                })
            })
            .transpose()?
    } else {
        None
    };
    #[cfg(not(feature = "plot-core"))]
    if crate::output_count::current_output_count() == Some(0)
        || (crate::output_context::requested_output_count() == Some(0)
            && crate::output_count::current_output_count().is_none())
    {
        if let Value::GpuTensor(handle) = &x {
            super::resident_output::owner(handle, || envelope_error(&ENVELOPE_ERROR_INTERNAL))?;
        } else {
            let input = value_to_signal_input(prepared_integer.take().unwrap_or(x)).await?;
            let _ = evaluate(input, &options, &mut cooperation).await?;
        }
        return Err(crate::plotting_hooks::unavailable_presentation());
    }
    let source = if let Value::GpuTensor(handle) = &x {
        super::resident_output::owner(handle, || envelope_error(&ENVELOPE_ERROR_INTERNAL))?;
        Some(handle.clone())
    } else {
        None
    };
    let _source_residency = source
        .as_ref()
        .map(runmat_accelerate_api::ResidencyRestoreGuard::new);
    let restore = |value| {
        let Some(source) = source.as_ref() else {
            return Ok(value);
        };
        let protected = rest
            .iter()
            .filter_map(|value| {
                if let Value::GpuTensor(handle) = value {
                    Some(handle)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        super::resident_output::restore(source, value, BUILTIN_NAME, &protected)
    };

    if crate::output_context::requested_output_count() == Some(0)
        && crate::output_count::current_output_count().is_none()
    {
        let input = value_to_signal_input(prepared_integer.take().unwrap_or(x)).await?;
        let eval = evaluate(input, &options, &mut cooperation).await?;
        render_envelope_plot(&eval).await?;
        cooperation.completed(0).await?;
        return Ok(Value::output_list(Vec::new()));
    }

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            let input = value_to_signal_input(prepared_integer.take().unwrap_or(x)).await?;
            let eval = evaluate(input, &options, &mut cooperation).await?;
            render_envelope_plot(&eval).await?;
            cooperation.completed(0).await?;
            return Ok(Value::output_list(Vec::new()));
        }
        if let Value::GpuTensor(handle) = &x {
            if let Some(value) =
                try_gpu_envelope(handle, &options, Some(out_count), &mut cooperation).await?
            {
                return Ok(value);
            }
        }
        let input = value_to_signal_input(prepared_integer.take().unwrap_or(x)).await?;
        let eval = evaluate(input, &options, &mut cooperation).await?;
        if out_count == 1 {
            return restore(Value::output_list(vec![eval.upper_value()?]));
        }
        return restore(crate::output_count::output_list_with_padding(
            out_count,
            vec![eval.upper_value()?, eval.lower_value()?],
        ));
    }

    if let Value::GpuTensor(handle) = &x {
        if let Some(value) = try_gpu_envelope(handle, &options, None, &mut cooperation).await? {
            return Ok(value);
        }
    }
    let input = value_to_signal_input(prepared_integer.take().unwrap_or(x)).await?;
    let eval = evaluate(input, &options, &mut cooperation).await?;
    restore(eval.upper_value()?)
}

async fn try_gpu_envelope(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    options: &EnvelopeOptions,
    out_count: Option<usize>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<Value>> {
    if runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
        || runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Ok(None);
    }
    let provider =
        super::resident_output::owner(handle, || envelope_error(&ENVELOPE_ERROR_INTERNAL))?;
    let Some(method) = provider_envelope_method(options) else {
        return Ok(None);
    };
    let Some((channel_len, channel_count, output_shape)) = gpu_signal_shape(handle)? else {
        return Ok(None);
    };
    let request = ProviderEnvelopeRequest {
        input: handle,
        channel_len,
        channel_count,
        output_shape: &output_shape,
        method,
    };
    if !provider.supports_signal_envelope(&request) {
        return Ok(None);
    }
    cooperation.completed(0).await?;
    let result = match provider.signal_envelope(&request).await {
        Ok(result) => result,
        Err(error) => {
            let descriptor = if provider_rejected_nonfinite_signal(&error) {
                &ENVELOPE_ERROR_INVALID_SIGNAL
            } else {
                &ENVELOPE_ERROR_INTERNAL
            };
            return Err(crate::runtime_error::with_provider_execution_source(
                envelope_error_with_detail(
                    descriptor,
                    format!("provider envelope failed: {error}"),
                ),
                error,
            ));
        }
    };
    if let Err(error) = cooperation.completed(0).await {
        return Err(gpu_helpers::reject_gpu_output(
            result.lower,
            gpu_helpers::reject_gpu_output(result.upper, error),
        ));
    }
    let contract = gpu_helpers::BinaryGpuOutputContract {
        shape: output_shape,
        storage: runmat_accelerate_api::GpuTensorStorage::Real,
        precision: runmat_accelerate_api::handle_precision(handle),
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    let error = || {
        envelope_error_with_detail(
            &ENVELOPE_ERROR_INTERNAL,
            "provider envelope returned malformed output",
        )
    };
    let context = super::resident_output::OutputContext {
        input: handle,
        provider,
        caller_inputs: &[],
    };
    let upper = context.admit(result.upper, &contract, &[&result.lower], error)?;
    let lower = context.admit(result.lower, &contract, &[&upper], error)?;
    Ok(Some(gpu_envelope_value(
        ProviderEnvelopeResult { upper, lower },
        out_count,
    )?))
}

fn provider_envelope_method(options: &EnvelopeOptions) -> Option<ProviderEnvelopeMethod> {
    match options.method {
        EnvelopeMethod::Analytic => Some(match options.parameter {
            Some(filter_len) => ProviderEnvelopeMethod::AnalyticFir { filter_len },
            None => ProviderEnvelopeMethod::Analytic,
        }),
        EnvelopeMethod::Rms => Some(ProviderEnvelopeMethod::Rms {
            window_len: options.parameter.unwrap_or(1),
        }),
        EnvelopeMethod::Peak => None,
    }
}

fn gpu_signal_shape(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Option<(usize, usize, Vec<usize>)>> {
    if handle.shape().is_empty() {
        return Ok(None);
    }
    let Some(handle_len) = handle
        .shape()
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    else {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
    };
    let (output_shape, rows, cols) = signal_geometry(handle.shape(), handle_len)?;
    if rows == 1 || cols == 1 {
        Ok(Some((handle_len, 1, output_shape)))
    } else {
        Ok(Some((rows, cols, output_shape)))
    }
}

fn gpu_envelope_value(
    result: ProviderEnvelopeResult,
    out_count: Option<usize>,
) -> BuiltinResult<Value> {
    let ProviderEnvelopeResult { upper, lower } = result;
    match out_count {
        None => {
            drop(lower);
            gpu_helpers::resident_gpu_value(upper)
        }
        Some(1) => {
            drop(lower);
            Ok(Value::output_list(vec![gpu_helpers::resident_gpu_value(
                upper,
            )?]))
        }
        Some(count) => Ok(crate::output_count::output_list_with_padding(
            count,
            vec![
                gpu_helpers::resident_gpu_value(upper)?,
                gpu_helpers::resident_gpu_value(lower)?,
            ],
        )),
    }
}

async fn value_to_signal_input(value: Value) -> BuiltinResult<SignalInput> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| {
                    let mut error =
                        envelope_error_with_detail(&ENVELOPE_ERROR_INVALID_SIGNAL, err.message());
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error.source = Some(Box::new(err));
                    error
                })?;
            tensor_to_signal_input(tensor)
        }
        Value::Tensor(tensor) => tensor_to_signal_input(tensor),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INVALID_SIGNAL, err))?;
            tensor_to_signal_input(tensor)
        }
        Value::Num(n) => scalar_signal_input(n),
        Value::Int(i) => scalar_signal_input(i.to_f64()),
        Value::Bool(b) => scalar_signal_input(f64::from(u8::from(b))),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL))
        }
        _ => Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL)),
    }
}

fn scalar_signal_input(value: f64) -> BuiltinResult<SignalInput> {
    if !value.is_finite() {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
    }
    Ok(SignalInput {
        data: vec![value],
        shape: vec![1, 1],
        rows: 1,
        cols: 1,
        vector: true,
        single_output: false,
    })
}

fn tensor_to_signal_input(tensor: Tensor) -> BuiltinResult<SignalInput> {
    let len = tensor.len();
    let shape_meta = tensor.shape().to_vec();
    let (shape, rows, cols) = signal_geometry(&shape_meta, len)?;
    let single_output = tensor.numeric_dtype() == NumericDType::F32;
    let data = tensor::tensor_into_values_f64(tensor);
    if data.iter().any(|value| !value.is_finite()) {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
    }
    Ok(SignalInput {
        data,
        shape,
        rows,
        cols,
        vector: rows == 1 || cols == 1,
        single_output,
    })
}

async fn parse_options(
    args: crate::call::FunctionArgumentView<'_>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<EnvelopeOptions> {
    match args.len() {
        0 => Ok(EnvelopeOptions {
            method: EnvelopeMethod::Analytic,
            parameter: None,
        }),
        1 | 2 => {
            let resident_control = if let Value::GpuTensor(handle) = &args[0] {
                if runmat_accelerate_api::handle_storage(handle)
                    != runmat_accelerate_api::GpuTensorStorage::Real
                    || handle
                        .shape()
                        .iter()
                        .try_fold(1usize, |n, &extent| n.checked_mul(extent))
                        != Some(1)
                {
                    return Err(envelope_error(&ENVELOPE_ERROR_INVALID_LENGTH));
                }
                let control =
                    gpu_helpers::gather_value_async(&args[0])
                        .await
                        .map_err(|source| {
                            let mut builder = build_runtime_error(format!(
                                "{}: control download failed: {}",
                                ENVELOPE_ERROR_INVALID_LENGTH.message,
                                source.message()
                            ))
                            .with_builtin(BUILTIN_NAME)
                            .with_source(source)
                            .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
                            if let Some(identifier) = ENVELOPE_ERROR_INVALID_LENGTH.identifier {
                                builder = builder.with_identifier(identifier);
                            }
                            builder.build()
                        })?;
                cooperation.completed(0).await?;
                Some(control)
            } else {
                None
            };
            let parameter =
                parse_positive_integer("n", resident_control.as_ref().unwrap_or(&args[0]))?;
            let method = if args.len() == 2 {
                parse_method(&args[1])?
            } else {
                EnvelopeMethod::Analytic
            };
            Ok(EnvelopeOptions {
                method,
                parameter: Some(parameter),
            })
        }
        _ => Err(envelope_error(&ENVELOPE_ERROR_ARG_COUNT)),
    }
}

fn parse_positive_integer(label: &str, value: &Value) -> BuiltinResult<usize> {
    let parsed = parse_nonnegative_integer(BUILTIN_NAME, label, value)
        .map_err(|_| envelope_error(&ENVELOPE_ERROR_INVALID_LENGTH))?;
    if parsed == 0 {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_LENGTH));
    }
    Ok(parsed)
}

fn parse_method(value: &Value) -> BuiltinResult<EnvelopeMethod> {
    match keyword(value)
        .map_err(|source| {
            let mut error = envelope_error(&ENVELOPE_ERROR_INVALID_METHOD);
            error.source = Some(Box::new(source));
            error
        })?
        .as_deref()
    {
        Some("analytic") => Ok(EnvelopeMethod::Analytic),
        Some("rms") => Ok(EnvelopeMethod::Rms),
        Some("peak") => Ok(EnvelopeMethod::Peak),
        _ => Err(envelope_error(&ENVELOPE_ERROR_INVALID_METHOD)),
    }
}

async fn evaluate(
    input: SignalInput,
    options: &EnvelopeOptions,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<EnvelopeEvaluation> {
    let mut upper = vec![0.0; input.data.len()];
    let mut lower = vec![0.0; input.data.len()];
    for channel in input.channels() {
        cooperation.completed(0).await?;
        let mut work = NumericalWork::new(cooperation);
        let (up, lo) = match options.method {
            EnvelopeMethod::Analytic => {
                analytic_envelope(channel.values, options.parameter, &mut work).await?
            }
            EnvelopeMethod::Rms => {
                rms_envelope(channel.values, options.parameter.unwrap_or(1), &mut work).await?
            }
            EnvelopeMethod::Peak => {
                peak_envelope(channel.values, options.parameter.unwrap_or(1), &mut work).await?
            }
        };
        for (offset, (up_value, lo_value)) in up.into_iter().zip(lo).enumerate() {
            upper[channel.start + offset] = up_value;
            lower[channel.start + offset] = lo_value;
            work.tick().await?;
        }
        work.boundary().await?;
    }
    Ok(EnvelopeEvaluation {
        input,
        upper,
        lower,
    })
}

struct Channel<'a> {
    values: &'a [f64],
    start: usize,
}

impl SignalInput {
    fn channels(&self) -> Vec<Channel<'_>> {
        if self.vector {
            return vec![Channel {
                values: &self.data,
                start: 0,
            }];
        }
        (0..self.cols)
            .map(|col| {
                let start = col * self.rows;
                Channel {
                    values: &self.data[start..start + self.rows],
                    start,
                }
            })
            .collect()
    }
}

impl EnvelopeEvaluation {
    fn upper_value(&self) -> BuiltinResult<Value> {
        value_from_data(
            self.upper.clone(),
            self.input.shape.clone(),
            self.input.single_output,
        )
    }

    fn lower_value(&self) -> BuiltinResult<Value> {
        value_from_data(
            self.lower.clone(),
            self.input.shape.clone(),
            self.input.single_output,
        )
    }

    #[cfg(feature = "plot-core")]
    fn input_value(&self) -> BuiltinResult<Value> {
        value_from_data(
            self.input.data.clone(),
            self.input.shape.clone(),
            self.input.single_output,
        )
    }
}

fn value_from_data(data: Vec<f64>, shape: Vec<usize>, single_output: bool) -> BuiltinResult<Value> {
    if data.len() == 1 && !single_output {
        return Ok(Value::Num(data[0]));
    }
    let storage = if single_output {
        NumericStorage::F32(data.into_iter().map(|value| value as f32).collect())
    } else {
        NumericStorage::F64(data)
    };
    Tensor::from_numeric_storage(storage, shape)
        .map(Value::Tensor)
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err))
}

#[path = "envelope/numerics.rs"]
mod numerics;
use numerics::{analytic_envelope, peak_envelope, rms_envelope, NumericalWork};

#[cfg(feature = "plot-core")]
async fn render_envelope_plot(eval: &EnvelopeEvaluation) -> BuiltinResult<()> {
    let mut args = Vec::new();
    let x = if eval.input.vector {
        Tensor::new(
            (1..=eval.input.data.len()).map(|idx| idx as f64).collect(),
            vec![eval.input.data.len(), 1],
        )
        .map(Value::Tensor)
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err))?
    } else {
        Tensor::new(
            (1..=eval.input.rows).map(|idx| idx as f64).collect(),
            vec![eval.input.rows, 1],
        )
        .map(Value::Tensor)
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err))?
    };

    if eval.input.vector {
        args.extend([
            x.clone(),
            eval.input_value()?,
            x.clone(),
            eval.upper_value()?,
            x,
            eval.lower_value()?,
        ]);
    } else {
        for col in 0..eval.input.cols {
            let start = col * eval.input.rows;
            let end = start + eval.input.rows;
            args.push(x.clone());
            args.push(column_value(&eval.input.data[start..end])?);
            args.push(x.clone());
            args.push(column_value(&eval.upper[start..end])?);
            args.push(x.clone());
            args.push(column_value(&eval.lower[start..end])?);
        }
    }

    let _handle = crate::builtins::plotting::plot::plot_builtin(args)
        .await
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err.message()))?;
    Ok(())
}

#[cfg(not(feature = "plot-core"))]
async fn render_envelope_plot(_eval: &EnvelopeEvaluation) -> BuiltinResult<()> {
    Err(crate::plotting_hooks::unavailable_presentation())
}

#[cfg(feature = "plot-core")]
fn column_value(values: &[f64]) -> BuiltinResult<Value> {
    Tensor::new(values.to_vec(), vec![values.len(), 1])
        .map(Value::Tensor)
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err))
}

#[cfg(test)]
#[path = "envelope/tests.rs"]
mod tests;
