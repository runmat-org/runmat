//! MATLAB-compatible Welch power spectral density estimate.

use self::signal::preflight_signal_carrier;
use num_complex::Complex;
use runmat_accelerate_api::{
    ProviderSpectralFrameMode, ProviderSpectralRange, ProviderSpectralRequest,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, PwelchControlCursor, PwelchControlRole,
    PwelchKeyword, PWELCH_ERROR_ARG_COUNT, PWELCH_ERROR_INTERNAL, PWELCH_ERROR_INVALID_FS,
    PWELCH_ERROR_INVALID_NFFT, PWELCH_ERROR_INVALID_OVERLAP, PWELCH_ERROR_INVALID_PARENT,
    PWELCH_ERROR_INVALID_RANGE, PWELCH_ERROR_INVALID_SIGNAL, PWELCH_ERROR_INVALID_WINDOW,
    PWELCH_INTEGER_CONTROL_EXTENSION, PWELCH_INTEGER_SIGNAL_EXTENSION,
    PWELCH_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
use rustfft::FftPlanner;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::math::signal::common::output_precision::SignalPrecision;
use crate::builtins::math::signal::common::{
    centered_frequency_offset, centered_shift, gpu_matrix_shape, hamming_coefficient,
    parse_nonnegative_integer, parse_scalar_f64, symmetric_window_values, value_to_complex_vector,
    window_allocation_error, SignalCooperation, WindowArgError, SIGNAL_WORK_CHUNK,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "pwelch";
const DEFAULT_NFFT_MIN: usize = 256;
const MAX_NFFT: usize = 1 << 22;
#[derive(Default)]
struct CompletedWork {
    pending: u64,
}

impl CompletedWork {
    fn record(&mut self, units: u64) -> Option<u64> {
        self.pending = self.pending.saturating_add(units);
        if self.pending >= SIGNAL_WORK_CHUNK {
            Some(std::mem::take(&mut self.pending))
        } else {
            None
        }
    }

    async fn finish(&mut self, cooperation: &mut SignalCooperation) -> BuiltinResult<()> {
        cooperation
            .completed(std::mem::take(&mut self.pending))
            .await
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::pwelch")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pwelch",
    op_kind: GpuOpKind::Custom("welch-psd"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("uniform_spectral_estimate")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uniform-grid Welch spectral estimation stays GPU-resident for input framing, FFTs, power scaling, and segment averaging; explicit-frequency forms fall back to host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::pwelch")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pwelch",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "pwelch materialises PSD and frequency vectors and is not fused.",
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrequencyRange {
    Onesided,
    Twosided,
    Centered,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpectrumScale {
    Psd,
    Power,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WelchTrace {
    Mean,
    MaxHold,
    MinHold,
}

#[derive(Clone, Debug)]
struct SignalColumns {
    columns: Vec<Vec<Complex<f64>>>,
    rows: usize,
    cols: usize,
    is_complex: bool,
    output_precision: SignalPrecision,
}

#[derive(Clone, Debug)]
struct PwelchOptions {
    window: Vec<f64>,
    noverlap: usize,
    grid: FrequencyGrid,
    fs: Option<f64>,
    range: FrequencyRange,
    scale: SpectrumScale,
    trace: WelchTrace,
}

#[derive(Clone, Debug)]
enum FrequencyGrid {
    Uniform { nfft: usize },
    Explicit { frequencies: Vec<f64> },
}

#[derive(Clone, Debug)]
struct PwelchEvaluation {
    pxx: Vec<f64>,
    f: Vec<f64>,
    columns: usize,
}

fn pwelch_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    pwelch_error_with_message(error.message, error)
}

fn pwelch_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    pwelch_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn pwelch_control_error(
    error: &'static BuiltinErrorDescriptor,
    source: RuntimeError,
) -> RuntimeError {
    let mut error = pwelch_error_with_detail(error, source.message());
    error.source = Some(Box::new(source));
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

fn pwelch_window_generation_error(error: WindowArgError) -> RuntimeError {
    match error {
        WindowArgError::Allocation(cause) => {
            window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause)
        }
        WindowArgError::InvalidLength => pwelch_error(&PWELCH_ERROR_INVALID_WINDOW),
        WindowArgError::TensorBuild(detail) => {
            pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, detail)
        }
        WindowArgError::Text(source) => {
            let mut error = pwelch_error(&PWELCH_ERROR_INVALID_WINDOW);
            error.source = Some(Box::new(source));
            error
        }
        WindowArgError::TooManyOptions
        | WindowArgError::InvalidOptionType
        | WindowArgError::UnknownOption(_) => pwelch_error(&PWELCH_ERROR_INTERNAL),
    }
}

fn pwelch_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "pwelch",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::pwelch"
)]
async fn pwelch_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate_in_runtime(runtime.as_ref(), x, rest.view()).await
}

#[cfg(test)]
async fn evaluate(x: Value, rest: &[Value]) -> BuiltinResult<Value> {
    evaluate_in_runtime(None, x, crate::call::FunctionArgumentView::from_slice(rest)).await
}

async fn evaluate_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    x: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime, BUILTIN_NAME);
    evaluate_with_cooperation(x, rest, &mut cooperation).await
}

async fn evaluate_with_cooperation(
    x: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    cooperation.completed(0).await?;
    let named = split_named_controls(rest)?;
    let rest = named.positional.as_slice();
    let tokens = named.tokens.as_slice();
    if rest.len() > 7 {
        return Err(pwelch_error(&PWELCH_ERROR_ARG_COUNT));
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&x, BUILTIN_NAME)?;
    for value in rest.iter().copied() {
        crate::builtins::common::validation::reject_typed_complex_integer(value, BUILTIN_NAME)?;
    }
    ensure_pwelch_class_boundary(&x, &PWELCH_INTEGER_SIGNAL_EXTENSION)?;
    for value in rest.iter().take(4).copied() {
        ensure_pwelch_class_boundary(value, &PWELCH_INTEGER_CONTROL_EXTENSION)?;
    }
    let complex_input = preflight_signal_carrier(&x)?;
    let mut prepared_controls =
        preflight_host_controls(rest, tokens, complex_input, cooperation).await?;
    #[cfg(not(feature = "plot-core"))]
    if named.wants_plot() {
        return Err(crate::plotting_hooks::unavailable_presentation());
    }
    // All borrowed operands must be admitted before exactness checks or parsers
    // can download any one of them.
    let resident_inputs = std::iter::once(&x)
        .chain(rest.iter().copied())
        .filter_map(|value| {
            if let Value::GpuTensor(handle) = value {
                Some(handle)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    for handle in &resident_inputs {
        super::resident_output::owner(handle, || pwelch_error(&PWELCH_ERROR_INTERNAL))?;
    }
    let _preflight_guards = resident_inputs
        .iter()
        .map(|handle| runmat_accelerate_api::ResidencyRestoreGuard::new(handle))
        .collect::<Vec<_>>();
    cooperation.completed(0).await?;
    let prepared_signal_payload = ensure_pwelch_integer_boundary(
        &x,
        "signal",
        &PWELCH_ERROR_INVALID_SIGNAL,
        &PWELCH_INTEGER_SIGNAL_EXTENSION,
        cooperation,
    )
    .await?;
    cooperation.completed(0).await?;
    let materialized_controls =
        materialize_resident_controls(rest, &mut prepared_controls, cooperation).await?;
    let controls = rest
        .iter()
        .zip(&materialized_controls)
        .map(|(original, materialized)| materialized.as_ref().unwrap_or(*original))
        .collect::<Vec<_>>();
    let source_handle = match &x {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let protected = rest
        .iter()
        .copied()
        .filter_map(|value| {
            if let Value::GpuTensor(handle) = value {
                Some(handle)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let _guards = source_handle
        .iter()
        .chain(protected.iter().copied())
        .map(runmat_accelerate_api::ResidencyRestoreGuard::new)
        .collect::<Vec<_>>();
    let parsed = if let Some(handle) = source_handle.as_ref() {
        super::resident_output::owner(handle, || pwelch_error(&PWELCH_ERROR_INTERNAL))?;
        let (rows, cols) = gpu_matrix_shape(BUILTIN_NAME, "x", handle)
            .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INVALID_SIGNAL, err.message()))?;
        let complex_input = runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
        let options = parse_options(
            &controls,
            tokens,
            rows,
            complex_input,
            &mut prepared_controls,
            cooperation,
        )
        .await?;
        if !named.wants_plot() {
            if let Some(value) =
                output_gpu(handle, rows, cols, &options, &protected, cooperation).await?
            {
                return Ok(value);
            }
        }
        Some(options)
    } else {
        None
    };
    drop(_preflight_guards);
    drop(resident_inputs);
    let input = value_to_signal_columns(x, prepared_signal_payload, cooperation).await?;
    if input.rows == 0 || input.cols == 0 {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_SIGNAL));
    }
    let options = match parsed {
        Some(options) => options,
        None => {
            parse_options(
                &controls,
                tokens,
                input.rows,
                input.is_complex,
                &mut prepared_controls,
                cooperation,
            )
            .await?
        }
    };
    let eval = compute_pwelch(&input.columns, &options, cooperation).await?;
    #[cfg(feature = "plot-core")]
    let prepared_plot = if named.wants_plot() {
        Some(prepare_pwelch_plot(&eval, &options, named.parent, cooperation).await?)
    } else {
        None
    };
    let output = output_eval(eval, input.output_precision)?;
    let output = match source_handle.as_ref() {
        Some(source) => super::resident_output::restore(source, output, BUILTIN_NAME, &protected),
        None => Ok(output),
    }?;
    #[cfg(feature = "plot-core")]
    if let Some(plot) = prepared_plot {
        publish_pwelch_plot(plot)?;
    }
    Ok(output)
}

fn ensure_pwelch_class_boundary(
    value: &Value,
    extension: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    use crate::builtins::common::validation::{
        value_has_logical_class, value_has_native_integer_class,
    };
    if value_has_logical_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PWELCH_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)?;
    }
    Ok(())
}

async fn ensure_pwelch_integer_boundary(
    value: &Value,
    role: &str,
    error: &'static BuiltinErrorDescriptor,
    extension: &'static BuiltinExtensionDescriptor,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<crate::builtins::common::validation::PreparedIntegerPayload>> {
    let prepared = cooperation
        .prepare_integer_receipt(value, extension)
        .await?;
    if !prepared.is_exact() {
        return Err(pwelch_error_with_detail(
            error,
            format!("integer {role} values must be exactly representable as double"),
        ));
    }
    Ok(prepared.into_resident_payload())
}

fn output_eval(eval: PwelchEvaluation, precision: SignalPrecision) -> BuiltinResult<Value> {
    if statement_form_call() {
        return Ok(Value::output_list(Vec::new()));
    }
    let freq_len = eval.f.len();
    let pxx = precision
        .real(eval.pxx, vec![freq_len, eval.columns])
        .map(Value::Tensor)
        .map_err(|e| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, e))?;
    let f = precision
        .real(eval.f, vec![freq_len, 1])
        .map(Value::Tensor)
        .map_err(|e| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, e))?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 1 {
            return Ok(Value::output_list(vec![pxx]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![pxx, f],
        ));
    }
    Ok(pxx)
}

mod resident;
use resident::output_gpu;

mod signal;
use signal::value_to_signal_columns;

mod controls;
use controls::*;

mod named;
use named::*;

#[cfg(feature = "plot-core")]
mod presentation;
#[cfg(feature = "plot-core")]
use presentation::{prepare_pwelch_plot, publish_pwelch_plot};

mod numerical;
use numerical::*;

mod scaling;
use scaling::normalized_window;

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => tensor_utils::tensor_element_len(t) == 0,
        Value::ComplexTensor(t) => tensor_utils::complex_tensor_element_len(t) == 0,
        _ => false,
    }
}

#[cfg(test)]
mod tests;
