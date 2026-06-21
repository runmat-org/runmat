//! MATLAB-compatible periodogram power spectral density estimate.

use num_complex::Complex;
use runmat_accelerate_api::{
    ProviderSpectralFrameMode, ProviderSpectralRange, ProviderSpectralRequest,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use rustfft::FftPlanner;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::signal::common::{
    centered_frequency_offset, centered_shift, gpu_matrix_shape, parse_nonnegative_integer,
    parse_scalar_f64, value_to_complex_vector,
};
use crate::builtins::math::signal::type_resolvers::periodogram_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "periodogram";
const DEFAULT_NFFT_MIN: usize = 256;
const MAX_NFFT: usize = 1 << 22;
const EPS: f64 = 1.0e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::periodogram")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "periodogram",
    op_kind: GpuOpKind::Custom("periodogram-psd"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("uniform_spectral_estimate")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uniform-grid periodograms window, FFT, select, and power-scale on GPU for resident inputs; explicit-frequency forms fall back to host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::periodogram")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "periodogram",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "periodogram materialises PSD and frequency vectors and is not fused.",
};

const PERIODOGRAM_OUTPUT_PXX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pxx",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Power spectral density or power spectrum estimate.",
}];

const PERIODOGRAM_OUTPUT_PXX_F: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "pxx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Power spectral density or power spectrum estimate.",
    },
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Frequency vector in radians/sample or Hz when fs is supplied.",
    },
];

const PERIODOGRAM_INPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal vector or matrix with one signal per column.",
    },
    BuiltinParamDescriptor {
        name: "window",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Window vector. [] uses a rectangular window matching the signal length.",
    },
    BuiltinParamDescriptor {
        name: "nfft",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "DFT length or explicit frequency vector.",
    },
    BuiltinParamDescriptor {
        name: "fs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Sampling frequency in Hz.",
    },
    BuiltinParamDescriptor {
        name: "freqrange",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Frequency range selector (`onesided`, `twosided`, or `centered`) and spectrum type (`psd` or `power`).",
    },
];

const PERIODOGRAM_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pxx = periodogram(x, window, nfft, fs, freqrange)",
        inputs: &PERIODOGRAM_INPUTS,
        outputs: &PERIODOGRAM_OUTPUT_PXX,
    },
    BuiltinSignatureDescriptor {
        label: "[pxx, f] = periodogram(x, window, nfft, fs, freqrange)",
        inputs: &PERIODOGRAM_INPUTS,
        outputs: &PERIODOGRAM_OUTPUT_PXX_F,
    },
];

const PERIODOGRAM_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.ARG_COUNT",
    identifier: Some("RunMat:periodogram:ArgCount"),
    when: "The argument count is outside supported forms.",
    message:
        "periodogram: expected periodogram(x, [window, [nfft_or_f, [fs, [freqrange_or_scale]]]])",
};

const PERIODOGRAM_ERROR_INVALID_SIGNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INVALID_SIGNAL",
    identifier: Some("RunMat:periodogram:InvalidSignal"),
    when: "Input signal is not a numeric vector or matrix.",
    message: "periodogram: x must be a nonempty numeric vector or 2-D matrix",
};

const PERIODOGRAM_ERROR_INVALID_WINDOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INVALID_WINDOW",
    identifier: Some("RunMat:periodogram:InvalidWindow"),
    when: "Window input is invalid.",
    message: "periodogram: window must be a finite real vector matching the signal length",
};

const PERIODOGRAM_ERROR_INVALID_NFFT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INVALID_NFFT",
    identifier: Some("RunMat:periodogram:InvalidNfft"),
    when: "DFT length or frequency vector is invalid.",
    message: "periodogram: nfft must be a positive integer or finite real frequency vector",
};

const PERIODOGRAM_ERROR_INVALID_FS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INVALID_FS",
    identifier: Some("RunMat:periodogram:InvalidFs"),
    when: "Sampling frequency is invalid.",
    message: "periodogram: fs must be a positive finite scalar",
};

const PERIODOGRAM_ERROR_INVALID_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INVALID_RANGE",
    identifier: Some("RunMat:periodogram:InvalidRange"),
    when: "Frequency range selector or spectrum type is invalid.",
    message: "periodogram: freqrange must be 'onesided', 'twosided', or 'centered'; spectrum type must be 'psd' or 'power'",
};

const PERIODOGRAM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERIODOGRAM.INTERNAL",
    identifier: Some("RunMat:periodogram:Internal"),
    when: "Output tensor construction fails internally.",
    message: "periodogram: internal error",
};

const PERIODOGRAM_ERRORS: [BuiltinErrorDescriptor; 7] = [
    PERIODOGRAM_ERROR_ARG_COUNT,
    PERIODOGRAM_ERROR_INVALID_SIGNAL,
    PERIODOGRAM_ERROR_INVALID_WINDOW,
    PERIODOGRAM_ERROR_INVALID_NFFT,
    PERIODOGRAM_ERROR_INVALID_FS,
    PERIODOGRAM_ERROR_INVALID_RANGE,
    PERIODOGRAM_ERROR_INTERNAL,
];

pub const PERIODOGRAM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PERIODOGRAM_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PERIODOGRAM_ERRORS,
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

#[derive(Clone, Debug)]
enum FrequencyGrid {
    Uniform { nfft: usize },
    Explicit { frequencies: Vec<f64> },
}

#[derive(Clone, Debug)]
struct SignalColumns {
    columns: Vec<Vec<Complex<f64>>>,
    rows: usize,
    cols: usize,
    is_complex: bool,
}

#[derive(Clone, Debug)]
struct PeriodogramOptions {
    window: Vec<f64>,
    grid: FrequencyGrid,
    frequency_units: FrequencyUnits,
    range: FrequencyRange,
    scale: SpectrumScale,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FrequencyUnits {
    NormalizedRadians,
    Hz(f64),
}

#[derive(Clone, Debug)]
struct PeriodogramEvaluation {
    pxx: Vec<f64>,
    f: Vec<f64>,
    columns: usize,
}

fn periodogram_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    periodogram_error_with_message(error.message, error)
}

fn periodogram_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    periodogram_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn periodogram_error_with_message(
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
    name = "periodogram",
    category = "math/signal",
    summary = "Estimate power spectral density with a periodogram.",
    keywords = "periodogram,psd,power spectrum,spectral density,signal processing",
    type_resolver(periodogram_type),
    descriptor(crate::builtins::math::signal::periodogram::PERIODOGRAM_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::periodogram"
)]
async fn periodogram_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(x, &rest).await
}

pub async fn evaluate(x: Value, rest: &[Value]) -> BuiltinResult<Value> {
    if rest.len() > 6 {
        return Err(periodogram_error(&PERIODOGRAM_ERROR_ARG_COUNT));
    }
    if let Value::GpuTensor(handle) = &x {
        let (rows, cols) = gpu_matrix_shape(BUILTIN_NAME, "x", handle).map_err(|err| {
            periodogram_error_with_detail(&PERIODOGRAM_ERROR_INVALID_SIGNAL, err.message())
        })?;
        let complex_input = runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
        let options = parse_options(rest, rows, complex_input).await?;
        if let Ok(value) = output_gpu(handle, rows, cols, &options).await {
            return Ok(value);
        }
    }
    let input = value_to_signal_columns(x).await?;
    if input.rows == 0 || input.cols == 0 {
        return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_SIGNAL));
    }
    let options = parse_options(rest, input.rows, input.is_complex).await?;
    let eval = compute_periodogram(&input.columns, &options)?;
    output_eval(eval)
}

fn output_eval(eval: PeriodogramEvaluation) -> BuiltinResult<Value> {
    let freq_len = eval.f.len();
    let pxx = Tensor::new(eval.pxx, vec![freq_len, eval.columns])
        .map(Value::Tensor)
        .map_err(|e| periodogram_error_with_detail(&PERIODOGRAM_ERROR_INTERNAL, e))?;
    let f = Tensor::new(eval.f, vec![freq_len, 1])
        .map(Value::Tensor)
        .map_err(|e| periodogram_error_with_detail(&PERIODOGRAM_ERROR_INTERNAL, e))?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![pxx]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![pxx, f],
        ));
    }
    Ok(pxx)
}

async fn output_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    rows: usize,
    cols: usize,
    options: &PeriodogramOptions,
) -> BuiltinResult<Value> {
    let FrequencyGrid::Uniform { nfft } = options.grid else {
        return Err(periodogram_error(&PERIODOGRAM_ERROR_INTERNAL));
    };
    let window_energy = options.window.iter().map(|v| v * v).sum::<f64>();
    let coherent_gain = options.window.iter().sum::<f64>();
    if window_energy <= 0.0 || !window_energy.is_finite() {
        return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_WINDOW));
    }
    let denominator = match options.scale {
        SpectrumScale::Psd => frequency_scale(options.frequency_units) * window_energy,
        SpectrumScale::Power => {
            let gain_sq = coherent_gain * coherent_gain;
            if gain_sq <= EPS || !gain_sq.is_finite() {
                return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_WINDOW));
            }
            gain_sq
        }
    };
    let input_len = rows
        .checked_mul(cols)
        .ok_or_else(|| periodogram_error(&PERIODOGRAM_ERROR_INTERNAL))?;
    let estimate = runmat_accelerate_api::uniform_spectral_estimate(ProviderSpectralRequest {
        input: handle,
        input_len,
        input_complex: runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
        window: &options.window,
        nfft,
        frame_count: cols,
        frame_mode: ProviderSpectralFrameMode::FoldedColumns { input_rows: rows },
        range: gpu_range(options.range),
        denominator,
    })
    .await
    .map_err(|err| periodogram_error_with_detail(&PERIODOGRAM_ERROR_INTERNAL, err.to_string()))?;
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(periodogram_error(&PERIODOGRAM_ERROR_INTERNAL));
    };
    let f_values = frequency_vector(nfft, options.frequency_units, options.range);
    let f_shape = [estimate.rows, 1usize];
    let f = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &f_values,
            shape: &f_shape,
        })
        .map_err(|e| periodogram_error_with_detail(&PERIODOGRAM_ERROR_INTERNAL, e.to_string()))?;
    let pxx = crate::builtins::common::gpu_helpers::resident_gpu_value(estimate.ps);
    let f = crate::builtins::common::gpu_helpers::resident_gpu_value(f);
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![pxx]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![pxx, f],
        ));
    }
    Ok(pxx)
}

async fn value_to_signal_columns(value: Value) -> BuiltinResult<SignalColumns> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = crate::builtins::common::gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| {
                    periodogram_error_with_detail(&PERIODOGRAM_ERROR_INVALID_SIGNAL, err.message())
                })?;
            tensor_to_signal_columns(tensor)
        }
        Value::Tensor(tensor) => tensor_to_signal_columns(tensor),
        Value::ComplexTensor(tensor) => complex_tensor_to_signal_columns(tensor),
        Value::LogicalArray(logical) => {
            let tensor =
                crate::builtins::common::tensor::logical_to_tensor(&logical).map_err(|err| {
                    periodogram_error_with_detail(&PERIODOGRAM_ERROR_INVALID_SIGNAL, err)
                })?;
            tensor_to_signal_columns(tensor)
        }
        Value::Num(n) => Ok(SignalColumns {
            columns: vec![vec![Complex::new(n, 0.0)]],
            rows: 1,
            cols: 1,
            is_complex: false,
        }),
        Value::Int(i) => Ok(SignalColumns {
            columns: vec![vec![Complex::new(i.to_f64(), 0.0)]],
            rows: 1,
            cols: 1,
            is_complex: false,
        }),
        Value::Bool(b) => Ok(SignalColumns {
            columns: vec![vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)]],
            rows: 1,
            cols: 1,
            is_complex: false,
        }),
        Value::Complex(re, im) => Ok(SignalColumns {
            columns: vec![vec![Complex::new(re, im)]],
            rows: 1,
            cols: 1,
            is_complex: im.abs() > EPS,
        }),
        other => Err(periodogram_error_with_detail(
            &PERIODOGRAM_ERROR_INVALID_SIGNAL,
            format!("unsupported input type {other:?}"),
        )),
    }
}

fn tensor_to_signal_columns(tensor: Tensor) -> BuiltinResult<SignalColumns> {
    if tensor.shape.len() > 2 {
        return Err(periodogram_error_with_detail(
            &PERIODOGRAM_ERROR_INVALID_SIGNAL,
            "x must be a vector or 2-D matrix",
        ));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    if rows == 1 || cols == 1 {
        return Ok(SignalColumns {
            columns: vec![tensor
                .data
                .into_iter()
                .map(|value| Complex::new(value, 0.0))
                .collect()],
            rows: rows.max(cols),
            cols: 1,
            is_complex: false,
        });
    }
    let mut columns = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut column = Vec::with_capacity(rows);
        for row in 0..rows {
            column.push(Complex::new(tensor.data[row + col * rows], 0.0));
        }
        columns.push(column);
    }
    Ok(SignalColumns {
        columns,
        rows,
        cols,
        is_complex: false,
    })
}

fn complex_tensor_to_signal_columns(
    tensor: runmat_builtins::ComplexTensor,
) -> BuiltinResult<SignalColumns> {
    if tensor.shape.len() > 2 {
        return Err(periodogram_error_with_detail(
            &PERIODOGRAM_ERROR_INVALID_SIGNAL,
            "x must be a vector or 2-D matrix",
        ));
    }
    let rows = tensor.rows;
    let cols = tensor.cols;
    let is_complex = tensor.data.iter().any(|(_, im)| im.abs() > EPS);
    if rows == 1 || cols == 1 {
        return Ok(SignalColumns {
            columns: vec![tensor
                .data
                .into_iter()
                .map(|(re, im)| Complex::new(re, im))
                .collect()],
            rows: rows.max(cols),
            cols: 1,
            is_complex,
        });
    }
    let mut columns = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut column = Vec::with_capacity(rows);
        for row in 0..rows {
            let (re, im) = tensor.data[row + col * rows];
            column.push(Complex::new(re, im));
        }
        columns.push(column);
    }
    Ok(SignalColumns {
        columns,
        rows,
        cols,
        is_complex,
    })
}

async fn parse_options(
    rest: &[Value],
    signal_len: usize,
    complex_input: bool,
) -> BuiltinResult<PeriodogramOptions> {
    let mut idx = 0usize;
    let window = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            vec![1.0; signal_len]
        } else if keyword(value).is_some() {
            vec![1.0; signal_len]
        } else {
            idx += 1;
            parse_window(value.clone(), signal_len).await?
        }
    } else {
        vec![1.0; signal_len]
    };
    let window_len = window.len();

    let grid = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            FrequencyGrid::Uniform {
                nfft: default_nfft(window_len)?,
            }
        } else if keyword(value).is_some() {
            FrequencyGrid::Uniform {
                nfft: default_nfft(window_len)?,
            }
        } else {
            idx += 1;
            parse_frequency_grid(value.clone()).await?
        }
    } else {
        FrequencyGrid::Uniform {
            nfft: default_nfft(window_len)?,
        }
    };

    let frequency_units = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            FrequencyUnits::Hz(1.0)
        } else if keyword(value).is_some() {
            FrequencyUnits::NormalizedRadians
        } else {
            idx += 1;
            let parsed = parse_scalar_f64(BUILTIN_NAME, "fs", value).map_err(|err| {
                periodogram_error_with_detail(&PERIODOGRAM_ERROR_INVALID_FS, err.message())
            })?;
            if parsed <= 0.0 {
                return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_FS));
            }
            FrequencyUnits::Hz(parsed)
        }
    } else {
        FrequencyUnits::NormalizedRadians
    };

    let default_range = if complex_input {
        FrequencyRange::Twosided
    } else {
        FrequencyRange::Onesided
    };
    let mut range = default_range;
    let mut scale = SpectrumScale::Psd;
    while let Some(value) = rest.get(idx) {
        let Some(keyword) = keyword(value) else {
            return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_RANGE));
        };
        idx += 1;
        match keyword.as_str() {
            "onesided" | "twosided" | "centered"
                if matches!(grid, FrequencyGrid::Explicit { .. }) =>
            {
                return Err(periodogram_error_with_detail(
                    &PERIODOGRAM_ERROR_INVALID_RANGE,
                    "range selectors cannot be combined with an explicit frequency vector",
                ));
            }
            "onesided" if complex_input => {
                return Err(periodogram_error_with_detail(
                    &PERIODOGRAM_ERROR_INVALID_RANGE,
                    "onesided spectra are only supported for real-valued signals",
                ));
            }
            "onesided" => range = FrequencyRange::Onesided,
            "twosided" => range = FrequencyRange::Twosided,
            "centered" => range = FrequencyRange::Centered,
            "psd" => scale = SpectrumScale::Psd,
            "power" => scale = SpectrumScale::Power,
            _ => return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_RANGE)),
        }
    }

    Ok(PeriodogramOptions {
        window,
        grid,
        frequency_units,
        range,
        scale,
    })
}

async fn parse_window(value: Value, signal_len: usize) -> BuiltinResult<Vec<f64>> {
    let vector = value_to_complex_vector(BUILTIN_NAME, "window", value)
        .await
        .map_err(|err| {
            periodogram_error_with_detail(&PERIODOGRAM_ERROR_INVALID_WINDOW, err.message())
        })?;
    if vector.data.len() != signal_len {
        return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_WINDOW));
    }
    let mut out = Vec::with_capacity(vector.data.len());
    for value in vector.data {
        if value.im.abs() > EPS || !value.re.is_finite() {
            return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_WINDOW));
        }
        out.push(value.re);
    }
    Ok(out)
}

async fn parse_frequency_grid(value: Value) -> BuiltinResult<FrequencyGrid> {
    if is_scalar_numeric(&value) {
        let parsed = parse_nonnegative_integer(BUILTIN_NAME, "nfft", &value).map_err(|err| {
            periodogram_error_with_detail(&PERIODOGRAM_ERROR_INVALID_NFFT, err.message())
        })?;
        if parsed == 0 || parsed > MAX_NFFT {
            return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_NFFT));
        }
        return Ok(FrequencyGrid::Uniform { nfft: parsed });
    }

    let vector = value_to_complex_vector(BUILTIN_NAME, "nfft", value)
        .await
        .map_err(|err| {
            periodogram_error_with_detail(&PERIODOGRAM_ERROR_INVALID_NFFT, err.message())
        })?;
    if vector.data.is_empty() || vector.data.len() > MAX_NFFT {
        return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_NFFT));
    }
    let mut frequencies = Vec::with_capacity(vector.data.len());
    for value in vector.data {
        if value.im.abs() > EPS || !value.re.is_finite() {
            return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_NFFT));
        }
        frequencies.push(value.re);
    }
    Ok(FrequencyGrid::Explicit { frequencies })
}

fn is_scalar_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor.data.len() == 1,
        Value::ComplexTensor(tensor) => tensor.data.len() == 1,
        Value::LogicalArray(logical) => logical.data.len() == 1,
        _ => false,
    }
}

fn default_nfft(window_len: usize) -> BuiltinResult<usize> {
    let next_pow2 = window_len
        .checked_next_power_of_two()
        .ok_or_else(|| periodogram_error(&PERIODOGRAM_ERROR_INVALID_NFFT))?;
    let nfft = DEFAULT_NFFT_MIN.max(next_pow2);
    if nfft > MAX_NFFT {
        return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_NFFT));
    }
    Ok(nfft)
}

fn compute_periodogram(
    columns: &[Vec<Complex<f64>>],
    options: &PeriodogramOptions,
) -> BuiltinResult<PeriodogramEvaluation> {
    let window_energy = options.window.iter().map(|v| v * v).sum::<f64>();
    let coherent_gain = options.window.iter().sum::<f64>();
    if window_energy <= 0.0 || !window_energy.is_finite() {
        return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_WINDOW));
    }
    let denominator = match options.scale {
        SpectrumScale::Psd => frequency_scale(options.frequency_units) * window_energy,
        SpectrumScale::Power => {
            let gain_sq = coherent_gain * coherent_gain;
            if gain_sq <= EPS || !gain_sq.is_finite() {
                return Err(periodogram_error(&PERIODOGRAM_ERROR_INVALID_WINDOW));
            }
            gain_sq
        }
    };

    let mut planner = FftPlanner::<f64>::new();
    let mut fft_plan = match &options.grid {
        FrequencyGrid::Uniform { nfft } => Some((
            planner.plan_fft_forward(*nfft),
            vec![Complex::new(0.0, 0.0); *nfft],
        )),
        FrequencyGrid::Explicit { .. } => None,
    };
    let mut selected_columns = Vec::new();
    let mut selected_f = Vec::new();

    for column in columns {
        let selected = match (&options.grid, &mut fft_plan) {
            (FrequencyGrid::Uniform { nfft }, Some((fft, buffer))) => {
                let power = uniform_periodogram(column, &options.window, denominator, fft, buffer);
                select_range(power, *nfft, options.frequency_units, options.range)
            }
            (FrequencyGrid::Explicit { frequencies }, None) => explicit_frequency_periodogram(
                column,
                &options.window,
                denominator,
                frequencies,
                options.frequency_units,
            ),
            _ => return Err(periodogram_error(&PERIODOGRAM_ERROR_INTERNAL)),
        };
        if selected_f.is_empty() {
            selected_f = selected.f;
        }
        selected_columns.push(selected.pxx);
    }

    let rows = selected_f.len();
    let cols = selected_columns.len();
    let mut pxx = Vec::with_capacity(rows * cols);
    for column in selected_columns {
        pxx.extend(column);
    }
    Ok(PeriodogramEvaluation {
        pxx,
        f: selected_f,
        columns: cols,
    })
}

fn uniform_periodogram(
    column: &[Complex<f64>],
    window: &[f64],
    denominator: f64,
    fft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    buffer: &mut [Complex<f64>],
) -> Vec<f64> {
    buffer.fill(Complex::new(0.0, 0.0));
    for idx in 0..window.len() {
        let sample = column
            .get(idx)
            .copied()
            .unwrap_or_else(|| Complex::new(0.0, 0.0));
        let slot = idx % buffer.len();
        buffer[slot] += sample * window[idx];
    }
    fft.process(buffer);
    buffer
        .iter()
        .map(|value| value.norm_sqr() / denominator)
        .collect()
}

fn explicit_frequency_periodogram(
    column: &[Complex<f64>],
    window: &[f64],
    denominator: f64,
    frequencies: &[f64],
    frequency_units: FrequencyUnits,
) -> PeriodogramEvaluation {
    let mut pxx = Vec::with_capacity(frequencies.len());
    for &frequency in frequencies {
        let omega = angular_frequency(frequency, frequency_units);
        let mut sum = Complex::new(0.0, 0.0);
        for idx in 0..window.len() {
            let sample = column
                .get(idx)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            let phase = -omega * idx as f64;
            let twiddle = Complex::new(phase.cos(), phase.sin());
            sum += sample * window[idx] * twiddle;
        }
        pxx.push(sum.norm_sqr() / denominator);
    }
    PeriodogramEvaluation {
        pxx,
        f: frequencies.to_vec(),
        columns: 1,
    }
}

fn select_range(
    mut power: Vec<f64>,
    nfft: usize,
    frequency_units: FrequencyUnits,
    range: FrequencyRange,
) -> PeriodogramEvaluation {
    let freq_scale = frequency_scale(frequency_units);
    match range {
        FrequencyRange::Twosided => {
            let f = (0..nfft)
                .map(|idx| freq_scale * idx as f64 / nfft as f64)
                .collect();
            PeriodogramEvaluation {
                pxx: power,
                f,
                columns: 1,
            }
        }
        FrequencyRange::Centered => {
            let shift = centered_shift(nfft);
            power.rotate_left(shift);
            let offset = centered_frequency_offset(nfft);
            let f = (0..nfft)
                .map(|idx| freq_scale * (idx as isize - offset) as f64 / nfft as f64)
                .collect();
            PeriodogramEvaluation {
                pxx: power,
                f,
                columns: 1,
            }
        }
        FrequencyRange::Onesided => {
            let len = nfft / 2 + 1;
            let even = nfft.is_multiple_of(2);
            let mut pxx = power[..len].to_vec();
            for (idx, value) in pxx.iter_mut().enumerate() {
                let is_dc = idx == 0;
                let is_nyquist = even && idx == len - 1;
                if !is_dc && !is_nyquist {
                    *value *= 2.0;
                }
            }
            let f = (0..len)
                .map(|idx| freq_scale * idx as f64 / nfft as f64)
                .collect();
            PeriodogramEvaluation { pxx, f, columns: 1 }
        }
    }
}

fn frequency_scale(units: FrequencyUnits) -> f64 {
    match units {
        FrequencyUnits::NormalizedRadians => 2.0 * std::f64::consts::PI,
        FrequencyUnits::Hz(fs) => fs,
    }
}

fn angular_frequency(frequency: f64, units: FrequencyUnits) -> f64 {
    match units {
        FrequencyUnits::NormalizedRadians => frequency,
        FrequencyUnits::Hz(fs) => 2.0 * std::f64::consts::PI * frequency / fs,
    }
}

fn gpu_range(range: FrequencyRange) -> ProviderSpectralRange {
    match range {
        FrequencyRange::Onesided => ProviderSpectralRange::Onesided,
        FrequencyRange::Twosided => ProviderSpectralRange::Twosided,
        FrequencyRange::Centered => ProviderSpectralRange::Centered,
    }
}

fn frequency_vector(nfft: usize, units: FrequencyUnits, range: FrequencyRange) -> Vec<f64> {
    let freq_scale = frequency_scale(units);
    match range {
        FrequencyRange::Twosided => (0..nfft)
            .map(|idx| freq_scale * idx as f64 / nfft as f64)
            .collect(),
        FrequencyRange::Centered => {
            let offset = centered_frequency_offset(nfft);
            (0..nfft)
                .map(|idx| freq_scale * (idx as isize - offset) as f64 / nfft as f64)
                .collect()
        }
        FrequencyRange::Onesided => {
            let len = nfft / 2 + 1;
            (0..len)
                .map(|idx| freq_scale * idx as f64 / nfft as f64)
                .collect()
        }
    }
}

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => t.data.is_empty(),
        Value::ComplexTensor(t) => t.data.is_empty(),
        _ => false,
    }
}

fn keyword(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.trim().to_ascii_lowercase()),
        Value::StringArray(array) if array.data.len() == 1 => {
            Some(array.data[0].trim().to_ascii_lowercase())
        }
        Value::CharArray(chars) if chars.rows <= 1 => Some(
            chars
                .data
                .iter()
                .collect::<String>()
                .trim()
                .to_ascii_lowercase(),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_builtins::builtin_function_by_name;

    fn call(x: Value, rest: &[Value], outputs: Option<usize>) -> BuiltinResult<Value> {
        let _guard = outputs.map(|count| crate::output_count::push_output_count(Some(count)));
        block_on(evaluate(x, rest))
    }

    fn output_pair(value: Value) -> (Vec<f64>, Vec<f64>) {
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        let Value::Tensor(pxx) = &values[0] else {
            panic!("expected pxx tensor");
        };
        let Value::Tensor(f) = &values[1] else {
            panic!("expected f tensor");
        };
        (pxx.data.clone(), f.data.clone())
    }

    fn output_pair_with_pxx_shape(value: Value) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        let Value::Tensor(pxx) = &values[0] else {
            panic!("expected pxx tensor");
        };
        let Value::Tensor(f) = &values[1] else {
            panic!("expected f tensor");
        };
        (pxx.data.clone(), pxx.shape.clone(), f.data.clone())
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("periodogram builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| { sig.label == "[pxx, f] = periodogram(x, window, nfft, fs, freqrange)" }));
    }

    #[test]
    fn periodogram_detects_sinusoid_bin() {
        let x = (0..32)
            .map(|idx| (2.0 * std::f64::consts::PI * 4.0 * idx as f64 / 32.0).sin())
            .collect::<Vec<_>>();
        let out = call(
            Value::Tensor(Tensor::new(x, vec![1, 32]).unwrap()),
            &[
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(32.0),
                Value::Num(32.0),
            ],
            Some(2),
        )
        .unwrap();
        let (pxx, f) = output_pair(out);
        let (peak_idx, _) = pxx
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        assert_eq!(peak_idx, 4);
        assert_eq!(f[peak_idx], 4.0);
    }

    #[test]
    fn periodogram_power_reports_mean_square_sinusoid_power() {
        let amp = 1.8;
        let x = (0..32)
            .map(|idx| amp * (2.0 * std::f64::consts::PI * 4.0 * idx as f64 / 32.0).cos())
            .collect::<Vec<_>>();
        let out = call(
            Value::Tensor(Tensor::new(x, vec![1, 32]).unwrap()),
            &[
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(32.0),
                Value::Num(32.0),
                Value::from("power"),
            ],
            Some(2),
        )
        .unwrap();
        let (pxx, _) = output_pair(out);
        let peak = pxx.iter().copied().fold(0.0, f64::max);
        assert!((peak - amp * amp / 2.0).abs() < 1e-12);
    }

    #[test]
    fn periodogram_accepts_matrix_columns() {
        let data = vec![
            1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0,
        ];
        let out = call(
            Value::Tensor(Tensor::new(data, vec![8, 2]).unwrap()),
            &[
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(8.0),
            ],
            Some(2),
        )
        .unwrap();
        let (pxx, shape, f) = output_pair_with_pxx_shape(out);
        assert_eq!(shape, vec![5, 2]);
        assert_eq!(pxx.len(), 10);
        assert_eq!(f.len(), 5);
        assert!(pxx[0] > pxx[5]);
    }

    #[test]
    fn periodogram_accepts_explicit_frequency_vector() {
        let x = Tensor::new(vec![1.0; 4], vec![1, 4]).unwrap();
        let frequencies = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let out = call(
            Value::Tensor(x),
            &[
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Tensor(frequencies),
            ],
            Some(2),
        )
        .unwrap();
        let (pxx, f) = output_pair(out);
        assert_eq!(f, vec![0.0, std::f64::consts::PI]);
        assert_eq!(pxx.len(), 2);
        assert!(pxx[0] > 0.5);
        assert!(pxx[1].abs() < 1e-12);
    }

    #[test]
    fn periodogram_centered_odd_nfft_uses_fftshift_order() {
        let data = (0..5)
            .map(|idx| (idx as f64, if idx == 0 { 0.0 } else { 0.25 }))
            .collect::<Vec<_>>();
        let x = runmat_builtins::ComplexTensor::new(data, vec![1, 5]).unwrap();
        let out = call(
            Value::ComplexTensor(x),
            &[
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(5.0),
                Value::from("centered"),
            ],
            Some(2),
        )
        .unwrap();
        let (_, f) = output_pair(out);
        assert_eq!(f.len(), 5);
        assert!((f[0] + 4.0 * std::f64::consts::PI / 5.0).abs() < 1e-12);
        assert_eq!(f[2], 0.0);
        assert!((f[4] - 4.0 * std::f64::consts::PI / 5.0).abs() < 1e-12);
    }

    #[test]
    fn periodogram_centered_even_nfft_uses_matlab_interval() {
        let x = Tensor::new(vec![1.0, 0.0, 0.0, 0.0], vec![1, 4]).unwrap();
        let out = call(
            Value::Tensor(x),
            &[
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(4.0),
                Value::Num(4.0),
                Value::from("centered"),
            ],
            Some(2),
        )
        .unwrap();
        let (_, f) = output_pair(out);
        assert_eq!(f, vec![-1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn periodogram_rejects_wrong_window_length() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let window = Tensor::new(vec![1.0; 4], vec![1, 4]).unwrap();
        let err = call(Value::Tensor(x), &[Value::Tensor(window)], Some(2)).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:periodogram:InvalidWindow"));
    }

    #[test]
    fn periodogram_folds_signal_when_nfft_is_shorter_than_window() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let out = call(
            Value::Tensor(x),
            &[
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(4.0),
            ],
            Some(2),
        )
        .unwrap();
        let (pxx, f) = output_pair(out);
        assert_eq!(pxx.len(), 3);
        assert_eq!(f.len(), 3);
        assert!(pxx[0] > 1.0);
        assert!(pxx[1..].iter().all(|value| value.abs() < 1e-12));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn periodogram_wgpu_keeps_uniform_outputs_resident() {
        let Some(provider) = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
            .expect("wgpu provider")
        else {
            return;
        };
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let data = (0..32)
            .map(|idx| (2.0 * std::f64::consts::PI * 4.0 * idx as f64 / 32.0).sin())
            .collect::<Vec<_>>();
        let shape = [1usize, 32usize];
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload");
        let out = call(
            Value::GpuTensor(handle.clone()),
            &[
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(32.0),
                Value::Num(32.0),
            ],
            Some(2),
        )
        .expect("periodogram gpu");
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        for value in &values {
            assert!(matches!(value, Value::GpuTensor(_)));
        }
        let pxx =
            crate::builtins::common::test_support::gather(values[0].clone()).expect("gather pxx");
        let f = crate::builtins::common::test_support::gather(values[1].clone()).expect("gather f");
        assert_eq!(pxx.shape, vec![17, 1]);
        assert_eq!(f.data[4], 4.0);
        let peak = pxx
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(peak, 4);
        provider.free(&handle).ok();
    }

    #[test]
    fn periodogram_explicit_empty_fs_uses_one_hz_units() {
        let x = Tensor::new(vec![1.0; 4], vec![1, 4]).unwrap();
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        let out = call(
            Value::Tensor(x),
            &[empty.clone(), empty.clone(), empty],
            Some(2),
        )
        .unwrap();
        let (pxx, f) = output_pair(out);
        assert!((pxx[0] - 4.0).abs() < 1e-12);
        assert_eq!(f[0], 0.0);
        assert!((f[1] - 1.0 / 256.0).abs() < 1e-12);
    }

    #[test]
    fn periodogram_rejects_onesided_complex_input() {
        let x = runmat_builtins::ComplexTensor::new(vec![(1.0, 1.0); 8], vec![1, 8]).unwrap();
        let err = call(
            Value::ComplexTensor(x),
            &[
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(8.0),
                Value::from("onesided"),
            ],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:periodogram:InvalidRange"));
    }

    #[test]
    fn default_nfft_uses_next_power_of_two_with_minimum() {
        assert_eq!(default_nfft(4).unwrap(), 256);
        assert_eq!(default_nfft(300).unwrap(), 512);
    }
}
