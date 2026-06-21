//! MATLAB-compatible Welch power spectral density estimate.

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
    gpu_matrix_shape, parse_nonnegative_integer, parse_scalar_f64, value_to_complex_vector,
};
use crate::builtins::math::signal::type_resolvers::pwelch_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "pwelch";
const DEFAULT_NFFT_MIN: usize = 256;
const MAX_NFFT: usize = 1 << 22;
const EPS: f64 = 1.0e-12;

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

const PWELCH_OUTPUT_PXX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pxx",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Power spectral density estimate.",
}];

const PWELCH_OUTPUT_PXX_F: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "pxx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Power spectral density estimate.",
    },
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Frequency vector in radians/sample or Hz when fs is supplied.",
    },
];

const PWELCH_INPUTS: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal vector or matrix with one signal per column.",
    },
    BuiltinParamDescriptor {
        name: "window",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Window vector or scalar segment length.",
    },
    BuiltinParamDescriptor {
        name: "noverlap",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Number of overlapped samples.",
    },
    BuiltinParamDescriptor {
        name: "nfft",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "FFT length or explicit frequency vector.",
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

const PWELCH_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pxx = pwelch(x, window, noverlap, nfft, fs, freqrange)",
        inputs: &PWELCH_INPUTS,
        outputs: &PWELCH_OUTPUT_PXX,
    },
    BuiltinSignatureDescriptor {
        label: "[pxx, f] = pwelch(x, window, noverlap, nfft, fs, freqrange)",
        inputs: &PWELCH_INPUTS,
        outputs: &PWELCH_OUTPUT_PXX_F,
    },
];

const PWELCH_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.ARG_COUNT",
    identifier: Some("RunMat:pwelch:ArgCount"),
    when: "The argument count is outside supported forms.",
    message:
        "pwelch: expected pwelch(x, [window, [noverlap, [nfft_or_f, [fs, [freqrange_or_scale]]]]])",
};

const PWELCH_ERROR_INVALID_SIGNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_SIGNAL",
    identifier: Some("RunMat:pwelch:InvalidSignal"),
    when: "Input signal is not a numeric vector or matrix.",
    message: "pwelch: x must be a nonempty numeric vector or 2-D matrix",
};

const PWELCH_ERROR_INVALID_WINDOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_WINDOW",
    identifier: Some("RunMat:pwelch:InvalidWindow"),
    when: "Window input is invalid.",
    message: "pwelch: window must be a nonempty vector or positive integer length",
};

const PWELCH_ERROR_INVALID_OVERLAP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_OVERLAP",
    identifier: Some("RunMat:pwelch:InvalidOverlap"),
    when: "Overlap is invalid.",
    message: "pwelch: noverlap must be a nonnegative integer less than window length",
};

const PWELCH_ERROR_INVALID_NFFT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_NFFT",
    identifier: Some("RunMat:pwelch:InvalidNfft"),
    when: "FFT length is invalid.",
    message: "pwelch: nfft must be a positive integer at least as large as the window length",
};

const PWELCH_ERROR_INVALID_FS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_FS",
    identifier: Some("RunMat:pwelch:InvalidFs"),
    when: "Sampling frequency is invalid.",
    message: "pwelch: fs must be a positive finite scalar",
};

const PWELCH_ERROR_INVALID_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INVALID_RANGE",
    identifier: Some("RunMat:pwelch:InvalidRange"),
    when: "Frequency range selector is invalid.",
    message: "pwelch: freqrange must be 'onesided', 'twosided', or 'centered'; spectrum type must be 'psd' or 'power'",
};

const PWELCH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PWELCH.INTERNAL",
    identifier: Some("RunMat:pwelch:Internal"),
    when: "Output tensor construction fails internally.",
    message: "pwelch: internal error",
};

const PWELCH_ERRORS: [BuiltinErrorDescriptor; 8] = [
    PWELCH_ERROR_ARG_COUNT,
    PWELCH_ERROR_INVALID_SIGNAL,
    PWELCH_ERROR_INVALID_WINDOW,
    PWELCH_ERROR_INVALID_OVERLAP,
    PWELCH_ERROR_INVALID_NFFT,
    PWELCH_ERROR_INVALID_FS,
    PWELCH_ERROR_INVALID_RANGE,
    PWELCH_ERROR_INTERNAL,
];

pub const PWELCH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PWELCH_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PWELCH_ERRORS,
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
struct SignalColumns {
    columns: Vec<Vec<Complex<f64>>>,
    rows: usize,
    cols: usize,
    is_complex: bool,
}

#[derive(Clone, Debug)]
struct PwelchOptions {
    window: Vec<f64>,
    noverlap: usize,
    grid: FrequencyGrid,
    fs: Option<f64>,
    range: FrequencyRange,
    scale: SpectrumScale,
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
    category = "math/signal",
    summary = "Estimate power spectral density with Welch's method.",
    keywords = "pwelch,welch,periodogram,psd,spectral density,signal processing",
    type_resolver(pwelch_type),
    descriptor(crate::builtins::math::signal::pwelch::PWELCH_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::pwelch"
)]
async fn pwelch_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(x, &rest).await
}

pub async fn evaluate(x: Value, rest: &[Value]) -> BuiltinResult<Value> {
    if rest.len() > 7 {
        return Err(pwelch_error(&PWELCH_ERROR_ARG_COUNT));
    }
    if let Value::GpuTensor(handle) = &x {
        let (rows, cols) = gpu_matrix_shape(BUILTIN_NAME, "x", handle)
            .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INVALID_SIGNAL, err.message()))?;
        let complex_input = runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
        let options = parse_options(rest, rows, complex_input).await?;
        if let Ok(value) = output_gpu(handle, rows, cols, &options).await {
            return Ok(value);
        }
    }
    let input = value_to_signal_columns(x).await?;
    if input.rows == 0 || input.cols == 0 {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_SIGNAL));
    }
    let options = parse_options(rest, input.rows, input.is_complex).await?;
    let eval = compute_pwelch(&input.columns, &options)?;
    output_eval(eval)
}

fn output_eval(eval: PwelchEvaluation) -> BuiltinResult<Value> {
    let freq_len = eval.f.len();
    let pxx = Tensor::new(eval.pxx, vec![freq_len, eval.columns])
        .map(Value::Tensor)
        .map_err(|e| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, e))?;
    let f = Tensor::new(eval.f, vec![freq_len, 1])
        .map(Value::Tensor)
        .map_err(|e| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, e))?;
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
    options: &PwelchOptions,
) -> BuiltinResult<Value> {
    let FrequencyGrid::Uniform { nfft } = options.grid else {
        return Err(pwelch_error(&PWELCH_ERROR_INTERNAL));
    };
    let window_len = options.window.len();
    let step = window_len - options.noverlap;
    let starts = segment_starts(rows, window_len, step);
    let segment_count = starts.len();
    let window_energy = options.window.iter().map(|v| v * v).sum::<f64>();
    if window_energy <= 0.0 || !window_energy.is_finite() {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
    }
    let denominator = match options.scale {
        SpectrumScale::Psd => options.fs.unwrap_or(2.0 * std::f64::consts::PI) * window_energy,
        SpectrumScale::Power => window_energy,
    };
    let input_len = rows
        .checked_mul(cols)
        .ok_or_else(|| pwelch_error(&PWELCH_ERROR_INTERNAL))?;
    let frame_count = segment_count
        .checked_mul(cols)
        .ok_or_else(|| pwelch_error(&PWELCH_ERROR_INTERNAL))?;
    let range = gpu_range(options.range);
    let estimate = runmat_accelerate_api::uniform_spectral_estimate(ProviderSpectralRequest {
        input: handle,
        input_len,
        input_complex: runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
        window: &options.window,
        nfft,
        frame_count,
        frame_mode: ProviderSpectralFrameMode::ColumnSliding {
            hop: step,
            input_rows: rows,
            frames_per_column: segment_count,
        },
        range,
        denominator,
    })
    .await
    .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, err.to_string()))?;
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(pwelch_error(&PWELCH_ERROR_INTERNAL));
    };

    let ps_frames = provider
        .reshape(&estimate.ps, &[estimate.rows, segment_count, cols])
        .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, err.to_string()))?;
    let ps_mean = provider
        .reduce_mean_nd(&ps_frames, &[1])
        .await
        .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, err.to_string()))?;
    let pxx = provider
        .reshape(&ps_mean, &[estimate.rows, cols])
        .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, err.to_string()))?;

    let f_values = frequency_vector(nfft, options.fs, options.range);
    let f_shape = [estimate.rows, 1usize];
    let f = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &f_values,
            shape: &f_shape,
        })
        .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, err.to_string()))?;

    provider.free(&estimate.s).ok();
    provider.free(&ps_frames).ok();

    let pxx = crate::builtins::common::gpu_helpers::resident_gpu_value(pxx);
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
                    pwelch_error_with_detail(&PWELCH_ERROR_INVALID_SIGNAL, err.message())
                })?;
            tensor_to_signal_columns(tensor)
        }
        Value::Tensor(tensor) => tensor_to_signal_columns(tensor),
        Value::ComplexTensor(tensor) => complex_tensor_to_signal_columns(tensor),
        Value::LogicalArray(logical) => {
            let tensor = crate::builtins::common::tensor::logical_to_tensor(&logical)
                .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INVALID_SIGNAL, err))?;
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
        other => Err(pwelch_error_with_detail(
            &PWELCH_ERROR_INVALID_SIGNAL,
            format!("unsupported input type {other:?}"),
        )),
    }
}

fn tensor_to_signal_columns(tensor: Tensor) -> BuiltinResult<SignalColumns> {
    if tensor.shape.len() > 2 {
        return Err(pwelch_error_with_detail(
            &PWELCH_ERROR_INVALID_SIGNAL,
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
        return Err(pwelch_error_with_detail(
            &PWELCH_ERROR_INVALID_SIGNAL,
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
    input_len: usize,
    complex_input: bool,
) -> BuiltinResult<PwelchOptions> {
    let mut idx = 0usize;
    let window = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            default_window(input_len)
        } else if let Some(keyword) = keyword(value) {
            return Err(pwelch_error_with_detail(
                &PWELCH_ERROR_INVALID_WINDOW,
                format!("unexpected option '{keyword}' before window arguments"),
            ));
        } else {
            idx += 1;
            parse_window(value.clone()).await?
        }
    } else {
        default_window(input_len)
    };
    let window_len = window.len();

    let noverlap = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            window_len / 2
        } else if keyword(value).is_some() {
            window_len / 2
        } else {
            idx += 1;
            let parsed =
                parse_nonnegative_integer(BUILTIN_NAME, "noverlap", value).map_err(|err| {
                    pwelch_error_with_detail(&PWELCH_ERROR_INVALID_OVERLAP, err.message())
                })?;
            if parsed >= window_len {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_OVERLAP));
            }
            parsed
        }
    } else {
        window_len / 2
    };

    let grid = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            FrequencyGrid::Uniform {
                nfft: DEFAULT_NFFT_MIN.max(window_len),
            }
        } else if keyword(value).is_some() {
            FrequencyGrid::Uniform {
                nfft: DEFAULT_NFFT_MIN.max(window_len),
            }
        } else {
            idx += 1;
            parse_frequency_grid(value.clone(), window_len).await?
        }
    } else {
        FrequencyGrid::Uniform {
            nfft: DEFAULT_NFFT_MIN.max(window_len),
        }
    };
    if let FrequencyGrid::Uniform { nfft } = &grid {
        if *nfft < window_len || *nfft > MAX_NFFT {
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_NFFT));
        }
    }

    let fs = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            None
        } else if keyword(value).is_some() {
            None
        } else {
            idx += 1;
            let parsed = parse_scalar_f64(BUILTIN_NAME, "fs", value)
                .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INVALID_FS, err.message()))?;
            if parsed <= 0.0 {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_FS));
            }
            Some(parsed)
        }
    } else {
        None
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
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_RANGE));
        };
        idx += 1;
        match keyword.as_str() {
            "onesided" | "twosided" | "centered"
                if matches!(grid, FrequencyGrid::Explicit { .. }) =>
            {
                return Err(pwelch_error_with_detail(
                    &PWELCH_ERROR_INVALID_RANGE,
                    "range selectors cannot be combined with an explicit frequency vector",
                ));
            }
            "onesided" => range = FrequencyRange::Onesided,
            "twosided" => range = FrequencyRange::Twosided,
            "centered" => range = FrequencyRange::Centered,
            "psd" => scale = SpectrumScale::Psd,
            "power" => scale = SpectrumScale::Power,
            _ => return Err(pwelch_error(&PWELCH_ERROR_INVALID_RANGE)),
        }
    }

    Ok(PwelchOptions {
        window,
        noverlap,
        grid,
        fs,
        range,
        scale,
    })
}

async fn parse_frequency_grid(value: Value, window_len: usize) -> BuiltinResult<FrequencyGrid> {
    if is_scalar_numeric(&value) {
        let parsed = parse_nonnegative_integer(BUILTIN_NAME, "nfft", &value)
            .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INVALID_NFFT, err.message()))?;
        if parsed == 0 || parsed < window_len || parsed > MAX_NFFT {
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_NFFT));
        }
        return Ok(FrequencyGrid::Uniform { nfft: parsed });
    }

    let vector = value_to_complex_vector(BUILTIN_NAME, "nfft", value)
        .await
        .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INVALID_NFFT, err.message()))?;
    if vector.data.is_empty() || vector.data.len() > MAX_NFFT {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_NFFT));
    }
    let mut frequencies = Vec::with_capacity(vector.data.len());
    for value in vector.data {
        if value.im.abs() > EPS || !value.re.is_finite() {
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_NFFT));
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

async fn parse_window(value: Value) -> BuiltinResult<Vec<f64>> {
    match &value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let len = parse_nonnegative_integer(BUILTIN_NAME, "window", &value).map_err(|err| {
                pwelch_error_with_detail(&PWELCH_ERROR_INVALID_WINDOW, err.message())
            })?;
            if len == 0 {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
            }
            return Ok(hamming_window(len));
        }
        Value::Tensor(t) if t.data.len() == 1 => {
            let len = parse_nonnegative_integer(BUILTIN_NAME, "window", &value).map_err(|err| {
                pwelch_error_with_detail(&PWELCH_ERROR_INVALID_WINDOW, err.message())
            })?;
            if len == 0 {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
            }
            return Ok(hamming_window(len));
        }
        _ => {}
    }
    let vector = value_to_complex_vector(BUILTIN_NAME, "window", value)
        .await
        .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INVALID_WINDOW, err.message()))?;
    if vector.data.is_empty() {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
    }
    let mut out = Vec::with_capacity(vector.data.len());
    for value in vector.data {
        if value.im.abs() > EPS || !value.re.is_finite() {
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
        }
        out.push(value.re);
    }
    Ok(out)
}

fn compute_pwelch(
    columns: &[Vec<Complex<f64>>],
    options: &PwelchOptions,
) -> BuiltinResult<PwelchEvaluation> {
    let window_len = options.window.len();
    let step = window_len - options.noverlap;
    let scale = options.window.iter().map(|v| v * v).sum::<f64>();
    if scale <= 0.0 || !scale.is_finite() {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
    }
    let density_scale = match options.scale {
        SpectrumScale::Psd => options.fs.unwrap_or(2.0 * std::f64::consts::PI) * scale,
        SpectrumScale::Power => scale,
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
                let accum = column_uniform_periodogram(
                    column,
                    &options.window,
                    step,
                    density_scale,
                    *nfft,
                    fft,
                    buffer,
                );
                select_range(accum, *nfft, options.fs, options.range)
            }
            (FrequencyGrid::Explicit { frequencies }, None) => {
                column_explicit_frequency_periodogram(
                    column,
                    &options.window,
                    step,
                    density_scale,
                    frequencies,
                    options.fs,
                )
            }
            _ => return Err(pwelch_error(&PWELCH_ERROR_INTERNAL)),
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
    Ok(PwelchEvaluation {
        pxx,
        f: selected_f,
        columns: cols,
    })
}

fn column_uniform_periodogram(
    column: &[Complex<f64>],
    window: &[f64],
    step: usize,
    density_scale: f64,
    nfft: usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    buffer: &mut [Complex<f64>],
) -> Vec<f64> {
    let starts = segment_starts(column.len(), window.len(), step);
    let mut accum = vec![0.0; nfft];
    for &start in &starts {
        buffer.fill(Complex::new(0.0, 0.0));
        for idx in 0..window.len() {
            let sample = column
                .get(start + idx)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            buffer[idx] = sample * window[idx];
        }
        fft.process(buffer);
        for (dst, value) in accum.iter_mut().zip(buffer.iter()) {
            *dst += value.norm_sqr() / density_scale;
        }
    }
    let segment_count = starts.len() as f64;
    for value in &mut accum {
        *value /= segment_count;
    }
    accum
}

fn column_explicit_frequency_periodogram(
    column: &[Complex<f64>],
    window: &[f64],
    step: usize,
    density_scale: f64,
    frequencies: &[f64],
    fs: Option<f64>,
) -> PwelchEvaluation {
    let starts = segment_starts(column.len(), window.len(), step);
    let mut pxx = vec![0.0; frequencies.len()];
    let fs_scale = fs.map(|fs| 2.0 * std::f64::consts::PI / fs);
    for &start in &starts {
        for (freq_idx, &frequency) in frequencies.iter().enumerate() {
            let omega = fs_scale.map_or(frequency, |scale| frequency * scale);
            let mut sum = Complex::new(0.0, 0.0);
            for idx in 0..window.len() {
                let sample = column
                    .get(start + idx)
                    .copied()
                    .unwrap_or_else(|| Complex::new(0.0, 0.0));
                let phase = -omega * idx as f64;
                let twiddle = Complex::new(phase.cos(), phase.sin());
                sum += sample * window[idx] * twiddle;
            }
            pxx[freq_idx] += sum.norm_sqr() / density_scale;
        }
    }
    let segment_count = starts.len() as f64;
    for value in &mut pxx {
        *value /= segment_count;
    }
    PwelchEvaluation {
        pxx,
        f: frequencies.to_vec(),
        columns: 1,
    }
}

fn select_range(
    mut power: Vec<f64>,
    nfft: usize,
    fs: Option<f64>,
    range: FrequencyRange,
) -> PwelchEvaluation {
    let freq_scale = fs.unwrap_or(2.0 * std::f64::consts::PI);
    match range {
        FrequencyRange::Twosided => {
            let f = (0..nfft)
                .map(|idx| freq_scale * idx as f64 / nfft as f64)
                .collect();
            PwelchEvaluation {
                pxx: power,
                f,
                columns: 1,
            }
        }
        FrequencyRange::Centered => {
            let shift = nfft.div_ceil(2);
            power.rotate_left(shift);
            let offset = nfft / 2;
            let f = (0..nfft)
                .map(|idx| freq_scale * (idx as f64 - offset as f64) / nfft as f64)
                .collect();
            PwelchEvaluation {
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
            PwelchEvaluation { pxx, f, columns: 1 }
        }
    }
}

fn frequency_vector(nfft: usize, fs: Option<f64>, range: FrequencyRange) -> Vec<f64> {
    select_range(vec![0.0; nfft], nfft, fs, range).f
}

fn gpu_range(range: FrequencyRange) -> ProviderSpectralRange {
    match range {
        FrequencyRange::Onesided => ProviderSpectralRange::Onesided,
        FrequencyRange::Twosided => ProviderSpectralRange::Twosided,
        FrequencyRange::Centered => ProviderSpectralRange::Centered,
    }
}

fn segment_starts(signal_len: usize, window_len: usize, step: usize) -> Vec<usize> {
    if signal_len <= window_len {
        return vec![0];
    }
    let mut starts = Vec::new();
    let mut start = 0usize;
    while start + window_len <= signal_len {
        starts.push(start);
        start += step;
    }
    if starts.is_empty() {
        starts.push(0);
    }
    starts
}

fn default_window(input_len: usize) -> Vec<f64> {
    let section_len = if input_len <= 1 {
        1
    } else {
        ((2 * input_len) / 9).clamp(1, DEFAULT_NFFT_MIN)
    };
    hamming_window(section_len)
}

fn hamming_window(len: usize) -> Vec<f64> {
    if len == 1 {
        return vec![1.0];
    }
    (0..len)
        .map(|idx| {
            let phase = 2.0 * std::f64::consts::PI * idx as f64 / (len - 1) as f64;
            0.54 - 0.46 * phase.cos()
        })
        .collect()
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
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("pwelch builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| sig.label == "[pxx, f] = pwelch(x, window, noverlap, nfft, fs, freqrange)"));
    }

    #[test]
    fn pwelch_constant_signal_has_dc_peak() {
        let x = Tensor::new(vec![1.0; 16], vec![1, 16]).unwrap();
        let window = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let out = call(
            Value::Tensor(x),
            &[
                Value::Tensor(window),
                Value::Num(4.0),
                Value::Num(8.0),
                Value::Num(8.0),
            ],
            Some(2),
        )
        .unwrap();
        let (pxx, f) = output_pair(out);
        assert_eq!(pxx.len(), 5);
        assert_eq!(f.len(), 5);
        assert!(pxx[0] > 0.9);
        assert!(pxx[1..].iter().all(|value| value.abs() < 1e-12));
        assert_eq!(f[0], 0.0);
        assert_eq!(f[4], 4.0);
    }

    #[test]
    fn pwelch_detects_sinusoid_bin() {
        let x = (0..32)
            .map(|idx| (2.0 * std::f64::consts::PI * 4.0 * idx as f64 / 32.0).sin())
            .collect::<Vec<_>>();
        let out = call(
            Value::Tensor(Tensor::new(x, vec![1, 32]).unwrap()),
            &[
                Value::Num(32.0),
                Value::Num(0.0),
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
    fn pwelch_twosided_and_centered_ranges() {
        let data = (0..8)
            .map(|idx| (idx as f64, if idx % 2 == 0 { 0.0 } else { 0.25 }))
            .collect::<Vec<_>>();
        let x = runmat_builtins::ComplexTensor::new(data, vec![1, 8]).unwrap();
        let two = call(
            Value::ComplexTensor(x.clone()),
            &[
                Value::Num(8.0),
                Value::Num(0.0),
                Value::Num(8.0),
                Value::from("twosided"),
            ],
            Some(2),
        )
        .unwrap();
        let (pxx, f) = output_pair(two);
        assert_eq!(pxx.len(), 8);
        assert_eq!(f.len(), 8);

        let centered = call(
            Value::ComplexTensor(x),
            &[
                Value::Num(8.0),
                Value::Num(0.0),
                Value::Num(8.0),
                Value::from("centered"),
            ],
            Some(2),
        )
        .unwrap();
        let (_, f) = output_pair(centered);
        assert_eq!(f[0], -std::f64::consts::PI);
        assert_eq!(f[4], 0.0);
    }

    #[test]
    fn pwelch_centered_odd_nfft_uses_fftshift_order() {
        let x = runmat_builtins::ComplexTensor::new(
            (0..5).map(|idx| (idx as f64, 0.25)).collect(),
            vec![1, 5],
        )
        .unwrap();
        let out = call(
            Value::ComplexTensor(x),
            &[
                Value::Num(5.0),
                Value::Num(0.0),
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
    fn pwelch_accepts_matrix_columns() {
        let data = vec![
            1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0,
        ];
        let x = Tensor::new(data, vec![8, 2]).unwrap();
        let out = call(
            Value::Tensor(x),
            &[Value::Num(8.0), Value::Num(0.0), Value::Num(8.0)],
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
    fn pwelch_accepts_explicit_frequency_vector() {
        let x = Tensor::new(vec![1.0; 4], vec![1, 4]).unwrap();
        let window = Tensor::new(vec![1.0; 4], vec![1, 4]).unwrap();
        let frequencies = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let out = call(
            Value::Tensor(x),
            &[
                Value::Tensor(window),
                Value::Num(0.0),
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
    fn pwelch_rejects_nfft_shorter_than_window() {
        let x = Tensor::new(vec![1.0; 16], vec![1, 16]).unwrap();
        let err = call(
            Value::Tensor(x),
            &[Value::Num(8.0), Value::Num(0.0), Value::Num(4.0)],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:pwelch:InvalidNfft"));
    }

    #[test]
    fn pwelch_power_scale_omits_density_bandwidth() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let window = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let psd = call(
            Value::Tensor(x.clone()),
            &[
                Value::Tensor(window.clone()),
                Value::Num(0.0),
                Value::Num(8.0),
                Value::Num(8.0),
                Value::from("psd"),
            ],
            Some(2),
        )
        .unwrap();
        let power = call(
            Value::Tensor(x),
            &[
                Value::Tensor(window),
                Value::Num(0.0),
                Value::Num(8.0),
                Value::Num(8.0),
                Value::from("power"),
            ],
            Some(2),
        )
        .unwrap();
        let (psd, _) = output_pair(psd);
        let (power, _) = output_pair(power);
        assert!((power[0] / psd[0] - 8.0).abs() < 1e-12);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn pwelch_wgpu_keeps_uniform_outputs_resident_and_matches_cpu() {
        let Some(provider) = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
            .expect("wgpu provider")
        else {
            return;
        };
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

        let rows = 64usize;
        let mut data = Vec::with_capacity(rows * 2);
        data.extend(
            (0..rows).map(|idx| (2.0 * std::f64::consts::PI * 4.0 * idx as f64 / 32.0).sin()),
        );
        data.extend(std::iter::repeat_n(1.0, rows));
        let cpu_x = Tensor::new(data.clone(), vec![rows, 2]).unwrap();
        let cpu = call(
            Value::Tensor(cpu_x),
            &[
                Value::Num(16.0),
                Value::Num(8.0),
                Value::Num(32.0),
                Value::Num(32.0),
            ],
            Some(2),
        )
        .expect("pwelch cpu");
        let (cpu_pxx, cpu_f) = output_pair(cpu);

        let shape = [rows, 2usize];
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload");
        let gpu = call(
            Value::GpuTensor(handle.clone()),
            &[
                Value::Num(16.0),
                Value::Num(8.0),
                Value::Num(32.0),
                Value::Num(32.0),
            ],
            Some(2),
        )
        .expect("pwelch gpu");
        let Value::OutputList(values) = gpu else {
            panic!("expected output list");
        };
        for value in &values {
            assert!(matches!(value, Value::GpuTensor(_)));
        }
        let gpu_pxx =
            crate::builtins::common::test_support::gather(values[0].clone()).expect("gather pxx");
        let gpu_f =
            crate::builtins::common::test_support::gather(values[1].clone()).expect("gather f");

        assert_eq!(gpu_pxx.shape, vec![17, 2]);
        assert_eq!(gpu_f.shape, vec![17, 1]);
        assert_eq!(gpu_f.data, cpu_f);
        for (actual, expected) in gpu_pxx.data.iter().zip(cpu_pxx.iter()) {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "actual={actual} expected={expected}"
            );
        }
        provider.free(&handle).ok();
    }

    #[test]
    fn default_window_uses_eight_overlapped_sections() {
        assert_eq!(default_window(900).len(), 200);
        assert_eq!(default_window(1).len(), 1);
    }

    #[test]
    fn pwelch_rejects_invalid_overlap() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let err = call(
            Value::Tensor(x),
            &[Value::Num(8.0), Value::Num(8.0)],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:pwelch:InvalidOverlap"));
    }
}
