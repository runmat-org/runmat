//! MATLAB-compatible spectrogram using a short-time Fourier transform.

use num_complex::Complex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use rustfft::FftPlanner;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::signal::common::{
    centered_frequency_offset, centered_shift, gpu_uniform_spectral_estimate, gpu_vector_len,
    parse_nonnegative_integer, parse_scalar_f64, selected_frequency_len, value_to_complex_vector,
    GpuSpectralFrameMode, GpuSpectralRange, GpuSpectralRequest,
};
use crate::builtins::math::signal::type_resolvers::spectrogram_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "spectrogram";
const DEFAULT_NFFT_MIN: usize = 256;
const MAX_NFFT: usize = 1 << 22;
const EPS: f64 = 1.0e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::spectrogram")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "spectrogram",
    op_kind: GpuOpKind::Custom("stft-spectrogram"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("fft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uniform-grid STFT spectrograms frame/window, FFT, select, and power-scale on GPU for resident inputs; unsupported forms fall back to host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::spectrogram")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "spectrogram",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "spectrogram materialises STFT and spectral estimate matrices and is not fused.",
};

const SPECTROGRAM_OUTPUT_S: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex short-time Fourier transform matrix.",
}];

const SPECTROGRAM_OUTPUT_FULL: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "s",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex short-time Fourier transform matrix.",
    },
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Frequencies in radians/sample or Hz when fs is supplied.",
    },
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Segment midpoint sample indices or time instants when fs is supplied.",
    },
    BuiltinParamDescriptor {
        name: "ps",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Power spectral density or power spectrum estimate for each segment.",
    },
    BuiltinParamDescriptor {
        name: "fc",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Center frequency matrix for reassigned-output compatibility.",
    },
    BuiltinParamDescriptor {
        name: "tc",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Center time matrix for reassigned-output compatibility.",
    },
];

const SPECTROGRAM_INPUTS: [BuiltinParamDescriptor; 7] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal vector.",
    },
    BuiltinParamDescriptor {
        name: "win",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Hamming-window length, window vector, or [] for the default window.",
    },
    BuiltinParamDescriptor {
        name: "nOverlap",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Number of overlapped samples between adjacent segments.",
    },
    BuiltinParamDescriptor {
        name: "freqSpec",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "DFT length or explicit frequency vector.",
    },
    BuiltinParamDescriptor {
        name: "Fs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Sampling frequency in Hz.",
    },
    BuiltinParamDescriptor {
        name: "freqRange",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Frequency range selector: `onesided`, `twosided`, or `centered`.",
    },
    BuiltinParamDescriptor {
        name: "spectrumType",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("psd"),
        description: "Spectral estimate scaling: `psd` or `power`.",
    },
];

const SPECTROGRAM_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "s = spectrogram(x, win, nOverlap, freqSpec, Fs, freqRange, spectrumType)",
        inputs: &SPECTROGRAM_INPUTS,
        outputs: &SPECTROGRAM_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "[s, f, t, ps, fc, tc] = spectrogram(x, win, nOverlap, freqSpec, Fs, freqRange, spectrumType)",
        inputs: &SPECTROGRAM_INPUTS,
        outputs: &SPECTROGRAM_OUTPUT_FULL,
    },
];

const SPECTROGRAM_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.ARG_COUNT",
    identifier: Some("RunMat:spectrogram:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "spectrogram: expected spectrogram(x, [win, [nOverlap, [freqSpec, [Fs, [freqRange_or_spectrumType]]]]])",
};

const SPECTROGRAM_ERROR_INVALID_SIGNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_SIGNAL",
    identifier: Some("RunMat:spectrogram:InvalidSignal"),
    when: "Input signal is not a numeric vector.",
    message: "spectrogram: x must be a nonempty numeric vector",
};

const SPECTROGRAM_ERROR_INVALID_WINDOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_WINDOW",
    identifier: Some("RunMat:spectrogram:InvalidWindow"),
    when: "Window input is invalid.",
    message:
        "spectrogram: win must be [] or a nonempty finite real vector or positive integer length",
};

const SPECTROGRAM_ERROR_INVALID_OVERLAP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_OVERLAP",
    identifier: Some("RunMat:spectrogram:InvalidOverlap"),
    when: "Overlap is invalid.",
    message: "spectrogram: nOverlap must be a nonnegative integer smaller than the window length",
};

const SPECTROGRAM_ERROR_INVALID_FREQ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_FREQ",
    identifier: Some("RunMat:spectrogram:InvalidFreqSpec"),
    when: "Frequency specification is invalid.",
    message: "spectrogram: freqSpec must be a positive integer or finite real frequency vector with at least two elements",
};

const SPECTROGRAM_ERROR_INVALID_FS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_FS",
    identifier: Some("RunMat:spectrogram:InvalidFs"),
    when: "Sampling frequency is invalid.",
    message: "spectrogram: Fs must be a positive finite scalar",
};

const SPECTROGRAM_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INVALID_OPTION",
    identifier: Some("RunMat:spectrogram:InvalidOption"),
    when: "Frequency range, spectrum type, or display option is invalid.",
    message: "spectrogram: expected 'onesided', 'twosided', 'centered', 'psd', 'power', 'reassigned', or 'yaxis'",
};

const SPECTROGRAM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPECTROGRAM.INTERNAL",
    identifier: Some("RunMat:spectrogram:Internal"),
    when: "Output tensor construction fails internally.",
    message: "spectrogram: internal error",
};

const SPECTROGRAM_ERRORS: [BuiltinErrorDescriptor; 8] = [
    SPECTROGRAM_ERROR_ARG_COUNT,
    SPECTROGRAM_ERROR_INVALID_SIGNAL,
    SPECTROGRAM_ERROR_INVALID_WINDOW,
    SPECTROGRAM_ERROR_INVALID_OVERLAP,
    SPECTROGRAM_ERROR_INVALID_FREQ,
    SPECTROGRAM_ERROR_INVALID_FS,
    SPECTROGRAM_ERROR_INVALID_OPTION,
    SPECTROGRAM_ERROR_INTERNAL,
];

pub const SPECTROGRAM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPECTROGRAM_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPECTROGRAM_ERRORS,
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

#[derive(Clone, Copy, Debug, PartialEq)]
enum FrequencyUnits {
    NormalizedRadians,
    Hz(f64),
}

#[derive(Clone, Debug)]
enum FrequencyGrid {
    Uniform { nfft: usize },
    Explicit { frequencies: Vec<f64> },
}

#[derive(Clone, Debug)]
struct SpectrogramOptions {
    window: Vec<f64>,
    n_overlap: usize,
    grid: FrequencyGrid,
    units: FrequencyUnits,
    range: FrequencyRange,
    scale: SpectrumScale,
    reassigned: bool,
}

#[derive(Clone, Debug)]
struct SpectrogramEvaluation {
    s: Vec<Complex<f64>>,
    f: Vec<f64>,
    t: Vec<f64>,
    ps: Vec<f64>,
    fc: Vec<f64>,
    tc: Vec<f64>,
}

fn spectrogram_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    spectrogram_error_with_message(error.message, error)
}

fn spectrogram_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    spectrogram_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn spectrogram_error_with_message(
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
    name = "spectrogram",
    category = "math/signal",
    summary = "Compute a short-time Fourier transform spectrogram.",
    keywords = "spectrogram,stft,psd,power spectrum,time frequency,signal processing",
    type_resolver(spectrogram_type),
    descriptor(crate::builtins::math::signal::spectrogram::SPECTROGRAM_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::spectrogram"
)]
async fn spectrogram_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(x, &rest).await
}

pub async fn evaluate(x: Value, rest: &[Value]) -> BuiltinResult<Value> {
    if rest.len() > 9 {
        return Err(spectrogram_error(&SPECTROGRAM_ERROR_ARG_COUNT));
    }
    if let Value::GpuTensor(handle) = &x {
        let signal_len = gpu_vector_len(BUILTIN_NAME, "x", handle).map_err(|err| {
            spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INVALID_SIGNAL, err.message())
        })?;
        if signal_len == 0 {
            return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_SIGNAL));
        }
        let complex_input = runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
        let options = parse_options(rest, signal_len, complex_input).await?;
        if let Ok(value) = output_gpu(handle, &options, signal_len).await {
            return Ok(value);
        }
    }
    let signal = value_to_complex_vector(BUILTIN_NAME, "x", x)
        .await
        .map_err(|err| {
            spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INVALID_SIGNAL, err.message())
        })?;
    if signal.data.is_empty() {
        return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_SIGNAL));
    }
    let options = parse_options(rest, signal.data.len(), signal.is_complex).await?;
    let eval = compute_spectrogram(&signal.data, &options)?;
    output_eval(eval, options.reassigned)
}

fn output_eval(eval: SpectrogramEvaluation, reassigned: bool) -> BuiltinResult<Value> {
    let rows = eval.f.len();
    let cols = eval.t.len();
    let s = ComplexTensor::new(
        eval.s.into_iter().map(|z| (z.re, z.im)).collect(),
        vec![rows, cols],
    )
    .map(Value::ComplexTensor)
    .map_err(|e| spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INTERNAL, e))?;
    let f = Tensor::new(eval.f, vec![rows, 1])
        .map(Value::Tensor)
        .map_err(|e| spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INTERNAL, e))?;
    let t = Tensor::new(eval.t, vec![1, cols])
        .map(Value::Tensor)
        .map_err(|e| spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INTERNAL, e))?;
    let ps = Tensor::new(eval.ps, vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|e| spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INTERNAL, e))?;

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let mut values = vec![s, f, t, ps];
        if reassigned {
            let fc = Tensor::new(eval.fc, vec![rows, cols])
                .map(Value::Tensor)
                .map_err(|e| spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INTERNAL, e))?;
            let tc = Tensor::new(eval.tc, vec![rows, cols])
                .map(Value::Tensor)
                .map_err(|e| spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INTERNAL, e))?;
            values.push(fc);
            values.push(tc);
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count, values,
        ));
    }
    Ok(s)
}

async fn output_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    options: &SpectrogramOptions,
    signal_len: usize,
) -> BuiltinResult<Value> {
    let FrequencyGrid::Uniform { nfft } = options.grid else {
        return Err(spectrogram_error(&SPECTROGRAM_ERROR_INTERNAL));
    };
    if options.reassigned {
        return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_OPTION));
    }
    let window_len = options.window.len();
    let step = window_len - options.n_overlap;
    let starts = segment_starts(signal_len, window_len, step);
    let window_energy = options.window.iter().map(|v| v * v).sum::<f64>();
    let coherent_gain = options.window.iter().sum::<f64>();
    if window_energy <= 0.0 || !window_energy.is_finite() {
        return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_WINDOW));
    }
    let denominator = match options.scale {
        SpectrumScale::Psd => frequency_scale(options.units) * window_energy,
        SpectrumScale::Power => {
            let gain_sq = coherent_gain * coherent_gain;
            if gain_sq <= EPS || !gain_sq.is_finite() {
                return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_WINDOW));
            }
            gain_sq
        }
    };
    let range = gpu_range(options.range);
    let estimate = gpu_uniform_spectral_estimate(GpuSpectralRequest {
        input: handle,
        input_len: signal_len,
        input_complex: runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
        window: &options.window,
        nfft,
        frame_count: starts.len(),
        frame_mode: GpuSpectralFrameMode::Sliding { hop: step },
        range,
        denominator,
    })
    .await
    .map_err(|err| spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INTERNAL, err.message()))?;
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(spectrogram_error(&SPECTROGRAM_ERROR_INTERNAL));
    };
    let f_values = frequency_vector(nfft, options.units, options.range);
    let t_values = starts
        .iter()
        .map(|&start| segment_time(start, window_len, options.units))
        .collect::<Vec<_>>();
    let f_shape = [estimate.rows, 1usize];
    let t_shape = [1usize, estimate.cols];
    let f = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &f_values,
            shape: &f_shape,
        })
        .map_err(|e| spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INTERNAL, e.to_string()))?;
    let t = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &t_values,
            shape: &t_shape,
        })
        .map_err(|e| spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INTERNAL, e.to_string()))?;
    let s = crate::builtins::common::gpu_helpers::resident_gpu_value(estimate.s);
    let f = crate::builtins::common::gpu_helpers::resident_gpu_value(f);
    let t = crate::builtins::common::gpu_helpers::resident_gpu_value(t);
    let ps = crate::builtins::common::gpu_helpers::resident_gpu_value(estimate.ps);
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![s, f, t, ps],
        ));
    }
    Ok(s)
}

async fn parse_options(
    rest: &[Value],
    signal_len: usize,
    complex_input: bool,
) -> BuiltinResult<SpectrogramOptions> {
    let mut idx = 0usize;
    let window = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            default_window(signal_len)
        } else if keyword(value).is_some() {
            default_window(signal_len)
        } else {
            idx += 1;
            parse_window(value.clone()).await?
        }
    } else {
        default_window(signal_len)
    };
    let window_len = window.len();

    let n_overlap = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            window_len / 2
        } else if keyword(value).is_some() {
            window_len / 2
        } else {
            idx += 1;
            let parsed =
                parse_nonnegative_integer(BUILTIN_NAME, "nOverlap", value).map_err(|err| {
                    spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INVALID_OVERLAP, err.message())
                })?;
            if parsed >= window_len {
                return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_OVERLAP));
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

    let units = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            FrequencyUnits::Hz(1.0)
        } else if keyword(value).is_some() {
            FrequencyUnits::NormalizedRadians
        } else {
            idx += 1;
            let parsed = parse_scalar_f64(BUILTIN_NAME, "Fs", value).map_err(|err| {
                spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INVALID_FS, err.message())
            })?;
            if parsed <= 0.0 {
                return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_FS));
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
    let reassigned = false;
    while let Some(value) = rest.get(idx) {
        let Some(keyword) = keyword(value) else {
            return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_OPTION));
        };
        idx += 1;
        match keyword.as_str() {
            "onesided" | "twosided" | "centered"
                if matches!(grid, FrequencyGrid::Explicit { .. }) =>
            {
                return Err(spectrogram_error_with_detail(
                    &SPECTROGRAM_ERROR_INVALID_OPTION,
                    "range selectors cannot be combined with an explicit frequency vector",
                ));
            }
            "onesided" if complex_input => {
                return Err(spectrogram_error_with_detail(
                    &SPECTROGRAM_ERROR_INVALID_OPTION,
                    "onesided spectra are only supported for real-valued signals",
                ));
            }
            "onesided" => range = FrequencyRange::Onesided,
            "twosided" => range = FrequencyRange::Twosided,
            "centered" => range = FrequencyRange::Centered,
            "psd" => scale = SpectrumScale::Psd,
            "power" => scale = SpectrumScale::Power,
            "reassigned" => {
                return Err(spectrogram_error_with_detail(
                    &SPECTROGRAM_ERROR_INVALID_OPTION,
                    "reassigned spectrogram outputs require center-of-energy reassignment and are not implemented yet",
                ));
            }
            "yaxis" => {}
            _ => return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_OPTION)),
        }
    }

    Ok(SpectrogramOptions {
        window,
        n_overlap,
        grid,
        units,
        range,
        scale,
        reassigned,
    })
}

async fn parse_window(value: Value) -> BuiltinResult<Vec<f64>> {
    if is_scalar_numeric(&value) {
        let len = parse_nonnegative_integer(BUILTIN_NAME, "win", &value).map_err(|err| {
            spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INVALID_WINDOW, err.message())
        })?;
        if len == 0 {
            return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_WINDOW));
        }
        return Ok(hamming_window(len));
    }
    let vector = value_to_complex_vector(BUILTIN_NAME, "win", value)
        .await
        .map_err(|err| {
            spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INVALID_WINDOW, err.message())
        })?;
    if vector.data.is_empty() {
        return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_WINDOW));
    }
    let mut out = Vec::with_capacity(vector.data.len());
    for value in vector.data {
        if value.im.abs() > EPS || !value.re.is_finite() {
            return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_WINDOW));
        }
        out.push(value.re);
    }
    Ok(out)
}

async fn parse_frequency_grid(value: Value) -> BuiltinResult<FrequencyGrid> {
    if is_scalar_numeric(&value) {
        let parsed =
            parse_nonnegative_integer(BUILTIN_NAME, "freqSpec", &value).map_err(|err| {
                spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INVALID_FREQ, err.message())
            })?;
        if parsed == 0 || parsed > MAX_NFFT {
            return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_FREQ));
        }
        return Ok(FrequencyGrid::Uniform { nfft: parsed });
    }
    let vector = value_to_complex_vector(BUILTIN_NAME, "freqSpec", value)
        .await
        .map_err(|err| {
            spectrogram_error_with_detail(&SPECTROGRAM_ERROR_INVALID_FREQ, err.message())
        })?;
    if vector.data.len() < 2 || vector.data.len() > MAX_NFFT {
        return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_FREQ));
    }
    let mut frequencies = Vec::with_capacity(vector.data.len());
    for value in vector.data {
        if value.im.abs() > EPS || !value.re.is_finite() {
            return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_FREQ));
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

fn compute_spectrogram(
    signal: &[Complex<f64>],
    options: &SpectrogramOptions,
) -> BuiltinResult<SpectrogramEvaluation> {
    let window_len = options.window.len();
    let step = window_len - options.n_overlap;
    let starts = segment_starts(signal.len(), window_len, step);
    let window_energy = options.window.iter().map(|v| v * v).sum::<f64>();
    let coherent_gain = options.window.iter().sum::<f64>();
    if window_energy <= 0.0 || !window_energy.is_finite() {
        return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_WINDOW));
    }
    let denominator = match options.scale {
        SpectrumScale::Psd => frequency_scale(options.units) * window_energy,
        SpectrumScale::Power => {
            let gain_sq = coherent_gain * coherent_gain;
            if gain_sq <= EPS || !gain_sq.is_finite() {
                return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_WINDOW));
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

    let mut s_columns = Vec::new();
    let mut ps_columns = Vec::new();
    let mut f = Vec::new();
    let mut t = Vec::with_capacity(starts.len());

    for &start in &starts {
        let selected = match (&options.grid, &mut fft_plan) {
            (FrequencyGrid::Uniform { nfft }, Some((fft, buffer))) => {
                let full = segment_fft(signal, &options.window, start, *nfft, fft, buffer);
                select_range(full, *nfft, options.units, options.range)
            }
            (FrequencyGrid::Explicit { frequencies }, None) => {
                explicit_frequency_stft(signal, &options.window, start, frequencies, options.units)
            }
            _ => return Err(spectrogram_error(&SPECTROGRAM_ERROR_INTERNAL)),
        };
        if f.is_empty() {
            f = selected.f;
        }
        let ps = selected_power(
            &selected.s,
            denominator,
            selected.double_one_sided,
            selected.has_nyquist,
        );
        s_columns.push(selected.s);
        ps_columns.push(ps);
        t.push(segment_time(start, window_len, options.units));
    }

    let rows = f.len();
    let cols = starts.len();
    let mut s = Vec::with_capacity(rows * cols);
    let mut ps = Vec::with_capacity(rows * cols);
    for column in s_columns {
        s.extend(column);
    }
    for column in ps_columns {
        ps.extend(column);
    }
    let (fc, tc) = coordinate_matrices(&f, &t);
    Ok(SpectrogramEvaluation {
        s,
        f,
        t,
        ps,
        fc,
        tc,
    })
}

#[derive(Clone, Debug)]
struct SelectedStft {
    s: Vec<Complex<f64>>,
    f: Vec<f64>,
    double_one_sided: bool,
    has_nyquist: bool,
}

fn segment_fft(
    signal: &[Complex<f64>],
    window: &[f64],
    start: usize,
    nfft: usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    buffer: &mut [Complex<f64>],
) -> Vec<Complex<f64>> {
    buffer.fill(Complex::new(0.0, 0.0));
    let copy_len = window.len().min(nfft);
    for idx in 0..copy_len {
        let sample = signal
            .get(start + idx)
            .copied()
            .unwrap_or_else(|| Complex::new(0.0, 0.0));
        buffer[idx] = sample * window[idx];
    }
    fft.process(buffer);
    buffer.to_vec()
}

fn explicit_frequency_stft(
    signal: &[Complex<f64>],
    window: &[f64],
    start: usize,
    frequencies: &[f64],
    units: FrequencyUnits,
) -> SelectedStft {
    let mut s = Vec::with_capacity(frequencies.len());
    for &frequency in frequencies {
        let omega = angular_frequency(frequency, units);
        let mut sum = Complex::new(0.0, 0.0);
        for idx in 0..window.len() {
            let sample = signal
                .get(start + idx)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            let phase = -omega * idx as f64;
            let twiddle = Complex::new(phase.cos(), phase.sin());
            sum += sample * window[idx] * twiddle;
        }
        s.push(sum);
    }
    SelectedStft {
        s,
        f: frequencies.to_vec(),
        double_one_sided: false,
        has_nyquist: false,
    }
}

fn select_range(
    mut full: Vec<Complex<f64>>,
    nfft: usize,
    units: FrequencyUnits,
    range: FrequencyRange,
) -> SelectedStft {
    let freq_scale = frequency_scale(units);
    match range {
        FrequencyRange::Twosided => {
            let f = (0..nfft)
                .map(|idx| freq_scale * idx as f64 / nfft as f64)
                .collect();
            SelectedStft {
                s: full,
                f,
                double_one_sided: false,
                has_nyquist: false,
            }
        }
        FrequencyRange::Centered => {
            let shift = centered_shift(nfft);
            full.rotate_left(shift);
            let offset = centered_frequency_offset(nfft);
            let f = (0..nfft)
                .map(|idx| freq_scale * (idx as isize - offset) as f64 / nfft as f64)
                .collect();
            SelectedStft {
                s: full,
                f,
                double_one_sided: false,
                has_nyquist: false,
            }
        }
        FrequencyRange::Onesided => {
            let len = nfft / 2 + 1;
            let f = (0..len)
                .map(|idx| freq_scale * idx as f64 / nfft as f64)
                .collect();
            SelectedStft {
                s: full[..len].to_vec(),
                f,
                double_one_sided: true,
                has_nyquist: nfft.is_multiple_of(2),
            }
        }
    }
}

fn selected_power(
    s: &[Complex<f64>],
    denominator: f64,
    double_one_sided: bool,
    has_nyquist: bool,
) -> Vec<f64> {
    let mut out = s
        .iter()
        .map(|value| value.norm_sqr() / denominator)
        .collect::<Vec<_>>();
    if double_one_sided {
        for (idx, value) in out.iter_mut().enumerate() {
            let is_dc = idx == 0;
            let is_nyquist = has_nyquist && idx == s.len() - 1;
            if !is_dc && !is_nyquist {
                *value *= 2.0;
            }
        }
    }
    out
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

fn segment_time(start: usize, window_len: usize, units: FrequencyUnits) -> f64 {
    match units {
        FrequencyUnits::NormalizedRadians => {
            (start as f64 + window_len as f64 / 2.0) / (2.0 * std::f64::consts::PI)
        }
        FrequencyUnits::Hz(fs) => (start as f64 + window_len as f64 / 2.0) / fs,
    }
}

fn gpu_range(range: FrequencyRange) -> GpuSpectralRange {
    match range {
        FrequencyRange::Onesided => GpuSpectralRange::Onesided,
        FrequencyRange::Twosided => GpuSpectralRange::Twosided,
        FrequencyRange::Centered => GpuSpectralRange::Centered,
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
        FrequencyRange::Onesided => (0..selected_frequency_len(nfft, GpuSpectralRange::Onesided))
            .map(|idx| freq_scale * idx as f64 / nfft as f64)
            .collect(),
    }
}

fn coordinate_matrices(f: &[f64], t: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut fc = Vec::with_capacity(f.len() * t.len());
    let mut tc = Vec::with_capacity(f.len() * t.len());
    for &time in t {
        for &frequency in f {
            fc.push(frequency);
            tc.push(time);
        }
    }
    (fc, tc)
}

fn default_window(signal_len: usize) -> Vec<f64> {
    let len = if signal_len <= 1 {
        1
    } else {
        ((2 * signal_len) / 9).max(1)
    };
    hamming_window(len)
}

fn default_nfft(window_len: usize) -> BuiltinResult<usize> {
    let next_pow2 = window_len
        .checked_next_power_of_two()
        .ok_or_else(|| spectrogram_error(&SPECTROGRAM_ERROR_INVALID_FREQ))?;
    let nfft = DEFAULT_NFFT_MIN.max(next_pow2);
    if nfft > MAX_NFFT {
        return Err(spectrogram_error(&SPECTROGRAM_ERROR_INVALID_FREQ));
    }
    Ok(nfft)
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

    fn empty() -> Value {
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap())
    }

    fn call(x: Value, rest: &[Value], outputs: Option<usize>) -> BuiltinResult<Value> {
        let _guard = outputs.map(|count| crate::output_count::push_output_count(Some(count)));
        block_on(evaluate(x, rest))
    }

    fn output_list(value: Value) -> Vec<Value> {
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        values
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("spectrogram builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert!(descriptor.signatures.iter().any(|sig| {
            sig.label == "[s, f, t, ps, fc, tc] = spectrogram(x, win, nOverlap, freqSpec, Fs, freqRange, spectrumType)"
        }));
    }

    #[test]
    fn spectrogram_detects_time_frequency_bin() {
        let x = (0..64)
            .map(|idx| (2.0 * std::f64::consts::PI * 4.0 * idx as f64 / 32.0).sin())
            .collect::<Vec<_>>();
        let out = call(
            Value::Tensor(Tensor::new(x, vec![1, 64]).unwrap()),
            &[
                Value::Num(32.0),
                Value::Num(16.0),
                Value::Num(32.0),
                Value::Num(32.0),
            ],
            Some(4),
        )
        .unwrap();
        let values = output_list(out);
        let Value::ComplexTensor(s) = &values[0] else {
            panic!("expected stft");
        };
        let Value::Tensor(f) = &values[1] else {
            panic!("expected f");
        };
        let Value::Tensor(t) = &values[2] else {
            panic!("expected t");
        };
        let Value::Tensor(ps) = &values[3] else {
            panic!("expected ps");
        };
        assert_eq!(s.shape, vec![17, 3]);
        assert_eq!(ps.shape, vec![17, 3]);
        assert_eq!(f.data[4], 4.0);
        assert_eq!(t.data, vec![0.5, 1.0, 1.5]);
        let first_column_peak = ps.data[..17]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(first_column_peak, 4);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn spectrogram_wgpu_keeps_uniform_stft_outputs_resident() {
        let Some(provider) = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
            .expect("wgpu provider")
        else {
            return;
        };
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let data = (0..64)
            .map(|idx| (2.0 * std::f64::consts::PI * 4.0 * idx as f64 / 32.0).sin())
            .collect::<Vec<_>>();
        let shape = [1usize, 64usize];
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload");
        let out = call(
            Value::GpuTensor(handle.clone()),
            &[
                Value::Num(32.0),
                Value::Num(16.0),
                Value::Num(32.0),
                Value::Num(32.0),
            ],
            Some(4),
        )
        .expect("spectrogram gpu");
        let values = output_list(out);
        let Value::GpuTensor(s) = &values[0] else {
            panic!("expected resident stft");
        };
        assert_eq!(s.shape, vec![17, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(s),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        for value in &values[1..=3] {
            assert!(matches!(value, Value::GpuTensor(_)));
        }
        let ps =
            crate::builtins::common::test_support::gather(values[3].clone()).expect("gather ps");
        assert_eq!(ps.shape, vec![17, 3]);
        let peak = ps.data[..17]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(peak, 4);
        provider.free(&handle).ok();
    }

    #[test]
    fn spectrogram_defaults_match_documented_windowing_rule() {
        let x = Tensor::new(vec![1.0; 1024], vec![1, 1024]).unwrap();
        let out = call(Value::Tensor(x), &[], Some(3)).unwrap();
        let values = output_list(out);
        let Value::ComplexTensor(s) = &values[0] else {
            panic!("expected stft");
        };
        let Value::Tensor(t) = &values[2] else {
            panic!("expected t");
        };
        assert_eq!(default_window(1024).len(), 227);
        assert_eq!(s.shape[0], 129);
        assert_eq!(s.shape[1], t.data.len());
    }

    #[test]
    fn spectrogram_complex_defaults_to_twosided_and_rejects_onesided() {
        let data = (0..16).map(|idx| (idx as f64, 0.5)).collect();
        let x = runmat_builtins::ComplexTensor::new(data, vec![1, 16]).unwrap();
        let out = call(
            Value::ComplexTensor(x.clone()),
            &[Value::Num(8.0), Value::Num(4.0), Value::Num(8.0)],
            Some(3),
        )
        .unwrap();
        let values = output_list(out);
        let Value::ComplexTensor(s) = &values[0] else {
            panic!("expected stft");
        };
        assert_eq!(s.shape[0], 8);

        let err = call(
            Value::ComplexTensor(x),
            &[
                Value::Num(8.0),
                Value::Num(4.0),
                Value::Num(8.0),
                Value::from("onesided"),
            ],
            Some(3),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:spectrogram:InvalidOption"));
    }

    #[test]
    fn spectrogram_accepts_explicit_frequency_vector() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let freqs = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let out = call(
            Value::Tensor(x),
            &[Value::Num(4.0), Value::Num(2.0), Value::Tensor(freqs)],
            Some(4),
        )
        .unwrap();
        let values = output_list(out);
        let Value::ComplexTensor(s) = &values[0] else {
            panic!("expected stft");
        };
        let Value::Tensor(f) = &values[1] else {
            panic!("expected f");
        };
        assert_eq!(s.shape, vec![2, 3]);
        assert_eq!(f.data, vec![0.0, std::f64::consts::PI]);
    }

    #[test]
    fn spectrogram_power_scale_uses_coherent_gain() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let out = call(
            Value::Tensor(x),
            &[
                Value::Tensor(Tensor::new(vec![1.0; 4], vec![1, 4]).unwrap()),
                Value::Num(0.0),
                Value::Num(4.0),
                Value::Num(4.0),
                Value::from("power"),
            ],
            Some(4),
        )
        .unwrap();
        let values = output_list(out);
        let Value::Tensor(ps) = &values[3] else {
            panic!("expected ps");
        };
        assert!((ps.data[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn spectrogram_explicit_empty_fs_uses_one_hz_units() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let out = call(
            Value::Tensor(x),
            &[Value::Num(4.0), Value::Num(2.0), empty(), empty()],
            Some(4),
        )
        .unwrap();
        let values = output_list(out);
        let Value::Tensor(f) = &values[1] else {
            panic!("expected f");
        };
        let Value::Tensor(t) = &values[2] else {
            panic!("expected t");
        };
        assert_eq!(f.data[1], 1.0 / 256.0);
        assert_eq!(t.data[0], 2.0);
    }

    #[test]
    fn spectrogram_omitted_fs_uses_normalized_time_units() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let out = call(
            Value::Tensor(x),
            &[Value::Num(4.0), Value::Num(2.0), Value::Num(4.0)],
            Some(4),
        )
        .unwrap();
        let values = output_list(out);
        let Value::Tensor(t) = &values[2] else {
            panic!("expected t");
        };
        assert!((t.data[0] - (1.0 / std::f64::consts::PI)).abs() < 1e-12);
    }

    #[test]
    fn spectrogram_centered_even_nfft_uses_matlab_interval() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let out = call(
            Value::Tensor(x),
            &[
                Value::Num(4.0),
                Value::Num(0.0),
                Value::Num(4.0),
                Value::Num(4.0),
                Value::from("centered"),
            ],
            Some(2),
        )
        .unwrap();
        let values = output_list(out);
        let Value::Tensor(f) = &values[1] else {
            panic!("expected f");
        };
        assert_eq!(f.data, vec![-1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn spectrogram_reassigned_is_rejected_until_true_reassignment_exists() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let err = call(
            Value::Tensor(x),
            &[
                Value::Num(4.0),
                Value::Num(2.0),
                Value::Num(4.0),
                Value::from("reassigned"),
            ],
            Some(6),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:spectrogram:InvalidOption"));
    }

    #[test]
    fn spectrogram_rejects_invalid_overlap() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
        let err = call(
            Value::Tensor(x),
            &[Value::Num(4.0), Value::Num(4.0)],
            Some(3),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:spectrogram:InvalidOverlap"));
    }
}
