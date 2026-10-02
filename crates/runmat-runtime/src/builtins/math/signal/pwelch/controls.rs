use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
enum WelchKeyword {
    Onesided,
    Twosided,
    Centered,
    Psd,
    Power,
    Unknown(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControlRole {
    Window,
    Overlap,
    Grid,
    SampleRate,
    Option,
}

#[derive(Default)]
pub(super) struct PreparedHostControls {
    window: Option<Vec<f64>>,
    overlap: Option<usize>,
    grid: Option<FrequencyGrid>,
    sample_rate: Option<f64>,
}

/// Validate every host-visible control before a resident operand is read.
/// Parsed host payloads are retained, so the execution parser never repeats
/// large window or frequency-vector work.
pub(super) async fn preflight_host_controls(rest: &[Value]) -> BuiltinResult<PreparedHostControls> {
    let mut prepared = PreparedHostControls::default();
    let mut next_positional = ControlRole::Window;
    let mut options_started = false;
    for (index, value) in rest.iter().enumerate() {
        let decoded = keyword(value);
        let role = if options_started || (decoded.is_some() && index > 0) {
            options_started = true;
            ControlRole::Option
        } else {
            let role = next_positional;
            next_positional = match role {
                ControlRole::Window => ControlRole::Overlap,
                ControlRole::Overlap => ControlRole::Grid,
                ControlRole::Grid => ControlRole::SampleRate,
                ControlRole::SampleRate | ControlRole::Option => ControlRole::Option,
            };
            role
        };
        if role == ControlRole::Option {
            if !matches!(
                decoded,
                Some(
                    WelchKeyword::Onesided
                        | WelchKeyword::Twosided
                        | WelchKeyword::Centered
                        | WelchKeyword::Psd
                        | WelchKeyword::Power
                )
            ) {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_RANGE));
            }
            continue;
        }
        if matches!(value, Value::GpuTensor(_)) || is_empty(value) {
            continue;
        }
        match role {
            ControlRole::Window => {
                // A text value in the first position has its owning window error.
                if let Some(option) = decoded {
                    return Err(pwelch_error_with_detail(
                        &PWELCH_ERROR_INVALID_WINDOW,
                        format!(
                            "unexpected option '{}' before window arguments",
                            keyword_label(&option)
                        ),
                    ));
                }
                prepared.window = Some(parse_window(value.clone()).await?);
            }
            ControlRole::Overlap => {
                prepared.overlap = Some(
                    parse_nonnegative_integer(BUILTIN_NAME, "noverlap", value).map_err(
                        |error| {
                            pwelch_error_with_detail(&PWELCH_ERROR_INVALID_OVERLAP, error.message())
                        },
                    )?,
                );
            }
            ControlRole::Grid => {
                prepared.grid = Some(parse_frequency_grid(value.clone(), 0).await?);
            }
            ControlRole::SampleRate => {
                let rate = parse_scalar_f64(BUILTIN_NAME, "fs", value).map_err(|error| {
                    pwelch_error_with_detail(&PWELCH_ERROR_INVALID_FS, error.message())
                })?;
                if rate <= 0.0 {
                    return Err(pwelch_error(&PWELCH_ERROR_INVALID_FS));
                }
                prepared.sample_rate = Some(rate);
            }
            ControlRole::Option => unreachable!("handled above"),
        }
    }
    Ok(prepared)
}

pub(super) async fn materialize_resident_controls(
    rest: &[Value],
) -> BuiltinResult<Vec<Option<Value>>> {
    let mut materialized = Vec::with_capacity(rest.len());
    for value in rest {
        if let Value::GpuTensor(handle) = value {
            let provider = super::super::resident_output::owner(handle, || {
                pwelch_error(&PWELCH_ERROR_INTERNAL)
            })?;
            let host =
                crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
                    provider, handle,
                )
                .await?;
            materialized.push(Some(host));
        } else {
            materialized.push(None);
        }
    }
    Ok(materialized)
}

fn keyword(value: &Value) -> Option<WelchKeyword> {
    let text = match value {
        Value::String(text) => text,
        Value::StringArray(array) if array.data.len() == 1 => &array.data[0],
        Value::CharArray(chars) if chars.rows <= 1 => {
            let text = chars.data.iter().collect::<String>();
            return Some(decode_keyword(&text));
        }
        _ => return None,
    };
    Some(decode_keyword(text))
}

fn decode_keyword(text: &str) -> WelchKeyword {
    let normalized = text.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "onesided" => WelchKeyword::Onesided,
        "twosided" => WelchKeyword::Twosided,
        "centered" => WelchKeyword::Centered,
        "psd" => WelchKeyword::Psd,
        "power" => WelchKeyword::Power,
        _ => WelchKeyword::Unknown(normalized),
    }
}

fn keyword_label(keyword: &WelchKeyword) -> &str {
    match keyword {
        WelchKeyword::Onesided => "onesided",
        WelchKeyword::Twosided => "twosided",
        WelchKeyword::Centered => "centered",
        WelchKeyword::Psd => "psd",
        WelchKeyword::Power => "power",
        WelchKeyword::Unknown(value) => value,
    }
}

pub(super) async fn parse_options(
    rest: &[&Value],
    input_len: usize,
    complex_input: bool,
    prepared: &mut PreparedHostControls,
) -> BuiltinResult<PwelchOptions> {
    let keywords = rest.iter().map(|value| keyword(value)).collect::<Vec<_>>();
    let mut idx = 0usize;
    let window = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            default_window(input_len)
        } else if let Some(keyword) = keywords.get(idx).and_then(Option::as_ref) {
            return Err(pwelch_error_with_detail(
                &PWELCH_ERROR_INVALID_WINDOW,
                format!(
                    "unexpected option '{}' before window arguments",
                    keyword_label(&keyword)
                ),
            ));
        } else {
            idx += 1;
            match prepared.window.take() {
                Some(window) => window,
                None => parse_window((*value).clone()).await?,
            }
        }
    } else {
        default_window(input_len)
    };
    let window_len = window.len();

    let noverlap = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            window_len / 2
        } else if keywords.get(idx).and_then(Option::as_ref).is_some() {
            window_len / 2
        } else {
            idx += 1;
            let parsed = prepared.overlap.take().map(Ok).unwrap_or_else(|| {
                parse_nonnegative_integer(BUILTIN_NAME, "noverlap", value).map_err(|err| {
                    pwelch_error_with_detail(&PWELCH_ERROR_INVALID_OVERLAP, err.message())
                })
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
        } else if keywords.get(idx).and_then(Option::as_ref).is_some() {
            FrequencyGrid::Uniform {
                nfft: DEFAULT_NFFT_MIN.max(window_len),
            }
        } else {
            idx += 1;
            match prepared.grid.take() {
                Some(grid) => grid,
                None => parse_frequency_grid((*value).clone(), window_len).await?,
            }
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
        } else if keywords.get(idx).and_then(Option::as_ref).is_some() {
            None
        } else {
            idx += 1;
            let parsed = prepared.sample_rate.take().map(Ok).unwrap_or_else(|| {
                parse_scalar_f64(BUILTIN_NAME, "fs", value).map_err(|err| {
                    pwelch_error_with_detail(&PWELCH_ERROR_INVALID_FS, err.message())
                })
            })?;
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
    while rest.get(idx).is_some() {
        let Some(keyword) = keywords.get(idx).and_then(Option::as_ref) else {
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_RANGE));
        };
        idx += 1;
        match keyword {
            WelchKeyword::Onesided | WelchKeyword::Twosided | WelchKeyword::Centered
                if matches!(grid, FrequencyGrid::Explicit { .. }) =>
            {
                return Err(pwelch_error_with_detail(
                    &PWELCH_ERROR_INVALID_RANGE,
                    "range selectors cannot be combined with an explicit frequency vector",
                ));
            }
            WelchKeyword::Onesided => range = FrequencyRange::Onesided,
            WelchKeyword::Twosided => range = FrequencyRange::Twosided,
            WelchKeyword::Centered => range = FrequencyRange::Centered,
            WelchKeyword::Psd => scale = SpectrumScale::Psd,
            WelchKeyword::Power => scale = SpectrumScale::Power,
            WelchKeyword::Unknown(_) => return Err(pwelch_error(&PWELCH_ERROR_INVALID_RANGE)),
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

pub(super) async fn parse_frequency_grid(
    value: Value,
    window_len: usize,
) -> BuiltinResult<FrequencyGrid> {
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
        .map_err(|source| pwelch_control_error(&PWELCH_ERROR_INVALID_NFFT, source))?;
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

pub(super) fn is_scalar_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor_utils::is_scalar_tensor(tensor),
        Value::ComplexTensor(tensor) => tensor_utils::is_scalar_complex_tensor(tensor),
        Value::LogicalArray(logical) => logical.data.len() == 1,
        _ => false,
    }
}

pub(super) async fn parse_window(value: Value) -> BuiltinResult<Vec<f64>> {
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
        Value::Tensor(t) if tensor_utils::is_scalar_tensor(t) => {
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
        .map_err(|source| pwelch_control_error(&PWELCH_ERROR_INVALID_WINDOW, source))?;
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
