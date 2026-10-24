use super::*;

use PwelchControlRole as ControlRole;
use PwelchKeyword as WelchKeyword;

fn role_at_positional_index(index: usize) -> ControlRole {
    match index {
        0 => ControlRole::Window,
        1 => ControlRole::Overlap,
        2 => ControlRole::Grid,
        3 => ControlRole::SampleRate,
        _ => ControlRole::Option,
    }
}

fn integer_boundary(role: ControlRole) -> Option<(&'static str, &'static BuiltinErrorDescriptor)> {
    match role {
        ControlRole::Window => Some(("window", &PWELCH_ERROR_INVALID_WINDOW)),
        ControlRole::Overlap => Some(("overlap", &PWELCH_ERROR_INVALID_OVERLAP)),
        ControlRole::Grid => Some(("frequency or DFT-length", &PWELCH_ERROR_INVALID_NFFT)),
        ControlRole::SampleRate => Some(("sample-rate", &PWELCH_ERROR_INVALID_FS)),
        ControlRole::Option => None,
    }
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
pub(super) async fn preflight_host_controls(
    rest: &[&Value],
    tokens: &[Option<PwelchKeyword>],
    complex_input: bool,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PreparedHostControls> {
    let mut prepared = PreparedHostControls::default();
    let mut cursor = PwelchControlCursor::default();
    for (value, decoded) in rest.iter().zip(tokens) {
        cooperation.completed(1).await?;
        let role = cursor.next(decoded.as_ref());
        if role == ControlRole::Option {
            if complex_input && matches!(decoded, Some(WelchKeyword::Onesided)) {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_RANGE));
            }
            if !decoded.as_ref().is_some_and(PwelchKeyword::is_selector) {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_RANGE));
            }
            continue;
        }
        if matches!(value, Value::GpuTensor(_)) || is_empty(value) {
            continue;
        }
        let (name, error) = integer_boundary(role).expect("option handled above");
        ensure_pwelch_integer_boundary(
            value,
            name,
            error,
            &PWELCH_INTEGER_CONTROL_EXTENSION,
            cooperation,
        )
        .await?;
        match role {
            ControlRole::Window => {
                // A text value in the first position has its owning window error.
                if let Some(option) = decoded {
                    return Err(pwelch_error_with_detail(
                        &PWELCH_ERROR_INVALID_WINDOW,
                        format!(
                            "unexpected option '{}' before window arguments",
                            option.label()
                        ),
                    ));
                }
                prepared.window = Some(parse_window(value, cooperation).await?);
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
                prepared.grid = Some(parse_frequency_grid(value, cooperation).await?);
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
    rest: &[&Value],
    prepared: &mut PreparedHostControls,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Option<Value>>> {
    let mut materialized = Vec::with_capacity(rest.len());
    for (index, value) in rest.iter().enumerate() {
        cooperation.completed(1).await?;
        if let Value::GpuTensor(handle) = value {
            let provider = super::super::resident_output::owner(handle, || {
                pwelch_error(&PWELCH_ERROR_INTERNAL)
            })?;
            let host =
                crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
                    provider, handle,
                )
                .await?;
            cooperation.completed(0).await?;
            if let Some((role, error)) = integer_boundary(role_at_positional_index(index)) {
                ensure_pwelch_integer_boundary(
                    &host,
                    role,
                    error,
                    &PWELCH_INTEGER_CONTROL_EXTENSION,
                    cooperation,
                )
                .await?;
            }
            if !is_empty(&host) {
                match role_at_positional_index(index) {
                    ControlRole::Window => {
                        prepared.window = Some(parse_window(&host, cooperation).await?);
                    }
                    ControlRole::Overlap => {
                        prepared.overlap = Some(
                            parse_nonnegative_integer(BUILTIN_NAME, "noverlap", &host).map_err(
                                |error| {
                                    pwelch_error_with_detail(
                                        &PWELCH_ERROR_INVALID_OVERLAP,
                                        error.message(),
                                    )
                                },
                            )?,
                        );
                    }
                    ControlRole::Grid => {
                        prepared.grid = Some(parse_frequency_grid(&host, cooperation).await?);
                    }
                    ControlRole::SampleRate => {
                        let rate =
                            parse_scalar_f64(BUILTIN_NAME, "fs", &host).map_err(|error| {
                                pwelch_error_with_detail(&PWELCH_ERROR_INVALID_FS, error.message())
                            })?;
                        if rate <= 0.0 {
                            return Err(pwelch_error(&PWELCH_ERROR_INVALID_FS));
                        }
                        prepared.sample_rate = Some(rate);
                    }
                    ControlRole::Option => return Err(pwelch_error(&PWELCH_ERROR_INVALID_RANGE)),
                }
            }
            materialized.push(Some(host));
        } else {
            materialized.push(None);
        }
    }
    Ok(materialized)
}

pub(super) async fn parse_options(
    rest: &[&Value],
    keywords: &[Option<PwelchKeyword>],
    input_len: usize,
    complex_input: bool,
    prepared: &mut PreparedHostControls,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PwelchOptions> {
    let mut idx = 0usize;
    let window = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            default_window(input_len, cooperation).await?
        } else if let Some(keyword) = keywords.get(idx).and_then(Option::as_ref) {
            if matches!(keyword, WelchKeyword::Unknown(_)) {
                return Err(pwelch_error_with_detail(
                    &PWELCH_ERROR_INVALID_WINDOW,
                    format!("unknown window option '{}'", keyword.label()),
                ));
            }
            default_window(input_len, cooperation).await?
        } else {
            idx += 1;
            match prepared.window.take() {
                Some(window) => window,
                None => parse_window(value, cooperation).await?,
            }
        }
    } else {
        default_window(input_len, cooperation).await?
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
                nfft: default_nfft(window_len)?,
            }
        } else if keywords.get(idx).and_then(Option::as_ref).is_some() {
            FrequencyGrid::Uniform {
                nfft: default_nfft(window_len)?,
            }
        } else {
            idx += 1;
            match prepared.grid.take() {
                Some(grid) => grid,
                None => parse_frequency_grid(value, cooperation).await?,
            }
        }
    } else {
        FrequencyGrid::Uniform {
            nfft: default_nfft(window_len)?,
        }
    };
    if let FrequencyGrid::Uniform { nfft } = &grid {
        if *nfft == 0 || *nfft > MAX_NFFT {
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_NFFT));
        }
    }

    let fs = if let Some(value) = rest.get(idx) {
        if is_empty(value) {
            idx += 1;
            // An explicit empty Fs selects one cycle per unit time; omission
            // retains the normalized angular-frequency default.
            Some(1.0)
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
    let mut trace = WelchTrace::Mean;
    while rest.get(idx).is_some() {
        cooperation.completed(1).await?;
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
            WelchKeyword::Mean => trace = WelchTrace::Mean,
            WelchKeyword::MaxHold => trace = WelchTrace::MaxHold,
            WelchKeyword::MinHold => trace = WelchTrace::MinHold,
            WelchKeyword::Parent => return Err(pwelch_error(&PWELCH_ERROR_INVALID_RANGE)),
            WelchKeyword::Unknown(_) => return Err(pwelch_error(&PWELCH_ERROR_INVALID_RANGE)),
        }
    }

    if complex_input && range == FrequencyRange::Onesided {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_RANGE));
    }

    Ok(PwelchOptions {
        window,
        noverlap,
        grid,
        fs,
        range,
        scale,
        trace,
    })
}

fn default_nfft(window_len: usize) -> BuiltinResult<usize> {
    window_len
        .checked_next_power_of_two()
        .map(|power_of_two| DEFAULT_NFFT_MIN.max(power_of_two))
        .filter(|nfft| *nfft <= MAX_NFFT)
        .ok_or_else(|| pwelch_error(&PWELCH_ERROR_INVALID_NFFT))
}

pub(super) async fn parse_frequency_grid(
    value: &Value,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<FrequencyGrid> {
    cooperation.completed(0).await?;
    let over_limit = match value {
        Value::Tensor(tensor) => tensor_utils::tensor_element_len(tensor) > MAX_NFFT,
        Value::ComplexTensor(tensor) => tensor_utils::complex_tensor_element_len(tensor) > MAX_NFFT,
        Value::LogicalArray(array) => array.data.len() > MAX_NFFT,
        _ => false,
    };
    if over_limit {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_NFFT));
    }
    if is_scalar_numeric(value) {
        let parsed = parse_nonnegative_integer(BUILTIN_NAME, "nfft", value)
            .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INVALID_NFFT, err.message()))?;
        if parsed == 0 || parsed > MAX_NFFT {
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_NFFT));
        }
        return Ok(FrequencyGrid::Uniform { nfft: parsed });
    }

    let vector = value_to_complex_vector(BUILTIN_NAME, "nfft", value.clone())
        .await
        .map_err(|source| pwelch_control_error(&PWELCH_ERROR_INVALID_NFFT, source))?;
    if vector.data.is_empty() || vector.data.len() > MAX_NFFT {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_NFFT));
    }
    let mut frequencies = Vec::new();
    frequencies
        .try_reserve_exact(vector.data.len())
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    let mut work = CompletedWork::default();
    for value in vector.data {
        if let Some(units) = work.record(1) {
            cooperation.completed(units).await?;
        }
        if value.im != 0.0 || !value.re.is_finite() {
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_NFFT));
        }
        frequencies.push(value.re);
    }
    work.finish(cooperation).await?;
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

pub(super) async fn parse_window(
    value: &Value,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<f64>> {
    cooperation.completed(0).await?;
    let over_limit = match value {
        Value::Tensor(tensor) => tensor_utils::tensor_element_len(tensor) > MAX_NFFT,
        Value::ComplexTensor(tensor) => tensor_utils::complex_tensor_element_len(tensor) > MAX_NFFT,
        Value::LogicalArray(array) => array.data.len() > MAX_NFFT,
        _ => false,
    };
    if over_limit {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
    }
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let len = parse_nonnegative_integer(BUILTIN_NAME, "window", value).map_err(|err| {
                pwelch_error_with_detail(&PWELCH_ERROR_INVALID_WINDOW, err.message())
            })?;
            if len == 0 {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
            }
            if len > MAX_NFFT {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
            }
            return symmetric_window_values(len, hamming_coefficient, cooperation)
                .await?
                .map_err(pwelch_window_generation_error);
        }
        Value::Tensor(t) if tensor_utils::is_scalar_tensor(t) => {
            let len = parse_nonnegative_integer(BUILTIN_NAME, "window", value).map_err(|err| {
                pwelch_error_with_detail(&PWELCH_ERROR_INVALID_WINDOW, err.message())
            })?;
            if len == 0 {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
            }
            if len > MAX_NFFT {
                return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
            }
            return symmetric_window_values(len, hamming_coefficient, cooperation)
                .await?
                .map_err(pwelch_window_generation_error);
        }
        _ => {}
    }
    let vector = value_to_complex_vector(BUILTIN_NAME, "window", value.clone())
        .await
        .map_err(|source| pwelch_control_error(&PWELCH_ERROR_INVALID_WINDOW, source))?;
    if vector.data.is_empty() || vector.data.len() > MAX_NFFT {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(vector.data.len())
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    let mut work = CompletedWork::default();
    for value in vector.data {
        if let Some(units) = work.record(1) {
            cooperation.completed(units).await?;
        }
        if value.im != 0.0 || !value.re.is_finite() {
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
        }
        out.push(value.re);
    }
    work.finish(cooperation).await?;
    Ok(out)
}
