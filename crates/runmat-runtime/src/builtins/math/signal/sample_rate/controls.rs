use super::*;

#[derive(Clone, Copy)]
pub(super) enum FilterPreparation {
    ValidateOnly,
    Prepare,
}

pub(super) async fn parse_resample_options(
    rest: crate::call::BuiltinArguments,
    p: usize,
    q: usize,
) -> BuiltinResult<ResampleOptions> {
    parse_resample_options_inner(rest, p, q, FilterPreparation::Prepare)
        .await
        .map(|options| options.expect("prepared filter mode produces options"))
}

pub(super) async fn validate_resample_options(
    rest: crate::call::BuiltinArguments,
    p: usize,
    q: usize,
) -> BuiltinResult<()> {
    parse_resample_options_inner(rest, p, q, FilterPreparation::ValidateOnly)
        .await
        .map(|_| ())
}

async fn parse_resample_options_inner(
    rest: crate::call::BuiltinArguments,
    p: usize,
    q: usize,
    preparation: FilterPreparation,
) -> BuiltinResult<Option<ResampleOptions>> {
    let mut idx = 0usize;
    let mut positional = Vec::new();
    let mut dim = None;
    while idx < rest.len() {
        if let Some(key) = value_as_text(&rest[idx])? {
            if key.eq_ignore_ascii_case("dimension") {
                let Some(value) = rest.get(idx + 1) else {
                    return Err(sample_error_with_detail(
                        RESAMPLE_NAME,
                        &SAMPLE_ERROR_INVALID_OPTION,
                        "Dimension name-value option requires a value",
                    ));
                };
                dim = Some(
                    tensor::parse_dimension(value, RESAMPLE_NAME)
                        .map_err(|err| {
                            sample_error_with_detail(
                                RESAMPLE_NAME,
                                &SAMPLE_ERROR_INVALID_OPTION,
                                err,
                            )
                        })?
                        .saturating_sub(1),
                );
                idx += 2;
                continue;
            }
        }
        positional.push(rest[idx].clone());
        idx += 1;
    }

    let filter = parse_resample_filter_spec(&positional, p, q, preparation).await?;
    Ok(filter.map(|filter| ResampleOptions { filter, dim }))
}

async fn parse_resample_filter_spec(
    args: &[Value],
    p: usize,
    q: usize,
    preparation: FilterPreparation,
) -> BuiltinResult<Option<ResampleFilter>> {
    match args.len() {
        0 => prepare_filter_design(p, q, DEFAULT_RESAMPLE_N, DEFAULT_RESAMPLE_BETA, preparation),
        1 => {
            if let Some(n) = scalar_integer_option(&args[0]).await? {
                prepare_filter_design(p, q, n, DEFAULT_RESAMPLE_BETA, preparation)
            } else {
                parse_filter_vector(args[0].clone(), preparation).await
            }
        }
        2 => {
            let Some(n) = scalar_integer_option(&args[0]).await? else {
                return Err(sample_error_with_detail(
                    RESAMPLE_NAME,
                    &SAMPLE_ERROR_INVALID_OPTION,
                    "N must be a nonnegative integer scalar when BETA is provided",
                ));
            };
            let beta = scalar_f64_option(&args[1]).await?;
            prepare_filter_design(p, q, n, beta, preparation)
        }
        _ => Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_ARG_COUNT,
            "resample accepts X, P, Q, optional N/BETA or B, and optional Dimension",
        )),
    }
}

fn prepare_filter_design(
    p: usize,
    q: usize,
    n: usize,
    beta: f64,
    preparation: FilterPreparation,
) -> BuiltinResult<Option<ResampleFilter>> {
    if matches!(preparation, FilterPreparation::ValidateOnly) {
        resample_filter_len(p, q, n)?;
        Ok(None)
    } else {
        design_resample_filter(p, q, n, beta)
            .map(ResampleFilter::Host)
            .map(Some)
    }
}

/// Authenticate a resident scalar before any provider call, then materialize
/// its native class exactly once. Integer structure is decoded from the
/// resulting host value rather than round-tripped through binary64.
async fn admitted_scalar_control(
    value: &Value,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<Value> {
    let Value::GpuTensor(handle) = value else {
        return Ok(value.clone());
    };
    if checked_product(handle.shape()) != Some(1)
        || runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Err(sample_error_with_detail(
            builtin,
            error,
            "resident control must be a real numeric scalar",
        ));
    }
    let provider = resolved_actual_downsample_owner(handle).ok_or_else(|| {
        sample_error_with_detail(
            builtin,
            error,
            "resident control has no valid live provider owner or numeric descriptor",
        )
    })?;
    gpu_helpers::download_value_preserving_residency_async(provider, handle)
        .await
        .map_err(|source| {
            sample_error_with_source(builtin, error, "resident control download failed", source)
        })
}

pub(super) async fn scalar_integer_option(value: &Value) -> BuiltinResult<Option<usize>> {
    match value {
        Value::Tensor(tensor) if !tensor::is_scalar_tensor(tensor) => return Ok(None),
        Value::GpuTensor(handle) if checked_product(handle.shape()) != Some(1) => return Ok(None),
        Value::ComplexTensor(_) => return Ok(None),
        _ => {}
    }
    let admitted =
        admitted_scalar_control(value, RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION).await?;
    if let Some(integer) = tensor::scalar_integer_value(&admitted) {
        return parse_nonnegative_integer_value(
            integer,
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
        )
        .map(Some);
    }
    let Some(raw) = tensor::scalar_f64_from_host_value(&admitted).map_err(|err| {
        sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION, err)
    })?
    else {
        return Ok(None);
    };
    parse_nonnegative_integer(raw, RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION).map(Some)
}

pub(super) async fn scalar_f64_option(value: &Value) -> BuiltinResult<f64> {
    let admitted =
        admitted_scalar_control(value, RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION).await?;
    if tensor::scalar_integer_value(&admitted)
        .is_some_and(|integer| !crate::builtins::common::validation::integer_is_exact_f64(&integer))
    {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "typed integer BETA must be exactly representable as double",
        ));
    }
    let Some(raw) = tensor::scalar_f64_from_host_value(&admitted).map_err(|err| {
        sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION, err)
    })?
    else {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            format!("expected scalar option, received {value:?}"),
        ));
    };
    if !raw.is_finite() || raw < 0.0 {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "BETA must be a finite nonnegative scalar",
        ));
    }
    Ok(raw)
}

pub(super) async fn parse_filter_vector(
    value: Value,
    preparation: FilterPreparation,
) -> BuiltinResult<Option<ResampleFilter>> {
    if let Value::GpuTensor(handle) = &value {
        let len = checked_product(handle.shape())
            .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
        validate_filter_shape(len, handle.shape())?;
        if runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
            || resolved_actual_downsample_owner(handle).is_none()
        {
            return Err(sample_error_with_detail(
                RESAMPLE_NAME,
                &SAMPLE_ERROR_INVALID_OPTION,
                "resident filter must have real storage and a valid live provider owner",
            ));
        }
    }
    if is_typed_integer_value(&value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await
            .map_err(|source| {
                sample_error_with_source(
                    RESAMPLE_NAME,
                    &SAMPLE_ERROR_INVALID_OPTION,
                    "filter exactness admission failed",
                    source,
                )
            })?
    {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "typed integer filter coefficients must be exactly representable as double",
        ));
    }
    if let Value::GpuTensor(handle) = value {
        let len = checked_product(handle.shape())
            .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
        return Ok(matches!(preparation, FilterPreparation::Prepare)
            .then_some(ResampleFilter::Gpu { handle, len }));
    }

    if matches!(preparation, FilterPreparation::ValidateOnly) {
        if let Value::LogicalArray(logical) = &value {
            // The prepared path converts every byte to f64. A discarded output
            // needs only the authoritative logical payload/shape admission.
            if checked_product(&logical.shape) != Some(logical.len()) {
                return Err(sample_error_with_detail(
                    RESAMPLE_NAME,
                    &SAMPLE_ERROR_INVALID_OPTION,
                    "logical filter shape does not match its payload",
                ));
            }
            validate_filter_shape(logical.len(), &logical.shape)?;
            return Ok(None);
        }
    }

    let tensor = tensor::value_into_tensor_for(RESAMPLE_NAME, value).map_err(|err| {
        sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION, err)
    })?;
    let shape = tensor.shape().to_vec();
    validate_filter_shape(tensor.len(), &shape)?;
    Ok(matches!(preparation, FilterPreparation::Prepare)
        .then(|| ResampleFilter::Host(tensor::tensor_into_values_f64(tensor))))
}

pub(super) fn validate_filter_shape(len: usize, shape: &[usize]) -> BuiltinResult<()> {
    if len == 0 {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "filter coefficient vector cannot be empty",
        ));
    }
    if shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "filter coefficients must be a vector",
        ));
    }
    Ok(())
}

pub(super) async fn parse_factor(value: &Value, builtin: &'static str) -> BuiltinResult<usize> {
    let admitted = admitted_scalar_control(value, builtin, &SAMPLE_ERROR_INVALID_FACTOR).await?;
    if let Some(integer) = tensor::scalar_integer_value(&admitted) {
        return parse_positive_integer_value(integer, builtin, &SAMPLE_ERROR_INVALID_FACTOR);
    }
    let Some(raw) = tensor::scalar_f64_from_host_value(&admitted).map_err(|detail| {
        sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_FACTOR, detail)
    })?
    else {
        return Err(sample_error_with_detail(
            builtin,
            &SAMPLE_ERROR_INVALID_FACTOR,
            format!("received {value:?}"),
        ));
    };
    parse_positive_integer(raw, builtin, &SAMPLE_ERROR_INVALID_FACTOR)
}

pub(super) async fn parse_phase(
    value: &Value,
    factor: usize,
    builtin: &'static str,
) -> BuiltinResult<usize> {
    let admitted = admitted_scalar_control(value, builtin, &SAMPLE_ERROR_INVALID_PHASE).await?;
    let phase = if let Some(integer) = tensor::scalar_integer_value(&admitted) {
        parse_nonnegative_integer_value(integer, builtin, &SAMPLE_ERROR_INVALID_PHASE)?
    } else {
        let Some(raw) = tensor::scalar_f64_from_host_value(&admitted).map_err(|detail| {
            sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_PHASE, detail)
        })?
        else {
            return Err(sample_error_with_detail(
                builtin,
                &SAMPLE_ERROR_INVALID_PHASE,
                format!("received {value:?}"),
            ));
        };
        parse_nonnegative_integer(raw, builtin, &SAMPLE_ERROR_INVALID_PHASE)?
    };
    if phase >= factor {
        return Err(sample_error_with_detail(
            builtin,
            &SAMPLE_ERROR_INVALID_PHASE,
            format!("phase {phase} is not less than N ({factor})"),
        ));
    }
    Ok(phase)
}

pub(super) fn parse_positive_integer(
    raw: f64,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<usize> {
    let value = parse_nonnegative_integer(raw, builtin, error)?;
    if value == 0 {
        return Err(sample_error(builtin, error));
    }
    Ok(value)
}

pub(super) fn parse_positive_integer_value(
    value: runmat_value::IntValue,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<usize> {
    let value = parse_nonnegative_integer_value(value, builtin, error)?;
    if value == 0 {
        return Err(sample_error(builtin, error));
    }
    Ok(value)
}

pub(super) fn parse_nonnegative_integer_value(
    value: runmat_value::IntValue,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .ok_or_else(|| sample_error(builtin, error))
}

pub(super) fn parse_nonnegative_integer(
    raw: f64,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<usize> {
    if !raw.is_finite() || raw < 0.0 {
        return Err(sample_error(builtin, error));
    }
    if raw.trunc() != raw
        || raw > usize::MAX as f64
        || (usize::BITS == 64 && raw == usize::MAX as f64)
    {
        return Err(sample_error(builtin, error));
    }
    Ok(raw as usize)
}

pub(super) fn value_as_text(value: &Value) -> crate::BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => Some(text.as_str().to_owned()),
        Value::CharArray(chars) if chars.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                let mut error = sample_error_with_detail(
                    RESAMPLE_NAME,
                    &SAMPLE_ERROR_INVALID_OPTION,
                    source.to_string(),
                );
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
}

pub(super) fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a.max(1)
}
