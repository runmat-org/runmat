use super::*;

#[derive(Debug)]
pub(super) struct CorrcoefArgs {
    pub(super) first: Value,
    pub(super) second: Option<Value>,
    pub(super) rows: CorrcoefRows,
    pub(super) alpha: f64,
}

impl CorrcoefArgs {
    pub(super) fn parse(first: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Self> {
        let mut values = rest.into_iter();
        let mut second = None;
        let mut rows = CorrcoefRows::All;
        let mut alpha = 0.05;
        let mut rows_seen = false;
        let mut alpha_seen = false;
        let mut pending = values.next();

        if pending.as_ref().is_some_and(|value| !is_text_value(value)) {
            second = pending.take();
            pending = values.next();
        }

        while let Some(name) = pending {
            let key = value_to_string(&name)
                .ok_or_else(|| {
                    if second.is_some() {
                        corrcoef_error(&CORRCOEF_ERROR_TOO_MANY_INPUT_ARRAYS)
                    } else {
                        corrcoef_error(&CORRCOEF_ERROR_INVALID_ARGUMENT)
                    }
                })?
                .trim()
                .to_ascii_lowercase();
            let option = values.next().ok_or_else(|| {
                corrcoef_error_with_detail(
                    &CORRCOEF_ERROR_ROWS_OPTION_MALFORMED,
                    format!("option '{key}' requires a value"),
                )
            })?;
            match CorrcoefOptionKey::parse(&key) {
                Some(CorrcoefOptionKey::Rows) => {
                    if rows_seen {
                        return Err(corrcoef_error_with_detail(
                            &CORRCOEF_ERROR_OPTION_DUPLICATE,
                            "'Rows'",
                        ));
                    }
                    let choice = value_to_string(&option).ok_or_else(|| {
                        corrcoef_error_with_detail(
                            &CORRCOEF_ERROR_ROWS_OPTION_MALFORMED,
                            "Rows must be a string value",
                        )
                    })?;
                    rows = parse_rows_option(choice.trim())?;
                    rows_seen = true;
                }
                Some(CorrcoefOptionKey::Alpha) => {
                    if alpha_seen {
                        return Err(corrcoef_error_with_detail(
                            &CORRCOEF_ERROR_OPTION_DUPLICATE,
                            "'Alpha'",
                        ));
                    }
                    alpha = parse_alpha(&option)?;
                    alpha_seen = true;
                }
                None => {
                    return Err(corrcoef_error_with_detail(
                        &CORRCOEF_ERROR_OPTION_UNKNOWN,
                        format!("'{key}'"),
                    ))
                }
            }
            pending = values.next();
        }

        Ok(Self {
            first,
            second,
            rows,
            alpha,
        })
    }
}

fn is_text_value(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::ComplexTensor(tensor) if tensor.integer_storage().is_some()
        )
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(
            value,
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle)
        )
}

pub(super) fn ensure_corrcoef_extensions(args: &CorrcoefArgs) -> BuiltinResult<()> {
    if is_typed_integer_value(&args.first)
        || args.second.as_ref().is_some_and(is_typed_integer_value)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CORRCOEF_INTEGER_DATA_EXTENSION,
            NAME,
        )?;
    }
    if is_logical_value(&args.first) || args.second.as_ref().is_some_and(is_logical_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CORRCOEF_LOGICAL_DATA_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

/// Check the data carriers and geometry without evaluating a correlation.
/// Discarded-output calls must observe the same input errors as ordinary
/// calls, but should not upload, download, or invoke a numerical provider.
pub(super) fn admit_corrcoef_inputs(args: &CorrcoefArgs) -> BuiltinResult<()> {
    for handle in std::iter::once(&args.first)
        .chain(args.second.as_ref())
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
    {
        if runmat_accelerate_api::numeric_readback_for_handle(handle).is_err() {
            let mut error =
                corrcoef_internal_error("corrcoef: resident operand has no exact owning provider");
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(error);
        }
        handle.allocation_lease().validate_live().map_err(|error| {
            corrcoef_provider_error(runmat_accelerate_api::ProviderExecutionError::from_source(
                error,
            ))
        })?;
        let element =
            gpu_helpers::expected_handle_numeric_element_type(handle).map_err(|error| {
                let mut fault = corrcoef_internal_error(format!(
                    "corrcoef: invalid resident physical element type: {error}"
                ));
                fault.gpu_gather_retry = crate::GpuGatherRetry::Never;
                fault
            })?;
        let storage = runmat_accelerate_api::handle_storage(handle);
        let precision = runmat_accelerate_api::handle_precision(handle);
        let integer = runmat_accelerate_api::handle_integer_type(handle);
        let logical = runmat_accelerate_api::handle_is_logical(handle);
        if handle.view().validate(handle.physical()).is_err()
            || !gpu_helpers::numeric_descriptor_matches(handle, element, storage)
            || !gpu_helpers::gpu_class_metadata_matches(handle, precision, integer, logical)
        {
            let mut error = corrcoef_internal_error("corrcoef: invalid resident descriptor");
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(error);
        }
    }
    let first_shape = admissible_shape(&args.first)?;
    if first_shape.len() > 2 {
        return Err(corrcoef_error_with_detail(
            &CORRCOEF_ERROR_INVALID_ARGUMENT,
            "A must be a matrix",
        ));
    }
    if let Some(second) = &args.second {
        let second_shape = admissible_shape(second)?;
        if !same_size(first_shape, second_shape) {
            return Err(corrcoef_error(&CORRCOEF_ERROR_SIZE_MISMATCH));
        }
    }
    Ok(())
}

fn admissible_shape(value: &Value) -> BuiltinResult<&[usize]> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.shape()),
        Value::ComplexTensor(tensor) => Ok(tensor.shape()),
        Value::LogicalArray(array) => Ok(&array.shape),
        Value::GpuTensor(handle) => Ok(handle.shape()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(&[1, 1]),
        other => Err(corrcoef_internal_error(format!(
            "corrcoef: unsupported input type {other:?}; expected numeric or logical values"
        ))),
    }
}

fn parse_rows_option(value: &str) -> BuiltinResult<CorrcoefRows> {
    match CorrcoefRowsOption::parse(value) {
        Some(CorrcoefRowsOption::All) => Ok(CorrcoefRows::All),
        Some(CorrcoefRowsOption::Complete) => Ok(CorrcoefRows::Complete),
        Some(CorrcoefRowsOption::Pairwise) => Ok(CorrcoefRows::Pairwise),
        None => Err(corrcoef_error_with_detail(
            &CORRCOEF_ERROR_ROWS_OPTION_UNKNOWN,
            format!("'{value}'"),
        )),
    }
}

fn parse_alpha(value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) if value.is_finite() && *value > 0.0 && *value < 1.0 => Ok(*value),
        Value::Tensor(tensor)
            if tensor.len() == 1
                && tensor.integer_storage().is_none()
                && matches!(
                    tensor.numeric_dtype(),
                    runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64
                ) =>
        {
            let value = tensor.materialize_f64()[0];
            if value.is_finite() && value > 0.0 && value < 1.0 {
                Ok(value)
            } else {
                Err(corrcoef_error(&CORRCOEF_ERROR_ALPHA_INVALID))
            }
        }
        _ => Err(corrcoef_error(&CORRCOEF_ERROR_ALPHA_INVALID)),
    }
}
