use super::*;

#[derive(Debug)]
pub(super) struct CovArgs {
    pub(super) first: Value,
    pub(super) second: Option<Value>,
    pub(super) normalization: CovNormalization,
    pub(super) rows: CovRows,
    pub(super) weight_vector: Option<Value>,
    pub(super) typed_normalization: bool,
}

impl CovArgs {
    pub(super) fn parse(first: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Self> {
        let mut second_candidate: Option<Value> = None;
        let mut weight_candidate: Option<Value> = None;
        let mut normalization = CovNormalization::Unbiased;
        let mut normalization_explicit = false;
        let mut typed_normalization = false;
        let mut rows = CovRows::All;

        let iter = rest.into_iter();
        for arg in iter {
            match arg {
                Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                    let key = tensor::value_to_string(&arg)
                        .ok_or_else(|| cov_error(&COV_ERROR_INVALID_ARGUMENT))?;
                    let lowered = key.trim().to_ascii_lowercase();
                    rows = parse_rows_option(&lowered)?;
                }
                Value::Tensor(_) | Value::LogicalArray(_) | Value::GpuTensor(_) => {
                    if second_candidate.is_none() {
                        second_candidate = Some(arg);
                    } else if weight_candidate.is_none() {
                        weight_candidate = Some(arg);
                    } else {
                        return Err(cov_error(&COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS));
                    }
                }
                Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
                    if normalization_explicit || weight_candidate.is_some() {
                        return Err(cov_error(&COV_ERROR_NORMALIZATION_DUPLICATE));
                    }
                    typed_normalization = matches!(arg, Value::Int(_) | Value::Bool(_));
                    normalization = parse_normalization(arg)?;
                    normalization_explicit = true;
                }
                Value::ComplexTensor(_) => {
                    return Err(cov_error(&COV_ERROR_COMPLEX_UNSUPPORTED));
                }
                other => {
                    return Err(cov_error_with_detail(
                        &COV_ERROR_INVALID_ARGUMENT,
                        format!("{other:?}"),
                    ))
                }
            }
        }

        if let Some(weight_array) = weight_candidate {
            // Explicit weight vector always takes precedence over dataset detection.
            return Ok(Self {
                first,
                second: second_candidate,
                normalization,
                rows,
                weight_vector: Some(weight_array),
                typed_normalization,
            });
        }

        let mut second = second_candidate;
        let mut weight_vector: Option<Value> = None;

        if let Some(candidate) = second.take() {
            if should_treat_as_weight(&first, &candidate, normalization_explicit, rows)? {
                weight_vector = Some(candidate);
            } else {
                second = Some(candidate);
            }
        }

        Ok(Self {
            first,
            second,
            normalization,
            rows,
            weight_vector,
            typed_normalization,
        })
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
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

pub(super) fn ensure_cov_extensions(args: &CovArgs) -> BuiltinResult<()> {
    if is_typed_integer_value(&args.first)
        || args.second.as_ref().is_some_and(is_typed_integer_value)
    {
        crate::compatibility::ensure_builtin_extension_enabled(&COV_INTEGER_DATA_EXTENSION, NAME)?;
    }
    if is_logical_value(&args.first) || args.second.as_ref().is_some_and(is_logical_value) {
        crate::compatibility::ensure_builtin_extension_enabled(&COV_LOGICAL_DATA_EXTENSION, NAME)?;
    }
    if args.typed_normalization {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COV_TYPED_NORMALIZATION_EXTENSION,
            NAME,
        )?;
    }
    if args.weight_vector.is_some() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COV_VECTOR_WEIGHTS_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

fn parse_rows_option(value: &str) -> BuiltinResult<CovRows> {
    match CovRowsOption::parse(value) {
        Some(CovRowsOption::All) => Ok(CovRows::All),
        Some(CovRowsOption::OmitRows) => Ok(CovRows::OmitRows),
        Some(CovRowsOption::PartialRows) => Ok(CovRows::PartialRows),
        None => Err(cov_error_with_detail(
            &COV_ERROR_ROWS_OPTION_UNKNOWN,
            format!("'{value}'"),
        )),
    }
}

fn parse_normalization(value: Value) -> BuiltinResult<CovNormalization> {
    match value {
        Value::Int(i) => match i.to_i128() {
            0 => Ok(CovNormalization::Unbiased),
            1 => Ok(CovNormalization::Biased),
            other => Err(cov_error_with_detail(
                &COV_ERROR_NORMALIZATION_INVALID,
                format!("expected 0 or 1, received {other}"),
            )),
        },
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(cov_error_with_detail(
                    &COV_ERROR_NORMALIZATION_INVALID,
                    "value must be finite",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > 1.0e-12 {
                return Err(cov_error_with_detail(
                    &COV_ERROR_NORMALIZATION_INVALID,
                    "value must be an integer",
                ));
            }
            match rounded as i64 {
                0 => Ok(CovNormalization::Unbiased),
                1 => Ok(CovNormalization::Biased),
                other => Err(cov_error_with_detail(
                    &COV_ERROR_NORMALIZATION_INVALID,
                    format!("expected 0 or 1, received {other}"),
                )),
            }
        }
        Value::Bool(flag) => Ok(if flag {
            CovNormalization::Biased
        } else {
            CovNormalization::Unbiased
        }),
        other => Err(cov_error_with_detail(
            &COV_ERROR_NORMALIZATION_INVALID,
            format!("value must be numeric, received {other:?}"),
        )),
    }
}

fn should_treat_as_weight(
    first: &Value,
    candidate: &Value,
    normalization_explicit: bool,
    rows_option: CovRows,
) -> BuiltinResult<bool> {
    let (rows_first, cols_first) = value_rows_cols(first)?;
    let (rows_candidate, cols_candidate) = value_rows_cols(candidate)?;

    let is_vector = rows_candidate == 1
        || cols_candidate == 1
        || rows_candidate * cols_candidate == rows_candidate
            && (rows_candidate == rows_first || cols_candidate == rows_first);

    if !is_vector {
        return Ok(false);
    }

    if rows_candidate != rows_first && cols_candidate != rows_first {
        // Length mismatch, treat as dataset so the later validation emits the proper error.
        return Ok(false);
    }

    if (rows_first == 1 || cols_first == 1)
        && !normalization_explicit
        && matches!(rows_option, CovRows::All)
    {
        // Ambiguous `cov(x, y)` case – prefer dataset semantics for compatibility.
        return Ok(false);
    }

    Ok(true)
}

pub(super) fn value_rows_cols(value: &Value) -> BuiltinResult<(usize, usize)> {
    match value {
        Value::Tensor(tensor) => Ok((tensor.rows(), tensor.cols())),
        Value::LogicalArray(array) => {
            if array.shape.len() > 2 {
                return Err(cov_error_with_detail(
                    &COV_ERROR_INVALID_ARGUMENT,
                    "inputs must be 2-D matrices or vectors",
                ));
            }
            let rows = if array.shape.is_empty() {
                1
            } else {
                array.shape[0]
            };
            let cols = if array.shape.len() >= 2 {
                array.shape[1]
            } else {
                1
            };
            Ok((rows, cols))
        }
        Value::GpuTensor(handle) => {
            if handle.shape().len() > 2 {
                return Err(cov_error_with_detail(
                    &COV_ERROR_INVALID_ARGUMENT,
                    "inputs must be 2-D matrices or vectors",
                ));
            }
            let rows = if handle.shape().is_empty() {
                1
            } else {
                handle.shape()[0]
            };
            let cols = if handle.shape().len() >= 2 {
                handle.shape()[1]
            } else {
                1
            };
            Ok((rows, cols))
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok((1, 1)),
        other => Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            format!("unsupported input type for shape inspection: {other:?}"),
        )),
    }
}
