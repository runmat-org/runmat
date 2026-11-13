use super::*;

pub(super) struct AdmittedPartition {
    pub(super) kind: PartitionKind,
    pub(super) custom_first: bool,
    pub(super) option_start: usize,
    pub(super) stratify: Vec<AdmittedBoolean>,
}

pub(super) enum AdmittedBoolean {
    Known(bool),
    Resident(usize),
}

pub(super) fn admit_cvpartition(
    first: &Value,
    second: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<AdmittedPartition> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(invalid_argument("cvpartition: too many output arguments"));
    }
    let first_method = value_partition_method(first);
    let custom_first = first_method == Some(PartitionKind::Custom);
    let kind = if custom_first {
        PartitionKind::Custom
    } else {
        let text = scalar_text(second, "partition kind")?;
        parse_kind(&text)?
    };
    let option_start = match kind {
        PartitionKind::Custom => {
            if !rest.is_empty() {
                return Err(invalid_argument(
                    "cvpartition: CustomPartition accepts exactly one test-set argument",
                ));
            }
            0
        }
        PartitionKind::KFold | PartitionKind::Holdout => {
            if rest.is_empty() {
                return Err(invalid_argument(format!(
                    "cvpartition: {} requires a partition value",
                    kind.property_name()
                )));
            }
            1
        }
        PartitionKind::Leaveout | PartitionKind::Resubstitution => 0,
    };
    let options = rest.range(option_start..)?;
    if !options.len().is_multiple_of(2) {
        return Err(invalid_argument(
            "cvpartition: name-value options must be supplied in pairs",
        ));
    }
    let mut stratify_indices = Vec::new();
    let mut needs_boolean_extension = false;
    for (pair_index, pair) in options.chunks_exact(2).enumerate() {
        let name = scalar_text(&pair[0], "option name")?;
        let option = CvPartitionOption::parse(&name).ok_or_else(|| {
            invalid_argument(format!(
                "cvpartition: unsupported option '{}'",
                canonical(&name)
            ))
        })?;
        match option {
            CvPartitionOption::Stratify => {
                if !documented_logical_scalar(&pair[1]) {
                    needs_boolean_extension = true;
                }
                stratify_indices.push(option_start + pair_index * 2 + 1);
            }
        }
    }
    ensure_cvpartition_extensions(first, second, rest, kind, custom_first)?;
    if needs_boolean_extension {
        enable_extension(&CVPARTITION_BOOLEAN_ALIAS_EXTENSION)?;
    }
    preflight_host_partition(first, second, rest, kind, custom_first)?;
    let mut stratify = Vec::with_capacity(stratify_indices.len());
    for index in stratify_indices {
        stratify.push(if matches!(&rest[index], Value::GpuTensor(_)) {
            AdmittedBoolean::Resident(index)
        } else {
            AdmittedBoolean::Known(scalar_bool(&rest[index], "Stratify")?)
        });
    }
    Ok(AdmittedPartition {
        kind,
        custom_first,
        option_start,
        stratify,
    })
}

pub(super) fn preflight_host_partition(
    first: &Value,
    second: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    kind: PartitionKind,
    custom_first: bool,
) -> BuiltinResult<()> {
    if kind == PartitionKind::Custom {
        let payload = if custom_first { second } else { first };
        if !matches!(
            payload,
            Value::GpuTensor(_) | Value::LogicalArray(_) | Value::Bool(_) | Value::Tensor(_)
        ) {
            return Err(invalid_argument(
                "cvpartition: CustomPartition requires logical or numeric test sets",
            ));
        }
        return Ok(());
    }
    let n = inspect_host_observation_count(first)?;
    match kind {
        PartitionKind::KFold => {
            if !matches!(&rest[0], Value::GpuTensor(_)) {
                let k = positive_integer(&rest[0], "KFold")?;
                if k < 2 || n.is_some_and(|n| k >= n) {
                    return Err(invalid_argument(
                        "cvpartition: KFold must be an integer in the interval [2, n)",
                    ));
                }
                if let Some(n) = n {
                    materialized_mask_len(n, k, "KFold")?;
                }
            }
        }
        PartitionKind::Holdout => {
            if !matches!(&rest[0], Value::GpuTensor(_)) {
                if let Some(n) = n {
                    holdout_count(&rest[0], n)?;
                } else {
                    validate_holdout_control_without_n(&rest[0])?;
                }
            }
        }
        PartitionKind::Leaveout => {
            if let Some(n) = n {
                materialized_mask_len(n, n, "Leaveout")?;
            }
        }
        PartitionKind::Resubstitution | PartitionKind::Custom => {}
    }
    Ok(())
}

pub(super) fn validate_holdout_control_without_n(value: &Value) -> BuiltinResult<()> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        if integer.try_to_usize().is_some_and(|count| count > 0) {
            return Ok(());
        }
    } else {
        let raw = scalar_number(value, "Holdout")?;
        if (raw > 0.0 && raw < 1.0) || (raw.is_finite() && raw.fract() == 0.0 && raw >= 1.0) {
            return Ok(());
        }
    }
    Err(invalid_argument(
        "cvpartition: Holdout must be a fraction in (0,1) or an integer in [1,n)",
    ))
}

pub(super) fn inspect_host_observation_count(value: &Value) -> BuiltinResult<Option<usize>> {
    let vector_len = |shape: &[usize], len: usize| -> BuiltinResult<usize> {
        if len == 0 || shape.iter().filter(|dim| **dim > 1).count() > 1 {
            return Err(invalid_argument(
                "cvpartition: stratification variable must be a nonempty vector",
            ));
        }
        Ok(len)
    };
    let n = match value {
        Value::GpuTensor(_) => return Ok(None),
        Value::Num(number) => positive_integer_number(*number, "n")?,
        Value::Int(integer) => integer.try_to_usize().filter(|n| *n > 0)
            .ok_or_else(|| invalid_argument("cvpartition: n must be a positive integer scalar"))?,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(integer) = tensor::scalar_integer_value(value) {
                integer.try_to_usize().filter(|n| *n > 0)
                    .ok_or_else(|| invalid_argument("cvpartition: n must be a positive integer scalar"))?
            } else {
                positive_integer_number(tensor::tensor_value_f64(tensor, 0), "n")?
            }
        }
        Value::Tensor(tensor) => vector_len(tensor.shape(), tensor::tensor_element_len(tensor))?,
        Value::LogicalArray(array) => vector_len(&array.shape, array.data.len())?,
        Value::StringArray(array) => vector_len(&array.shape, array.data().len())?,
        Value::CharArray(chars) if chars.rows > 0 => chars.rows,
        Value::Bool(_) | Value::String(_) => 1,
        other => return Err(invalid_argument(format!(
            "cvpartition: first input must be an observation count or grouping variable, got {other:?}"
        ))),
    };
    Ok(Some(n))
}

pub(super) fn enable_extension(extension: &BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, CVPARTITION_NAME)
}

pub(super) fn typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

pub(super) fn scalar_value(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_) | Value::Bool(_))
        || matches!(value, Value::Tensor(value_tensor) if tensor::is_scalar_tensor(value_tensor))
        || matches!(value, Value::LogicalArray(array) if array.data.len() == 1)
        || matches!(value, Value::GpuTensor(handle) if tensor::element_count(handle.shape()) == 1)
}

pub(super) fn documented_logical_scalar(value: &Value) -> bool {
    matches!(value, Value::Bool(_))
        || matches!(value, Value::LogicalArray(array) if array.data.len() == 1)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle) && tensor::element_count(handle.shape()) == 1)
}

pub(super) fn ensure_cvpartition_extensions(
    first: &Value,
    second: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    kind: PartitionKind,
    custom_first: bool,
) -> BuiltinResult<()> {
    if custom_first {
        if typed_integer_value(second) {
            enable_extension(&CVPARTITION_INTEGER_CUSTOM_EXTENSION)?;
        }
        return Ok(());
    }

    if typed_integer_value(first) {
        if scalar_value(first) {
            enable_extension(&CVPARTITION_INTEGER_N_EXTENSION)?;
        } else {
            enable_extension(&CVPARTITION_INTEGER_STRATIFICATION_EXTENSION)?;
        }
    }

    if matches!(kind, PartitionKind::KFold | PartitionKind::Holdout)
        && rest.first().is_some_and(typed_integer_value)
    {
        enable_extension(&CVPARTITION_INTEGER_CONTROL_EXTENSION)?;
    }
    Ok(())
}
