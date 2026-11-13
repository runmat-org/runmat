use super::*;

pub(super) fn mask_output_source(
    rest: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    crate::builtins::common::gpu_helpers::select_resident_output_source(
        rest.iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        builtin,
    )
}

pub(super) fn restore_mask_output(
    output: Value,
    source: Option<&GpuTensorHandle>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match source {
        Some(source) => crate::builtins::common::gpu_helpers::restore_class_preserving_value(
            source, output, builtin,
        ),
        None => Ok(output),
    }
}

pub(super) fn ensure_mask_extensions(
    rest: crate::call::FunctionArgumentView<'_>,
    training: bool,
) -> BuiltinResult<()> {
    let Some(index) = rest.first() else {
        return Ok(());
    };
    let (integer, logical) = if training {
        (
            &TRAINING_INTEGER_INDEX_EXTENSION,
            &TRAINING_LOGICAL_INDEX_EXTENSION,
        )
    } else {
        (&TEST_INTEGER_INDEX_EXTENSION, &TEST_LOGICAL_INDEX_EXTENSION)
    };
    if typed_integer_value(index) {
        crate::compatibility::ensure_builtin_extension_enabled(
            integer,
            if training { "training" } else { "test" },
        )?;
    }
    if is_logical_index(index) {
        crate::compatibility::ensure_builtin_extension_enabled(
            logical,
            if training { "training" } else { "test" },
        )?;
    }
    Ok(())
}

pub(super) fn is_logical_index(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

pub(super) struct AdmittedMask {
    selected_host: Option<Vec<usize>>,
    resubstitution: bool,
}

pub(super) fn admit_mask(
    c: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<AdmittedMask> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(invalid_argument(
            "cvpartition: training/test return one output",
        ));
    }
    if rest.len() > 1 {
        return Err(invalid_argument(
            "cvpartition: training/test accept at most one index argument",
        ));
    }
    let object = match c {
        Value::Object(object) if object.is_class(CVPARTITION_CLASS) => object,
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: expected cvpartition object, got {other:?}"
            )))
        }
    };
    let test_sets = match object.properties().get(TEST_SETS_PROPERTY) {
        Some(Value::LogicalArray(array)) => array,
        _ => {
            return Err(invalid_argument(
                "cvpartition: missing internal test-set metadata",
            ))
        }
    };
    let (rows, cols) = logical_shape(test_sets)?;
    if rows.checked_mul(cols) != Some(test_sets.data.len()) {
        return Err(invalid_argument(
            "cvpartition: object test-set metadata is inconsistent",
        ));
    }
    let active_rows = match object.properties().get(ACTIVE_ROWS_PROPERTY) {
        Some(Value::LogicalArray(array)) => array,
        _ => {
            return Err(invalid_argument(
                "cvpartition: missing internal active-row metadata",
            ))
        }
    };
    let (active_count, active_columns) = logical_shape(active_rows)?;
    if active_count != rows || active_columns != 1 || active_rows.data.len() != rows {
        return Err(invalid_argument(
            "cvpartition: object active-row metadata is inconsistent",
        ));
    }
    let kind = object
        .properties()
        .get("Type")
        .map(|value| scalar_text(value, "partition type"))
        .transpose()?
        .and_then(|name| PartitionKind::from_property_name(&name))
        .ok_or_else(|| invalid_argument("cvpartition: object partition type is invalid"))?;
    let selected_host =
        match rest.first() {
            Some(Value::GpuTensor(_)) => None,
            Some(index) => Some(selected_indices(index, cols)?),
            None if cols == 1 => Some(vec![0]),
            None => return Err(invalid_argument(
                "cvpartition: an index or 'all' is required for partitions with multiple test sets",
            )),
        };
    if let Some(selected) = &selected_host {
        materialized_mask_len(rows, selected.len(), "training/test output")?;
    }
    Ok(AdmittedMask {
        selected_host,
        resubstitution: kind == PartitionKind::Resubstitution,
    })
}

pub(super) fn mask_from_partition(
    c: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    admitted: AdmittedMask,
    training: bool,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(invalid_argument(
            "cvpartition: training/test accept at most one index argument",
        ));
    }
    let object = match c {
        Value::Object(object) if object.is_class(CVPARTITION_CLASS) => object,
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: expected cvpartition object, got {other:?}"
            )))
        }
    };
    let test_sets = test_sets_property(&object)?;
    let active_rows = active_rows_property(&object)?;
    let resubstitution = admitted.resubstitution;
    let (rows, cols) = logical_shape(&test_sets)?;
    if active_rows.data.len() != rows {
        return Err(invalid_argument(
            "cvpartition: object active-row metadata is inconsistent",
        ));
    }
    let selected = if let Some(selected) = admitted.selected_host {
        selected
    } else if let Some(index) = rest.first() {
        selected_indices(index, cols)?
    } else if cols == 1 {
        vec![0usize]
    } else {
        return Err(invalid_argument(
            "cvpartition: an index or 'all' is required for partitions with multiple test sets",
        ));
    };
    let _ = materialized_mask_len(rows, selected.len(), "training/test output")?;
    let mut data = Vec::with_capacity(rows * selected.len());
    for col in &selected {
        let base = *col * rows;
        for row in 0..rows {
            let active = active_rows.data[row] != 0;
            let is_test = test_sets.data[base + row] != 0;
            data.push(u8::from(if training {
                active && (resubstitution || !is_test)
            } else {
                active && is_test
            }));
        }
    }
    LogicalArray::new(
        data,
        vec![rows, if rest.is_empty() { 1 } else { selected.len() }],
    )
    .map(Value::LogicalArray)
    .map_err(|err| internal_error(format!("cvpartition: {err}")))
}

pub(super) fn selected_indices(value: &Value, cols: usize) -> BuiltinResult<Vec<usize>> {
    if scalar_text(value, "index")
        .ok()
        .and_then(|text| MaskSelectionKeyword::parse(&text))
        == Some(MaskSelectionKeyword::All)
    {
        return Ok((0..cols).collect());
    }
    if let Value::Int(integer) = value {
        return integer
            .try_to_usize()
            .filter(|index| *index >= 1 && *index <= cols)
            .map(|index| vec![index - 1])
            .ok_or_else(|| {
                invalid_argument(format!(
                    "cvpartition: index must be an integer between 1 and {cols}"
                ))
            });
    }
    if let Value::Tensor(tensor) = value {
        if let Some(storage) = tensor.integer_storage() {
            if storage.is_empty() {
                return Err(invalid_argument("cvpartition: index must be nonempty"));
            }
            return storage
                .exact_values()
                .iter()
                .map(|value| {
                    value
                        .try_to_usize()
                        .filter(|index| *index >= 1 && *index <= cols)
                        .map(|index| index - 1)
                        .ok_or_else(|| {
                            invalid_argument(format!(
                                "cvpartition: index must be an integer between 1 and {cols}"
                            ))
                        })
                })
                .collect();
        }
    }
    let raw = match value {
        Value::Num(number) => vec![*number],
        Value::Tensor(tensor) => tensor::tensor_values_f64(tensor),
        Value::LogicalArray(array) => array
            .data
            .iter()
            .map(|flag| if *flag == 0 { 0.0 } else { 1.0 })
            .collect(),
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: index must be numeric, logical, or 'all', got {other:?}"
            )))
        }
    };
    let mut indices = Vec::with_capacity(raw.len());
    for value in raw {
        if !value.is_finite() || value.fract() != 0.0 || value < 1.0 || value > cols as f64 {
            return Err(invalid_argument(format!(
                "cvpartition: index must be an integer between 1 and {cols}"
            )));
        }
        indices.push(value as usize - 1);
    }
    if indices.is_empty() {
        return Err(invalid_argument("cvpartition: index must be nonempty"));
    }
    Ok(indices)
}

pub(super) fn test_sets_property(object: &ObjectInstance) -> BuiltinResult<LogicalArray> {
    match object.properties().get(TEST_SETS_PROPERTY) {
        Some(Value::LogicalArray(array)) => Ok(array.clone()),
        _ => Err(invalid_argument(
            "cvpartition: object is missing internal test-set data",
        )),
    }
}

pub(super) fn active_rows_property(object: &ObjectInstance) -> BuiltinResult<LogicalArray> {
    match object.properties().get(ACTIVE_ROWS_PROPERTY) {
        Some(Value::LogicalArray(array)) => Ok(array.clone()),
        _ => Err(invalid_argument(
            "cvpartition: object is missing internal active-row data",
        )),
    }
}

pub(super) fn logical_shape(array: &LogicalArray) -> BuiltinResult<(usize, usize)> {
    match array.shape.as_slice() {
        [] => Ok((array.data.len(), 1)),
        [rows] => Ok((*rows, 1)),
        [1, cols] => Ok((*cols, 1)),
        [rows, 1] => Ok((*rows, 1)),
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(invalid_argument(
            "cvpartition: logical test sets must be a vector or 2-D matrix",
        )),
    }
}
