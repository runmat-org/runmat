use super::numeric::{
    char_rows, integer_size_to_usize, is_missing_text, scalar_text, scalar_usize,
};
use super::*;

pub(super) fn parse_size_args(args: &[Value]) -> BuiltinResult<Vec<usize>> {
    if args.is_empty() {
        return Ok(vec![1, 1]);
    }
    if args.len() == 1 {
        match &args[0] {
            Value::Tensor(tensor) => return tensor_shape_as_size(tensor),
            Value::Int(_) | Value::Num(_) => {
                let n = scalar_usize(&args[0], "missing size")?;
                return Ok(vec![n, n]);
            }
            Value::String(s) if s.eq_ignore_ascii_case("like") => {
                return Err(invalid_argument(
                    "missing: 'like' requires a prototype value",
                ));
            }
            _ => {}
        }
    }
    let mut out = Vec::with_capacity(args.len());
    let mut idx = 0;
    while idx < args.len() {
        if scalar_text(&args[idx])
            .map(|text| text.eq_ignore_ascii_case("like"))
            .unwrap_or(false)
        {
            idx += 2;
            continue;
        }
        out.push(scalar_usize(&args[idx], "missing size")?);
        idx += 1;
    }
    if out.is_empty() {
        Ok(vec![1, 1])
    } else {
        Ok(out)
    }
}

pub(super) fn tensor_shape_as_size(tensor: &Tensor) -> BuiltinResult<Vec<usize>> {
    if let Some(storage) = tensor.integer_storage() {
        return (0..storage.len())
            .map(|index| {
                let value = storage.value_at(index).ok_or_else(|| {
                    internal_error("missing: integer size vector storage length mismatch")
                })?;
                integer_size_to_usize(&value, "missing size")
            })
            .collect();
    }
    let values = tensor_utils::tensor_values_f64_cow(tensor);
    if values.is_empty() {
        return Ok(vec![0, 0]);
    }
    values
        .iter()
        .map(|value| {
            if !value.is_finite() || *value < 0.0 || value.fract() != 0.0 {
                return Err(invalid_argument(
                    "missing: sizes must be nonnegative finite integers",
                ));
            }
            if *value > usize::MAX as f64 {
                return Err(invalid_argument("missing: size exceeds platform limits"));
            }
            if usize::BITS == 64 && *value == usize::MAX as f64 {
                return Err(invalid_argument("missing: size exceeds platform limits"));
            }
            Ok(*value as usize)
        })
        .collect()
}

pub(super) fn missing_string_array(shape: Vec<usize>) -> BuiltinResult<Value> {
    let count = element_count(&shape)?;
    let array =
        StringArray::new(vec![MISSING_TEXT.to_string(); count], shape).map_err(internal_error)?;
    Ok(Value::StringArray(array))
}

pub(super) fn element_count(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| invalid_argument("array size is too large"))
    })
}

pub(super) fn ismissing_value(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(n) => Ok(Value::Bool(n.is_nan())),
        Value::Complex(re, im) => Ok(Value::Bool(re.is_nan() || im.is_nan())),
        Value::Int(_) | Value::Bool(_) | Value::FunctionHandle(_) | Value::ClassRef(_) => {
            Ok(Value::Bool(false))
        }
        Value::String(s) => Ok(Value::Bool(is_missing_text(s))),
        Value::CharArray(array) => Ok(Value::LogicalArray(
            LogicalArray::new(
                char_rows(array)
                    .into_iter()
                    .map(|text| u8::from(text.trim().is_empty() || is_missing_text(&text)))
                    .collect(),
                vec![array.rows, 1],
            )
            .map_err(internal_error)?,
        )),
        Value::StringArray(array) => logical_from_iter(
            array.data.iter().map(|text| is_missing_text(text)),
            array.shape.clone(),
        ),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => logical_from_iter(
            vec![false; tensor_utils::tensor_element_len(tensor)],
            tensor.shape.clone(),
        ),
        Value::Tensor(tensor) => {
            let values = tensor_utils::tensor_values_f64_cow(tensor);
            logical_from_iter(
                values.iter().map(|value| value.is_nan()),
                tensor.shape.clone(),
            )
        }
        Value::ComplexTensor(tensor) => logical_from_iter(
            tensor
                .materialize_f64()
                .iter()
                .map(|(re, im)| re.is_nan() || im.is_nan()),
            tensor.shape.clone(),
        ),
        Value::SparseTensor(tensor) => {
            let mut data = vec![0u8; tensor.rows * tensor.cols];
            if tensor.integer_storage().is_none() {
                for col in 0..tensor.cols {
                    for idx in tensor.col_ptrs[col]..tensor.col_ptrs[col + 1] {
                        if tensor
                            .numeric_value_at(idx)
                            .expect("sparse storage index is valid")
                            .materialize_f64()
                            .is_nan()
                        {
                            data[tensor.row_indices[idx] + col * tensor.rows] = 1;
                        }
                    }
                }
            }
            Ok(Value::LogicalArray(
                LogicalArray::new(data, vec![tensor.rows, tensor.cols]).map_err(internal_error)?,
            ))
        }
        Value::LogicalArray(array) => Ok(Value::LogicalArray(LogicalArray::zeros(
            array.shape.clone(),
        ))),
        Value::Cell(cell) => {
            let mut data = Vec::with_capacity(cell.data.len());
            for item in &cell.data {
                data.push(u8::from(any_missing(item)?));
            }
            Ok(Value::LogicalArray(
                LogicalArray::new(data, vec![cell.rows, cell.cols]).map_err(internal_error)?,
            ))
        }
        Value::Struct(st) => {
            let mut out = StructValue::new();
            for (name, field) in &st.fields {
                out.insert(name.clone(), ismissing_value(field)?);
            }
            Ok(Value::Struct(out))
        }
        Value::StructArray(array) => array
            .clone()
            .try_map_values(|field| ismissing_value(&field))
            .map(Value::StructArray),
        Value::Object(object) if is_tabular_object(object) => ismissing_table(object),
        Value::Object(object) if object.is_class(runmat_types::standard::DATETIME) => {
            let serials = crate::builtins::datetime::serials_from_datetime_value(value)?;
            let values = tensor_utils::tensor_values_f64_cow(&serials);
            logical_from_iter(
                values.iter().map(|serial| serial.is_nan()),
                serials.shape.clone(),
            )
        }
        Value::Object(object) if object.is_class(runmat_types::standard::DURATION) => {
            let days = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
            let values = tensor_utils::tensor_values_f64_cow(&days);
            logical_from_iter(values.iter().map(|day| day.is_nan()), days.shape.clone())
        }
        Value::OutputList(values) => {
            let mut data = Vec::with_capacity(values.len());
            for item in values {
                data.push(u8::from(any_missing(item)?));
            }
            Ok(Value::LogicalArray(
                LogicalArray::new(data, vec![1, values.len()]).map_err(internal_error)?,
            ))
        }
        _ => Ok(Value::Bool(false)),
    }
}

pub(super) fn ismissing_table(object: &ObjectInstance) -> BuiltinResult<Value> {
    let height = table_height(object)?;
    let width = table_width(object)?;
    let names = table_variable_names_from_object(object)?;
    let variables = table_variables(object)?;
    let mut data = vec![0u8; height * width];
    for (col, name) in names.iter().enumerate() {
        let Some(value) = variables.fields.get(name) else {
            continue;
        };
        let mask = logical_mask_for_rows(value, height)?;
        for row in 0..height {
            if mask.get(row).copied().unwrap_or(0) != 0 {
                data[row + col * height] = 1;
            }
        }
    }
    Ok(Value::LogicalArray(
        LogicalArray::new(data, vec![height, width]).map_err(internal_error)?,
    ))
}

pub(super) fn logical_from_iter<I>(iter: I, shape: Vec<usize>) -> BuiltinResult<Value>
where
    I: IntoIterator<Item = bool>,
{
    Ok(Value::LogicalArray(
        LogicalArray::new(iter.into_iter().map(u8::from).collect(), shape)
            .map_err(internal_error)?,
    ))
}

pub(super) fn any_missing(value: &Value) -> BuiltinResult<bool> {
    match ismissing_value(value)? {
        Value::Bool(flag) => Ok(flag),
        Value::LogicalArray(array) => Ok(array.data.iter().any(|flag| *flag != 0)),
        Value::Struct(st) => {
            for field in st.fields.values() {
                if any_missing(field)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Value::StructArray(array) => {
            for field in array.field_names() {
                if let Some(values) = array.field_values(field) {
                    for value in values {
                        if any_missing(value)? {
                            return Ok(true);
                        }
                    }
                }
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

pub(super) fn logical_mask_for_rows(value: &Value, expected_rows: usize) -> BuiltinResult<Vec<u8>> {
    let mask_value = ismissing_value(value)?;
    match mask_value {
        Value::Bool(flag) => Ok(vec![u8::from(flag); expected_rows]),
        Value::LogicalArray(mask) => logical_array_mask_for_rows(&mask, expected_rows),
        _ => Err(unsupported_type("cannot build row missing mask for value")),
    }
}

pub(super) fn logical_array_mask_for_rows(
    mask: &LogicalArray,
    expected_rows: usize,
) -> BuiltinResult<Vec<u8>> {
    let rows = mask.shape.first().copied().unwrap_or(mask.data.len());
    let cols = mask.shape.get(1).copied().unwrap_or(1);
    if rows == expected_rows {
        let mut out = vec![0u8; expected_rows];
        for col in 0..cols {
            for (row, slot) in out.iter_mut().enumerate().take(expected_rows) {
                let idx = row + col * rows;
                if mask.data.get(idx).copied().unwrap_or(0) != 0 {
                    *slot = 1;
                }
            }
        }
        Ok(out)
    } else if mask.data.len() == expected_rows {
        Ok(mask.data.to_vec())
    } else {
        Err(invalid_argument(
            "missing mask shape does not match table height",
        ))
    }
}
