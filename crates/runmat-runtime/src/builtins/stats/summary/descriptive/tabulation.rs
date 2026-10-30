//! Frequency tabulation with exact integer labels and typed cell output.
use super::{descriptive_error, gathered, is_real_typed_integer_value};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_builtins::{TABULATE_GPU_INPUT_EXTENSION, TABULATE_INTEGER_DATA_EXTENSION};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, IntValue, StringArray, Tensor, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

// Existing frequency-table expansion policy. It bounds the dense enumeration
// of unobserved positive levels, not the number of observed input elements.
const MAX_EXPANDED_LEVELS: u128 = 1_000_000;

pub(super) fn expanded_level_count(maximum: u128) -> Option<usize> {
    (maximum <= MAX_EXPANDED_LEVELS)
        .then(|| usize::try_from(maximum).ok())
        .flatten()
}

fn compare_same_class_integers(left: &IntValue, right: &IntValue) -> Ordering {
    match (left, right) {
        (IntValue::I8(left), IntValue::I8(right)) => left.cmp(right),
        (IntValue::I16(left), IntValue::I16(right)) => left.cmp(right),
        (IntValue::I32(left), IntValue::I32(right)) => left.cmp(right),
        (IntValue::I64(left), IntValue::I64(right)) => left.cmp(right),
        (IntValue::U8(left), IntValue::U8(right)) => left.cmp(right),
        (IntValue::U16(left), IntValue::U16(right)) => left.cmp(right),
        (IntValue::U32(left), IntValue::U32(right)) => left.cmp(right),
        (IntValue::U64(left), IntValue::U64(right)) => left.cmp(right),
        _ => unreachable!("integer tensor storage is homogeneous"),
    }
}

fn positive_integer_magnitude(value: &IntValue) -> Option<u128> {
    match value {
        IntValue::I8(value) if *value >= 1 => Some(*value as u128),
        IntValue::I16(value) if *value >= 1 => Some(*value as u128),
        IntValue::I32(value) if *value >= 1 => Some(*value as u128),
        IntValue::I64(value) if *value >= 1 => Some(*value as u128),
        IntValue::U8(value) if *value >= 1 => Some(*value as u128),
        IntValue::U16(value) if *value >= 1 => Some(*value as u128),
        IntValue::U32(value) if *value >= 1 => Some(*value as u128),
        IntValue::U64(value) if *value >= 1 => Some(*value as u128),
        _ => None,
    }
}

fn positive_integer_like(prototype: &IntValue, value: u128) -> IntValue {
    match prototype {
        IntValue::I8(_) => IntValue::I8(value as i8),
        IntValue::I16(_) => IntValue::I16(value as i16),
        IntValue::I32(_) => IntValue::I32(value as i32),
        IntValue::I64(_) => IntValue::I64(value as i64),
        IntValue::U8(_) => IntValue::U8(value as u8),
        IntValue::U16(_) => IntValue::U16(value as u16),
        IntValue::U32(_) => IntValue::U32(value as u32),
        IntValue::U64(_) => IntValue::U64(value as u64),
    }
}

fn exact_integer_tabulate(name: &str, mut values: Vec<IntValue>) -> BuiltinResult<Value> {
    let total = values.len();
    if values.is_empty() {
        return CellArray::new(Vec::new(), 0, 3)
            .map(Value::Cell)
            .map_err(|err| descriptive_error(name, format!("{name}: {err}")));
    }
    let positive_max = values
        .iter()
        .map(positive_integer_magnitude)
        .collect::<Option<Vec<_>>>()
        .and_then(|values| values.into_iter().max());
    let rows = if let Some(max) = positive_max.and_then(expanded_level_count) {
        let prototype = values[0].clone();
        let counts =
            values
                .into_iter()
                .fold(HashMap::<IntValue, usize>::new(), |mut counts, value| {
                    *counts.entry(value).or_insert(0) += 1;
                    counts
                });
        (1..=max)
            .map(|value| {
                let value = positive_integer_like(&prototype, value as u128);
                let count = counts.get(&value).copied().unwrap_or(0);
                (value, count)
            })
            .collect::<Vec<_>>()
    } else {
        values.sort_by(compare_same_class_integers);
        let mut rows = Vec::new();
        let mut idx = 0usize;
        while idx < values.len() {
            let value = values[idx].clone();
            let mut end = idx + 1;
            while end < values.len() && compare_same_class_integers(&values[end], &value).is_eq() {
                end += 1;
            }
            rows.push((value, end - idx));
            idx = end;
        }
        rows
    };
    let row_count = rows.len();
    let cells = row_count
        .checked_mul(3)
        .ok_or_else(|| descriptive_error(name, format!("{name}: table size overflowed")))?;
    let mut data = Vec::new();
    data.try_reserve_exact(cells)
        .map_err(|err| descriptive_error(name, format!("{name}: cannot allocate table: {err}")))?;
    for (value, count) in rows {
        data.push(Value::Int(value));
        data.push(Value::Num(count as f64));
        data.push(Value::Num(count as f64 * 100.0 / total as f64));
    }
    CellArray::new(data, row_count, 3)
        .map(Value::Cell)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn numeric_tabulate(name: &str, tensor: Tensor) -> BuiltinResult<Value> {
    if let Some(storage) = tensor.integer_storage() {
        return exact_integer_tabulate(name, storage.exact_values());
    }
    let mut values = tensor::tensor_into_values_f64(tensor)
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    let n = values.len() as f64;
    values.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    });
    let positive_integers = !values.is_empty()
        && values
            .iter()
            .all(|value| value.is_finite() && *value >= 1.0 && value.fract() == 0.0);
    let mut rows = Vec::<(f64, f64)>::new();
    if positive_integers {
        let maximum = values.last().copied().unwrap_or(0.0);
        if maximum > MAX_EXPANDED_LEVELS as f64 {
            return Err(descriptive_error(
                name,
                format!("{name}: positive-level expansion exceeds the supported table size"),
            ));
        }
        let max_value = expanded_level_count(maximum as u128).ok_or_else(|| {
            descriptive_error(
                name,
                format!("{name}: positive-level expansion exceeds the supported table size"),
            )
        })?;
        let count_len = max_value
            .checked_add(1)
            .ok_or_else(|| descriptive_error(name, format!("{name}: table size overflowed")))?;
        let mut counts = Vec::new();
        counts.try_reserve_exact(count_len).map_err(|err| {
            descriptive_error(name, format!("{name}: cannot allocate frequencies: {err}"))
        })?;
        counts.resize(count_len, 0usize);
        for value in values {
            counts[value as usize] += 1;
        }
        rows.extend((1..=max_value).map(|idx| (idx as f64, counts[idx] as f64)));
    } else {
        let mut idx = 0usize;
        while idx < values.len() {
            let value = values[idx];
            let mut end = idx + 1;
            while end < values.len()
                && ((value.is_nan() && values[end].is_nan()) || values[end] == value)
            {
                end += 1;
            }
            rows.push((value, (end - idx) as f64));
            idx = end;
        }
    }
    let rows_len = rows.len();
    let cells = rows_len
        .checked_mul(3)
        .ok_or_else(|| descriptive_error(name, format!("{name}: table size overflowed")))?;
    let mut data = Vec::new();
    data.try_reserve_exact(cells)
        .map_err(|err| descriptive_error(name, format!("{name}: cannot allocate table: {err}")))?;
    data.extend(rows.iter().map(|(value, _)| *value));
    data.extend(rows.iter().map(|(_, count)| *count));
    data.extend(rows.iter().map(|(_, count)| {
        if n == 0.0 {
            f64::NAN
        } else {
            *count * 100.0 / n
        }
    }));
    Tensor::new(data, vec![rows_len, 3])
        .map(tensor::tensor_into_value)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn logical_tabulate(name: &str, data: Vec<u8>) -> BuiltinResult<Value> {
    let false_count = data.iter().filter(|value| **value == 0).count();
    let true_count = data.len() - false_count;
    let total = data.len() as f64;
    let mut rows = Vec::new();
    if false_count > 0 {
        rows.push((false, false_count));
    }
    if true_count > 0 {
        rows.push((true, true_count));
    }
    let mut cell_data = Vec::with_capacity(rows.len() * 3);
    for (value, count) in &rows {
        cell_data.push(Value::Bool(*value));
        cell_data.push(Value::Num(*count as f64));
        cell_data.push(Value::Num(if total == 0.0 {
            f64::NAN
        } else {
            *count as f64 * 100.0 / total
        }));
    }
    CellArray::new(cell_data, rows.len(), 3)
        .map(Value::Cell)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

pub(super) fn label_for_value(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(text) => text.as_str().map(str::to_owned),
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().map(str::to_owned)
        }
        Value::CharArray(chars) if chars.rows == 1 => Some(
            crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
                descriptive_error("tabulate", "tabulate: value is not valid UTF-16 text")
            })?,
        ),
        Value::Num(num) => Some(num.to_string()),
        Value::Int(int) => Some(int.decimal_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    })
}

// Missing observations are excluded before this census. Literal spellings and
// empty present strings remain ordinary labels in the denominator.
fn string_tabulate(name: &str, labels: impl IntoIterator<Item = String>) -> BuiltinResult<Value> {
    let mut counts = BTreeMap::<String, usize>::new();
    let mut total = 0usize;
    for label in labels {
        *counts.entry(label).or_insert(0) += 1;
        total += 1;
    }
    let rows = counts.len();
    let mut data = Vec::with_capacity(rows * 3);
    for (label, count) in counts {
        data.push(Value::StringArray(
            StringArray::new(vec![label], vec![1, 1])
                .map_err(|err| descriptive_error(name, format!("{name}: {err}")))?,
        ));
        data.push(Value::Num(count as f64));
        data.push(Value::Num(if total == 0 {
            f64::NAN
        } else {
            count as f64 * 100.0 / total as f64
        }));
    }
    CellArray::new(data, rows, 3)
        .map(Value::Cell)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn tabulate_value(name: &str, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => numeric_tabulate(name, tensor),
        Value::LogicalArray(logical) => logical_tabulate(name, logical.data.into_vec()),
        Value::Int(value) => exact_integer_tabulate(name, vec![value]),
        Value::Num(_) | Value::Bool(_) => tensor::value_into_tensor_for(name, value)
            .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
            .and_then(|tensor| numeric_tabulate(name, tensor)),
        Value::String(text) => string_tabulate(name, text.as_str().map(str::to_owned)),
        Value::StringArray(array) => string_tabulate(
            name,
            array
                .data()
                .iter()
                .filter_map(|text| text.as_str().map(str::to_owned)),
        ),
        Value::CharArray(chars) if chars.rows == 1 => {
            let text = crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
                descriptive_error(name, format!("{name}: input is not valid UTF-16 text"))
            })?;
            string_tabulate(name, text.chars().map(|ch| ch.to_string()))
        }
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data().len());
            for entry in cell.data() {
                if matches!(entry, Value::String(runmat_value::StringElement::Missing))
                    || matches!(entry, Value::StringArray(array) if array.data().len() == 1 && array.data()[0].is_missing())
                {
                    continue;
                }
                let Some(label) = label_for_value(entry)? else {
                    return Err(descriptive_error(
                        name,
                        format!("{name}: cell entries must be scalar strings or scalars"),
                    ));
                };
                labels.push(label);
            }
            string_tabulate(name, labels)
        }
        other => Err(descriptive_error(
            name,
            format!("{name}: unsupported input type {other:?}"),
        )),
    }
}

#[cfg(test)]
mod missing_string_tests {
    use super::*;
    use runmat_value::StringElement;

    #[test]
    fn missing_is_excluded_but_empty_and_literal_labels_are_counted() {
        let elements = vec![
            StringElement::text("<missing>"),
            StringElement::Missing,
            StringElement::text(""),
            StringElement::text("<missing>"),
        ];
        let array = Value::StringArray(
            StringArray::from_elements_unbounded(elements.clone(), vec![4, 1]).unwrap(),
        );
        let cell = Value::Cell(
            CellArray::new(elements.into_iter().map(Value::String).collect(), 4, 1).unwrap(),
        );
        for input in [array, cell] {
            let Value::Cell(table) = tabulate_value("tabulate", input).unwrap() else {
                panic!("text tabulation must return a cell array");
            };
            assert_eq!(table.data().len(), 6);
            for (index, expected) in [(0, ""), (1, "<missing>")] {
                let Value::StringArray(label) = table.get(index, 0).unwrap() else {
                    panic!("tabulated label must retain string class");
                };
                assert_eq!(label.data()[0].as_str(), Some(expected));
            }
            assert!(matches!(table.get(0, 1).unwrap(), Value::Num(1.0)));
            assert!(matches!(table.get(1, 1).unwrap(), Value::Num(2.0)));
            assert!(
                matches!(table.get(1, 2).unwrap(), Value::Num(value) if (value - 200.0 / 3.0).abs() < 1e-12)
            );
        }
    }
}

pub mod tabulate {
    use super::*;

    #[runtime_builtin(
        name = "tabulate",
        binding_variant = "default",
        builtin_path = "crate::builtins::stats::summary::descriptive::tabulation::tabulate"
    )]
    pub(crate) async fn tabulate_builtin(value: Value) -> BuiltinResult<Value> {
        if super::is_real_typed_integer_value(&value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &TABULATE_INTEGER_DATA_EXTENSION,
                "tabulate",
            )?;
        }
        if matches!(value, Value::GpuTensor(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &TABULATE_GPU_INPUT_EXTENSION,
                "tabulate",
            )?;
        }
        if let Value::Cell(cell) = &value {
            if cell
                .data()
                .iter()
                .any(crate::dispatcher::value_contains_gpu)
            {
                return Err(descriptive_error(
                    "tabulate",
                    "tabulate: cell entries must be scalar strings or scalars",
                ));
            }
        }
        let value = super::gathered(value, "tabulate").await?;
        super::tabulate_value("tabulate", value)
    }
}
