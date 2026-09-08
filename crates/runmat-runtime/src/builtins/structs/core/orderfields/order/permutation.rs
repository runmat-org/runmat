use crate::builtins::common::tensor as tensor_support;
use runmat_value::{NumericScalar, Tensor, Value};
use std::collections::HashSet;

pub(super) fn element_count(value: &Value) -> Option<usize> {
    match value {
        Value::Int(_) | Value::Num(_) => Some(1),
        Value::Tensor(values) => Some(values.len()),
        _ => None,
    }
}

pub(super) fn parse(
    current: &[String],
    value: &Value,
) -> crate::BuiltinResult<Option<Vec<String>>> {
    match value {
        Value::Int(value) => value
            .try_to_usize()
            .ok_or_else(super::super::error::index_out_of_range)
            .and_then(|index| scalar(current, index))
            .map(Some),
        Value::Num(value) => floating_index(*value)
            .and_then(|index| scalar(current, index))
            .map(Some),
        Value::Tensor(values) => parse_tensor(current, values).map(Some),
        _ => Ok(None),
    }
}

fn scalar(current: &[String], index: usize) -> crate::BuiltinResult<Vec<String>> {
    if current.len() != 1 {
        return Err(super::super::error::invalid_permutation());
    }
    if index != 1 {
        return Err(super::super::error::index_out_of_range());
    }
    Ok(current.to_vec())
}

fn parse_tensor(current: &[String], values: &Tensor) -> crate::BuiltinResult<Vec<String>> {
    let len = tensor_support::tensor_element_len(values);
    if len != current.len() || !is_vector(values) {
        return Err(super::super::error::invalid_permutation());
    }
    let mut seen = HashSet::with_capacity(len);
    let mut order = Vec::with_capacity(len);
    for position in 0..len {
        let index = numeric_index(
            values
                .numeric_value_at(position)
                .ok_or_else(super::super::error::index_out_of_range)?,
        )?;
        if index == 0 || index > current.len() {
            return Err(super::super::error::index_out_of_range());
        }
        let source = index - 1;
        if !seen.insert(source) {
            return Err(super::super::error::duplicate_index());
        }
        order.push(current[source].clone());
    }
    Ok(order)
}

fn is_vector(values: &Tensor) -> bool {
    values.is_empty()
        || (values.shape.len() <= 2
            && values
                .shape
                .iter()
                .filter(|dimension| **dimension > 1)
                .count()
                <= 1)
}

fn numeric_index(value: NumericScalar) -> crate::BuiltinResult<usize> {
    match value {
        NumericScalar::F64(value) => floating_index(value),
        NumericScalar::F32(value) => floating_index(f64::from(value)),
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_usize())
            .ok_or_else(super::super::error::index_out_of_range),
    }
}

fn floating_index(value: f64) -> crate::BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(super::super::error::index_not_integer());
    }
    if value < 1.0 || value > usize::MAX as f64 {
        return Err(super::super::error::index_out_of_range());
    }
    Ok(value as usize)
}
