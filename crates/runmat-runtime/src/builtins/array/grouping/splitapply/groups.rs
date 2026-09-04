use runmat_value::{IntValue, NumericScalar, Value};

use crate::BuiltinResult;

use super::error;

const MAX_GROUPS: usize = 50_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SplitAxis {
    Rows,
    Columns,
}

pub(super) struct Groups {
    pub(super) axis: SplitAxis,
    pub(super) observation_count: usize,
    pub(super) rows_by_group: Vec<Vec<usize>>,
}

impl Groups {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Self> {
        let (values, shape) = numeric_values(value)?;
        validate_vector(&shape)?;
        let axis = if shape.first() == Some(&1) && values.len() > 1 {
            SplitAxis::Columns
        } else {
            SplitAxis::Rows
        };
        let assignments = values
            .into_iter()
            .map(group_number)
            .collect::<BuiltinResult<Vec<_>>>()?;
        let group_count = assignments.iter().flatten().copied().max().unwrap_or(0);
        if group_count == 0 {
            return Err(error::invalid(
                "splitapply: G must contain at least one positive group number",
            ));
        }
        if group_count > MAX_GROUPS {
            return Err(error::invalid("splitapply: G contains too many groups"));
        }
        let mut rows_by_group = vec![Vec::new(); group_count];
        for (observation, group) in assignments.into_iter().enumerate() {
            if let Some(group) = group {
                rows_by_group[group - 1].push(observation);
            }
        }
        if let Some(missing) = rows_by_group.iter().position(Vec::is_empty) {
            return Err(error::invalid(format!(
                "splitapply: G must include group {} because group numbers cannot have gaps",
                missing + 1
            )));
        }
        Ok(Self {
            axis,
            observation_count: shape.iter().product(),
            rows_by_group,
        })
    }
}

fn numeric_values(value: &Value) -> BuiltinResult<(Vec<NumericScalar>, Vec<usize>)> {
    match value {
        Value::Num(value) => Ok((vec![NumericScalar::F64(*value)], vec![1, 1])),
        Value::Int(value) => Ok((vec![NumericScalar::from(value.clone())], vec![1, 1])),
        Value::Tensor(value) => Ok((
            (0..value.len())
                .map(|index| {
                    value.numeric_value_at(index).ok_or_else(|| {
                        error::invalid("splitapply: G storage is inconsistent with its shape")
                    })
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            value.shape.clone(),
        )),
        other => Err(error::invalid(format!(
            "splitapply: G must be a real numeric vector, got {other:?}"
        ))),
    }
}

fn validate_vector(shape: &[usize]) -> BuiltinResult<()> {
    if shape.iter().filter(|dimension| **dimension > 1).count() <= 1 {
        Ok(())
    } else {
        Err(error::invalid("splitapply: G must be a vector"))
    }
}

fn group_number(value: NumericScalar) -> BuiltinResult<Option<usize>> {
    if let Some(value) = value.into_int_value() {
        return integer_group_number(&value).map(Some);
    }
    let value = value.materialize_f64();
    if value.is_nan() {
        return Ok(None);
    }
    if !value.is_finite()
        || value <= 0.0
        || value.fract() != 0.0
        || value > usize::MAX as f64
        || (usize::BITS == 64 && value == usize::MAX as f64)
    {
        return Err(error::invalid(
            "splitapply: G values must be positive integers or NaN",
        ));
    }
    let value = value as usize;
    if value == 0 {
        return Err(error::invalid(
            "splitapply: G values exceed the supported integer range",
        ));
    }
    Ok(Some(value))
}

fn integer_group_number(value: &IntValue) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|value| *value > 0)
        .ok_or_else(|| error::invalid("splitapply: G values must be positive integers"))
}
