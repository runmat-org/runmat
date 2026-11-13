//! Typed shared summaries for pivot, groupsummary, and table heatmap.

use super::{
    compare_integer_values, invalid_argument, invalid_index, invalid_variable, BuiltinResult,
    Ordering, Tensor, Value,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SummaryMethod {
    Mean,
    Sum,
    Min,
    Max,
    Median,
    Count,
    Numel,
}

impl SummaryMethod {
    pub(super) fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "mean" => Some(Self::Mean),
            "sum" => Some(Self::Sum),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "median" => Some(Self::Median),
            "count" => Some(Self::Count),
            "numel" => Some(Self::Numel),
            _ => None,
        }
    }

    pub(super) fn output_prefix(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Sum => "sum",
            Self::Min => "min",
            Self::Max => "max",
            Self::Median => "median",
            Self::Count => "count",
            Self::Numel => "numel",
        }
    }
}

pub(super) fn summarize_groups_value(
    value: &Value,
    groups: &[&Vec<usize>],
    method: SummaryMethod,
) -> BuiltinResult<Value> {
    let Value::Tensor(tensor) = value else {
        return Err(invalid_variable(
            "groupsummary: summary data variables must be numeric column vectors",
        ));
    };
    if tensor.cols() != 1 {
        return Err(invalid_variable(
            "groupsummary: summary data variables must be numeric column vectors",
        ));
    }
    let Some(storage) = tensor.integer_storage() else {
        let values = summarize_groups(value, groups.iter().copied(), method)?;
        return Tensor::new(values, vec![groups.len(), 1])
            .map(Value::Tensor)
            .map_err(invalid_variable);
    };
    if matches!(method, SummaryMethod::Min | SummaryMethod::Max) {
        let mut extrema = Vec::with_capacity(groups.len());
        for rows in groups {
            let mut values = rows.iter().map(|row| {
                storage
                    .value_at(*row)
                    .ok_or_else(|| invalid_index("groupsummary: integer row out of bounds"))
            });
            let mut selected = values.next().transpose()?.ok_or_else(|| {
                invalid_argument("groupsummary: observed integer groups cannot be empty")
            })?;
            for value in values {
                let value = value?;
                let ordering = compare_integer_values(&value, &selected);
                if (method == SummaryMethod::Min && ordering == Ordering::Less)
                    || (method == SummaryMethod::Max && ordering == Ordering::Greater)
                {
                    selected = value;
                }
            }
            extrema.push(selected);
        }
        let output = storage
            .from_exact_values_like(extrema)
            .map_err(invalid_variable)?;
        return Tensor::new_integer(output, vec![groups.len(), 1])
            .map(Value::Tensor)
            .map_err(invalid_variable);
    }
    if matches!(method, SummaryMethod::Count | SummaryMethod::Numel) {
        return Tensor::new(
            groups.iter().map(|rows| rows.len() as f64).collect(),
            vec![groups.len(), 1],
        )
        .map(Value::Tensor)
        .map_err(invalid_variable);
    }
    let mut output = Vec::with_capacity(groups.len());
    for rows in groups {
        let mut values = rows
            .iter()
            .map(|row| {
                let value = storage
                    .value_at(*row)
                    .ok_or_else(|| invalid_index("groupsummary: integer row out of bounds"))?;
                if !crate::builtins::common::validation::integer_is_exact_f64(&value) {
                    return Err(invalid_argument(
                        "groupsummary: integer data must be exactly representable as double for floating summary methods",
                    ));
                }
                Ok(value.to_f64())
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        let value = match method {
            SummaryMethod::Mean => values.iter().sum::<f64>() / values.len() as f64,
            SummaryMethod::Sum => values.iter().sum(),
            SummaryMethod::Median => {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let mid = values.len() / 2;
                if values.len().is_multiple_of(2) {
                    (values[mid - 1] + values[mid]) / 2.0
                } else {
                    values[mid]
                }
            }
            SummaryMethod::Min
            | SummaryMethod::Max
            | SummaryMethod::Count
            | SummaryMethod::Numel => unreachable!("summary method handled above"),
        };
        output.push(value);
    }
    Tensor::new(output, vec![groups.len(), 1])
        .map(Value::Tensor)
        .map_err(invalid_variable)
}

pub(super) fn summarize_groups<'a>(
    value: &Value,
    groups: impl Iterator<Item = &'a Vec<usize>>,
    method: SummaryMethod,
) -> BuiltinResult<Vec<f64>> {
    let tensor = match value {
        Value::Tensor(tensor) if tensor.cols() == 1 => tensor,
        _ => {
            return Err(invalid_variable(
                "groupsummary: summary data variables must be numeric column vectors",
            ))
        }
    };
    groups
        .map(|rows| {
            let mut values = rows
                .iter()
                .map(|row| tensor.get2(*row, 0).map_err(invalid_index))
                .collect::<BuiltinResult<Vec<_>>>()?;
            values.retain(|value| !value.is_nan());
            let result = match method {
                SummaryMethod::Mean => {
                    if values.is_empty() {
                        f64::NAN
                    } else {
                        values.iter().sum::<f64>() / values.len() as f64
                    }
                }
                SummaryMethod::Sum => values.iter().sum(),
                SummaryMethod::Min => values.into_iter().fold(f64::INFINITY, f64::min),
                SummaryMethod::Max => values.into_iter().fold(f64::NEG_INFINITY, f64::max),
                SummaryMethod::Median => {
                    if values.is_empty() {
                        f64::NAN
                    } else {
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                        let mid = values.len() / 2;
                        if values.len() % 2 == 0 {
                            (values[mid - 1] + values[mid]) / 2.0
                        } else {
                            values[mid]
                        }
                    }
                }
                SummaryMethod::Count | SummaryMethod::Numel => values.len() as f64,
            };
            Ok(result)
        })
        .collect()
}
