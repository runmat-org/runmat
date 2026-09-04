mod equal_width;
mod labels;

use runmat_value::{NumericScalar, StringArray, Value};

use crate::builtins::array::binning::numeric;
use crate::builtins::array::grouping::variables::GroupColumn;
use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

use super::error;
use super::options::{scalar_text, IncludedEdge};

pub(super) struct BinLevels {
    pub(super) labels: Vec<String>,
}

pub(super) fn apply(
    columns: &mut [GroupColumn],
    positional: &[Value],
    edge: IncludedEdge,
) -> BuiltinResult<Option<BinLevels>> {
    if positional.is_empty() {
        return Ok(None);
    }
    if positional.len() != 1 || columns.len() != 1 {
        return Err(error::invalid(
            "groupcounts: current binning support requires one bin specification and one numeric grouping vector",
        ));
    }
    let specification = unwrap_specification(&positional[0])?;
    if scalar_text(specification).is_some_and(|text| text.eq_ignore_ascii_case("none")) {
        return Ok(None);
    }
    let values = numeric_values(&columns[0].value)?;
    let included_right = edge == IncludedEdge::Right;
    let (assignments, labels) = if is_bin_count(specification) {
        equal_width::assign(&values, bin_count(specification)?, included_right)?
    } else {
        explicit_edges(&values, specification, included_right)?
    };
    let rows = assignments
        .into_iter()
        .map(|bin| {
            bin.map(|index| labels[index].clone())
                .unwrap_or_else(|| "<missing>".into())
        })
        .collect();
    let values = Value::StringArray(
        StringArray::new(rows, vec![columns[0].rows, 1]).map_err(error::internal)?,
    );
    columns[0].replace_value(values).map_err(error::internal)?;
    Ok(Some(BinLevels { labels }))
}

fn unwrap_specification(value: &Value) -> BuiltinResult<&Value> {
    match value {
        Value::Cell(value) if value.data.len() == 1 => Ok(&value.data[0]),
        Value::Cell(_) => Err(error::invalid(
            "groupcounts: one grouping vector requires one bin specification",
        )),
        value => Ok(value),
    }
}

fn explicit_edges(
    values: &[NumericScalar],
    specification: &Value,
    included_right: bool,
) -> BuiltinResult<(Vec<Option<usize>>, Vec<String>)> {
    let edges = numeric_values(specification)?;
    if edges.len() < 2 {
        return Err(error::invalid(
            "groupcounts: numeric bin edges must contain at least two values",
        ));
    }
    if edges
        .windows(2)
        .any(|pair| numeric::compare(pair[0], pair[1]) != Some(std::cmp::Ordering::Less))
    {
        return Err(error::invalid(
            "groupcounts: numeric bin edges must be finite and strictly increasing",
        ));
    }
    let labels = edges
        .windows(2)
        .enumerate()
        .map(|(index, pair)| {
            labels::interval(pair[0], pair[1], index, edges.len() - 1, included_right)
        })
        .collect();
    let assignments = values
        .iter()
        .map(|value| numeric::assign(*value, &edges, included_right).map(|index| index - 1))
        .collect();
    Ok((assignments, labels))
}

fn numeric_values(value: &Value) -> BuiltinResult<Vec<NumericScalar>> {
    match value {
        Value::Num(value) => Ok(vec![NumericScalar::F64(*value)]),
        Value::Int(value) => Ok(vec![NumericScalar::from(value.clone())]),
        Value::Tensor(value) => (0..value.len())
            .map(|index| {
                value
                    .numeric_value_at(index)
                    .ok_or_else(|| error::invalid("groupcounts: numeric storage is malformed"))
            })
            .collect(),
        _ => Err(error::invalid(
            "groupcounts: binning requires real numeric data and numeric edges or a bin count",
        )),
    }
}

fn is_bin_count(value: &Value) -> bool {
    tensor_utils::scalar_integer_value(value)
        .and_then(|value| value.try_to_usize())
        .is_some_and(|value| value > 0)
        || matches!(value, Value::Num(number) if positive_integer(*number).is_some())
        || matches!(value, Value::Tensor(tensor) if tensor.len() == 1 && positive_integer(tensor.materialize_f64()[0]).is_some())
}

fn bin_count(value: &Value) -> BuiltinResult<usize> {
    tensor_utils::scalar_integer_value(value)
        .and_then(|value| value.try_to_usize())
        .or_else(|| match value {
            Value::Num(value) => positive_integer(*value),
            Value::Tensor(value) if value.len() == 1 => {
                positive_integer(value.materialize_f64()[0])
            }
            _ => None,
        })
        .filter(|count| *count <= 1_000_000)
        .ok_or_else(|| {
            error::invalid(
                "groupcounts: number of bins must be a positive integer no greater than 1000000",
            )
        })
}

fn positive_integer(value: f64) -> Option<usize> {
    (value.is_finite() && value >= 1.0 && value.fract() == 0.0 && value <= usize::MAX as f64)
        .then_some(value as usize)
}
