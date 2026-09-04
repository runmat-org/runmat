use std::cmp::Ordering;

use runmat_value::{NumericScalar, Value};

use crate::BuiltinResult;

use super::{arguments, error, labels::Labels, numeric};
use crate::builtins::array::binning::numeric as numeric_order;

const MAX_COMPUTED_EDGES: usize = 50_000_000;

pub(super) struct Plan {
    pub edges: Vec<NumericScalar>,
    pub labels: Option<Labels>,
    pub included_right: bool,
    pub computed: bool,
}

pub(super) fn plan(
    values: &[NumericScalar],
    edges_or_count: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Plan> {
    let computed = bin_count(&edges_or_count).is_some();
    let edges = match bin_count(&edges_or_count) {
        Some(count) => equal_width_edges(values, count)?
            .into_iter()
            .map(NumericScalar::F64)
            .collect(),
        None => numeric::values(&edges_or_count, "edges")?,
    };
    if edges.len() < 2 {
        return Err(error::invalid(
            "discretize: at least two bin edges are required",
        ));
    }
    for pair in edges.windows(2) {
        let ordering = numeric_order::compare(pair[0], pair[1])
            .ok_or_else(|| error::invalid("discretize: bin edges must not contain NaN"))?;
        if ordering == Ordering::Greater {
            return Err(error::invalid(
                "discretize: bin edges must be monotonically increasing",
            ));
        }
    }

    let (labels, included_right) = parse_tail(rest)?;
    if let Some(labels) = &labels {
        if labels.len() != edges.len() - 1 {
            return Err(error::invalid(
                "discretize: number of replacement values must match number of bins",
            ));
        }
    }
    Ok(Plan {
        edges,
        labels,
        included_right,
        computed,
    })
}

fn parse_tail(rest: Vec<Value>) -> BuiltinResult<(Option<Labels>, bool)> {
    let mut labels = None;
    let mut included_right = false;
    let mut index = 0;
    if let Some(first) = rest.first() {
        if !arguments::is_option_name(first) {
            labels = Some(Labels::from_value(first)?);
            index = 1;
        }
    }
    while index < rest.len() {
        if index + 1 >= rest.len() {
            return Err(error::invalid(
                "discretize: name-value options must be provided in pairs",
            ));
        }
        let name = arguments::scalar_text(&rest[index], "option name")?;
        if !name.eq_ignore_ascii_case("IncludedEdge") {
            return Err(error::invalid(format!(
                "discretize: unsupported option '{name}'"
            )));
        }
        let side = arguments::scalar_text(&rest[index + 1], "IncludedEdge")?;
        included_right = match side.to_ascii_lowercase().as_str() {
            "left" => false,
            "right" => true,
            other => {
                return Err(error::invalid(format!(
                    "discretize: unsupported IncludedEdge '{other}'"
                )))
            }
        };
        index += 2;
    }
    Ok((labels, included_right))
}

fn bin_count(value: &Value) -> Option<usize> {
    match value {
        Value::Num(value)
            if value.is_finite()
                && *value > 0.0
                && value.fract() == 0.0
                && *value <= usize::MAX as f64 =>
        {
            Some(*value as usize)
        }
        Value::Int(value) => value.try_to_usize().filter(|value| *value > 0),
        _ => None,
    }
}

fn equal_width_edges(values: &[NumericScalar], bins: usize) -> BuiltinResult<Vec<f64>> {
    if bins >= MAX_COMPUTED_EDGES {
        return Err(error::too_large(
            "discretize: requested number of bins is too large",
        ));
    }
    let finite = values
        .iter()
        .map(|value| value.materialize_f64())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Err(error::invalid(
            "discretize: cannot infer equal-width bins from all-missing data",
        ));
    }
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if min == max {
        return Ok((0..=bins)
            .map(|index| min - 0.5 + index as f64 / bins as f64)
            .collect());
    }
    let step = (max - min) / bins as f64;
    Ok((0..=bins).map(|index| min + index as f64 * step).collect())
}
