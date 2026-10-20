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
    let count = bin_count(&edges_or_count)?;
    let computed = count.is_some();
    let edges = match count {
        Some(count) => equal_width_edges(values, count)?
            .into_iter()
            .map(NumericScalar::F64)
            .collect(),
        None => numeric::values(&edges_or_count, "edges")?,
    };
    validate_edges(&edges)?;
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

pub(super) fn validate_edges(edges: &[NumericScalar]) -> BuiltinResult<()> {
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

    Ok(())
}

pub(super) fn parse_tail(rest: Vec<Value>) -> BuiltinResult<(Option<Labels>, bool)> {
    let (labels, included_right) =
        tail_arguments(crate::call::FunctionArgumentView::from_slice(&rest))?;
    Ok((labels.map(Labels::from_value).transpose()?, included_right))
}

pub(super) fn tail_arguments(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(Option<&Value>, bool)> {
    let mut labels = None;
    let mut included_right = false;
    let mut index = 0;
    if let Some(first) = rest.first() {
        if !arguments::is_option_name(first) {
            labels = Some(first);
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

pub(super) fn bin_count(value: &Value) -> BuiltinResult<Option<usize>> {
    let scalar = match value {
        Value::Num(value) => Some(NumericScalar::F64(*value)),
        Value::Int(value) => Some(NumericScalar::from(value.clone())),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor.numeric_value_at(0),
        _ => return Ok(None),
    }
    .ok_or_else(|| error::invalid("discretize: invalid scalar bin count"))?;
    let count = if let Some(integer) = scalar.into_int_value() {
        integer.try_to_usize().filter(|count| *count > 0)
    } else {
        let number = scalar.materialize_f64();
        (number.is_finite() && number > 0.0 && number.fract() == 0.0 && number < usize::MAX as f64)
            .then_some(number as usize)
    }
    .ok_or_else(|| error::invalid("discretize: bin count must be a positive integer scalar"))?;
    if count >= MAX_COMPUTED_EDGES {
        return Err(error::too_large(
            "discretize: requested number of bins is too large",
        ));
    }
    Ok(Some(count))
}

fn equal_width_edges(values: &[NumericScalar], bins: usize) -> BuiltinResult<Vec<f64>> {
    if bins >= MAX_COMPUTED_EDGES {
        return Err(error::too_large(
            "discretize: requested number of bins is too large",
        ));
    }
    let (min, max) = super::computed_range::enclosing_bounds(values)?;
    Ok((0..=bins)
        .map(|index| {
            if index == 0 {
                return min;
            }
            if index == bins {
                return max;
            }
            let t = index as f64 / bins as f64;
            if min.is_sign_negative() != max.is_sign_negative() {
                min * (1.0 - t) + max * t
            } else {
                min + (max - min) * t
            }
        })
        .collect())
}
