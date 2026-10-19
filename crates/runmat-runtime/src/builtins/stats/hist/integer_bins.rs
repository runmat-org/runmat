//! Exact bin comparisons for fixed-width histogram observations.
//!
//! All RunMat integer classes fit in `i128`. Keep the observation in that
//! domain even when the public edge array is represented as double.

use std::cmp::Ordering;

use runmat_value::{IntValue, Value};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum UnitEdgeError {
    UnrepresentableBoundary,
    TooManyEdges,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GeneratedEdgeError {
    Nonfinite,
    Unrepresentable,
}

pub(super) fn reserve_double_edges(count: usize) -> Option<Vec<f64>> {
    let mut edges = Vec::new();
    edges.try_reserve_exact(count).ok()?;
    Some(edges)
}

pub(super) fn distinct_generated_double_edges(
    mut edges: Vec<f64>,
) -> Result<Vec<f64>, GeneratedEdgeError> {
    if edges.iter().any(|edge| !edge.is_finite()) {
        return Err(GeneratedEdgeError::Nonfinite);
    }
    if edges.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(GeneratedEdgeError::Unrepresentable);
    }
    let mut previous = None;
    edges.retain(|edge| {
        let distinct = previous.is_none_or(|last| *edge > last);
        if distinct {
            previous = Some(*edge);
        }
        distinct
    });
    if edges.len() == 1 {
        let centre = edges[0];
        let lower = next_down(centre);
        if lower.is_finite() {
            edges.insert(0, lower);
        } else {
            let upper = next_up(centre);
            if upper.is_finite() {
                edges.push(upper);
            }
        }
    }
    if edges.len() < 2 {
        return Err(GeneratedEdgeError::Unrepresentable);
    }
    Ok(edges)
}

/// Construct the integer-method grid only when every unit boundary has a
/// distinct double representation. The public generated-edge result is double;
/// silently collapsing adjacent integer bins would change its bin contract.
pub(super) fn unit_integer_edges(
    lower: f64,
    upper: f64,
    limits_supplied: bool,
) -> Result<Vec<f64>, UnitEdgeError> {
    const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_992.0;
    if !lower.is_finite()
        || !upper.is_finite()
        || lower < -MAX_EXACT_INTEGER
        || upper > MAX_EXACT_INTEGER
    {
        return Err(UnitEdgeError::UnrepresentableBoundary);
    }
    let first = if limits_supplied {
        lower.ceil()
    } else {
        lower.floor() + 1.0
    } as i128;
    let end = upper.ceil() as i128;
    let interior_count = end.saturating_sub(first);
    let capacity = interior_count
        .checked_add(2)
        .and_then(|count| usize::try_from(count).ok())
        .ok_or(UnitEdgeError::TooManyEdges)?;
    let mut edges = reserve_double_edges(capacity).ok_or(UnitEdgeError::TooManyEdges)?;
    edges.push(lower);
    for integer in first..end {
        let candidate = integer as f64;
        if candidate > lower + 1.0e-12 {
            edges.push(candidate);
        }
    }
    if edges.last().copied() != Some(upper) {
        edges.push(upper);
    }
    if edges.len() < 2 {
        return Err(UnitEdgeError::UnrepresentableBoundary);
    }
    Ok(edges)
}

pub(super) fn exact_integer_vector(value: &Value) -> Option<Vec<IntValue>> {
    match value {
        Value::Int(value) => Some(vec![value.clone()]),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| storage.exact_values()),
        _ => None,
    }
}

pub(super) fn edge_widths(floating: &[f64], exact: Option<&[IntValue]>) -> Vec<f64> {
    if let Some(exact) = exact {
        exact
            .windows(2)
            .map(|pair| (pair[1].to_i128() - pair[0].to_i128()) as f64)
            .collect()
    } else {
        floating.windows(2).map(|pair| pair[1] - pair[0]).collect()
    }
}

/// Expand only generated, unconstrained double boundaries when conversion of
/// an integer extreme rounded the displayed endpoint inward.
pub(super) fn cover_exact_integer_range(edges: &mut [f64], values: &[IntValue]) {
    let Some((minimum, maximum)) =
        values
            .iter()
            .map(IntValue::to_i128)
            .fold(None::<(i128, i128)>, |bounds, value| {
                Some(match bounds {
                    None => (value, value),
                    Some((low, high)) => (low.min(value), high.max(value)),
                })
            })
    else {
        return;
    };
    if edges.len() < 2 {
        return;
    }
    if compare_integer_float_i128(minimum, edges[0]) == Some(Ordering::Less) {
        let rounded = minimum as f64;
        edges[0] = if compare_integer_float_i128(minimum, rounded) == Some(Ordering::Less) {
            next_down(rounded)
        } else {
            rounded
        };
    }
    let last = edges.len() - 1;
    if compare_integer_float_i128(maximum, edges[last]) == Some(Ordering::Greater) {
        let rounded = maximum as f64;
        edges[last] = if compare_integer_float_i128(maximum, rounded) == Some(Ordering::Greater) {
            next_up(rounded)
        } else {
            rounded
        };
    }
}

pub(super) fn validate_exact_edges(edges: &[IntValue]) -> bool {
    edges.len() >= 2
        && edges
            .windows(2)
            .all(|pair| pair[0].to_i128() < pair[1].to_i128())
}

pub(super) fn exact_edge_bin(value: &IntValue, edges: &[IntValue]) -> Option<usize> {
    bin_from_comparison(edges.len(), |index| {
        Some(value.to_i128().cmp(&edges[index].to_i128()))
    })
}

pub(super) fn floating_edge_bin(value: &IntValue, edges: &[f64]) -> Option<usize> {
    bin_from_comparison(edges.len(), |index| {
        compare_integer_float(value, edges[index])
    })
}

pub(super) fn floating_observation_exact_bin(value: f64, edges: &[IntValue]) -> Option<usize> {
    bin_from_comparison(edges.len(), |index| {
        compare_integer_float(&edges[index], value).map(Ordering::reverse)
    })
}

fn bin_from_comparison(
    edge_count: usize,
    compare: impl Fn(usize) -> Option<Ordering>,
) -> Option<usize> {
    if edge_count < 2
        || compare(0)? == Ordering::Less
        || compare(edge_count - 1)? == Ordering::Greater
    {
        return None;
    }
    if compare(edge_count - 1)? == Ordering::Equal {
        return Some(edge_count - 2);
    }
    let mut low = 0;
    let mut high = edge_count - 1;
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        match compare(middle)? {
            Ordering::Less => high = middle,
            Ordering::Equal | Ordering::Greater => low = middle,
        }
    }
    Some(low)
}

fn compare_integer_float(value: &IntValue, edge: f64) -> Option<Ordering> {
    compare_integer_float_i128(value.to_i128(), edge)
}

fn compare_integer_float_i128(value: i128, edge: f64) -> Option<Ordering> {
    if edge.is_nan() {
        return None;
    }
    // These powers of two are exact binary64 values. They bracket every
    // fixed-width integer before converting the finite middle to i128.
    if edge < -9_223_372_036_854_775_808.0 {
        return Some(Ordering::Greater);
    }
    if edge >= 18_446_744_073_709_551_616.0 {
        return Some(Ordering::Less);
    }
    let whole = edge.trunc() as i128;
    let order = value.cmp(&whole);
    Some(if order == Ordering::Equal {
        if edge.fract() > 0.0 {
            Ordering::Less
        } else if edge.fract() < 0.0 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    } else {
        order
    })
}

pub(super) fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

pub(super) fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_method_grid_is_bounded_and_preserves_distinct_unit_edges() {
        assert_eq!(
            unit_integer_edges(0.0, 3.0, false),
            Ok(vec![0.0, 1.0, 2.0, 3.0])
        );
        assert_eq!(
            unit_integer_edges(0.5, 2.5, true),
            Ok(vec![0.5, 1.0, 2.0, 2.5])
        );
        assert_eq!(
            unit_integer_edges(-3.0, -1.0, false),
            Ok(vec![-3.0, -2.0, -1.0])
        );
        assert_eq!(
            unit_integer_edges(-0.5, 0.5, false),
            Ok(vec![-0.5, 0.0, 0.5])
        );
        assert_eq!(unit_integer_edges(1.0, 2.0, true), Ok(vec![1.0, 2.0]));
        assert_eq!(
            unit_integer_edges(9_007_199_254_740_992.0, 9_007_199_254_740_994.0, false),
            Err(UnitEdgeError::UnrepresentableBoundary)
        );
        assert_eq!(
            unit_integer_edges(-9_007_199_254_740_992.0, 9_007_199_254_740_992.0, false),
            Err(UnitEdgeError::TooManyEdges)
        );
    }

    #[test]
    fn generated_double_edges_compact_only_equal_rounding() {
        assert_eq!(
            distinct_generated_double_edges(vec![1.0, 1.0, 2.0]),
            Ok(vec![1.0, 2.0])
        );
        assert_eq!(
            distinct_generated_double_edges(vec![1.0, 0.0, 2.0]),
            Err(GeneratedEdgeError::Unrepresentable)
        );
        assert_eq!(
            distinct_generated_double_edges(vec![1.0, f64::NAN]),
            Err(GeneratedEdgeError::Nonfinite)
        );
    }

    #[test]
    fn adjacent_wide_edges_and_mixed_integer_classes_remain_distinct() {
        let base = 9_007_199_254_740_992_u64;
        let edges = [
            IntValue::U64(base),
            IntValue::U64(base + 1),
            IntValue::U64(base + 2),
        ];
        assert!(validate_exact_edges(&edges));
        assert_eq!(exact_edge_bin(&IntValue::U64(base), &edges), Some(0));
        assert_eq!(exact_edge_bin(&IntValue::U64(base + 1), &edges), Some(1));
        assert_eq!(exact_edge_bin(&IntValue::U64(base + 2), &edges), Some(1));
        assert_eq!(exact_edge_bin(&IntValue::I64(-1), &edges), None);
        assert_eq!(
            exact_edge_bin(
                &IntValue::I64(0),
                &[IntValue::U8(0), IntValue::U64(u64::MAX)]
            ),
            Some(0)
        );
        assert!(!validate_exact_edges(&[
            IntValue::U64(base),
            IntValue::U64(base)
        ]));
    }

    #[test]
    fn float_edges_do_not_round_integer_observations() {
        let base = 9_007_199_254_740_992_u64;
        let edges = [base as f64, (base + 2) as f64, (base + 4) as f64];
        assert_eq!(floating_edge_bin(&IntValue::U64(base + 1), &edges), Some(0));
        assert_eq!(floating_edge_bin(&IntValue::U64(base + 2), &edges), Some(1));
        assert_eq!(floating_edge_bin(&IntValue::U64(u64::MAX), &edges), None);
        assert_eq!(
            floating_edge_bin(&IntValue::I64(i64::MIN), &[f64::NEG_INFINITY, -1.0]),
            Some(0)
        );
        assert_eq!(floating_edge_bin(&IntValue::U8(0), &[-0.0, 1.0]), Some(0));
        assert_eq!(floating_edge_bin(&IntValue::U8(0), &[f64::NAN, 1.0]), None);
        let exact = [IntValue::U64(base + 1), IntValue::U64(base + 2)];
        assert_eq!(floating_observation_exact_bin(base as f64, &exact), None);
        assert_eq!(
            floating_observation_exact_bin((base + 2) as f64, &exact),
            Some(0)
        );
    }

    #[test]
    fn generated_display_bounds_cover_rounded_extremes() {
        let mut edges = [9_007_199_254_740_992.0, 9_007_199_254_740_996.0];
        cover_exact_integer_range(&mut edges, &[IntValue::U64(9_007_199_254_740_993)]);
        assert_eq!(edges, [9_007_199_254_740_992.0, 9_007_199_254_740_996.0]);

        let mut upper = [18_446_744_073_709_547_520.0, 18_446_744_073_709_549_568.0];
        cover_exact_integer_range(&mut upper, &[IntValue::U64(u64::MAX)]);
        assert_eq!(floating_edge_bin(&IntValue::U64(u64::MAX), &upper), Some(0));
    }
}
