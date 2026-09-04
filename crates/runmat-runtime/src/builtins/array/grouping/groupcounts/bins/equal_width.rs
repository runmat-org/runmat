use runmat_value::{IntValue, NumericScalar};

use crate::builtins::array::binning::numeric;
use crate::BuiltinResult;

use super::labels;
pub(super) fn assign(
    values: &[NumericScalar],
    count: usize,
    included_right: bool,
) -> BuiltinResult<(Vec<Option<usize>>, Vec<String>)> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| numeric::compare(*value, *value).is_some())
        .collect::<Vec<_>>();
    finite.sort_by(|left, right| {
        numeric::compare(*left, *right).unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(minimum) = finite.first().copied() else {
        return Ok((
            vec![None; values.len()],
            (1..=count).map(|index| format!("bin{index}")).collect(),
        ));
    };
    let maximum = finite[finite.len() - 1];
    let assignments = match (integer(minimum), integer(maximum)) {
        (Some(minimum), Some(maximum)) => {
            integer_assignments(values, minimum, maximum, count, included_right)
        }
        _ => floating_assignments(
            values,
            minimum.materialize_f64(),
            maximum.materialize_f64(),
            count,
            included_right,
        ),
    };
    let labels = if minimum == maximum {
        (1..=count).map(|index| format!("bin{index}")).collect()
    } else {
        equal_width_labels(minimum, maximum, count, included_right)
    };
    Ok((assignments, labels))
}

fn integer_assignments(
    values: &[NumericScalar],
    minimum: i128,
    maximum: i128,
    count: usize,
    included_right: bool,
) -> Vec<Option<usize>> {
    let range = (maximum - minimum) as u128;
    values
        .iter()
        .map(|value| {
            let value = integer(*value)?;
            if range == 0 {
                return Some(0);
            }
            let delta = (value - minimum) as u128;
            let scaled = delta * count as u128;
            let index = if included_right && delta > 0 {
                scaled.div_ceil(range).saturating_sub(1)
            } else {
                scaled / range
            };
            Some((index as usize).min(count - 1))
        })
        .collect()
}

fn floating_assignments(
    values: &[NumericScalar],
    minimum: f64,
    maximum: f64,
    count: usize,
    included_right: bool,
) -> Vec<Option<usize>> {
    values
        .iter()
        .map(|value| {
            let value = value.materialize_f64();
            if !value.is_finite() {
                None
            } else if minimum == maximum {
                Some(0)
            } else {
                let scaled = (value - minimum) * count as f64 / (maximum - minimum);
                let index = if included_right && value > minimum {
                    scaled.ceil() - 1.0
                } else {
                    scaled.floor()
                };
                Some((index.max(0.0) as usize).min(count - 1))
            }
        })
        .collect()
}

fn equal_width_labels(
    minimum: NumericScalar,
    maximum: NumericScalar,
    count: usize,
    included_right: bool,
) -> Vec<String> {
    let minimum = minimum.materialize_f64();
    let maximum = maximum.materialize_f64();
    let width = (maximum - minimum) / count as f64;
    (0..count)
        .map(|index| {
            labels::floating_interval(
                minimum + width * index as f64,
                if index + 1 == count {
                    maximum
                } else {
                    minimum + width * (index + 1) as f64
                },
                index,
                count,
                included_right,
            )
        })
        .collect()
}

fn integer(value: NumericScalar) -> Option<i128> {
    value.into_int_value().map(|value| match value {
        IntValue::I8(value) => i128::from(value),
        IntValue::I16(value) => i128::from(value),
        IntValue::I32(value) => i128::from(value),
        IntValue::I64(value) => i128::from(value),
        IntValue::U8(value) => i128::from(value),
        IntValue::U16(value) => i128::from(value),
        IntValue::U32(value) => i128::from(value),
        IntValue::U64(value) => i128::from(value),
    })
}
