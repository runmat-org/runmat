//! Unbiased integer creation on the representable binary64 coordinate domain.
//! Dense integer intervals retain ordinary exact integer uniform sampling;
//! wide intervals use integral Float64 ordinals rather than a flintmax cap.
use super::super::{invalid, random, ScalarCooperation};
use crate::BuiltinResult;
const DENSE: u64 = 1_u64 << 53;
const START_BITS: u64 = 0x4340_0000_0000_0000; // binary64 2^53
const LAST: u64 = DENSE + (0x7fef_ffff_ffff_ffff - START_BITS);
#[derive(Clone, Copy)]
struct Interval {
    first: u64,
    span: u128,
}
impl Interval {
    fn new(lower: f64, upper: f64) -> BuiltinResult<Self> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(invalid(
                "integer creation requires finite ordered working ranges",
            ));
        }
        let lower = lower.ceil();
        let upper = upper.floor();
        if lower > upper {
            return Err(invalid(
                "integer creation working range contains no integer coordinate",
            ));
        }
        let first = rank(lower);
        let last = rank(upper);
        Ok(Self {
            first,
            span: u128::from(last - first) + 1,
        })
    }
    fn sample(self, offset: u64) -> BuiltinResult<f64> {
        if u128::from(offset) >= self.span {
            return Err(invalid("integer RNG returned an out-of-range offset"));
        }
        let index = self
            .first
            .checked_add(offset)
            .ok_or_else(|| invalid("integer coordinate ordinal overflow"))?;
        let (positive, negative) = if index < LAST {
            (LAST - index, true)
        } else {
            (index - LAST, false)
        };
        let value = if positive <= DENSE {
            positive as f64
        } else {
            f64::from_bits(START_BITS + (positive - DENSE))
        };
        Ok(if negative { -value } else { value })
    }
}
fn rank(value: f64) -> u64 {
    let magnitude = value.abs();
    let positive = if magnitude <= DENSE as f64 {
        magnitude as u64
    } else {
        DENSE + (magnitude.to_bits() - START_BITS)
    };
    if value < 0. {
        LAST - positive
    } else {
        LAST + positive
    }
}

pub(in super::super) async fn fill(
    draft: &mut super::Draft,
    ranges: &[(f64, f64)],
    integers: &[usize],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if ranges.len() != draft.variables() {
        return Err(invalid("integer creation range extent differs from nvars"));
    }
    let mut previous = None;
    for &column in integers {
        if column >= draft.variables() || previous.is_some_and(|prior| prior >= column) {
            return Err(invalid(
                "integer creation columns must be sorted, distinct and in range",
            ));
        }
        Interval::new(ranges[column].0, ranges[column].1)?;
        previous = Some(column);
        cooperation.completed(1).await?;
    }
    for &(lower, upper) in ranges {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(invalid("creation requires finite ordered working ranges"));
        }
        cooperation.completed(1).await?;
    }
    let mut integer = 0;
    for (column, &(lower, upper)) in ranges.iter().enumerate() {
        let interval = if integers.get(integer) == Some(&column) {
            integer += 1;
            Some(Interval::new(lower, upper)?)
        } else {
            None
        };
        let mut row = draft.missing().start;
        while row < draft.members() {
            let count = (draft.members() - row).min(128);
            if let Some(interval) = interval {
                let offsets = random::integer_offsets(interval.span, count, cooperation).await?;
                draft
                    .fill_column_block(
                        column,
                        row,
                        offsets.into_iter().map(|offset| interval.sample(offset)),
                        cooperation,
                    )
                    .await?;
            } else {
                let draws = random::uniform(count, cooperation).await?;
                draft
                    .fill_column_block(
                        column,
                        row,
                        draws.into_iter().map(|draw| {
                            super::super::operators::crossover::interpolate(lower, upper, draw)
                                .ok_or_else(|| invalid("creation coordinate cannot be represented"))
                        }),
                        cooperation,
                    )
                    .await?;
            }
            row += count;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_ordinals_cover_dense_and_wide_coordinates_without_rounding_or_caps() {
        let small = Interval::new(-3., 3.).unwrap();
        assert_eq!(small.span, 7);
        for offset in 0..7 {
            assert_eq!(small.sample(offset).unwrap(), offset as f64 - 3.);
        }
        let boundary = Interval::new(DENSE as f64 - 1., DENSE as f64 + 4.).unwrap();
        assert_eq!(boundary.span, 4);
        assert_eq!(
            (0..4)
                .map(|offset| boundary.sample(offset).unwrap())
                .collect::<Vec<_>>(),
            [
                DENSE as f64 - 1.,
                DENSE as f64,
                DENSE as f64 + 2.,
                DENSE as f64 + 4.
            ]
        );
        let wide = Interval::new(1e308, 1e308).unwrap();
        assert_eq!(wide.span, 1);
        assert_eq!(wide.sample(0).unwrap(), 1e308);
        let all = Interval::new(-f64::MAX, f64::MAX).unwrap();
        assert!(all.span <= 1_u128 << 64);
        assert_eq!(all.sample(0).unwrap(), -f64::MAX);
        assert_eq!(all.sample((all.span - 1) as u64).unwrap(), f64::MAX);
    }
}
