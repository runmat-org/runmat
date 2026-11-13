//! Stable semantic ordering with one counted ordinal buffer and in-place cycles.
use super::{check_cancelled, reserve, slots};
use crate::constraints::ConstraintPlanError;
use std::cmp::Ordering;

pub(in crate::constraints) fn ordering_workspace_bytes(
    count: usize,
) -> Result<u64, ConstraintPlanError> {
    if count < 2 {
        Ok(0)
    } else {
        slots::<usize>(count)
    }
}
pub(in crate::constraints) fn sort_stable_by<T>(
    values: &mut [T],
    compare: impl Fn(&T, &T) -> Ordering,
) -> Result<(), ConstraintPlanError> {
    check_cancelled()?;
    let _bytes = ordering_workspace_bytes(values.len())?;
    if values.len() < 2 {
        return Ok(());
    }
    let mut order = reserve::<usize>(values.len())?;
    order.extend(0..values.len());
    order.sort_unstable_by(|a, b| compare(&values[*a], &values[*b]).then_with(|| a.cmp(b)));
    // order[destination] names its original source. Each cycle places that source
    // at the current destination; ordinal ties reproduce stable sorting exactly.
    for start in 0..values.len() {
        if start % 1024 == 0 {
            check_cancelled()?;
        }
        let mut current = start;
        while order[current] != current {
            let next = order[current];
            order[current] = current;
            if next == start {
                break;
            }
            values.swap(current, next);
            current = next;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counted_order_matches_standard_stable_reference_across_permutation_cycles() {
        for count in [0, 1, 2, 3, 7, 31, 128, 1025] {
            let mut actual: Vec<_> = (0..count).map(|i| ((i * 17 + 3) % 11, i)).collect();
            let mut expected = actual.clone();
            expected.sort_by_key(|v| v.0);
            let pointer = actual.as_ptr();
            sort_stable_by(&mut actual, |a, b| a.0.cmp(&b.0)).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(actual.as_ptr(), pointer);
        }
        assert!(ordering_workspace_bytes(usize::MAX).is_err());
    }
}
