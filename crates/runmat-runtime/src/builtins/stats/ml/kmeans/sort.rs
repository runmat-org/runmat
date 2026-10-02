//! Stable bounded-run ordering for clustering starts and median centroids.

use std::cmp::Ordering;

use super::{cooperation::KmeansCooperation, invalid};
use crate::BuiltinResult;

// A bounded uninterrupted library sort; the runtime execution service owns
// the actual turn policy and quantum.
const RUN_LEN: usize = 256;

pub(super) async fn stable_sort_by<T: Clone>(
    values: &mut Vec<T>,
    mut compare: impl FnMut(&T, &T) -> Ordering,
    cooperation: &mut KmeansCooperation,
) -> BuiltinResult<()> {
    cooperation.completed(0).await?;
    for run in values.chunks_mut(RUN_LEN) {
        run.sort_by(&mut compare);
        cooperation.completed(run.len() as u64).await?;
    }
    let mut width = RUN_LEN;
    while width < values.len() {
        let mut merged = Vec::new();
        merged
            .try_reserve_exact(values.len())
            .map_err(|_| invalid("kmeans: sorted observations exceed available memory"))?;
        let mut start = 0;
        while start < values.len() {
            let middle = start.saturating_add(width).min(values.len());
            let end = middle.saturating_add(width).min(values.len());
            let (mut left, mut right) = (start, middle);
            while left < middle && right < end {
                if compare(&values[left], &values[right]) != Ordering::Greater {
                    merged.push(values[left].clone());
                    left += 1;
                } else {
                    merged.push(values[right].clone());
                    right += 1;
                }
                cooperation.completed(1).await?;
            }
            while left < middle {
                merged.push(values[left].clone());
                left += 1;
                cooperation.completed(1).await?;
            }
            while right < end {
                merged.push(values[right].clone());
                right += 1;
                cooperation.completed(1).await?;
            }
            start = end;
        }
        *values = merged;
        width = width.saturating_mul(2);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_merge_preserves_equal_score_indices_across_runs() {
        let mut actual = (0..RUN_LEN + 3)
            .map(|index| (index, if index % 2 == 0 { 1.0 } else { 0.0 }))
            .collect::<Vec<_>>();
        let mut expected = actual.clone();
        expected.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap());
        futures::executor::block_on(stable_sort_by(
            &mut actual,
            |left, right| left.1.partial_cmp(&right.1).unwrap(),
            &mut KmeansCooperation::new(None),
        ))
        .unwrap();
        assert_eq!(actual, expected);
    }
}
