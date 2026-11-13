//! In-place metadata ordering with caller-owned scratch and bounded check gaps.
use std::cmp::Ordering;
const BLOCK: usize = 128;

/// Sort copyable metadata without hidden scratch allocations. Errors may leave
/// metadata partially ordered; the caller discards that speculative preparation.
/// Panics if scratch is not exactly the input length.
pub fn sort_copy_with_checkpoint<T: Copy, E>(
    values: &mut [T],
    scratch: &mut [T],
    compare: impl Fn(&T, &T) -> Ordering,
    mut checkpoint: impl FnMut() -> Result<(), E>,
) -> Result<(), E> {
    assert_eq!(
        values.len(),
        scratch.len(),
        "sort scratch must match the input"
    );
    for chunk in values.chunks_mut(BLOCK) {
        checkpoint()?;
        chunk.sort_unstable_by(&compare);
    }
    let mut width = BLOCK;
    while width < values.len() {
        for start in (0..values.len()).step_by(width.saturating_mul(2)) {
            let middle = start.saturating_add(width).min(values.len());
            let end = middle.saturating_add(width).min(values.len());
            let (mut left, mut right) = (start, middle);
            for (position, output) in scratch[start..end].iter_mut().enumerate() {
                if position.is_multiple_of(BLOCK) {
                    checkpoint()?;
                }
                if right == end
                    || (left < middle
                        && compare(&values[left], &values[right]) != Ordering::Greater)
                {
                    *output = values[left];
                    left += 1;
                } else {
                    *output = values[right];
                    right += 1;
                }
            }
        }
        for (target, source) in values.chunks_mut(BLOCK).zip(scratch.chunks(BLOCK)) {
            checkpoint()?;
            target.copy_from_slice(source);
        }
        width = width.saturating_mul(2);
    }
    checkpoint()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordering_matches_standard_sort_across_block_boundaries_and_cancels() {
        for count in [0usize, 1, 127, 128, 129, 1025, 8193] {
            let original = (0..count)
                .map(|value| ((value * 7919 + 17) % 101) as i32)
                .collect::<Vec<_>>();
            let mut expected = original.clone();
            expected.sort_unstable();
            let mut actual = original.clone();
            let mut scratch = original.clone();
            let mut total = 0;
            sort_copy_with_checkpoint(&mut actual, &mut scratch, Ord::cmp, || {
                total += 1;
                Ok::<_, ()>(())
            })
            .unwrap();
            assert_eq!(actual, expected);
            if count > 128 {
                for limit in [0, 1, total / 2, total - 1] {
                    let mut actual = original.clone();
                    let mut scratch = original.clone();
                    let mut calls = 0;
                    let result =
                        sort_copy_with_checkpoint(&mut actual, &mut scratch, Ord::cmp, || {
                            calls += 1;
                            if calls > limit {
                                Err(())
                            } else {
                                Ok(())
                            }
                        });
                    assert!(result.is_err());
                    assert_eq!(calls, limit + 1);
                }
            }
        }
    }
}
