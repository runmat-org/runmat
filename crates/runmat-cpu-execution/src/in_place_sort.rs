//! Cooperative metadata ordering without an additional full-size scratch buffer.
use std::cmp::Ordering;

/// Unstable in-place heapsort. Checks before work and at most every 128 heap
/// comparisons. An error leaves a permutation of the input, partially ordered.
pub fn sort_in_place_with_checkpoint<T, E>(
    values: &mut [T],
    compare: impl Fn(&T, &T) -> Ordering,
    mut checkpoint: impl FnMut() -> Result<(), E>,
) -> Result<(), E> {
    checkpoint()?;
    let mut comparisons = 0usize;
    let mut compare = |left: &T, right: &T| {
        if comparisons.is_multiple_of(128) {
            checkpoint()?;
        }
        comparisons = comparisons.wrapping_add(1);
        Ok(compare(left, right))
    };
    for root in (0..values.len() / 2).rev() {
        sift_down(values, root, &mut compare)?;
    }
    for end in (1..values.len()).rev() {
        values.swap(0, end);
        sift_down(&mut values[..end], 0, &mut compare)?;
    }
    checkpoint()
}
fn sift_down<T, E>(
    values: &mut [T],
    mut root: usize,
    compare: &mut impl FnMut(&T, &T) -> Result<Ordering, E>,
) -> Result<(), E> {
    while root < values.len() / 2 {
        let mut child = root * 2 + 1;
        if child + 1 < values.len()
            && compare(&values[child], &values[child + 1])? == Ordering::Less
        {
            child += 1;
        }
        if compare(&values[root], &values[child])? != Ordering::Less {
            break;
        }
        values.swap(root, child);
        root = child;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    // witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: In-place ordering matches standard ascending and descending order across block
    //   boundaries, and cancellation preserves the input inventory. The implementation uses no
    //   full-size scratch buffer; this finite primitive test is not an end-to-end resource or speed
    //   qualification.
    #[test]
    fn orders_both_directions_and_preserves_inventory_on_cancellation() {
        for length in [0, 1, 2, 127, 128, 129, 1025, 8193] {
            let original = (0..length)
                .map(|i| (i * 7919 + 17) % 101)
                .collect::<Vec<_>>();
            for reverse in [false, true] {
                let compare = |a: &usize, b: &usize| if reverse { b.cmp(a) } else { a.cmp(b) };
                let mut expected = original.clone();
                expected.sort_unstable_by(compare);
                let mut actual = original.clone();
                let mut calls = 0;
                sort_in_place_with_checkpoint(&mut actual, compare, || {
                    calls += 1;
                    Ok::<_, ()>(())
                })
                .unwrap();
                assert_eq!(actual, expected);
                for stop in [0, calls / 2, calls - 1] {
                    let mut actual = original.clone();
                    let mut count = 0;
                    assert!(sort_in_place_with_checkpoint(&mut actual, compare, || {
                        count += 1;
                        if count > stop {
                            Err(())
                        } else {
                            Ok(())
                        }
                    })
                    .is_err());
                    assert_eq!(count, stop + 1);
                    actual.sort_unstable();
                    let mut inventory = original.clone();
                    inventory.sort_unstable();
                    assert_eq!(actual, inventory);
                }
            }
        }
    }
}
