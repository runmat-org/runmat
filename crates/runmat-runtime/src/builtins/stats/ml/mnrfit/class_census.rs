//! Bounded cooperative ordering for observed response classes.

use std::cmp::Ordering;

use super::{invalid, BuiltinResult, FitCooperation};

mod text;
pub(super) use self::text::{clone_text, find_text_class, sort_unique_text};

const RUN: usize = 4_096;

/// Sort and deduplicate without an unbounded synchronous `sort` turn. Values
/// move between checked scratch vectors; each comparison keeps its caller's
/// exact class ordering and equality relation.
pub(super) async fn sort_unique<T: Default>(
    mut values: Vec<T>,
    compare: impl Fn(&T, &T) -> Ordering,
    equivalent: impl Fn(&T, &T) -> bool,
    role: &str,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Vec<T>> {
    for chunk in values.chunks_mut(RUN) {
        chunk.sort_by(&compare);
        cooperation.completed(chunk.len() as u64).await?;
    }
    let len = values.len();
    if len > RUN {
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(len)
            .map_err(|_| invalid(format!("mnrfit: {role} exceeds available host resources")))?;
        let mut width = RUN;
        while width < len {
            let mut start = 0;
            let mut completed = 0u64;
            while start < len {
                let middle = start.saturating_add(width).min(len);
                let end = middle.saturating_add(width).min(len);
                let (mut left, mut right) = (start, middle);
                while left < middle || right < end {
                    let take_left = right == end
                        || (left < middle
                            && compare(&values[left], &values[right]) != Ordering::Greater);
                    let selected = if take_left {
                        let selected = left;
                        left += 1;
                        selected
                    } else {
                        let selected = right;
                        right += 1;
                        selected
                    };
                    scratch.push(std::mem::take(&mut values[selected]));
                    completed += 1;
                    if completed == 16_384 {
                        cooperation.completed(completed).await?;
                        completed = 0;
                    }
                }
                start = end;
            }
            cooperation.completed(completed).await?;
            values.clear();
            std::mem::swap(&mut values, &mut scratch);
            width = width.saturating_mul(2);
        }
    }
    let mut unique = Vec::new();
    unique
        .try_reserve_exact(values.len())
        .map_err(|_| invalid(format!("mnrfit: {role} exceeds available host resources")))?;
    let mut completed = 0u64;
    for mut value in values {
        if unique.last().is_none_or(|last| !equivalent(last, &value)) {
            unique.push(std::mem::take(&mut value));
        }
        completed += 1;
        if completed == 16_384 {
            cooperation.completed(completed).await?;
            completed = 0;
        }
    }
    cooperation.completed(completed).await?;
    Ok(unique)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn class_order_and_equality_hold_across_multiple_cooperative_runs() {
        let mut values = (0..(RUN * 2 + 3))
            .rev()
            .map(|value| value as f64)
            .collect::<Vec<_>>();
        values.extend([-0.0, 0.0, 1.0, f64::from_bits(1.0f64.to_bits() + 1)]);
        let classes = block_on(sort_unique(
            values,
            f64::total_cmp,
            |left, right| left == right,
            "numeric class census",
            &mut FitCooperation::new(None),
        ))
        .unwrap();
        assert_eq!(classes.len(), RUN * 2 + 4);
        assert_eq!(classes[0], 0.0);
        assert_eq!(classes[0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(classes[2], f64::from_bits(1.0f64.to_bits() + 1));
        assert_eq!(*classes.last().unwrap(), (RUN * 2 + 2) as f64);

        let text = (0..(RUN + 2))
            .rev()
            .map(|i| format!("label-{i:05}"))
            .collect();
        let ordered = block_on(sort_unique_text(text, &mut FitCooperation::new(None))).unwrap();
        assert_eq!(ordered.first().unwrap(), "label-00000");
        assert_eq!(ordered.last().unwrap(), &format!("label-{:05}", RUN + 1));
    }
}
