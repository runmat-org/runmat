//! Cooperative text-class ordering. Long shared prefixes are charged by bytes.

use std::cmp::Ordering;

use super::{invalid, BuiltinResult, FitCooperation};

pub(in super::super) async fn sort_unique_text(
    mut values: Vec<String>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Vec<String>> {
    let len = values.len();
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(len)
        .map_err(|_| invalid("mnrfit: text class census exceeds available host resources"))?;
    let mut width = 1usize;
    while width < len {
        let mut start = 0usize;
        let mut moved = 0u64;
        while start < len {
            let middle = start.saturating_add(width).min(len);
            let end = middle.saturating_add(width).min(len);
            let (mut left, mut right) = (start, middle);
            while left < middle || right < end {
                let take_left = if right == end {
                    true
                } else if left == middle {
                    false
                } else {
                    compare_text(&values[left], &values[right], cooperation).await?
                        != Ordering::Greater
                };
                let selected = if take_left {
                    let index = left;
                    left += 1;
                    index
                } else {
                    let index = right;
                    right += 1;
                    index
                };
                scratch.push(std::mem::take(&mut values[selected]));
                moved += 1;
                if moved == 16_384 {
                    cooperation.completed(moved).await?;
                    moved = 0;
                }
            }
            start = end;
        }
        cooperation.completed(moved).await?;
        values.clear();
        std::mem::swap(&mut values, &mut scratch);
        width = width.saturating_mul(2);
    }
    let mut unique = Vec::<String>::new();
    unique
        .try_reserve_exact(values.len())
        .map_err(|_| invalid("mnrfit: text class census exceeds available host resources"))?;
    for value in values {
        let distinct = match unique.last() {
            Some(previous) => compare_text(previous, &value, cooperation).await? != Ordering::Equal,
            None => true,
        };
        if distinct {
            unique.push(value);
        }
    }
    Ok(unique)
}

async fn compare_text(
    left: &str,
    right: &str,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Ordering> {
    let left = left.as_bytes();
    let right = right.as_bytes();
    let common = left.len().min(right.len());
    let mut start = 0;
    while start < common {
        let end = start.saturating_add(16_384).min(common);
        let order = left[start..end].cmp(&right[start..end]);
        cooperation.completed((end - start) as u64).await?;
        if order != Ordering::Equal {
            return Ok(order);
        }
        start = end;
    }
    cooperation.completed(1).await?;
    Ok(left.len().cmp(&right.len()))
}

pub(in super::super) async fn find_text_class(
    classes: &[String],
    value: &str,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Option<usize>> {
    let (mut start, mut end) = (0, classes.len());
    while start < end {
        let middle = start + (end - start) / 2;
        match compare_text(classes[middle].as_str(), value, cooperation).await? {
            Ordering::Less => start = middle + 1,
            Ordering::Equal => return Ok(Some(middle)),
            Ordering::Greater => end = middle,
        }
    }
    Ok(None)
}

pub(in super::super) async fn clone_text(
    value: &str,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<String> {
    let mut copy = String::new();
    copy.try_reserve_exact(value.len())
        .map_err(|_| invalid("mnrfit: text class census exceeds available host resources"))?;
    let mut start = 0;
    while start < value.len() {
        let mut end = start.saturating_add(16_384).min(value.len());
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        copy.push_str(&value[start..end]);
        cooperation.completed((end - start) as u64).await?;
        start = end;
    }
    Ok(copy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use futures::task::noop_waker;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::task::{Context, Poll};

    #[test]
    fn long_common_prefixes_keep_exact_order_and_equality() {
        let prefix = "a".repeat(32_768);
        let first = format!("{prefix}x");
        let second = format!("{prefix}y");
        let ordered = block_on(sort_unique_text(
            vec![second.clone(), first.clone(), first.clone()],
            &mut FitCooperation::new(None),
        ))
        .unwrap();
        assert_eq!(ordered, vec![first, second]);
        let unicode = block_on(sort_unique_text(
            vec!["é".into(), "z".into(), "é".into(), "a".into()],
            &mut FitCooperation::new(None),
        ))
        .unwrap();
        assert_eq!(unicode, vec!["a".to_string(), "z".into(), "é".into()]);
        let copied = block_on(clone_text(
            &"é".repeat(16_385),
            &mut FitCooperation::new(None),
        ))
        .unwrap();
        assert_eq!(copied, "é".repeat(16_385));
        assert_eq!(
            block_on(find_text_class(
                &unicode,
                "z",
                &mut FitCooperation::new(None),
            ))
            .unwrap(),
            Some(1)
        );
        assert_eq!(
            block_on(find_text_class(
                &unicode,
                "zz",
                &mut FitCooperation::new(None),
            ))
            .unwrap(),
            None
        );
    }

    #[test]
    fn cancellation_interrupts_a_single_long_text_comparison() {
        let source = CancellationSource::new();
        let runtime = crate::context::RuntimeContext::with_cancellation(
            std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        let budget = crate::execution::CpuCooperationBudget::with_test_turn(
            crate::execution::CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(16_384).unwrap()),
            source.token(),
            || Box::pin(std::future::pending()),
        );
        let mut cooperation = FitCooperation::with_budget(&runtime, budget);
        let prefix = "a".repeat(32_768);
        let mut ordering = Box::pin(sort_unique_text(
            vec![format!("{prefix}y"), format!("{prefix}x")],
            &mut cooperation,
        ));
        let waker = noop_waker();
        // The first turn is due inside the shared prefix, not before comparison.
        assert!(ordering
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        source.cancel(CancellationReason::User);
        let Poll::Ready(Err(error)) = ordering.as_mut().poll(&mut Context::from_waker(&waker))
        else {
            panic!("cancellation must stop the comparison before class publication");
        };
        assert!(error.message().contains("cancelled"));
    }
}
