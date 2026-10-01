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
    let mut unique = Vec::new();
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
    let mut pending = 0u64;
    for index in 0..common {
        pending += 1;
        let order = left[index].cmp(&right[index]);
        if order != Ordering::Equal {
            cooperation.completed(pending).await?;
            return Ok(order);
        }
        if pending == 16_384 {
            cooperation.completed(pending).await?;
            pending = 0;
        }
    }
    cooperation.completed(pending + 1).await?;
    Ok(left.len().cmp(&right.len()))
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
    }

    #[test]
    fn cancellation_interrupts_a_single_long_text_comparison() {
        let source = CancellationSource::new();
        let runtime = crate::context::RuntimeContext::with_cancellation(
            std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        let budget = crate::execution::CpuCooperationBudget::with_test_turn(
            crate::execution::CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
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
