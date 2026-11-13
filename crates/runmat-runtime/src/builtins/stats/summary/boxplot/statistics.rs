//! Box quartiles, whiskers and outlier classification.
use super::*;
use crate::builtins::stats::summary::order_stats::{stable_sort_non_nan_f64, CooperativeSortError};

pub(super) async fn stats_for_values_cooperative(
    label: String,
    values: Vec<f64>,
    whisker: f64,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<BoxStats> {
    let mut finite = Vec::new();
    for value in values {
        if !value.is_nan() {
            if finite.len() == finite.capacity() {
                finite.try_reserve(1).map_err(|error| {
                    allocation_failure("boxplot: sample statistics exceed available memory", error)
                })?;
            }
            finite.push(value);
        }
        cooperation.completed(1).await?;
    }
    if finite.is_empty() {
        return Err(invalid(
            "boxplot: each plotted box must contain at least one non-NaN value",
        ));
    }
    stable_sort_non_nan_f64(&mut finite, cooperation.budget())
        .await
        .map_err(|error| match error {
            CooperativeSortError::Resource(error) => {
                allocation_failure("boxplot: sample statistics exceed available memory", error)
            }
            CooperativeSortError::Service(error) => cooperation.service_error(error),
        })?;
    let observation_count = finite.len();
    let q1 = percentile_sorted(&finite, 0.25);
    let median = percentile_sorted(&finite, 0.5);
    let q3 = percentile_sorted(&finite, 0.75);
    let iqr = q3 - q1;
    let (lower_fence, upper_fence) = if whisker == 0.0 {
        // An overflowing IQR still has exact zero-length whiskers at Q1/Q3.
        // Multiplying 0 by an infinite intermediate would produce NaN fences.
        (q1, q3)
    } else {
        (q1 - whisker * iqr, q3 + whisker * iqr)
    };
    let mut lower_whisker = None;
    let mut upper_whisker = None;
    let mut outliers = Vec::new();
    for value in finite {
        if value < lower_fence || value > upper_fence {
            if outliers.len() == outliers.capacity() {
                outliers.try_reserve(1).map_err(|error| {
                    allocation_failure("boxplot: sample outliers exceed available memory", error)
                })?;
            }
            outliers.push(value);
        } else {
            lower_whisker = Some(lower_whisker.map_or(value, |old: f64| old.min(value)));
            upper_whisker = Some(upper_whisker.map_or(value, |old: f64| old.max(value)));
        }
        cooperation.completed(1).await?;
    }
    Ok(BoxStats {
        label,
        group_key: Vec::new(),
        position: 0.0,
        width: 0.5,
        color: Vec4::ONE,
        q1,
        median,
        q3,
        lower_whisker: lower_whisker.unwrap_or(median),
        upper_whisker: upper_whisker.unwrap_or(median),
        outliers,
        observation_count,
    })
}

fn percentile_sorted(values: &[f64], p: f64) -> f64 {
    if values.len() == 1 {
        return values[0];
    }
    let pos = p * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let frac = pos - lo as f64;
        values[lo] * (1.0 - frac) + values[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    #[test]
    fn zero_whisker_keeps_extreme_outlier_when_iqr_overflows() {
        let max = f64::MAX;
        let stats = futures::executor::block_on(stats_for_values_cooperative(
            "A".into(),
            vec![-0.8 * max, -0.8 * max, 0.8 * max, 0.8 * max, max],
            0.0,
            &mut super::super::cooperation::BoxplotCooperation::new(None),
        ))
        .unwrap();
        assert_eq!(stats.q1, -0.8 * max);
        assert_eq!(stats.q3, 0.8 * max);
        assert_eq!(stats.outliers, vec![max]);
    }

    #[test]
    fn host_allocation_failure_is_internal_and_retains_original_cause() {
        let mut values: Vec<u8> = Vec::new();
        let cause = values.try_reserve(usize::MAX).unwrap_err();
        let error = allocation_failure("boxplot: sample statistics exceed available memory", cause);
        assert_eq!(error.identifier(), BOXPLOT_ERROR_INTERNAL.identifier);
        assert!(error.source.is_some());
    }

    #[test]
    fn cancellation_during_stable_sample_sort_prevents_statistics_publication() {
        let source = Rc::new(CancellationSource::new());
        let turns = Rc::new(Cell::new(0));
        let cancel_on_turn = source.clone();
        let counted = turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                counted.set(counted.get() + 1);
                if counted.get() == 602 {
                    cancel_on_turn.cancel(CancellationReason::User);
                }
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = super::super::cooperation::BoxplotCooperation::with_budget(budget);
        let values = (0..600).rev().map(f64::from).collect();
        let error = futures::executor::block_on(stats_for_values_cooperative(
            "A".into(),
            values,
            1.5,
            &mut cooperation,
        ))
        .expect_err("cancellation in the sort must not yield a partial box");
        assert_eq!(turns.get(), 602);
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    }
}
