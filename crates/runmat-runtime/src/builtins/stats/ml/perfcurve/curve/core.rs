use super::*;

pub(in crate::builtins::stats::ml::perfcurve) fn thresholds_for_scores(
    observations: &[Observation],
    explicit: Option<&Vec<f64>>,
) -> BuiltinResult<Vec<ThresholdSpec>> {
    if let Some(values) = explicit {
        let mut out: Vec<ThresholdSpec> =
            values.iter().copied().map(ThresholdSpec::Value).collect();
        out.sort_by(|left, right| {
            threshold_output_value(right).total_cmp(&threshold_output_value(left))
        });
        return Ok(out);
    }
    let mut values: Vec<f64> = observations
        .iter()
        .filter_map(|obs| obs.score.is_finite().then_some(obs.score))
        .collect();
    values.sort_by(|left, right| right.total_cmp(left));
    values.dedup_by(|left, right| left.total_cmp(right) == Ordering::Equal);
    if values.is_empty() {
        return Err(invalid(
            "perfcurve: at least one finite score is required to form thresholds",
        ));
    }
    let mut out = Vec::with_capacity(values.len() + 1);
    out.push(ThresholdSpec::AboveAll(values[0]));
    out.extend(values.into_iter().map(ThresholdSpec::Value));
    Ok(out)
}

pub(in crate::builtins::stats::ml::perfcurve) fn compute_curve(
    observations: &[Observation],
    thresholds: &[ThresholdSpec],
    xcrit: Criterion,
    ycrit: Criterion,
    cost: &[[f64; 2]; 2],
) -> BuiltinResult<Curve> {
    let mut x = Vec::with_capacity(thresholds.len());
    let mut y = Vec::with_capacity(thresholds.len());
    let mut t = Vec::with_capacity(thresholds.len());
    let counts = counts_for_thresholds(observations, thresholds, None);
    for (threshold, count) in thresholds.iter().zip(&counts) {
        x.push(criterion_value(*count, xcrit, cost));
        y.push(criterion_value(*count, ycrit, cost));
        t.push(threshold_output_value(threshold));
    }
    Ok(Curve {
        x,
        y,
        thresholds: t,
        counts,
        auc: 0.0,
    })
}

pub(super) fn counts_for_thresholds(
    observations: &[Observation],
    thresholds: &[ThresholdSpec],
    neg_index: Option<usize>,
) -> Vec<Counts> {
    let mut current = Counts {
        tp: 0.0,
        fp: 0.0,
        tn: 0.0,
        fn_: 0.0,
    };
    let mut finite = Vec::new();
    for obs in observations {
        if !observation_matches_subclass(obs, neg_index) {
            continue;
        }
        if obs.score.is_nan() {
            if obs.is_positive {
                current.fn_ += obs.weight;
            } else {
                current.fp += obs.weight;
            }
        } else {
            if obs.is_positive {
                current.fn_ += obs.weight;
            } else {
                current.tn += obs.weight;
            }
            finite.push(obs);
        }
    }
    finite.sort_by(|left, right| right.score.total_cmp(&left.score));

    let mut out = Vec::with_capacity(thresholds.len());
    let mut cursor = 0usize;
    for threshold in thresholds {
        match threshold {
            ThresholdSpec::AboveAll(_) => {}
            ThresholdSpec::Value(value) => {
                while cursor < finite.len() && finite[cursor].score >= *value {
                    let obs = finite[cursor];
                    if obs.is_positive {
                        current.fn_ -= obs.weight;
                        current.tp += obs.weight;
                    } else {
                        current.tn -= obs.weight;
                        current.fp += obs.weight;
                    }
                    cursor += 1;
                }
            }
        }
        out.push(current);
    }
    out
}

fn observation_matches_subclass(obs: &Observation, neg_index: Option<usize>) -> bool {
    obs.is_positive || neg_index.is_none() || obs.neg_index == neg_index
}

pub(super) fn criterion_value(counts: Counts, criterion: Criterion, cost: &[[f64; 2]; 2]) -> f64 {
    let pos = counts.tp + counts.fn_;
    let neg = counts.fp + counts.tn;
    match criterion {
        Criterion::Fpr => ratio(counts.fp, neg),
        Criterion::Tpr => ratio(counts.tp, pos),
        Criterion::Tnr => ratio(counts.tn, neg),
        Criterion::Fnr => ratio(counts.fn_, pos),
        Criterion::Precision => ratio(counts.tp, counts.tp + counts.fp),
        Criterion::Npv => ratio(counts.tn, counts.tn + counts.fn_),
        Criterion::Accuracy => ratio(counts.tp + counts.tn, counts.total()),
        Criterion::TruePositive => counts.tp,
        Criterion::FalseNegative => counts.fn_,
        Criterion::FalsePositive => counts.fp,
        Criterion::TrueNegative => counts.tn,
        Criterion::PredictedPositive => counts.tp + counts.fp,
        Criterion::RatePredictedPositive => ratio(counts.tp + counts.fp, counts.total()),
        Criterion::RatePredictedNegative => ratio(counts.tn + counts.fn_, counts.total()),
        Criterion::ExpectedCost => expected_cost(counts, cost),
    }
}

fn expected_cost(counts: Counts, cost: &[[f64; 2]; 2]) -> f64 {
    let total = counts.total();
    if total.is_nan() || total <= 0.0 {
        return f64::NAN;
    }
    let components = [
        (cost[0][0], counts.tn),
        (cost[0][1], counts.fp),
        (cost[1][0], counts.fn_),
        (cost[1][1], counts.tp),
    ];
    let maximum = cost.iter().flatten().copied().fold(0.0_f64, f64::max);
    let mut mean = 0.0_f64;
    for (value, count) in components {
        // Divide each nonnegative class mass by the total as part of a
        // scaled product. Multiplying value * count first can overflow even
        // though their weighted mean is finite.
        mean += numerics::product_ratio(value, count, total, 1.0);
    }
    // Rounded class fractions can add to slightly more than one. A weighted
    // mean of nonnegative costs cannot exceed the largest input cost.
    if mean.is_nan() {
        f64::NAN
    } else {
        mean.min(maximum)
    }
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::NAN
    }
}

pub(super) fn threshold_output_value(threshold: &ThresholdSpec) -> f64 {
    match threshold {
        ThresholdSpec::AboveAll(value) => *value,
        ThresholdSpec::Value(value) => *value,
    }
}
