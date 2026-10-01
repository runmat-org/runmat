use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ThresholdSpec {
    AboveAll(f64),
    Value(f64),
}

#[derive(Clone, Debug)]
pub(super) struct Options {
    pub(super) xcrit: Criterion,
    pub(super) ycrit: Criterion,
    pub(super) neg_class: Option<Vec<Label>>,
    pub(super) weights: Option<Vec<f64>>,
    pub(super) prior: PriorOption,
    pub(super) cost: [[f64; 2]; 2],
    pub(super) tvals: Option<Vec<f64>>,
    pub(super) xvals: Option<Vec<f64>>,
    pub(super) use_nearest: bool,
    pub(super) process_nan: ProcessNaN,
}

#[derive(Clone, Debug)]
pub(super) enum PriorOption {
    Empirical,
    Uniform,
    Values(f64, f64),
}

#[derive(Clone, Debug)]
pub(super) struct Observation {
    score: f64,
    weight: f64,
    pub(super) is_positive: bool,
    neg_index: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Counts {
    tp: f64,
    fp: f64,
    tn: f64,
    fn_: f64,
}

impl Counts {
    fn total(self) -> f64 {
        self.tp + self.fp + self.tn + self.fn_
    }
}

#[derive(Clone, Debug)]
pub(super) struct Curve {
    pub(super) x: Vec<f64>,
    pub(super) y: Vec<f64>,
    pub(super) thresholds: Vec<f64>,
    pub(super) counts: Vec<Counts>,
    pub(super) auc: f64,
}

pub(super) fn prepare_observations(
    labels: &LabelVector,
    scores: &[f64],
    posclass: &Label,
    neg_classes: &[Label],
    options: &Options,
) -> BuiltinResult<Vec<Observation>> {
    let weights = if let Some(weights) = &options.weights {
        if weights.len() != labels.labels.len() {
            return Err(invalid(
                "perfcurve: Weights must have one value per observation",
            ));
        }
        weights.clone()
    } else {
        vec![1.0; labels.labels.len()]
    };
    if weights
        .iter()
        .any(|value| value.is_nan() || *value < 0.0 || value.is_infinite())
    {
        return Err(invalid(
            "perfcurve: Weights must be finite nonnegative values",
        ));
    }
    let mut observations = Vec::with_capacity(labels.labels.len());
    for ((label, score), weight) in labels.labels.iter().zip(scores).zip(weights.iter()) {
        if is_missing_label(label) || *weight == 0.0 {
            continue;
        }
        if score.is_nan() && options.process_nan == ProcessNaN::Ignore {
            continue;
        }
        if score.is_infinite() {
            return Err(invalid("perfcurve: scores must be finite or NaN"));
        }
        let is_positive = same_label(label, posclass);
        let neg_index = if is_positive {
            None
        } else {
            neg_classes
                .iter()
                .position(|candidate| same_label(candidate, label))
        };
        if !is_positive && neg_index.is_none() {
            continue;
        }
        observations.push(Observation {
            score: *score,
            weight: *weight,
            is_positive,
            neg_index,
        });
    }
    apply_prior_weights(observations, &options.prior)
}

fn apply_prior_weights(
    mut observations: Vec<Observation>,
    prior: &PriorOption,
) -> BuiltinResult<Vec<Observation>> {
    let (pos_prior, neg_prior) = match prior {
        PriorOption::Empirical => {
            let mut total = 0.0_f64;
            for observation in &observations {
                total += observation.weight;
                if !total.is_finite() {
                    return Err(invalid(
                        "perfcurve: total observation weight exceeds finite double range",
                    ));
                }
            }
            return Ok(observations);
        }
        PriorOption::Uniform => (0.5, 0.5),
        PriorOption::Values(pos, neg) => (*pos, *neg),
    };
    let pos_max = observations
        .iter()
        .filter(|obs| obs.is_positive)
        .map(|obs| obs.weight)
        .fold(0.0_f64, f64::max);
    let neg_max = observations
        .iter()
        .filter(|obs| !obs.is_positive)
        .map(|obs| obs.weight)
        .fold(0.0_f64, f64::max);
    if pos_max == 0.0 || neg_max == 0.0 {
        return Ok(observations);
    }
    if pos_prior <= 0.0 || neg_prior <= 0.0 {
        return Err(invalid(
            "perfcurve: positive and negative prior probabilities must be nonzero",
        ));
    }
    let pos_scaled_total: f64 = observations
        .iter()
        .filter(|obs| obs.is_positive)
        .map(|obs| obs.weight / pos_max)
        .sum();
    let neg_scaled_total: f64 = observations
        .iter()
        .filter(|obs| !obs.is_positive)
        .map(|obs| obs.weight / neg_max)
        .sum();
    for obs in &mut observations {
        obs.weight = if obs.is_positive {
            (obs.weight / pos_max) * (pos_prior / pos_scaled_total)
        } else {
            (obs.weight / neg_max) * (neg_prior / neg_scaled_total)
        };
    }
    Ok(observations)
}

pub(super) fn thresholds_for_scores(
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

pub(super) fn compute_curve(
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

fn counts_for_thresholds(
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

fn criterion_value(counts: Counts, criterion: Criterion, cost: &[[f64; 2]; 2]) -> f64 {
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
        Criterion::ExpectedCost => ratio(
            (cost[0][0] * counts.tn)
                + (cost[0][1] * counts.fp)
                + (cost[1][0] * counts.fn_)
                + (cost[1][1] * counts.tp),
            counts.total(),
        ),
    }
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::NAN
    }
}

fn threshold_output_value(threshold: &ThresholdSpec) -> f64 {
    match threshold {
        ThresholdSpec::AboveAll(value) => *value,
        ThresholdSpec::Value(value) => *value,
    }
}

pub(super) fn auc(x: &[f64], y: &[f64]) -> f64 {
    let mut points: Vec<(f64, f64)> = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    points.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.total_cmp(&right.1)));
    let mut area = 0.0;
    for pair in points.windows(2) {
        let dx = pair[1].0 - pair[0].0;
        if dx >= 0.0 {
            area += dx * (pair[0].1 + pair[1].1) / 2.0;
        }
    }
    area
}

pub(super) fn auc_interval(x: &[f64], y: &[f64], xvals: &[f64]) -> f64 {
    let lo = xvals.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = xvals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut points: Vec<(f64, f64)> = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite() && *x >= lo && *x <= hi)
        .collect();
    if !points.iter().any(|(x, _)| *x == lo) {
        let (_, yi, _, _) = interpolate_from_xy(x, y, lo);
        if yi.is_finite() {
            points.push((lo, yi));
        }
    }
    if !points.iter().any(|(x, _)| *x == hi) {
        let (_, yi, _, _) = interpolate_from_xy(x, y, hi);
        if yi.is_finite() {
            points.push((hi, yi));
        }
    }
    points.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.total_cmp(&right.1)));
    let (xs, ys): (Vec<_>, Vec<_>) = points.into_iter().unzip();
    auc(&xs, &ys)
}

pub(super) fn optimal_point(curve: &Curve, options: &Options) -> (f64, f64) {
    if options.xcrit != Criterion::Fpr || options.ycrit != Criterion::Tpr {
        return (f64::NAN, f64::NAN);
    }
    let slope = roc_slope(curve, options);
    let mut best = (f64::NAN, f64::NAN);
    let mut best_score = f64::NEG_INFINITY;
    for ((x, y), counts) in curve.x.iter().zip(&curve.y).zip(&curve.counts) {
        let score = if slope.is_finite() {
            *y - slope * *x
        } else {
            -((criterion_value(*counts, Criterion::Fpr, &options.cost).powi(2))
                + (1.0 - criterion_value(*counts, Criterion::Tpr, &options.cost)).powi(2))
        };
        if score.is_finite() && score > best_score {
            best_score = score;
            best = (*x, *y);
        }
    }
    best
}

fn roc_slope(curve: &Curve, options: &Options) -> f64 {
    let Some(counts) = curve.counts.first().copied() else {
        return f64::NAN;
    };
    // Counts carry the effective class mass after empirical, uniform, or
    // explicit-prior weighting; the slope must use that same mass.
    let pos_prior = counts.tp + counts.fn_;
    let neg_prior = counts.fp + counts.tn;
    let false_positive_cost = options.cost[0][1] - options.cost[0][0];
    let false_negative_cost = options.cost[1][0] - options.cost[1][1];
    if false_positive_cost <= 0.0 || false_negative_cost <= 0.0 || pos_prior <= 0.0 {
        f64::NAN
    } else {
        (false_positive_cost * neg_prior) / (false_negative_cost * pos_prior)
    }
}

pub(super) fn apply_xvals(curve: &Curve, xvals: &[f64], use_nearest: bool) -> BuiltinResult<Curve> {
    let mut x = Vec::with_capacity(xvals.len());
    let mut y = Vec::with_capacity(xvals.len());
    let mut t = Vec::with_capacity(xvals.len());
    let mut counts = Vec::with_capacity(xvals.len());
    for target in xvals {
        let (xi, yi, ti, ci) = if use_nearest {
            nearest_at_x(curve, *target)
        } else {
            interpolate_at_x(curve, *target)
        };
        x.push(xi);
        y.push(yi);
        t.push(ti);
        counts.push(ci);
    }
    Ok(Curve {
        x,
        y,
        thresholds: t,
        counts,
        auc: 0.0,
    })
}

fn nearest_at_x(curve: &Curve, target: f64) -> (f64, f64, f64, Counts) {
    let mut best = 0usize;
    let mut best_dist = f64::INFINITY;
    for (idx, x) in curve.x.iter().enumerate() {
        let dist = (*x - target).abs();
        if dist < best_dist {
            best_dist = dist;
            best = idx;
        }
    }
    (
        curve.x[best],
        curve.y[best],
        curve.thresholds[best],
        curve.counts[best],
    )
}

fn interpolate_at_x(curve: &Curve, target: f64) -> (f64, f64, f64, Counts) {
    interpolate_from_points(
        curve
            .x
            .iter()
            .copied()
            .zip(curve.y.iter().copied())
            .zip(curve.thresholds.iter().copied())
            .zip(curve.counts.iter().copied())
            .map(|(((x, y), t), c)| (x, y, t, c))
            .collect(),
        target,
        curve.counts[0],
    )
}

fn interpolate_from_xy(x: &[f64], y: &[f64], target: f64) -> (f64, f64, f64, Counts) {
    interpolate_from_points(
        x.iter()
            .copied()
            .zip(y.iter().copied())
            .map(|(x, y)| {
                (
                    x,
                    y,
                    f64::NAN,
                    Counts {
                        tp: 0.0,
                        fp: 0.0,
                        tn: 0.0,
                        fn_: 0.0,
                    },
                )
            })
            .collect(),
        target,
        Counts {
            tp: 0.0,
            fp: 0.0,
            tn: 0.0,
            fn_: 0.0,
        },
    )
}

fn interpolate_from_points(
    raw_points: Vec<(f64, f64, f64, Counts)>,
    target: f64,
    fallback_counts: Counts,
) -> (f64, f64, f64, Counts) {
    let mut points: Vec<(f64, f64, f64, Counts)> = raw_points
        .into_iter()
        .filter(|(x, y, _, _)| x.is_finite() && y.is_finite())
        .collect();
    points.sort_by(|left, right| left.0.total_cmp(&right.0));
    if points.is_empty() {
        return (target, f64::NAN, f64::NAN, fallback_counts);
    }
    if target <= points[0].0 {
        return (target, points[0].1, points[0].2, points[0].3);
    }
    for pair in points.windows(2) {
        let (x0, y0, t0, c0) = pair[0];
        let (x1, y1, t1, c1) = pair[1];
        if target <= x1 {
            if x1 == x0 {
                return (target, y1, t1, c1);
            }
            let alpha = (target - x0) / (x1 - x0);
            return (target, y0 + alpha * (y1 - y0), t0 + alpha * (t1 - t0), c0);
        }
    }
    let last = points[points.len() - 1];
    (target, last.1, last.2, last.3)
}

pub(super) fn suby_output(
    observations: &[Observation],
    thresholds: &[ThresholdSpec],
    neg_classes: &[Label],
    xcrit: Criterion,
    ycrit: Criterion,
    cost: &[[f64; 2]; 2],
    xvals: Option<&Vec<f64>>,
    use_nearest: bool,
) -> BuiltinResult<Value> {
    let rows = xvals.map_or(thresholds.len(), Vec::len);
    let cols = neg_classes.len();
    let mut data = vec![0.0; rows * cols];
    for col in 0..cols {
        let mut x = Vec::with_capacity(thresholds.len());
        let mut y = Vec::with_capacity(thresholds.len());
        let counts = counts_for_thresholds(observations, thresholds, Some(col));
        for count in &counts {
            x.push(criterion_value(*count, xcrit, cost));
            y.push(criterion_value(*count, ycrit, cost));
        }
        let col_values = if let Some(xvals) = xvals {
            let curve = Curve {
                x,
                y,
                thresholds: thresholds.iter().map(threshold_output_value).collect(),
                counts,
                auc: 0.0,
            };
            xvals
                .iter()
                .map(|target| {
                    if use_nearest {
                        nearest_at_x(&curve, *target).1
                    } else {
                        interpolate_at_x(&curve, *target).1
                    }
                })
                .collect::<Vec<_>>()
        } else {
            y
        };
        for (row, value) in col_values.into_iter().enumerate() {
            data[row + col * rows] = value;
        }
    }
    Ok(Value::Tensor(
        Tensor::new(data, vec![rows, cols]).map_err(|err| internal(format!("perfcurve: {err}")))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_small_abscissae_interpolate_and_bound_auc_exactly() {
        let (_, y, _, _) = interpolate_from_xy(&[0.0, 1.0e-16], &[0.0, 1.0], 5.0e-17);
        assert!((y - 0.5).abs() < 1.0e-12);
        let area = auc_interval(&[0.0, 1.0e-16], &[0.0, 1.0], &[5.0e-17, 1.0e-16]);
        assert!((area - 3.75e-17).abs() < 1.0e-28);
    }

    #[test]
    fn optimal_roc_slope_uses_effective_empirical_class_mass() {
        let options = Options {
            xcrit: Criterion::Fpr,
            ycrit: Criterion::Tpr,
            neg_class: None,
            weights: None,
            prior: PriorOption::Empirical,
            cost: [[0.0, 1.0], [1.0, 0.0]],
            tvals: None,
            xvals: None,
            use_nearest: true,
            process_nan: ProcessNaN::Ignore,
        };
        let mut curve = Curve {
            x: vec![0.0],
            y: vec![0.0],
            thresholds: vec![1.0],
            counts: vec![Counts {
                tp: 0.0,
                fp: 0.0,
                tn: 3.0,
                fn_: 1.0,
            }],
            auc: 0.0,
        };
        assert_eq!(roc_slope(&curve, &options), 3.0);
        curve.counts[0] = Counts {
            tp: 0.0,
            fp: 0.0,
            tn: 0.5,
            fn_: 0.5,
        };
        assert_eq!(roc_slope(&curve, &options), 1.0);
    }
}
