use super::*;

pub(in crate::builtins::stats::ml::perfcurve) fn auc(x: &[f64], y: &[f64]) -> f64 {
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

pub(in crate::builtins::stats::ml::perfcurve) fn auc_interval(
    x: &[f64],
    y: &[f64],
    xvals: &[f64],
) -> f64 {
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

pub(in crate::builtins::stats::ml::perfcurve) fn optimal_point(
    curve: &Curve,
    options: &Options,
) -> (f64, f64) {
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

pub(super) fn roc_slope(curve: &Curve, options: &Options) -> f64 {
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
        numerics::product_ratio(
            false_positive_cost,
            neg_prior,
            false_negative_cost,
            pos_prior,
        )
    }
}

pub(in crate::builtins::stats::ml::perfcurve) fn apply_xvals(
    curve: &Curve,
    xvals: &[f64],
    use_nearest: bool,
) -> BuiltinResult<Curve> {
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

pub(super) fn nearest_at_x(curve: &Curve, target: f64) -> (f64, f64, f64, Counts) {
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

pub(super) fn interpolate_at_x(curve: &Curve, target: f64) -> (f64, f64, f64, Counts) {
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

pub(super) fn interpolate_from_xy(x: &[f64], y: &[f64], target: f64) -> (f64, f64, f64, Counts) {
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
