//! Reference KS statistic and documented large-sample approximations.
use super::*;
const SQRT_2: f64 = std::f64::consts::SQRT_2;
pub(super) fn evaluate_kstest(
    x: &Tensor,
    options: &KstestOptions,
) -> BuiltinResult<KstestEvaluation> {
    if !is_kstest_vector_shape(x.shape()) {
        return Err(kstest_invalid_argument("kstest: x must be a vector"));
    }
    let mut sample = tensor::tensor_values_f64(x)
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    sample.sort_by(f64::total_cmp);
    if sample.is_empty() {
        return Err(kstest_invalid_argument(
            "kstest: x must contain at least one non-NaN value",
        ));
    }
    validate_kstest_sample_domain(&sample, &options.cdf)?;
    let ksstat = kstest_statistic(&sample, options);
    let p = kstest_p_value(ksstat, sample.len(), options.tail);
    let cv = kstest_critical_value(options.alpha, sample.len(), options.tail);
    Ok(KstestEvaluation {
        h: Value::Bool(p.is_finite() && p <= options.alpha),
        p: Value::Num(p),
        ksstat: Value::Num(ksstat),
        cv: Value::Num(cv),
    })
}

pub(super) fn is_kstest_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|dim| **dim > 1).count() <= 1
}

fn validate_kstest_sample_domain(sample: &[f64], cdf: &CdfSpec) -> BuiltinResult<()> {
    if let CdfSpec::Table(points) = cdf {
        let min = points[0].0;
        let max = points[points.len() - 1].0;
        if sample[0] < min || sample[sample.len() - 1] > max {
            return Err(kstest_invalid_argument(
                "kstest: sample values must lie within the CDF table range",
            ));
        }
    }
    Ok(())
}

fn kstest_statistic(sample: &[f64], options: &KstestOptions) -> f64 {
    let n = sample.len() as f64;
    let mut d_plus: f64 = 0.0;
    let mut d_minus: f64 = 0.0;
    for (idx, value) in sample.iter().enumerate() {
        let f0 = cdf_value(&options.cdf, *value).clamp(0.0, 1.0);
        let i = idx as f64 + 1.0;
        d_plus = d_plus.max(i / n - f0);
        d_minus = d_minus.max(f0 - (i - 1.0) / n);
    }
    match options.tail {
        KstestTail::Unequal => d_plus.max(d_minus),
        KstestTail::Larger => d_plus,
        KstestTail::Smaller => d_minus,
    }
}

fn cdf_value(cdf: &CdfSpec, x: f64) -> f64 {
    match cdf {
        CdfSpec::StandardNormal => normal_cdf(x),
        CdfSpec::Table(points) => table_cdf_value(points, x),
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT_2)
}

fn table_cdf_value(points: &[(f64, f64)], x: f64) -> f64 {
    if x <= points[0].0 {
        return points[0].1;
    }
    let last = points.len() - 1;
    if x >= points[last].0 {
        return points[last].1;
    }
    let upper = points.partition_point(|(point_x, _)| *point_x < x);
    let (x0, f0) = points[upper - 1];
    let (x1, f1) = points[upper];
    let t = (x - x0) / (x1 - x0);
    f0 + t * (f1 - f0)
}

fn kstest_p_value(ksstat: f64, n: usize, tail: KstestTail) -> f64 {
    if !ksstat.is_finite() || n == 0 {
        return f64::NAN;
    }
    let n = n as f64;
    match tail {
        KstestTail::Unequal => {
            let z = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * ksstat;
            let mut sum = 0.0;
            for j in 1..=100 {
                let term = (-2.0 * (j as f64).powi(2) * z * z).exp();
                if j % 2 == 1 {
                    sum += term;
                } else {
                    sum -= term;
                }
                if term < 1.0e-14 {
                    break;
                }
            }
            (2.0 * sum).clamp(0.0, 1.0)
        }
        KstestTail::Larger | KstestTail::Smaller => (-2.0 * n * ksstat * ksstat).exp(),
    }
}

fn kstest_critical_value(alpha: f64, n: usize, tail: KstestTail) -> f64 {
    if !(alpha > 0.0 && alpha < 1.0) || n == 0 {
        return f64::NAN;
    }
    let n = n as f64;
    match tail {
        KstestTail::Unequal if (0.01..=0.20).contains(&alpha) => {
            (-0.5 * (alpha / 2.0).ln()).sqrt() / n.sqrt()
        }
        KstestTail::Larger | KstestTail::Smaller if (0.005..=0.10).contains(&alpha) => {
            (-0.5 * alpha.ln()).sqrt() / n.sqrt()
        }
        _ => f64::NAN,
    }
}
