use super::*;
pub(super) fn correlation_pvalue(rho: f64, n: usize, corr_type: CorrType, tail: Tail) -> f64 {
    if rho.is_nan() || n < 2 {
        return f64::NAN;
    }
    let cdf = match corr_type {
        CorrType::Pearson | CorrType::Spearman => {
            if n < 3 {
                return f64::NAN;
            }
            let df = (n - 2) as f64;
            let denom = 1.0 - rho * rho;
            let t = if denom <= 0.0 {
                if rho > 0.0 {
                    f64::INFINITY
                } else if rho < 0.0 {
                    f64::NEG_INFINITY
                } else {
                    0.0
                }
            } else {
                rho * f64::sqrt(df / denom)
            };
            student_t_cdf(t, df)
        }
        CorrType::Kendall => {
            if n < 3 {
                return f64::NAN;
            }
            let n = n as f64;
            let variance = 2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0));
            standard_normal_cdf(rho / f64::sqrt(variance))
        }
    };
    tail_pvalue(cdf, tail)
}

pub(super) fn tail_pvalue(cdf: f64, tail: Tail) -> f64 {
    if cdf.is_nan() {
        return f64::NAN;
    }
    match tail {
        Tail::Both => (2.0 * cdf.min(1.0 - cdf)).clamp(0.0, 1.0),
        Tail::Right => (1.0 - cdf).clamp(0.0, 1.0),
        Tail::Left => cdf.clamp(0.0, 1.0),
    }
}
