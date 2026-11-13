use super::*;

pub(super) fn coefficient_intervals(
    beta: &[f64],
    inv_xtx_active: &DMatrix<f64>,
    active_columns: &[usize],
    mse: f64,
    tcrit: f64,
) -> Vec<f64> {
    let cols = beta.len();
    let mut out = vec![0.0; cols * 2];
    for (active_idx, col) in active_columns.iter().copied().enumerate() {
        let variance = mse * inv_xtx_active[(active_idx, active_idx)];
        let half_width = if variance >= 0.0 {
            tcrit * variance.sqrt()
        } else {
            f64::NAN
        };
        out[col] = beta[col] - half_width;
        out[col + cols] = beta[col] + half_width;
    }
    out
}

pub(super) fn residual_intervals(
    design: &DMatrix<f64>,
    inv_xtx_active: &DMatrix<f64>,
    active_columns: &[usize],
    residuals: &[f64],
    sse: f64,
    dfe: f64,
    alpha: f64,
) -> Vec<f64> {
    let rows = design.nrows();
    let mut out = vec![f64::NAN; rows * 2];
    let residual_df = dfe - 1.0;
    let tcrit = if residual_df > 0.0 {
        student_t_inv(1.0 - alpha / 2.0, residual_df)
    } else {
        f64::NAN
    };
    for row in 0..rows {
        let mut leverage = 0.0;
        for (left_idx, left_col) in active_columns.iter().copied().enumerate() {
            for (right_idx, right_col) in active_columns.iter().copied().enumerate() {
                leverage += design[(row, left_col)]
                    * inv_xtx_active[(left_idx, right_idx)]
                    * design[(row, right_col)];
            }
        }
        let one_minus_h = 1.0 - leverage.clamp(0.0, 1.0);
        let deleted_sse = if one_minus_h > EPS {
            sse - residuals[row] * residuals[row] / one_minus_h
        } else {
            f64::NAN
        };
        let deleted_mse = if residual_df > 0.0 {
            (deleted_sse / residual_df).max(0.0)
        } else {
            f64::NAN
        };
        let variance = deleted_mse * one_minus_h;
        let half_width = if variance >= 0.0 && tcrit.is_finite() {
            tcrit * variance.sqrt()
        } else {
            f64::NAN
        };
        out[row] = residuals[row] - half_width;
        out[row + rows] = residuals[row] + half_width;
    }
    out
}

pub(super) fn stats_values(sse: f64, sst: f64, mse: f64, rows: usize, rank: usize) -> Vec<f64> {
    let r_squared = if sst > EPS {
        1.0 - sse / sst
    } else if sse <= EPS {
        1.0
    } else {
        f64::NAN
    };
    let dfe = rows as f64 - rank as f64;
    let df_model = rank.saturating_sub(1) as f64;
    let ssr = if sst.is_finite() && sse.is_finite() {
        sst - sse
    } else {
        f64::NAN
    };
    let f_stat = if df_model > 0.0 && dfe > 0.0 && mse > 0.0 {
        (ssr / df_model) / mse
    } else if df_model > 0.0 && dfe > 0.0 && mse == 0.0 && ssr > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    let p_value = if f_stat.is_finite() && f_stat >= 0.0 && df_model > 0.0 && dfe > 0.0 {
        f_upper_tail(f_stat, df_model, dfe)
    } else if f_stat == f64::INFINITY {
        0.0
    } else {
        f64::NAN
    };
    vec![r_squared, f_stat, p_value, mse]
}

pub(super) fn f_upper_tail(f: f64, df1: f64, df2: f64) -> f64 {
    if f < 0.0 || df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }
    if f == f64::INFINITY {
        return 0.0;
    }
    let x = (df1 * f) / (df1 * f + df2);
    (1.0 - regularized_beta(x, df1 / 2.0, df2 / 2.0)).clamp(0.0, 1.0)
}
