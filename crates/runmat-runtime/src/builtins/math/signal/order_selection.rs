use runmat_value::Value;

use crate::builtins::math::signal::common::{parse_scalar_f64, value_to_complex_vector};

const EPS: f64 = 1.0e-12;
const GOLDEN_RATIO_CONJUGATE: f64 = 0.381_966_011_250_105_1;
const DB_TO_NEPER: f64 = std::f64::consts::LN_10 / 10.0;
pub(crate) const MAX_ORDER: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FilterKind {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrderFamily {
    Butterworth,
    Chebyshev,
}

pub(crate) async fn real_edges(
    builtin_name: &'static str,
    label: &'static str,
    value: Value,
) -> Result<Vec<f64>, String> {
    let input = value_to_complex_vector(builtin_name, label, value)
        .await
        .map_err(|err| err.message().to_string())?;
    if input.data.len() != 1 && input.data.len() != 2 {
        return Err(format!("{label} must be a scalar or two-element vector"));
    }
    let mut out = Vec::with_capacity(input.data.len());
    for value in input.data {
        if value.im.abs() > EPS || !value.re.is_finite() || value.re <= 0.0 {
            return Err(format!(
                "{label} entries must be positive finite real values"
            ));
        }
        out.push(value.re);
    }
    if out.len() == 2 && out[0] >= out[1] {
        return Err(format!("{label} edge vector must be strictly increasing"));
    }
    Ok(out)
}

pub(crate) fn parse_positive_db(
    builtin_name: &'static str,
    label: &'static str,
    value: &Value,
) -> Result<f64, String> {
    let parsed =
        parse_scalar_f64(builtin_name, label, value).map_err(|err| err.message().to_string())?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(format!("{label} must be a positive finite scalar"));
    }
    Ok(parsed)
}

pub(crate) fn option_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.trim().to_ascii_lowercase()),
        Value::StringArray(array) if array.data.len() == 1 => {
            Some(array.data[0].trim().to_ascii_lowercase())
        }
        Value::CharArray(chars) if chars.rows <= 1 => Some(
            chars
                .data
                .iter()
                .collect::<String>()
                .trim()
                .to_ascii_lowercase(),
        ),
        _ => None,
    }
}

pub(crate) fn classify_kind(wp: &[f64], ws: &[f64]) -> Result<FilterKind, String> {
    match (wp.len(), ws.len()) {
        (1, 1) if wp[0] < ws[0] => Ok(FilterKind::Lowpass),
        (1, 1) if wp[0] > ws[0] => Ok(FilterKind::Highpass),
        (2, 2) if ws[0] < wp[0] && wp[1] < ws[1] => Ok(FilterKind::Bandpass),
        (2, 2) if wp[0] < ws[0] && ws[1] < wp[1] => Ok(FilterKind::Bandstop),
        _ => Err("passband and stopband edges do not define a supported response".to_string()),
    }
}

pub(crate) fn validate_domain(
    edges: &[f64],
    analog: bool,
    label: &'static str,
) -> Result<(), String> {
    if !analog && edges.iter().any(|&edge| edge >= 1.0) {
        return Err(format!(
            "{label} digital frequencies must be between 0 and 1"
        ));
    }
    Ok(())
}

pub(crate) fn transform_edges(edges: &[f64], analog: bool) -> Vec<f64> {
    if analog {
        edges.to_vec()
    } else {
        edges.iter().map(|&edge| prewarp_frequency(edge)).collect()
    }
}

pub(crate) fn prewarp_frequency(edge: f64) -> f64 {
    (std::f64::consts::PI * edge / 2.0).tan()
}

pub(crate) fn unwarp_frequency(edge: f64) -> f64 {
    2.0 * edge.atan() / std::f64::consts::PI
}

pub(crate) fn find_natural_frequency(
    kind: FilterKind,
    passband: &[f64],
    stopband: &[f64],
    rp: f64,
    rs: f64,
    family: OrderFamily,
) -> Result<(f64, Vec<f64>), String> {
    let mut passband = passband.to_vec();
    let nat_values = match kind {
        FilterKind::Lowpass => vec![stopband[0] / passband[0]],
        FilterKind::Highpass => vec![passband[0] / stopband[0]],
        FilterKind::Bandpass => {
            let bandwidth = passband[1] - passband[0];
            let center_sq = passband[0] * passband[1];
            stopband
                .iter()
                .map(|&edge| (edge * edge - center_sq) / (bandwidth * edge))
                .collect()
        }
        FilterKind::Bandstop => {
            passband = optimize_bandstop_passband(&passband, stopband, rp, rs, family)?;
            bandstop_natural_values(&passband, stopband)
        }
    };
    let nat = nat_values
        .into_iter()
        .map(f64::abs)
        .fold(f64::INFINITY, f64::min);
    if !nat.is_finite() || nat <= 1.0 {
        return Err("stopband must be separated from passband".to_string());
    }
    Ok((nat, passband))
}

pub(crate) fn db_excess(db: f64) -> Result<f64, String> {
    let x = db * DB_TO_NEPER;
    if !x.is_finite() || x <= 0.0 {
        return Err("attenuation must be positive and finite".to_string());
    }
    let excess = x.exp_m1();
    if !excess.is_finite() || excess <= 0.0 {
        return Err("attenuation is too large for stable order selection".to_string());
    }
    Ok(excess)
}

pub(crate) fn db_excess_log_ratio(rp: f64, rs: f64) -> Result<f64, String> {
    let pass = ln_expm1(rp * DB_TO_NEPER)?;
    let stop = ln_expm1(rs * DB_TO_NEPER)?;
    let ratio = stop - pass;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err("attenuation ratio must be positive and finite".to_string());
    }
    Ok(ratio)
}

pub(crate) fn chebyshev_v_pass_stop(log_ratio: f64) -> Result<f64, String> {
    let half = 0.5 * log_ratio;
    let value = if half < 350.0 {
        half.exp().acosh()
    } else {
        half + std::f64::consts::LN_2
    };
    if !value.is_finite() || value <= 0.0 {
        return Err("attenuation ratio is too large for stable order selection".to_string());
    }
    Ok(value)
}

pub(crate) fn checked_order(order: f64) -> Result<usize, String> {
    if !order.is_finite() || order <= 0.0 {
        return Err("computed filter order is not finite and positive".to_string());
    }
    let order = order.ceil().max(1.0);
    if order > MAX_ORDER as f64 {
        return Err(format!(
            "computed filter order exceeds RunMat's resource limit of {MAX_ORDER}"
        ));
    }
    Ok(order as usize)
}

pub(crate) fn validate_critical_frequencies(edges: &[f64], analog: bool) -> Result<(), String> {
    if edges.is_empty() || edges.len() > 2 {
        return Err("computed critical frequency has invalid shape".to_string());
    }
    for edge in edges {
        if !edge.is_finite() || *edge <= 0.0 {
            return Err("computed critical frequency must be positive and finite".to_string());
        }
        if !analog && *edge >= 1.0 {
            return Err("computed digital critical frequency must be between 0 and 1".to_string());
        }
    }
    if edges.len() == 2 && edges[0] >= edges[1] {
        return Err("computed critical frequency vector must be strictly increasing".to_string());
    }
    Ok(())
}

fn ln_expm1(x: f64) -> Result<f64, String> {
    if !x.is_finite() || x <= 0.0 {
        return Err("attenuation must be positive and finite".to_string());
    }
    if x < 36.0 {
        let value = x.exp_m1().ln();
        if value.is_finite() {
            return Ok(value);
        }
    }
    Ok(x)
}

fn bandstop_natural_values(passband: &[f64], stopband: &[f64]) -> Vec<f64> {
    stopband
        .iter()
        .map(|&edge| edge * (passband[0] - passband[1]) / (edge * edge - passband[0] * passband[1]))
        .collect()
}

fn optimize_bandstop_passband(
    passband: &[f64],
    stopband: &[f64],
    rp: f64,
    rs: f64,
    family: OrderFamily,
) -> Result<Vec<f64>, String> {
    let lower_right = stopband[0] - EPS;
    let upper_left = stopband[1] + EPS;
    if passband[0] >= lower_right || upper_left >= passband[1] {
        return Err("bandstop passband and stopband edges are not separated".to_string());
    }
    let left = fminbound(passband[0], lower_right, |candidate| {
        bandstop_order_objective(candidate, 0, passband, stopband, rp, rs, family)
    });
    let right = fminbound(upper_left, passband[1], |candidate| {
        bandstop_order_objective(candidate, 1, passband, stopband, rp, rs, family)
    });
    Ok(vec![left, right])
}

fn bandstop_order_objective(
    candidate: f64,
    index: usize,
    passband: &[f64],
    stopband: &[f64],
    rp: f64,
    rs: f64,
    family: OrderFamily,
) -> f64 {
    let mut adjusted = [passband[0], passband[1]];
    adjusted[index] = candidate;
    let nat = bandstop_natural_values(&adjusted, stopband)
        .into_iter()
        .map(f64::abs)
        .fold(f64::INFINITY, f64::min);
    if !nat.is_finite() || nat <= 1.0 {
        return f64::INFINITY;
    }
    let Ok(log_ratio) = db_excess_log_ratio(rp, rs) else {
        return f64::INFINITY;
    };
    match family {
        OrderFamily::Butterworth => log_ratio / (2.0 * nat.ln()),
        OrderFamily::Chebyshev => chebyshev_v_pass_stop(log_ratio)
            .map(|value| value / nat.acosh())
            .unwrap_or(f64::INFINITY),
    }
}

fn fminbound<F>(mut lower: f64, mut upper: f64, mut objective: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    let mut x1 = lower + GOLDEN_RATIO_CONJUGATE * (upper - lower);
    let mut x2 = upper - GOLDEN_RATIO_CONJUGATE * (upper - lower);
    let mut f1 = objective(x1);
    let mut f2 = objective(x2);
    for _ in 0..128 {
        if (upper - lower).abs() <= 1.0e-13 * (1.0 + lower.abs() + upper.abs()) {
            break;
        }
        if f1 < f2 {
            upper = x2;
            x2 = x1;
            f2 = f1;
            x1 = lower + GOLDEN_RATIO_CONJUGATE * (upper - lower);
            f1 = objective(x1);
        } else {
            lower = x1;
            x1 = x2;
            f1 = f2;
            x2 = upper - GOLDEN_RATIO_CONJUGATE * (upper - lower);
            f2 = objective(x2);
        }
    }
    (lower + upper) / 2.0
}
