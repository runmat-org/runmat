//! pn-junction helpers shared by diodes and bipolar transistors: Newton voltage limiting,
//! depletion charge, saturation-current temperature scaling and model-parameter lookup.

use super::K_OVER_Q;
use crate::netlist::FlatModel;
use crate::num::Dual;

/// Nominal temperature (27 °C) in kelvin.
pub const TNOM_K: f64 = 300.15;

/// Model parameters with defaults; unknown names are reported by [`Params::check`].
pub struct Params<'a> {
    pub model: Option<&'a FlatModel>,
}

impl Params<'_> {
    pub fn get(&self, key: &str, default: f64) -> f64 {
        self.model
            .and_then(|m| m.params.get(key).copied())
            .unwrap_or(default)
    }
    pub fn has(&self, key: &str) -> bool {
        self.model.is_some_and(|m| m.params.contains_key(key))
    }
    /// Names in the model that are not in `known` (and not in `ignored`, which are accepted
    /// for compatibility and have no effect).
    pub fn unknown(&self, known: &[&str], ignored: &[&str]) -> Vec<String> {
        self.model
            .map(|m| {
                m.params
                    .keys()
                    .filter(|k| !known.contains(&k.as_str()) && !ignored.contains(&k.as_str()))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Critical voltage `n·Vt·ln(n·Vt / (√2·Is))`, where the junction's current curvature peaks.
pub fn vcrit(nvt: f64, is: f64) -> f64 {
    nvt * (nvt / (std::f64::consts::SQRT_2 * is)).ln()
}

/// Limits a junction voltage step (Nagel's logarithmic damping). Returns the voltage to use and
/// whether it was limited.
pub fn pnjlim(vnew: f64, vold: f64, nvt: f64, vcrit: f64) -> (f64, bool) {
    if vnew > vcrit && (vnew - vold).abs() > 2.0 * nvt {
        if vold > 0.0 {
            let arg = 1.0 + (vnew - vold) / nvt;
            if arg > 0.0 {
                (vold + nvt * arg.ln(), true)
            } else {
                (vcrit, true)
            }
        } else {
            (nvt * (vnew / nvt).ln(), true)
        }
    } else {
        (vnew, false)
    }
}

/// `exp(x)`, continued linearly above `x = 80` so no iterate overflows.
pub fn explim<const N: usize>(x: Dual<N>) -> Dual<N> {
    const XMAX: f64 = 80.0;
    if x.v <= XMAX {
        x.exp()
    } else {
        let e = XMAX.exp();
        (x - XMAX + 1.0) * e
    }
}

/// Saturation current at temperature `t` (K).
pub fn is_at(is: f64, n: f64, eg: f64, xti: f64, t: f64, tnom: f64) -> f64 {
    let r = t / tnom;
    is * r.powf(xti / n) * ((r - 1.0) * eg / (n * K_OVER_Q * t)).exp()
}

/// Depletion charge of a junction with zero-bias capacitance `cj0`, built-in potential `vj`,
/// grading `m` and forward-bias coefficient `fc` (linear capacitance above `fc·vj`).
pub fn depletion<const N: usize>(v: Dual<N>, cj0: f64, vj: f64, m: f64, fc: f64) -> Dual<N> {
    if cj0 == 0.0 {
        return Dual::constant(0.0);
    }
    let fcv = fc * vj;
    if v.v < fcv {
        let x = 1.0 - v / vj;
        if (m - 1.0).abs() < 1e-9 {
            -(x.ln() * (cj0 * vj))
        } else {
            (1.0 - x.powf(1.0 - m)) * (cj0 * vj / (1.0 - m))
        }
    } else {
        let f1 = if (m - 1.0).abs() < 1e-9 {
            -vj * (1.0 - fc).ln()
        } else {
            vj / (1.0 - m) * (1.0 - (1.0 - fc).powf(1.0 - m))
        };
        let f2 = (1.0 - fc).powf(1.0 + m);
        let f3 = 1.0 - fc * (1.0 + m);
        let dv = v - fcv;
        let sq = v * v - fcv * fcv;
        (dv * f3 + sq * (m / (2.0 * vj))) * (cj0 / f2) + cj0 * f1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn depletion_charge_is_smooth_at_fc() {
        let (cj0, vj, m, fc) = (1e-12, 0.7, 0.5, 0.5);
        let at = |v: f64| depletion(Dual::<1>::var(v, 0), cj0, vj, m, fc);
        let lo = at(fc * vj - 1e-9);
        let hi = at(fc * vj + 1e-9);
        assert!((lo.v - hi.v).abs() < 1e-20);
        assert!((lo.d[0] - hi.d[0]).abs() / lo.d[0] < 1e-6);
        // C(0) = cj0.
        assert!((at(0.0).d[0] - cj0).abs() < 1e-24);
        // C(v) = cj0 / (1 - v/vj)^m below fc·vj.
        let c = at(-1.0).d[0];
        assert!((c - cj0 / (1.0f64 + 1.0 / 0.7).powf(0.5)).abs() < 1e-24);
    }
    #[test]
    fn limiting_damps_large_forward_steps() {
        let nvt = 0.02585;
        let vc = vcrit(nvt, 1e-14);
        let (v, lim) = pnjlim(5.0, 0.6, nvt, vc);
        assert!(lim && v < 0.75 && v > 0.6);
        assert_eq!(pnjlim(0.3, 0.29, nvt, vc), (0.3, false));
    }
}
