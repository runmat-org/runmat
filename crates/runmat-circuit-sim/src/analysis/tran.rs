//! Transient analysis with adaptive time steps.
//!
//! - Start: the operating point (with `.ic` nodes held), or with `uic` the `.ic` values and the
//!   devices' `ic=` charges directly.
//! - Each step: a linear predictor, then Newton at `t + h`. A failed Newton cuts the step by 8.
//! - Local truncation error: from divided differences of every charge and flux, for backward
//!   Euler `h²·|q[3]|`, trapezoidal `h³·|q[4]|/2`, Gear-2 `4h³·|q[4]|/3` (with `q[k]` the k-point
//!   divided difference), against `trtol·max(reltol·|i| + abstol, (reltol·|q| + chgtol)/h)`.
//!   A step whose error ratio exceeds 1 is rejected; the next step scales by
//!   `0.9·ratio^(−1/(order+1))`, at most doubling.
//! - Breakpoints (source corners) are hit exactly; the step after one uses backward Euler.

use super::engine::{Engine, LoadParams};
use super::{op, Probes};
use crate::devices::{Method, Mode};
use crate::results::RealWave;
use crate::SimError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranSpec {
    pub tstep: f64,
    pub tstop: f64,
    pub tstart: f64,
    pub tmax: Option<f64>,
    pub uic: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TranStats {
    pub accepted: u64,
    pub rejected_lte: u64,
    pub rejected_newton: u64,
}

pub fn run(e: &mut Engine, spec: &TranSpec) -> Result<(RealWave, TranStats), SimError> {
    let c = e.c;
    let o = c.options.clone();
    if !(spec.tstop > 0.0 && spec.tstep > 0.0 && spec.tstart >= 0.0 && spec.tstart < spec.tstop) {
        return Err(SimError::BadRequest(
            "transient needs tstep > 0, tstop > 0 and 0 ≤ tstart < tstop".into(),
        ));
    }
    let method = o.method;
    let mut x = vec![0.0; c.size()];
    if spec.uic {
        for &(k, v) in &c.ics {
            x[k] = v;
        }
    } else {
        let mut p = LoadParams::dc();
        p.force_ic = !c.ics.is_empty();
        op::solve(e, &mut x, p)?;
    }
    e.assemble(&x, &LoadParams::dc())?;
    if spec.uic {
        for d in &c.devices {
            d.apply_ic(&mut e.states);
        }
    }
    e.states.settle();
    let probes = Probes::new(c);
    let mut wave = RealWave::new("time".into(), &probes.names);
    if spec.tstart == 0.0 {
        wave.push(0.0, e.probe_values(&probes, &x)?);
    }
    let mut bps = vec![spec.tstop];
    for d in &c.devices {
        d.breakpoints(spec.tstop, &mut bps);
    }
    let eps = spec.tstop * 1e-12;
    bps.retain(|t| *t > eps && *t <= spec.tstop + eps);
    bps.sort_by(|a, b| a.total_cmp(b));
    bps.dedup_by(|a, b| (*a - *b).abs() <= eps);
    let hmax = spec
        .tmax
        .unwrap_or(spec.tstep.min((spec.tstop - spec.tstart) / 50.0))
        .min(spec.tstop / 50.0);
    let hmin = spec.tstop * 1e-12;
    let mut h = spec.tstep.min(hmax) / 100.0;
    let mut t = 0.0;
    let mut x_prev = x.clone();
    let mut h_prev = h;
    let mut times = [0.0f64; 4];
    let mut since_bp = 1usize;
    let mut order = 1usize;
    let mut stats = TranStats::default();
    let mut bp_i = 0usize;
    while t < spec.tstop - eps {
        while bp_i < bps.len() && bps[bp_i] <= t + eps {
            bp_i += 1;
        }
        let next_bp = bps.get(bp_i).copied().unwrap_or(spec.tstop);
        let mut hit = false;
        if t + h >= next_bp - 0.1 * h {
            h = next_bp - t;
            hit = true;
        }
        e.states.method = if order == 1 { Method::Euler } else { method };
        e.states.h = [h, h_prev];
        let mut xn = x.clone();
        if order > 1 {
            let r = h / h_prev;
            for (k, v) in xn.iter_mut().enumerate() {
                *v += (x[k] - x_prev[k]) * r;
            }
            // At the operation limit: the accepted points so far are the result.
            if e.charge_other(3 * x.len() as u64).is_err() {
                break;
            }
        }
        let saved_mem = e.mem.clone();
        let p = LoadParams {
            mode: Mode::Tran,
            time: t + h,
            ..LoadParams::dc()
        };
        let solved = match e.newton(&mut xn, &p, o.itl4) {
            // At the operation limit: the accepted points so far are the result.
            Err(SimError::Stopped) => break,
            r => r?,
        };
        if solved.is_none() {
            e.mem = saved_mem;
            stats.rejected_newton += 1;
            h /= 8.0;
            order = 1;
            if h < hmin {
                return Err(SimError::TimestepTooSmall { time: t });
            }
            continue;
        }
        // Truncation error.
        let pts = [t + h, times[0], times[1], times[2]];
        let need = if order == 1 { 3 } else { 4 };
        let mut ratio: Option<f64> = None;
        if since_bp + 1 >= need {
            if e.charge_other(30 * e.states.q.len() as u64).is_err() {
                break;
            }
            let mut worst = 0.0f64;
            for s in 0..e.states.q.len() {
                let q = e.states.q[s];
                let i = e.states.i[s];
                let dd1 = |a: usize| (q[a] - q[a + 1]) / (pts[a] - pts[a + 1]);
                let dd2 = |a: usize| (dd1(a) - dd1(a + 1)) / (pts[a] - pts[a + 2]);
                let lte = match (order, method) {
                    (1, _) => h * h * dd2(0).abs(),
                    (_, Method::Gear2) => {
                        4.0 / 3.0 * h.powi(3) * ((dd2(0) - dd2(1)) / (pts[0] - pts[3])).abs()
                    }
                    _ => 0.5 * h.powi(3) * ((dd2(0) - dd2(1)) / (pts[0] - pts[3])).abs(),
                };
                let abs = if e.states.is_flux[s] {
                    o.vntol
                } else {
                    o.abstol
                };
                let tol = o.trtol
                    * (o.reltol * i[0].abs().max(i[1].abs()) + abs)
                        .max((o.reltol * q[0].abs().max(q[1].abs()) + o.chgtol) / h);
                worst = worst.max(lte / h / tol);
            }
            ratio = Some(worst);
            if worst > 1.0 && h > hmin * 10.0 {
                e.mem = saved_mem;
                stats.rejected_lte += 1;
                h *= (0.9 * worst.powf(-1.0 / (order as f64 + 1.0))).max(0.25);
                continue;
            }
        }
        // Accept.
        times = [t + h, times[0], times[1], times[2]];
        t += h;
        x_prev = std::mem::replace(&mut x, xn);
        e.states.accept();
        e.accept(&x);
        stats.accepted += 1;
        since_bp += 1;
        if t >= spec.tstart - eps {
            match e.probe_values(&probes, &x) {
                Ok(v) => wave.push(t, v),
                // At the operation limit: the points before this one are the result.
                Err(_) => break,
            }
        }
        h_prev = h;
        let grow = match ratio {
            Some(r) if r > 0.0 => (0.9 * r.powf(-1.0 / (order as f64 + 1.0))).min(2.0),
            _ => 2.0,
        };
        let mut h_new = (h * grow).min(hmax);
        if hit {
            order = 1;
            since_bp = 1;
            let gap = bps.get(bp_i + 1).map(|b| b - t).unwrap_or(spec.tstop - t);
            h_new = h_new.min(h).min(0.1 * gap.max(hmin));
        } else {
            order = if method == Method::Euler { 1 } else { 2 };
        }
        h = h_new.max(hmin);
    }
    Ok((wave, stats))
}
