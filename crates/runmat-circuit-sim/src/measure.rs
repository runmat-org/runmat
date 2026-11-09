//! Measurements on results, in the spirit of SPICE's `.meas`.
//!
//! Signals are written `v(out)`, `i(r1)`, and for AC `vdb(out)` (dB), `vp(out)` (phase, degrees,
//! unwrapped), `vm(out)` (magnitude), `vr(out)`, `vi(out)`. Crossings interpolate linearly
//! between samples (in log frequency for AC).

use crate::results::{interp, SimResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    Rise,
    Fall,
    Either,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Measure {
    /// Value at an x (time, swept value or frequency).
    At {
        signal: String,
        x: f64,
    },
    Max {
        signal: String,
        from: Option<f64>,
        to: Option<f64>,
    },
    Min {
        signal: String,
        from: Option<f64>,
        to: Option<f64>,
    },
    PeakToPeak {
        signal: String,
        from: Option<f64>,
        to: Option<f64>,
    },
    Average {
        signal: String,
        from: Option<f64>,
        to: Option<f64>,
    },
    Rms {
        signal: String,
        from: Option<f64>,
        to: Option<f64>,
    },
    /// The x of the `nth` (1-based) crossing of `value`.
    When {
        signal: String,
        value: f64,
        edge: Edge,
        nth: usize,
    },
    /// `targ` crossing x minus `trig` crossing x.
    Delay {
        trig: (String, f64, Edge, usize),
        targ: (String, f64, Edge, usize),
    },
    /// 10 %–90 % (or `low`–`high` fractions) rise time between the first and last values in the
    /// `from`–`to` window (the whole run when unset), so a pulse train is measured on one edge.
    RiseTime {
        signal: String,
        low: f64,
        high: f64,
        from: Option<f64>,
        to: Option<f64>,
    },
    /// Time after which the signal stays within `band` (a fraction of the step) of its final value,
    /// from the window's start.
    Settling {
        signal: String,
        band: f64,
        from: Option<f64>,
        to: Option<f64>,
    },
    /// Peak excursion beyond the final value, in percent of the step, in the window.
    Overshoot {
        signal: String,
        from: Option<f64>,
        to: Option<f64>,
    },
    Final {
        signal: String,
    },
    /// The frequency where the magnitude falls `drop_db` below its first (lowest-frequency) value.
    Bandwidth {
        signal: String,
        drop_db: f64,
    },
    /// Frequency of the magnitude peak.
    PeakFrequency {
        signal: String,
    },
    /// Peak frequency over the bandwidth between the points 3 dB below the peak.
    Q {
        signal: String,
    },
    /// The frequency where the magnitude falls through 1 (0 dB).
    UnityGain {
        signal: String,
    },
    /// 180° plus the phase at unity gain.
    PhaseMargin {
        signal: String,
    },
}

/// A signal and how to read a complex sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    Plain,
    Db,
    Phase,
    Mag,
    Re,
    Im,
}

fn split(signal: &str) -> (String, Part) {
    let s = signal.trim().to_ascii_lowercase();
    for (p, part) in [
        ("vdb(", Part::Db),
        ("vp(", Part::Phase),
        ("vm(", Part::Mag),
        ("vr(", Part::Re),
        ("vi(", Part::Im),
        ("idb(", Part::Db),
        ("ip(", Part::Phase),
        ("im(", Part::Mag),
    ] {
        if let Some(rest) = s.strip_prefix(p) {
            return (format!("{}({rest}", &p[..1]), part);
        }
    }
    (s, Part::Plain)
}

/// The sweep and a signal's samples.
fn series(r: &SimResult, signal: &str) -> Result<(Vec<f64>, Vec<f64>, bool), String> {
    let (name, part) = split(signal);
    match r {
        SimResult::Op(m) => m
            .get(&name)
            .map(|v| (vec![0.0], vec![*v], false))
            .ok_or_else(|| format!("no signal `{name}`")),
        SimResult::Dc(w) | SimResult::Tran(w) => {
            let y = w.get(&name).ok_or_else(|| format!("no signal `{name}`"))?;
            Ok((w.x.clone(), y.to_vec(), false))
        }
        SimResult::Ac(w) => {
            let y = match part {
                Part::Db | Part::Plain => w.db(&name),
                Part::Phase => w.phase_deg(&name),
                Part::Mag => w.get(&name).map(|v| v.iter().map(|c| c.abs()).collect()),
                Part::Re => w.get(&name).map(|v| v.iter().map(|c| c.re).collect()),
                Part::Im => w.get(&name).map(|v| v.iter().map(|c| c.im).collect()),
            }
            .ok_or_else(|| format!("no signal `{name}`"))?;
            Ok((w.freq.clone(), y, true))
        }
    }
}

fn window(x: &[f64], y: &[f64], from: Option<f64>, to: Option<f64>) -> (Vec<f64>, Vec<f64>) {
    let lo = from.unwrap_or(f64::NEG_INFINITY);
    let hi = to.unwrap_or(f64::INFINITY);
    x.iter()
        .zip(y)
        .filter(|(a, _)| **a >= lo && **a <= hi)
        .map(|(a, b)| (*a, *b))
        .unzip()
}

/// The samples a step measure reads: the `from`–`to` window, with at least two points.
fn step_window(
    x: &[f64],
    y: &[f64],
    from: Option<f64>,
    to: Option<f64>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (xs, ys) = window(x, y, from, to);
    if xs.len() < 2 {
        return Err("the window holds no step (fewer than two points)".into());
    }
    Ok((xs, ys))
}

fn crossing(x: &[f64], y: &[f64], value: f64, edge: Edge, nth: usize, log_x: bool) -> Option<f64> {
    let mut count = 0;
    for k in 1..y.len() {
        let (a, b) = (y[k - 1] - value, y[k] - value);
        let rise = a < 0.0 && b >= 0.0;
        let fall = a > 0.0 && b <= 0.0;
        let hit = match edge {
            Edge::Rise => rise,
            Edge::Fall => fall,
            Edge::Either => rise || fall,
        };
        if hit {
            count += 1;
            if count == nth.max(1) {
                let f = a / (a - b);
                return Some(if log_x {
                    (x[k - 1].ln() + f * (x[k].ln() - x[k - 1].ln())).exp()
                } else {
                    x[k - 1] + f * (x[k] - x[k - 1])
                });
            }
        }
    }
    None
}

fn integral(x: &[f64], y: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(xs, ys)| 0.5 * (ys[0] + ys[1]) * (xs[1] - xs[0]))
        .sum()
}

pub fn evaluate(m: &Measure, r: &SimResult) -> Result<f64, String> {
    let none = |what: &str| format!("{what} not found in the result");
    match m {
        Measure::At { signal, x } => {
            let (xs, ys, _) = series(r, signal)?;
            interp(&xs, &ys, *x).ok_or_else(|| none("value"))
        }
        Measure::Max { signal, from, to }
        | Measure::Min { signal, from, to }
        | Measure::PeakToPeak { signal, from, to } => {
            let (xs, ys, _) = series(r, signal)?;
            let (_, w) = window(&xs, &ys, *from, *to);
            let max = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = w.iter().copied().fold(f64::INFINITY, f64::min);
            if w.is_empty() {
                return Err(none("window"));
            }
            Ok(match m {
                Measure::Max { .. } => max,
                Measure::Min { .. } => min,
                _ => max - min,
            })
        }
        Measure::Average { signal, from, to } | Measure::Rms { signal, from, to } => {
            let (xs, ys, _) = series(r, signal)?;
            let (wx, wy) = window(&xs, &ys, *from, *to);
            if wx.len() < 2 {
                return Err(none("window"));
            }
            let span = wx[wx.len() - 1] - wx[0];
            Ok(if matches!(m, Measure::Average { .. }) {
                integral(&wx, &wy) / span
            } else {
                let sq: Vec<f64> = wy.iter().map(|v| v * v).collect();
                (integral(&wx, &sq) / span).sqrt()
            })
        }
        Measure::When {
            signal,
            value,
            edge,
            nth,
        } => {
            let (xs, ys, log) = series(r, signal)?;
            crossing(&xs, &ys, *value, *edge, *nth, log).ok_or_else(|| none("crossing"))
        }
        Measure::Delay { trig, targ } => {
            let at = |t: &(String, f64, Edge, usize)| -> Result<f64, String> {
                let (xs, ys, log) = series(r, &t.0)?;
                crossing(&xs, &ys, t.1, t.2, t.3, log).ok_or_else(|| none("crossing"))
            };
            Ok(at(targ)? - at(trig)?)
        }
        Measure::RiseTime {
            signal,
            low,
            high,
            from,
            to,
        } => {
            let (xs, ys, _) = series(r, signal)?;
            let (xs, ys) = step_window(&xs, &ys, *from, *to)?;
            let (v0, v1) = (ys[0], ys[ys.len() - 1]);
            let lv = v0 + (v1 - v0) * low;
            let hv = v0 + (v1 - v0) * high;
            let e = if v1 >= v0 { Edge::Rise } else { Edge::Fall };
            let t1 = crossing(&xs, &ys, lv, e, 1, false).ok_or_else(|| none("low crossing"))?;
            let t2 = crossing(&xs, &ys, hv, e, 1, false).ok_or_else(|| none("high crossing"))?;
            Ok(t2 - t1)
        }
        Measure::Settling {
            signal,
            band,
            from,
            to,
        } => {
            let (xs, ys, _) = series(r, signal)?;
            let (xs, ys) = step_window(&xs, &ys, *from, *to)?;
            let (v0, v1) = (ys[0], ys[ys.len() - 1]);
            let tol = (v1 - v0).abs() * band;
            let last_out = ys.iter().rposition(|v| (v - v1).abs() > tol);
            Ok(match last_out {
                Some(k) if k + 1 < xs.len() => xs[k + 1] - xs[0],
                Some(_) => return Err("the signal does not settle in the window".into()),
                None => 0.0,
            })
        }
        Measure::Overshoot { signal, from, to } => {
            let (xs, ys, _) = series(r, signal)?;
            let (_, ys) = step_window(&xs, &ys, *from, *to)?;
            let (v0, v1) = (ys[0], ys[ys.len() - 1]);
            let step = v1 - v0;
            if step == 0.0 {
                return Err("no step".into());
            }
            let peak = ys
                .iter()
                .map(|v| (v - v0) / step)
                .fold(f64::NEG_INFINITY, f64::max);
            Ok(((peak - 1.0) * 100.0).max(0.0))
        }
        Measure::Final { signal } => {
            let (_, ys, _) = series(r, signal)?;
            ys.last().copied().ok_or_else(|| none("value"))
        }
        Measure::Bandwidth { signal, drop_db } => {
            let (xs, ys, _) = series(r, &db(signal))?;
            crossing(&xs, &ys, ys[0] - drop_db, Edge::Fall, 1, true)
                .ok_or_else(|| none("bandwidth edge"))
        }
        Measure::PeakFrequency { signal } => {
            let (xs, ys, _) = series(r, &db(signal))?;
            let k = argmax(&ys);
            Ok(xs[k])
        }
        Measure::Q { signal } => {
            let (xs, ys, _) = series(r, &db(signal))?;
            let k = argmax(&ys);
            let level = ys[k] - 3.0103;
            let (lx, ly) = (&xs[..=k], &ys[..=k]);
            let f_lo = crossing(lx, ly, level, Edge::Rise, count_rises(ly, level), true)
                .ok_or_else(|| none("lower −3 dB point"))?;
            let f_hi = crossing(&xs[k..], &ys[k..], level, Edge::Fall, 1, true)
                .ok_or_else(|| none("upper −3 dB point"))?;
            Ok(xs[k] / (f_hi - f_lo))
        }
        Measure::UnityGain { signal } => {
            let (xs, ys, _) = series(r, &db(signal))?;
            crossing(&xs, &ys, 0.0, Edge::Fall, 1, true).ok_or_else(|| none("unity-gain crossing"))
        }
        Measure::PhaseMargin { signal } => {
            let f = evaluate(
                &Measure::UnityGain {
                    signal: signal.clone(),
                },
                r,
            )?;
            let (xs, ph, _) = series(r, &phase(signal))?;
            let lx: Vec<f64> = xs.iter().map(|v| v.ln()).collect();
            Ok(180.0 + interp(&lx, &ph, f.ln()).ok_or_else(|| none("phase"))?)
        }
    }
}

fn argmax(ys: &[f64]) -> usize {
    ys.iter()
        .enumerate()
        .fold(0, |b, (k, v)| if *v > ys[b] { k } else { b })
}

fn count_rises(ys: &[f64], level: f64) -> usize {
    ys.windows(2)
        .filter(|w| w[0] < level && w[1] >= level)
        .count()
}

fn db(signal: &str) -> String {
    let (name, _) = split(signal);
    format!("{}db({}", &name[..1], &name[2..])
}

fn phase(signal: &str) -> String {
    let (name, _) = split(signal);
    format!("{}p({}", &name[..1], &name[2..])
}
