//! Independent-source time functions: `PULSE`, `SIN`, `EXP` and `PWL`, as SPICE defines them.

use crate::netlist::{parse_number, Tok};

#[derive(Clone, Debug, PartialEq)]
pub enum Wave {
    Pulse { v1: f64, v2: f64, td: f64, tr: f64, tf: f64, pw: f64, per: f64 },
    Sin { vo: f64, va: f64, freq: f64, td: f64, theta: f64, phase: f64 },
    Exp { v1: f64, v2: f64, td1: f64, tau1: f64, td2: f64, tau2: f64 },
    Pwl(Vec<(f64, f64)>),
}

/// A source's specification: DC value, AC phasor and optional time function.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceSpec {
    pub dc: Option<f64>,
    pub ac_mag: f64,
    pub ac_phase_deg: f64,
    pub wave: Option<Wave>,
}

impl SourceSpec {
    pub fn parse(args: &[Tok]) -> Result<SourceSpec, String> {
        let mut s = SourceSpec::default();
        let words: Vec<&str> = args.iter().filter_map(|t| t.word()).collect();
        let mut i = 0;
        let num = |i: usize| words.get(i).and_then(|w| parse_number(w));
        let nums_from = |i: usize| -> Vec<f64> {
            words[i..].iter().map_while(|w| parse_number(w)).collect()
        };
        while i < words.len() {
            match words[i] {
                "dc" => {
                    s.dc = Some(num(i + 1).ok_or("DC needs a value")?);
                    i += 2;
                }
                "ac" => {
                    s.ac_mag = num(i + 1).unwrap_or(1.0);
                    i += if num(i + 1).is_some() { 2 } else { 1 };
                    if let Some(p) = num(i) {
                        s.ac_phase_deg = p;
                        i += 1;
                    }
                }
                w @ ("pulse" | "sin" | "exp" | "pwl") => {
                    let v = nums_from(i + 1);
                    i += 1 + v.len();
                    let g = |k: usize, d: f64| v.get(k).copied().unwrap_or(d);
                    s.wave = Some(match w {
                        "pulse" => {
                            if v.len() < 2 {
                                return Err("PULSE needs at least V1 and V2".into());
                            }
                            Wave::Pulse { v1: v[0], v2: v[1], td: g(2, 0.0), tr: g(3, 0.0), tf: g(4, 0.0), pw: g(5, f64::INFINITY), per: g(6, f64::INFINITY) }
                        }
                        "sin" => {
                            if v.len() < 3 {
                                return Err("SIN needs VO, VA and FREQ".into());
                            }
                            Wave::Sin { vo: v[0], va: v[1], freq: v[2], td: g(3, 0.0), theta: g(4, 0.0), phase: g(5, 0.0) }
                        }
                        "exp" => {
                            if v.len() < 2 {
                                return Err("EXP needs V1 and V2".into());
                            }
                            Wave::Exp { v1: v[0], v2: v[1], td1: g(2, 0.0), tau1: g(3, 1e-9), td2: g(4, g(2, 0.0) + 1e-9), tau2: g(5, 1e-9) }
                        }
                        _ => {
                            if v.len() < 2 || v.len() % 2 != 0 {
                                return Err("PWL needs time-value pairs".into());
                            }
                            Wave::Pwl(v.chunks(2).map(|p| (p[0], p[1])).collect())
                        }
                    });
                }
                w => match parse_number(w) {
                    Some(v) if s.dc.is_none() => {
                        s.dc = Some(v);
                        i += 1;
                    }
                    _ => return Err(format!("unexpected `{w}` in a source")),
                },
            }
        }
        Ok(s)
    }

    /// The value for operating points: the DC value, else the time function at t = 0.
    pub fn dc_value(&self) -> f64 {
        self.dc.or_else(|| self.wave.as_ref().map(|w| w.at(0.0))).unwrap_or(0.0)
    }

    pub fn at(&self, t: f64) -> f64 {
        match &self.wave {
            Some(w) => w.at(t),
            None => self.dc.unwrap_or(0.0),
        }
    }
}

impl Wave {
    pub fn at(&self, t: f64) -> f64 {
        match *self {
            Wave::Pulse { v1, v2, td, tr, tf, pw, per } => {
                if t < td {
                    return v1;
                }
                let mut tt = t - td;
                if per.is_finite() && per > 0.0 {
                    tt %= per;
                }
                if tt < tr {
                    v1 + (v2 - v1) * tt / tr
                } else if tt < tr + pw {
                    v2
                } else if tt < tr + pw + tf {
                    v2 + (v1 - v2) * (tt - tr - pw) / tf
                } else {
                    v1
                }
            }
            Wave::Sin { vo, va, freq, td, theta, phase } => {
                let ph = phase.to_radians();
                if t < td {
                    vo + va * ph.sin()
                } else {
                    let tt = t - td;
                    vo + va * (-tt * theta).exp() * (2.0 * std::f64::consts::PI * freq * tt + ph).sin()
                }
            }
            Wave::Exp { v1, v2, td1, tau1, td2, tau2 } => {
                let mut v = v1;
                if t > td1 {
                    v += (v2 - v1) * (1.0 - (-(t - td1) / tau1).exp());
                }
                if t > td2 {
                    v += (v1 - v2) * (1.0 - (-(t - td2) / tau2).exp());
                }
                v
            }
            Wave::Pwl(ref p) => {
                if t <= p[0].0 {
                    return p[0].1;
                }
                for w in p.windows(2) {
                    if t <= w[1].0 {
                        let (t0, v0) = w[0];
                        let (t1, v1) = w[1];
                        return if t1 > t0 { v0 + (v1 - v0) * (t - t0) / (t1 - t0) } else { v1 };
                    }
                }
                p[p.len() - 1].1
            }
        }
    }

    /// Corners of the function up to `tstop`.
    pub fn breakpoints(&self, tstop: f64, out: &mut Vec<f64>) {
        match *self {
            Wave::Pulse { td, tr, tf, pw, per, .. } => {
                let mut start = td;
                loop {
                    for c in [0.0, tr, tr + pw, tr + pw + tf] {
                        let t = start + c;
                        if t.is_finite() && t <= tstop {
                            out.push(t);
                        }
                    }
                    if !(per.is_finite() && per > 0.0) || start + per > tstop {
                        break;
                    }
                    start += per;
                }
            }
            Wave::Sin { td, .. } => out.push(td),
            Wave::Exp { td1, td2, .. } => out.extend([td1, td2]),
            Wave::Pwl(ref p) => out.extend(p.iter().map(|x| x.0).filter(|t| *t <= tstop)),
        }
    }
}
