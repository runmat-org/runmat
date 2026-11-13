//! Analysis results: an operating point, real waveforms (DC sweep, transient) and complex
//! frequency responses (AC). Signals are named `v(<node>)` and `i(<device>)`, lowercase.

use crate::num::Complex;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RealWave {
    /// The swept quantity: `time`, or the swept source's name.
    pub x_name: String,
    pub x: Vec<f64>,
    pub names: Vec<String>,
    pub data: Vec<Vec<f64>>,
}

impl RealWave {
    pub fn new(x_name: String, names: &[String]) -> Self {
        RealWave {
            x_name,
            x: vec![],
            names: names.to_vec(),
            data: vec![vec![]; names.len()],
        }
    }
    pub fn push(&mut self, x: f64, values: Vec<f64>) {
        self.x.push(x);
        for (d, v) in self.data.iter_mut().zip(values) {
            d.push(v);
        }
    }
    pub fn get(&self, name: &str) -> Option<&[f64]> {
        let n = name.to_ascii_lowercase();
        self.names
            .iter()
            .position(|s| *s == n)
            .map(|k| self.data[k].as_slice())
    }
    /// Linear interpolation of a signal at `x`.
    pub fn at(&self, name: &str, x: f64) -> Option<f64> {
        interp(&self.x, self.get(name)?, x)
    }
}

pub fn interp(xs: &[f64], ys: &[f64], x: f64) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    if x <= xs[0] {
        return Some(ys[0]);
    }
    let k = xs.partition_point(|v| *v < x);
    if k >= xs.len() {
        return Some(ys[ys.len() - 1]);
    }
    let (x0, x1) = (xs[k - 1], xs[k]);
    Some(if x1 > x0 {
        ys[k - 1] + (ys[k] - ys[k - 1]) * (x - x0) / (x1 - x0)
    } else {
        ys[k]
    })
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AcWave {
    pub freq: Vec<f64>,
    pub names: Vec<String>,
    pub data: Vec<Vec<Complex>>,
}

impl AcWave {
    pub fn new(names: &[String]) -> Self {
        AcWave {
            freq: vec![],
            names: names.to_vec(),
            data: vec![vec![]; names.len()],
        }
    }
    pub fn push(&mut self, f: f64, values: Vec<Complex>) {
        self.freq.push(f);
        for (d, v) in self.data.iter_mut().zip(values) {
            d.push(v);
        }
    }
    pub fn get(&self, name: &str) -> Option<&[Complex]> {
        let n = name.to_ascii_lowercase();
        self.names
            .iter()
            .position(|s| *s == n)
            .map(|k| self.data[k].as_slice())
    }
    pub fn db(&self, name: &str) -> Option<Vec<f64>> {
        self.get(name)
            .map(|v| v.iter().map(|c| 20.0 * c.abs().log10()).collect())
    }
    /// Phase in degrees, unwrapped along the sweep.
    pub fn phase_deg(&self, name: &str) -> Option<Vec<f64>> {
        let v = self.get(name)?;
        let mut out = Vec::with_capacity(v.len());
        let mut prev: Option<f64> = None;
        let mut offset = 0.0;
        for c in v {
            let mut p = c.arg().to_degrees();
            if let Some(q) = prev {
                while p + offset - q > 180.0 {
                    offset -= 360.0;
                }
                while p + offset - q < -180.0 {
                    offset += 360.0;
                }
            }
            p += offset;
            prev = Some(p);
            out.push(p);
        }
        Some(out)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimResult {
    Op(BTreeMap<String, f64>),
    Dc(RealWave),
    Ac(AcWave),
    Tran(RealWave),
}

impl SimResult {
    pub fn op(&self) -> Option<&BTreeMap<String, f64>> {
        match self {
            SimResult::Op(m) => Some(m),
            _ => None,
        }
    }
    pub fn real(&self) -> Option<&RealWave> {
        match self {
            SimResult::Dc(w) | SimResult::Tran(w) => Some(w),
            _ => None,
        }
    }
    pub fn ac(&self) -> Option<&AcWave> {
        match self {
            SimResult::Ac(w) => Some(w),
            _ => None,
        }
    }
}
