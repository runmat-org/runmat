//! Small-signal noise (SPICE `.noise`): at each frequency, the output noise density at a node
//! pair and the same noise referred to an input voltage source.
//!
//! The circuit is linearized at its operating point as for AC. Each device reports its noise
//! generators there ([`crate::devices::Device::noise`]): thermal noise `4kT/R` of resistors and
//! of the series resistances of diodes, BJTs and MOSFETs, shot noise `2q|I|` of diode and BJT
//! junction currents, and MOSFET channel noise `(8/3)·kT·gm`. Flicker noise (`kf`, `af`) is not
//! modelled. Each generator is a current between two nodes; its transfer to the output comes
//! from one solve of the factored AC matrix, and the output density is
//! `Sout(f) = Σ |H_k(f)|²·S_k`. The input-referred density divides by the gain from the input
//! source, a unit AC voltage on its branch.
//!
//! The totals integrate the densities over the swept band (trapezoids on the frequency points)
//! into rms values.

use super::engine::{Engine, LoadParams};
use super::op;
use crate::devices::States;
use crate::num::{dual, Complex};
use crate::sparse::lu::Lu;
use crate::sparse::{order, Matrix, GND};
use crate::SimError;
use std::collections::BTreeMap;

/// Boltzmann's constant (J/K) and the electron charge (C).
pub const K_B: f64 = 1.380_649e-23;
pub const Q_E: f64 = 1.602_176_634e-19;

/// One noise generator: a current between `a` and `b` with density `psd` (A²/Hz).
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseSource {
    pub device: String,
    /// `thermal`, `shot` or `channel`.
    pub kind: &'static str,
    pub a: usize,
    pub b: usize,
    pub psd: f64,
}

/// Collects a device's generators.
pub struct NoiseSink<'a> {
    pub x: &'a [f64],
    pub states: &'a States,
    /// `k·T` at the circuit temperature (J).
    pub kt: f64,
    pub out: Vec<NoiseSource>,
    pub device: String,
}

impl NoiseSink<'_> {
    pub fn thermal(&mut self, a: usize, b: usize, g: f64) {
        if g > 0.0 && a != b {
            self.push("thermal", a, b, 4.0 * self.kt * g);
        }
    }
    pub fn shot(&mut self, a: usize, b: usize, current: f64) {
        if current != 0.0 && a != b {
            self.push("shot", a, b, 2.0 * Q_E * current.abs());
        }
    }
    pub fn push(&mut self, kind: &'static str, a: usize, b: usize, psd: f64) {
        self.out.push(NoiseSource { device: self.device.clone(), kind, a, b, psd });
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NoiseSpec {
    /// Output node and reference node (`0` for ground).
    pub output: (String, String),
    /// The input voltage source the noise is referred to.
    pub input: String,
    pub freqs: Vec<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NoiseWave {
    pub freq: Vec<f64>,
    /// Output noise density (V/√Hz).
    pub onoise: Vec<f64>,
    /// Input-referred density (V/√Hz).
    pub inoise: Vec<f64>,
    /// |gain| from the input source to the output.
    pub gain: Vec<f64>,
    /// Each device's share of the output density (V²/Hz), by device.
    pub contributions: BTreeMap<String, Vec<f64>>,
}

impl NoiseWave {
    /// The output noise integrated over the solved band (V rms).
    pub fn total_onoise(&self) -> f64 {
        integrate(&self.freq, &self.onoise)
    }
    /// The input-referred noise integrated over the solved band (V rms).
    pub fn total_inoise(&self) -> f64 {
        integrate(&self.freq, &self.inoise)
    }
}

/// `√∫ d²(f) df` by trapezoids.
fn integrate(f: &[f64], density: &[f64]) -> f64 {
    let mut s = 0.0;
    for k in 1..f.len().min(density.len()) {
        s += 0.5 * (density[k] * density[k] + density[k - 1] * density[k - 1]) * (f[k] - f[k - 1]);
    }
    s.sqrt()
}

pub fn run(e: &mut Engine, spec: &NoiseSpec) -> Result<NoiseWave, SimError> {
    let c = e.c;
    let node = |n: &str| -> Result<usize, SimError> {
        if n == "0" || n.eq_ignore_ascii_case("gnd") {
            return Ok(GND);
        }
        c.node(&n.to_ascii_lowercase()).ok_or_else(|| SimError::BadRequest(format!("noise: no node `{n}`")))
    };
    let (op_node, on_node) = (node(&spec.output.0)?, node(&spec.output.1)?);
    let input = c
        .device(&spec.input)
        .and_then(|d| d.branch())
        .ok_or_else(|| SimError::BadRequest(format!("noise: the input `{}` must be a voltage source", spec.input)))?;
    let mut x = vec![0.0; c.size()];
    op::solve(e, &mut x, LoadParams::dc())?;
    e.assemble(&x, &LoadParams::dc())?;
    let kt = K_B * (c.options.temp_c + 273.15);
    let mut sources = vec![];
    for d in &c.devices {
        let mut sink = NoiseSink { x: &x, states: &e.states, kt, out: vec![], device: d.name().to_string() };
        dual::uncounted(|| d.noise(&mut sink));
        sources.extend(sink.out);
    }
    let mut wave = NoiseWave::default();
    for s in &sources {
        wave.contributions.entry(s.device.clone()).or_default();
    }
    let mut mat: Matrix<Complex> = Matrix::new(e.mat.pattern.clone());
    let mut ord = order::minimum_degree(&mat.pattern);
    let mut lu: Option<Lu<Complex>> = None;
    let mut mem = e.mem.clone();
    let out_of = |b: &[Complex]| {
        let v = |n: usize| if n == GND { Complex::default() } else { b[n] };
        v(op_node) - v(on_node)
    };
    'freqs: for &f in &spec.freqs {
        match super::ac::point(e, &mut mat, &mut ord, &mut lu, &mut mem, &x, f) {
            Ok(_) => {}
            Err(SimError::Stopped) => break,
            Err(err) => return Err(err),
        }
        let l = lu.as_ref().expect("factored");
        let solve = |e: &mut Engine, b: &mut Vec<Complex>| -> bool {
            if !e.room(l.solve_ops()) {
                return false;
            }
            e.meter.solve += l.solve(b);
            true
        };
        let mut b = vec![Complex::default(); c.size()];
        b[input] = Complex::new(1.0, 0.0);
        if !solve(e, &mut b) {
            let _ = e.stop();
            break;
        }
        let gain = out_of(&b).abs();
        let mut total = 0.0;
        let mut shares: BTreeMap<&str, f64> = BTreeMap::new();
        for s in &sources {
            let mut b = vec![Complex::default(); c.size()];
            if s.a != GND {
                b[s.a] = Complex::new(1.0, 0.0);
            }
            if s.b != GND {
                b[s.b] = Complex::new(-1.0, 0.0);
            }
            if !solve(e, &mut b) {
                let _ = e.stop();
                break 'freqs;
            }
            let h = out_of(&b).abs();
            let p = h * h * s.psd;
            total += p;
            *shares.entry(&s.device).or_default() += p;
        }
        e.meter.other += 3 * sources.len() as u64 + 2;
        wave.freq.push(f);
        wave.onoise.push(total.sqrt());
        wave.inoise.push(if gain > 0.0 { total.sqrt() / gain } else { f64::INFINITY });
        wave.gain.push(gain);
        for (k, v) in wave.contributions.iter_mut() {
            v.push(shares.get(k.as_str()).copied().unwrap_or(0.0));
        }
    }
    Ok(wave)
}
