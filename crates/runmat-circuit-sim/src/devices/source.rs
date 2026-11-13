//! Independent voltage and current sources.

use super::waveform::SourceSpec;
use super::{Device, Load, Mode, States};
use crate::num::Complex;

/// The source value in the present analysis (DC sweeps write the DC value into `mem`).
fn value(spec: &SourceSpec, ld: &Load, mem: usize) -> f64 {
    match ld.mode {
        Mode::Tran if spec.wave.is_some() => spec.at(ld.time) * ld.src_scale,
        _ => ld.mem[mem] * ld.src_scale,
    }
}

fn phasor(spec: &SourceSpec) -> Complex {
    Complex::from_polar(spec.ac_mag, spec.ac_phase_deg.to_radians())
}

pub struct VSource {
    pub name: String,
    pub p: usize,
    pub n: usize,
    pub branch: usize,
    pub spec: SourceSpec,
    pub mem: usize,
}

impl Device for VSource {
    fn name(&self) -> &str {
        &self.name
    }
    fn load(&self, ld: &mut Load) {
        let (p, n, j) = (self.p, self.n, self.branch);
        ld.mat(p, j, 1.0);
        ld.mat(n, j, -1.0);
        ld.mat(j, p, 1.0);
        ld.mat(j, n, -1.0);
        if ld.is_ac() {
            ld.rhs_ac(j, phasor(&self.spec));
        } else {
            let v = value(&self.spec, ld, self.mem);
            ld.rhs(j, v);
        }
    }
    fn breakpoints(&self, tstop: f64, out: &mut Vec<f64>) {
        if let Some(w) = &self.spec.wave {
            w.breakpoints(tstop, out);
        }
    }
    fn branch(&self) -> Option<usize> {
        Some(self.branch)
    }
    fn set_dc(&self, mem: &mut [f64], v: f64) -> bool {
        mem[self.mem] = v;
        true
    }
}

pub struct ISource {
    pub name: String,
    pub p: usize,
    pub n: usize,
    pub spec: SourceSpec,
    pub mem: usize,
}

impl Device for ISource {
    fn name(&self) -> &str {
        &self.name
    }
    fn load(&self, ld: &mut Load) {
        // Positive current flows from p through the source to n.
        if ld.is_ac() {
            let i = phasor(&self.spec);
            ld.rhs_ac(self.p, -i);
            ld.rhs_ac(self.n, i);
        } else {
            let i = value(&self.spec, ld, self.mem);
            ld.rhs(self.p, -i);
            ld.rhs(self.n, i);
        }
    }
    fn breakpoints(&self, tstop: f64, out: &mut Vec<f64>) {
        if let Some(w) = &self.spec.wave {
            w.breakpoints(tstop, out);
        }
    }
    fn probe_current(&self, _x: &[f64], _s: &States) -> Option<f64> {
        self.spec.dc
    }
    fn set_dc(&self, mem: &mut [f64], v: f64) -> bool {
        mem[self.mem] = v;
        true
    }
}
