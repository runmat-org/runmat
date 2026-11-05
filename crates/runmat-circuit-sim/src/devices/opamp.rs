//! Operational amplifiers as built-in devices: `X<name> in+ in− out [v+ v−] <model>` with
//! `.model <model> OPAMP(...)`.
//!
//! - `ideal=1`: a nullor. The output supplies whatever current holds `V(in+) + vos = V(in−)`.
//! - Otherwise a single-pole macromodel: a transconductance `gm = a0` drives an internal node
//!   loaded by 1 Ω and `C1 = a0 / (2π·gbw)`, so the open-loop gain is `a0` with its pole at
//!   `gbw / a0`. With `sr` set the transconductance saturates at `sr·C1` (slew limiting, through
//!   `tanh`). The output is a voltage source with series `rout`, following the internal node
//!   and clamped smoothly (10 mV knee) between `V(v−) + vdrop_lo` and `V(v+) − vdrop_hi` when
//!   supply pins are given, or `vmin`/`vmax` when set.
//!
//! Parameters: `ideal a0 gbw sr rin cin rout vos ib vdrop vdrop_hi vdrop_lo vmax vmin`.
//! The output current returns through ground; supply current is not modelled.

use super::junction::Params;
use super::{Device, Load, Setup};
use crate::num::Dual;
use crate::sparse::GND;

pub const KNOWN: &[&str] = &[
    "ideal", "a0", "aol", "gbw", "sr", "rin", "cin", "rout", "vos", "ib", "vdrop", "vdrop_hi", "vdrop_lo", "vmax", "vmin", "level",
];

const KNEE: f64 = 0.01;

pub struct OpAmp {
    pub name: String,
    pub inp: usize,
    pub inn: usize,
    pub out: usize,
    pub rails: Option<(usize, usize)>,
    pub ideal: bool,
    pub gm: f64,
    pub c1: f64,
    pub isr: f64,
    pub gin: f64,
    pub cin: f64,
    pub rout: f64,
    pub vos: f64,
    pub ib: f64,
    pub drop_hi: f64,
    pub drop_lo: f64,
    pub vmax: f64,
    pub vmin: f64,
    pub n1: usize,
    pub branch: usize,
    pub q1: usize,
    pub qin: usize,
}

impl OpAmp {
    pub fn new(name: &str, nodes: &[usize], p: &Params, s: &mut Setup) -> Result<OpAmp, String> {
        if nodes.len() != 3 && nodes.len() != 5 {
            return Err(format!("`{name}`: an op-amp connects in+ in− out [v+ v−] (3 or 5 nodes), not {}", nodes.len()));
        }
        let a0 = p.get("a0", p.get("aol", 2e5));
        let gbw = p.get("gbw", 1e6);
        let c1 = a0 / (2.0 * std::f64::consts::PI * gbw);
        let sr = p.get("sr", f64::INFINITY);
        let drop = p.get("vdrop", 0.0);
        let ideal = p.get("ideal", 0.0) != 0.0;
        let n1 = if ideal { GND } else { s.internal(name, "gain") };
        let rin = p.get("rin", 0.0);
        Ok(OpAmp {
            name: name.into(),
            inp: nodes[0],
            inn: nodes[1],
            out: nodes[2],
            rails: (nodes.len() == 5).then(|| (nodes[3], nodes[4])),
            ideal,
            gm: a0,
            c1,
            isr: sr * c1,
            gin: if rin > 0.0 && rin.is_finite() { 1.0 / rin } else { 0.0 },
            cin: p.get("cin", 0.0),
            rout: p.get("rout", 0.0),
            vos: p.get("vos", 0.0),
            ib: p.get("ib", 0.0),
            drop_hi: p.get("vdrop_hi", drop),
            drop_lo: p.get("vdrop_lo", drop),
            vmax: p.get("vmax", f64::INFINITY),
            vmin: p.get("vmin", f64::NEG_INFINITY),
            n1,
            branch: s.branch(name),
            q1: s.state(false),
            qin: s.state(false),
        })
    }
}

fn softplus<const N: usize>(z: Dual<N>) -> Dual<N> {
    if z.v > 30.0 {
        z
    } else if z.v < -30.0 {
        Dual::constant(0.0)
    } else {
        (z.exp() + 1.0).ln()
    }
}

impl Device for OpAmp {
    fn name(&self) -> &str {
        &self.name
    }
    fn load(&self, ld: &mut Load) {
        let j = self.branch;
        ld.mat(self.out, j, 1.0);
        if self.gin > 0.0 {
            ld.conductance(self.inp, self.inn, self.gin);
        }
        if self.cin > 0.0 {
            let q = Dual::<1>::var(ld.v(self.inp) - ld.v(self.inn), 0) * self.cin;
            ld.charge(self.qin, self.inp, self.inn, q, [(self.inp, self.inn)]);
        }
        if self.ib != 0.0 && !ld.is_ac() {
            ld.rhs(self.inp, -self.ib);
            ld.rhs(self.inn, -self.ib);
        }
        if self.ideal {
            ld.mat(j, self.inp, 1.0);
            ld.mat(j, self.inn, -1.0);
            ld.rhs(j, -self.vos);
            return;
        }
        // Gain stage: current from ground into n1.
        let vd = ld.v(self.inp) - ld.v(self.inn) + self.vos;
        let x = Dual::<1>::var(vd, 0);
        let i = if self.isr.is_finite() { (x * (self.gm / self.isr)).tanh() * self.isr } else { x * self.gm };
        ld.current_at(GND, self.n1, i, [(self.inp, self.inn)], [vd - self.vos]);
        ld.conductance(self.n1, GND, 1.0);
        let q = Dual::<1>::var(ld.v(self.n1), 0) * self.c1;
        ld.charge(self.q1, self.n1, GND, q, [(self.n1, GND)]);
        // Output: V(out) + rout·j = clamp(V(n1)).
        let v1 = Dual::<3>::var(ld.v(self.n1), 0);
        let (hi, lo, cols) = match self.rails {
            Some((vp, vn)) => (
                Dual::<3>::var(ld.v(vp), 1) - self.drop_hi,
                Dual::<3>::var(ld.v(vn), 2) + self.drop_lo,
                [self.n1, vp, vn],
            ),
            None => (Dual::constant(self.vmax), Dual::constant(self.vmin), [self.n1, GND, GND]),
        };
        let mut vo = v1;
        if hi.v.is_finite() {
            vo = vo - softplus((v1 - hi) / KNEE) * KNEE;
        }
        if lo.v.is_finite() {
            vo = vo + softplus((lo - v1) / KNEE) * KNEE;
        }
        ld.mat(j, self.out, 1.0);
        ld.mat(j, j, self.rout);
        let mut feq = vo.v;
        for (k, &c) in cols.iter().enumerate() {
            if c != GND && vo.d[k] != 0.0 {
                ld.mat(j, c, -vo.d[k]);
                feq -= vo.d[k] * ld.v(c);
            }
        }
        ld.ops += 6;
        ld.rhs(j, feq);
    }
    fn branch(&self) -> Option<usize> {
        Some(self.branch)
    }
}
