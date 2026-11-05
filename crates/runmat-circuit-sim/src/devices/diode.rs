//! Junction diode: Shockley current with emission coefficient, series resistance, reverse
//! breakdown, depletion and diffusion (transit-time) charge, and temperature-scaled `Is`.
//!
//! Model `D` parameters: `is n rs cjo|cj0 vj m fc tt bv ibv eg xti`; the instance takes an area
//! factor and `off`.

use super::junction::{depletion, explim, is_at, pnjlim, vcrit, Params, TNOM_K};
use super::{node_v, Device, Load, Setup, States};
use crate::num::Dual;
use crate::sparse::GND;

pub const KNOWN: &[&str] = &[
    "is", "n", "rs", "cjo", "cj0", "cj", "vj", "m", "fc", "tt", "bv", "ibv", "eg", "xti", "tnom", "level",
];
pub const IGNORED: &[&str] = &["kf", "af", "iave", "vpk", "mfg", "type", "isr", "nr", "nbv", "ibvl", "nbvl", "tbv1", "trs1", "trs", "tikf", "ikf"];

pub struct Diode {
    pub name: String,
    pub a: usize,
    pub k: usize,
    /// Internal anode (after the series resistance), or `a` without one.
    pub ai: usize,
    pub is: f64,
    pub nvt: f64,
    pub gs: f64,
    pub cj0: f64,
    pub vj: f64,
    pub m: f64,
    pub fc: f64,
    pub tt: f64,
    pub bv: f64,
    pub ibv: f64,
    pub vcrit: f64,
    pub off: bool,
    pub state: usize,
    pub mem: usize,
}

impl Diode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(name: &str, a: usize, k: usize, p: &Params, area: f64, off: bool, temp_k: f64, s: &mut Setup) -> Diode {
        let n = p.get("n", 1.0);
        let vt = super::K_OVER_Q * temp_k;
        let tnom = p.get("tnom", 27.0) + 273.15;
        let is = is_at(p.get("is", 1e-14), n, p.get("eg", 1.11), p.get("xti", 3.0), temp_k, if p.has("tnom") { tnom } else { TNOM_K }) * area;
        let rs = p.get("rs", 0.0) / area;
        let ai = if rs > 0.0 { s.internal(name, "a") } else { a };
        let nvt = n * vt;
        Diode {
            name: name.into(),
            a,
            k,
            ai,
            is,
            nvt,
            gs: if rs > 0.0 { 1.0 / rs } else { 0.0 },
            cj0: p.get("cjo", p.get("cj0", p.get("cj", 0.0))) * area,
            vj: p.get("vj", 1.0),
            m: p.get("m", 0.5),
            fc: p.get("fc", 0.5),
            tt: p.get("tt", 0.0),
            bv: p.get("bv", f64::INFINITY),
            ibv: p.get("ibv", 1e-3) * area,
            vcrit: vcrit(nvt, is),
            off,
            state: s.state(false),
            mem: s.mem(0.0),
        }
    }

    fn id<const N: usize>(&self, vd: Dual<N>, gmin: f64) -> Dual<N> {
        let mut i = (explim(vd / self.nvt) - 1.0) * self.is + vd * gmin;
        if self.bv.is_finite() {
            i = i - explim((-vd - self.bv) / self.nvt) * self.ibv;
        }
        i
    }
}

impl Device for Diode {
    fn name(&self) -> &str {
        &self.name
    }
    fn load(&self, ld: &mut Load) {
        if self.gs > 0.0 {
            ld.conductance(self.a, self.ai, self.gs);
        }
        let raw = ld.v(self.ai) - ld.v(self.k);
        let vd = if ld.is_ac() {
            raw
        } else if ld.init_junctions {
            if self.off { 0.0 } else { self.vcrit }
        } else {
            let old = ld.mem[self.mem];
            let (mut v, mut lim) = pnjlim(raw, old, self.nvt, self.vcrit);
            if self.bv.is_finite() && v < (-self.bv + 10.0 * self.nvt).min(0.0) {
                let (r, l2) = pnjlim(-(v + self.bv), -(old + self.bv), self.nvt, self.vcrit);
                v = -(r + self.bv);
                lim |= l2;
            }
            ld.noncon |= lim;
            v
        };
        if !ld.is_ac() {
            ld.mem[self.mem] = vd;
        }
        let x = Dual::<1>::var(vd, 0);
        let ctrl = [(self.ai, self.k)];
        let i = self.id(x, ld.gmin);
        ld.current_at(self.ai, self.k, i, ctrl, [vd]);
        if self.cj0 > 0.0 || self.tt > 0.0 {
            let mut q = depletion(x, self.cj0, self.vj, self.m, self.fc);
            if self.tt > 0.0 {
                q = q + (explim(x / self.nvt) - 1.0) * (self.is * self.tt);
            }
            ld.charge_at(self.state, self.ai, self.k, q, ctrl, [vd]);
        }
    }
    fn probe_current(&self, x: &[f64], s: &States) -> Option<f64> {
        if self.gs > 0.0 {
            return Some((node_v(x, self.a) - node_v(x, self.ai)) * self.gs);
        }
        let vd = node_v(x, self.ai) - if self.k == GND { 0.0 } else { x[self.k] };
        Some(self.id(Dual::<1>::constant(vd), 0.0).v + s.i[self.state][0])
    }
}
