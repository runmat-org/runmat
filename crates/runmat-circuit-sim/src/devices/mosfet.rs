//! MOSFET, levels 1 (Shichman–Hodges: square-law drain current with body effect and
//! channel-length modulation), 2 and 3 (see [`super::mos_levels`]): symmetric source/drain,
//! drain and source resistances, and
//! constant gate capacitances (overlap plus two-thirds of the channel's oxide capacitance on the
//! gate–source side) and bulk junction capacitances.
//!
//! Model `NMOS`/`PMOS` parameters: `level vto kp gamma phi lambda rd rs cgso cgdo cgbo cbd cbs
//! tox ld uo nsub`, at level 2 also `delta ucrit uexp utra`, at level 3 `delta theta eta vmax
//! kappa xj`. Without `gamma` or `phi`, `nsub` (cm⁻³) and `tox` give them as SPICE does:
//! `γ = √(2·q·εsi·Nsub)/Cox`, `φ = 2·Vt·ln(Nsub/ni)`. The instance takes `l w` and `off`.

use super::junction::Params;
use super::mos_levels::{Geometry, Level2, Level3};
use super::{node_v, Device, Load, Setup, States};
use crate::num::Dual;

pub const KNOWN: &[&str] = &[
    "level", "vto", "vt0", "kp", "gamma", "phi", "lambda", "rd", "rs", "cgso", "cgdo", "cgbo", "cbd", "cbs", "tox", "ld", "uo", "u0", "l", "w",
    "nsub", "delta", "ucrit", "uexp", "utra", "theta", "eta", "vmax", "kappa", "xj",
];
pub const IGNORED: &[&str] = &[
    "is", "js", "pb", "cj", "mj", "cjsw", "mjsw", "fc", "kf", "af", "nss", "tpg", "rsh", "tnom", "mfg", "nfs", "neff",
];

/// The drain-current law of the model's level.
pub enum Law {
    One,
    Two(Level2),
    Three(Level3),
}

pub struct Mosfet {
    pub name: String,
    pub tp: f64,
    pub d: usize,
    pub g: usize,
    pub s: usize,
    pub b: usize,
    pub di: usize,
    pub si: usize,
    pub vto: f64,
    pub beta: f64,
    pub gamma: f64,
    pub phi: f64,
    pub lambda: f64,
    pub gd: f64,
    pub gs: f64,
    /// Gate–source, gate–drain, gate–bulk, bulk–drain, bulk–source capacitances.
    pub caps: [f64; 5],
    pub states: usize,
    pub mem: usize,
    pub off: bool,
    pub law: Law,
}

impl Mosfet {
    #[allow(clippy::too_many_arguments)]
    pub fn new(name: &str, pmos: bool, nodes: [usize; 4], p: &Params, l: Option<f64>, w: Option<f64>, off: bool, s: &mut Setup) -> Result<Mosfet, String> {
        let level = p.get("level", 1.0);
        if ![1.0, 2.0, 3.0].contains(&level) {
            return Err(format!("`{name}`: MOSFET level {level} is not supported yet (levels 1, 2 and 3 are)"));
        }
        let l = l.unwrap_or(p.get("l", 100e-6));
        let w = w.unwrap_or(p.get("w", 100e-6));
        let leff = l - 2.0 * p.get("ld", 0.0);
        if leff <= 0.0 {
            return Err(format!("`{name}`: effective channel length {leff} m is not positive"));
        }
        let cox = if p.has("tox") { 3.9 * 8.854_187_8e-12 / p.get("tox", 1e-7) } else { 0.0 };
        let kp = if p.has("kp") {
            p.get("kp", 2e-5)
        } else if cox > 0.0 && (p.has("uo") || p.has("u0")) {
            p.get("uo", p.get("u0", 600.0)) * 1e-4 * cox
        } else {
            2e-5
        };
        // Substrate doping gives γ and φ where the card leaves them out.
        let nsub = p.get("nsub", 0.0) * 1e6;
        let (eps_si, q, ni) = (11.7 * 8.854_187_8e-12, 1.602_176_634e-19, 1.45e16);
        let gamma = if p.has("gamma") || nsub <= 0.0 || cox <= 0.0 { p.get("gamma", 0.0) } else { (2.0 * q * eps_si * nsub).sqrt() / cox };
        let phi = if p.has("phi") || nsub <= ni { p.get("phi", 0.6) } else { 2.0 * super::K_OVER_Q * 300.15 * (nsub / ni).ln() };
        let vto = p.get("vto", p.get("vt0", 0.0));
        let beta = kp * w / leff;
        let geo = Geometry::new(p, leff, w, cox);
        let law = match level as i32 {
            2 => Law::Two(Level2::new(p, &geo, vto, beta, gamma, phi)),
            3 => Law::Three(Level3::new(p, &geo, vto, beta, gamma, phi)),
            _ => Law::One,
        };
        let [d, g, src, b] = nodes;
        let (rd, rs) = (p.get("rd", 0.0), p.get("rs", 0.0));
        let di = if rd > 0.0 { s.internal(name, "d") } else { d };
        let si = if rs > 0.0 { s.internal(name, "s") } else { src };
        let states = s.state(false);
        for _ in 1..5 {
            s.state(false);
        }
        let mem = s.mem(0.0);
        s.mem(0.0);
        s.mem(0.0);
        Ok(Mosfet {
            name: name.into(),
            tp: if pmos { -1.0 } else { 1.0 },
            d,
            g,
            s: src,
            b,
            di,
            si,
            vto,
            beta,
            gamma,
            phi,
            lambda: p.get("lambda", 0.0),
            gd: if rd > 0.0 { 1.0 / rd } else { 0.0 },
            gs: if rs > 0.0 { 1.0 / rs } else { 0.0 },
            caps: [
                p.get("cgso", 0.0) * w + 2.0 / 3.0 * cox * w * leff,
                p.get("cgdo", 0.0) * w,
                p.get("cgbo", 0.0) * leff,
                p.get("cbd", 0.0),
                p.get("cbs", 0.0),
            ],
            states,
            mem,
            off,
            law,
        })
    }

    /// Drain current (drain to source, NMOS orientation) for `vds ≥ 0`.
    fn ids(&self, vgs: Dual<3>, vds: Dual<3>, vbs: Dual<3>) -> Dual<3> {
        match &self.law {
            Law::Two(l) => return l.ids(vgs, vds, vbs),
            Law::Three(l) => return l.ids(vgs, vds, vbs),
            Law::One => {}
        }
        let sphi = self.phi.sqrt();
        let sarg = if vbs.v <= 0.0 {
            (self.phi - vbs).sqrt()
        } else {
            // Forward body bias: the smooth continuation SPICE level 1 uses.
            sphi / (1.0 + vbs / (2.0 * self.phi))
        };
        let vth = self.vto * self.tp + (sarg - sphi) * self.gamma;
        let vgst = vgs - vth;
        if vgst.v <= 0.0 {
            return Dual::constant(0.0);
        }
        let clm = 1.0 + vds * self.lambda;
        if vds.v < vgst.v {
            vds * (vgst - vds * 0.5) * clm * self.beta
        } else {
            vgst * vgst * clm * (self.beta * 0.5)
        }
    }
}

impl Device for Mosfet {
    fn name(&self) -> &str {
        &self.name
    }
    fn load(&self, ld: &mut Load) {
        if self.gd > 0.0 {
            ld.conductance(self.d, self.di, self.gd);
        }
        if self.gs > 0.0 {
            ld.conductance(self.s, self.si, self.gs);
        }
        let tp = self.tp;
        let (d, s, g, b) = (self.di, self.si, self.g, self.b);
        let mut vgs = tp * (ld.v(g) - ld.v(s));
        let mut vds = tp * (ld.v(d) - ld.v(s));
        let mut vbs = tp * (ld.v(b) - ld.v(s));
        if !ld.is_ac() {
            if ld.init_junctions {
                vgs = if self.off { 0.0 } else { self.vto * tp + 0.1 };
                vds = 0.1;
                vbs = 0.0;
            } else {
                // Damp large gate and drain steps.
                let lim = |v: f64, old: f64, max: f64| -> (f64, bool) {
                    if (v - old).abs() > max { (old + max.copysign(v - old), true) } else { (v, false) }
                };
                let (a, l1) = lim(vgs, ld.mem[self.mem], 1.0);
                let (c, l2) = lim(vds, ld.mem[self.mem + 1], 2.0);
                vgs = a;
                vds = c;
                ld.noncon |= l1 || l2;
            }
            ld.mem[self.mem] = vgs;
            ld.mem[self.mem + 1] = vds;
            ld.mem[self.mem + 2] = vbs;
        }
        let o = |x: usize, y: usize| if tp > 0.0 { (x, y) } else { (y, x) };
        if vds >= 0.0 {
            let i = self.ids(Dual::var(vgs, 0), Dual::var(vds, 1), Dual::var(vbs, 2)) + Dual::<3>::var(vds, 1) * ld.gmin;
            let (a, c) = o(d, s);
            ld.current_at(a, c, i, [o(g, s), o(d, s), o(b, s)], [vgs, vds, vbs]);
        } else {
            // Reverse mode: the source acts as drain.
            let (vgd, vsd, vbd) = (vgs - vds, -vds, vbs - vds);
            let i = self.ids(Dual::var(vgd, 0), Dual::var(vsd, 1), Dual::var(vbd, 2)) + Dual::<3>::var(vsd, 1) * ld.gmin;
            let (a, c) = o(s, d);
            ld.current_at(a, c, i, [o(g, d), o(s, d), o(b, d)], [vgd, vsd, vbd]);
        }
        let pairs = [(g, s), (g, d), (g, b), (b, d), (b, s)];
        for (k, (&cap, &(x, y))) in self.caps.iter().zip(&pairs).enumerate() {
            if cap > 0.0 {
                let q = Dual::<1>::var(ld.v(x) - ld.v(y), 0) * cap;
                ld.charge(self.states + k, x, y, q, [(x, y)]);
            }
        }
    }
    /// Drain current (into the drain terminal).
    fn probe_current(&self, x: &[f64], _s: &States) -> Option<f64> {
        let tp = self.tp;
        let v = |n| node_v(x, n);
        let (vgs, vds, vbs) = (tp * (v(self.g) - v(self.si)), tp * (v(self.di) - v(self.si)), tp * (v(self.b) - v(self.si)));
        let c = |a: f64| Dual::<3>::constant(a);
        let i = if vds >= 0.0 {
            self.ids(c(vgs), c(vds), c(vbs)).v
        } else {
            -self.ids(c(vgs - vds), c(-vds), c(vbs - vds)).v
        };
        Some(tp * i)
    }
}
