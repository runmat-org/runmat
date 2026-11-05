//! Bipolar transistor: the Gummel–Poon model (Ebers–Moll when the high-injection, Early and
//! leakage parameters are left at their defaults), with base, collector and emitter
//! resistances, depletion charges and transit-time diffusion charges.
//!
//! Model `NPN`/`PNP` parameters: `is bf nf vaf ikf ise ne br nr var ikr isc nc rb re rc cje vje
//! mje tf cjc vjc mjc tr fc eg xti`. The instance takes an optional substrate node (unused), an
//! area factor and `off`.

use super::junction::{depletion, explim, is_at, pnjlim, vcrit, Params, TNOM_K};
use super::{node_v, Device, Load, Setup, States, K_OVER_Q};
use crate::num::Dual;

pub const KNOWN: &[&str] = &[
    "is", "bf", "nf", "vaf", "va", "ikf", "ik", "ise", "ne", "br", "nr", "var", "vb", "ikr", "isc", "nc", "rb", "re", "rc",
    "cje", "vje", "pe", "mje", "me", "tf", "cjc", "vjc", "pc", "mjc", "mc", "tr", "fc", "eg", "xti", "tnom", "level",
];
pub const IGNORED: &[&str] = &[
    "irb", "rbm", "xtf", "vtf", "itf", "ptf", "xcjc", "cjs", "ccs", "vjs", "ps", "mjs", "ms", "xtb", "kf", "af", "iss", "ns",
    "nk", "qco", "rco", "vo", "gamma", "mfg", "vceo", "icrating", "type", "bvcbo", "bvebo", "bvceo", "nkf", "trb1", "tre1", "trc1", "quasimod", "iave", "vpk",
];

pub struct Bjt {
    pub name: String,
    /// +1 for NPN, −1 for PNP.
    pub tp: f64,
    pub c: usize,
    pub b: usize,
    pub e: usize,
    pub ci: usize,
    pub bi: usize,
    pub ei: usize,
    pub is: f64,
    pub bf: f64,
    pub br: f64,
    pub nf_vt: f64,
    pub nr_vt: f64,
    pub ise: f64,
    pub ne_vt: f64,
    pub isc: f64,
    pub nc_vt: f64,
    pub vaf: f64,
    pub var: f64,
    pub ikf: f64,
    pub ikr: f64,
    pub gb: f64,
    pub gc: f64,
    pub ge: f64,
    pub cje: f64,
    pub vje: f64,
    pub mje: f64,
    pub cjc: f64,
    pub vjc: f64,
    pub mjc: f64,
    pub tf: f64,
    pub tr: f64,
    pub fc: f64,
    pub vcrit: f64,
    pub off: bool,
    pub q_be: usize,
    pub q_bc: usize,
    pub mem: usize,
}

impl Bjt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(name: &str, pnp: bool, nodes: [usize; 3], p: &Params, area: f64, off: bool, temp_k: f64, s: &mut Setup) -> Bjt {
        let vt = K_OVER_Q * temp_k;
        let nf = p.get("nf", 1.0);
        let tnom = if p.has("tnom") { p.get("tnom", 27.0) + 273.15 } else { TNOM_K };
        let is = is_at(p.get("is", 1e-16), 1.0, p.get("eg", 1.11), p.get("xti", 3.0), temp_k, tnom) * area;
        let r = |k: &str| p.get(k, 0.0) / area;
        let (rb, rc, re) = (r("rb"), r("rc"), r("re"));
        let [c, b, e] = nodes;
        let ci = if rc > 0.0 { s.internal(name, "c") } else { c };
        let bi = if rb > 0.0 { s.internal(name, "b") } else { b };
        let ei = if re > 0.0 { s.internal(name, "e") } else { e };
        let inv = |v: f64| if v > 0.0 && v.is_finite() { 1.0 / v } else { 0.0 };
        let alt = |a: &str, b: &str, d: f64| p.get(a, p.get(b, d));
        Bjt {
            name: name.into(),
            tp: if pnp { -1.0 } else { 1.0 },
            c,
            b,
            e,
            ci,
            bi,
            ei,
            is,
            bf: p.get("bf", 100.0),
            br: p.get("br", 1.0),
            nf_vt: nf * vt,
            nr_vt: p.get("nr", 1.0) * vt,
            ise: p.get("ise", 0.0) * area,
            ne_vt: p.get("ne", 1.5) * vt,
            isc: p.get("isc", 0.0) * area,
            nc_vt: p.get("nc", 2.0) * vt,
            vaf: inv(alt("vaf", "va", f64::INFINITY)),
            var: inv(alt("var", "vb", f64::INFINITY)),
            ikf: inv(alt("ikf", "ik", f64::INFINITY) * area),
            ikr: inv(p.get("ikr", f64::INFINITY) * area),
            gb: inv(rb),
            gc: inv(rc),
            ge: inv(re),
            cje: p.get("cje", 0.0) * area,
            vje: alt("vje", "pe", 0.75),
            mje: alt("mje", "me", 0.33),
            cjc: p.get("cjc", 0.0) * area,
            vjc: alt("vjc", "pc", 0.75),
            mjc: alt("mjc", "mc", 0.33),
            tf: p.get("tf", 0.0),
            tr: p.get("tr", 0.0),
            fc: p.get("fc", 0.5),
            vcrit: vcrit(nf * vt, is),
            off,
            q_be: s.state(false),
            q_bc: s.state(false),
            mem: {
                let m = s.mem(0.0);
                s.mem(0.0);
                m
            },
        }
    }

    /// Orients a node pair for the device polarity.
    fn pair(&self, x: usize, y: usize) -> (usize, usize) {
        if self.tp > 0.0 {
            (x, y)
        } else {
            (y, x)
        }
    }

    /// Branch currents in NPN orientation: (base–emitter, base–collector, collector–emitter
    /// transport) and the junction charges.
    fn eval(&self, vbe: f64, vbc: f64, gmin: f64, charges: bool) -> [Dual<2>; 5] {
        let be = Dual::<2>::var(vbe, 0);
        let bc = Dual::<2>::var(vbc, 1);
        let ibe1 = (explim(be / self.nf_vt) - 1.0) * self.is;
        let ibc1 = (explim(bc / self.nr_vt) - 1.0) * self.is;
        let mut ibe = ibe1 / self.bf + be * gmin;
        let mut ibc = ibc1 / self.br + bc * gmin;
        if self.ise > 0.0 {
            ibe = ibe + (explim(be / self.ne_vt) - 1.0) * self.ise;
        }
        if self.isc > 0.0 {
            ibc = ibc + (explim(bc / self.nc_vt) - 1.0) * self.isc;
        }
        let q1 = 1.0 / (1.0 - bc * self.vaf - be * self.var);
        let qb = if self.ikf > 0.0 || self.ikr > 0.0 {
            let q2 = ibe1 * self.ikf + ibc1 * self.ikr;
            q1 * ((1.0 + (1.0 + q2 * 4.0).sqrt()) * 0.5)
        } else {
            q1
        };
        let ice = (ibe1 - ibc1) / qb;
        let (mut qbe, mut qbc) = (Dual::constant(0.0), Dual::constant(0.0));
        if charges {
            qbe = depletion(be, self.cje, self.vje, self.mje, self.fc);
            if self.tf > 0.0 {
                qbe = qbe + ibe1 / qb * self.tf;
            }
            qbc = depletion(bc, self.cjc, self.vjc, self.mjc, self.fc);
            if self.tr > 0.0 {
                qbc = qbc + ibc1 * self.tr;
            }
        }
        [ibe, ibc, ice, qbe, qbc]
    }

    fn has_charge(&self) -> bool {
        self.cje > 0.0 || self.cjc > 0.0 || self.tf > 0.0 || self.tr > 0.0
    }
}

impl Device for Bjt {
    fn name(&self) -> &str {
        &self.name
    }
    fn load(&self, ld: &mut Load) {
        if self.gb > 0.0 {
            ld.conductance(self.b, self.bi, self.gb);
        }
        if self.gc > 0.0 {
            ld.conductance(self.c, self.ci, self.gc);
        }
        if self.ge > 0.0 {
            ld.conductance(self.e, self.ei, self.ge);
        }
        let raw_be = self.tp * (ld.v(self.bi) - ld.v(self.ei));
        let raw_bc = self.tp * (ld.v(self.bi) - ld.v(self.ci));
        let (vbe, vbc) = if ld.is_ac() {
            (raw_be, raw_bc)
        } else if ld.init_junctions {
            if self.off { (0.0, 0.0) } else { (self.vcrit, 0.0) }
        } else {
            let (be, l1) = pnjlim(raw_be, ld.mem[self.mem], self.nf_vt, self.vcrit);
            let (bc, l2) = pnjlim(raw_bc, ld.mem[self.mem + 1], self.nr_vt, self.vcrit);
            ld.noncon |= l1 || l2;
            (be, bc)
        };
        if !ld.is_ac() {
            ld.mem[self.mem] = vbe;
            ld.mem[self.mem + 1] = vbc;
        }
        let [ibe, ibc, ice, qbe, qbc] = self.eval(vbe, vbc, ld.gmin, self.has_charge());
        let ctrl = [self.pair(self.bi, self.ei), self.pair(self.bi, self.ci)];
        let vals = [vbe, vbc];
        let (b, e) = self.pair(self.bi, self.ei);
        let (b2, c) = self.pair(self.bi, self.ci);
        let (c2, e2) = self.pair(self.ci, self.ei);
        ld.current_at(b, e, ibe, ctrl, vals);
        ld.current_at(b2, c, ibc, ctrl, vals);
        ld.current_at(c2, e2, ice, ctrl, vals);
        if self.has_charge() {
            ld.charge_at(self.q_be, b, e, qbe, ctrl, vals);
            ld.charge_at(self.q_bc, b2, c, qbc, ctrl, vals);
        }
    }
    /// Collector current (into the collector terminal).
    fn probe_current(&self, x: &[f64], s: &States) -> Option<f64> {
        let vbe = self.tp * (node_v(x, self.bi) - node_v(x, self.ei));
        let vbc = self.tp * (node_v(x, self.bi) - node_v(x, self.ci));
        let [_, ibc, ice, _, _] = self.eval(vbe, vbc, 0.0, false);
        Some(self.tp * (ice.v - ibc.v - s.i[self.q_bc][0]))
    }
}
