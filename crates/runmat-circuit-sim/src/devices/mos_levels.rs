//! MOSFET drain-current laws past level 1: level 2 (Grove–Frohman charge sheet) and level 3
//! (the short-channel semi-empirical model), as SPICE2/SPICE3 define them (Vladimirescu and Liu,
//! "The Simulation of MOS Integrated Circuits Using SPICE2", UCB/ERL M80/7, 1980).
//!
//! Each law takes `vgs`, `vds ≥ 0` and `vbs` in NMOS orientation and returns `Ids` as a
//! [`Dual`] of them, so the Jacobian is exact. Below threshold the current is zero (no `nfs`
//! weak-inversion term).
//!
//! - Level 2: `Ids = β·[(Vgs − Vbi − η·Vds/2)·Vds − ⅔·γs·((Vds + φ − Vbs)^{3/2} − (φ − Vbs)^{3/2})]`
//!   (φ is the model's `phi`, the surface potential 2φF) with `Vbi = Vto − γ·√φ`, `η = 1 + δ·π·εsi/(4·Cox·W)`, saturation at the `Vdsat` where
//!   `∂Ids/∂Vds = 0`, mobility `µs = µo·(Ucrit·εsi/(Cox·(Vgs − Vth − Utra·Vds)))^Uexp` (at most
//!   `µo`), and channel-length modulation `Ids/(1 − λ·(Vds − Vdsat))` in saturation.
//! - Level 3: `Vth = Vbi − σ·Vds + γ·Fs·√(φ − Vbs) + Fn·(φ − Vbs)`, `σ = η·8.15e−22/(Cox·L³)`,
//!   `µs = µo/(1 + θ·(Vgs − Vth))`, `Fb = γ·Fs/(4·√(φ − Vbs)) + Fn`,
//!   `Ids = β·(Vgs − Vth − (1 + Fb)/2·Vde)·Vde·Fd` with `Vde = min(Vds, Vdsat)`; with `vmax`,
//!   `Vc = vmax·L/µs`, `Vdsat = Vsat + Vc − √(Vsat² + Vc²)`, `Fd = 1/(1 + Vde/Vc)`; past
//!   saturation `ΔL = Xd·√(κ·(Vds − Vdsat))` shortens the channel (`Ids·L/(L − ΔL)`).

use super::junction::Params;
use crate::num::Dual;

type D = Dual<3>;

fn dmax(a: D, b: D) -> D {
    if a.v >= b.v {
        a
    } else {
        b
    }
}

fn dmin(a: D, b: D) -> D {
    if a.v <= b.v {
        a
    } else {
        b
    }
}

/// `η·(φ − Vbs)`.
fn pb_eta(vbs: D, phi: f64, eta: f64) -> D {
    (-vbs + phi) * eta
}

const EPS_SI: f64 = 11.7 * 8.854_187_8e-12;
const Q: f64 = 1.602_176_634e-19;

/// The physical quantities both levels derive from the model card.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub leff: f64,
    pub w: f64,
    pub cox: f64,
    /// `µo` (m²/V·s).
    pub uo: f64,
    /// Depletion width coefficient `√(2·εsi/(q·Nsub))` (m/√V); 0 without `nsub`.
    pub xd: f64,
}

impl Geometry {
    pub fn new(p: &Params, leff: f64, w: f64, cox: f64) -> Geometry {
        let nsub = p.get("nsub", 0.0) * 1e6;
        Geometry {
            leff,
            w,
            cox,
            uo: p.get("uo", p.get("u0", 600.0)) * 1e-4,
            xd: if nsub > 0.0 {
                (2.0 * EPS_SI / (Q * nsub)).sqrt()
            } else {
                0.0
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Level2 {
    pub vbi: f64,
    pub beta: f64,
    pub gamma: f64,
    pub phi: f64,
    pub eta: f64,
    pub lambda: f64,
    pub ucrit: f64,
    pub uexp: f64,
    pub utra: f64,
    pub cox: f64,
}

impl Level2 {
    pub fn new(p: &Params, g: &Geometry, vto: f64, beta: f64, gamma: f64, phi: f64) -> Level2 {
        let delta = p.get("delta", 0.0);
        Level2 {
            vbi: vto - gamma * phi.sqrt(),
            beta,
            gamma,
            phi,
            eta: 1.0 + delta * std::f64::consts::PI * EPS_SI / (4.0 * g.cox.max(1e-30) * g.w),
            lambda: p.get("lambda", 0.0),
            ucrit: p.get("ucrit", 1e4) * 1e2,
            uexp: p.get("uexp", 0.0),
            utra: p.get("utra", 0.0),
            cox: g.cox,
        }
    }

    /// The triode charge-sheet current.
    fn triode(&self, vgs: D, vds: D, vbs: D, beta: D) -> D {
        let a = (vds + (self.phi) - vbs).powf(1.5);
        let b = (-vbs + self.phi).powf(1.5);
        ((vgs - self.vbi - vds * (self.eta * 0.5)) * vds - (a - b) * (self.gamma * 2.0 / 3.0))
            * beta
    }

    pub fn ids(&self, vgs: D, vds: D, vbs: D) -> D {
        let vbs = if vbs.v > 0.0 { vbs * 0.0 } else { vbs };
        let sarg = (-vbs + self.phi).sqrt();
        let vth = sarg * self.gamma + self.vbi;
        let vgst = vgs - vth;
        if vgst.v <= 0.0 {
            return D::constant(0.0);
        }
        // Saturation: ∂Ids/∂Vds = 0 ⇔ Vgs − Vbi − η·Vds = γ·√(Vds + φ − Vbs). With
        // s = √(Vds + φ − Vbs): η·s² + γ·s − (Vgs − Vbi + η·(φ − Vbs)) = 0.
        let x = vgs - self.vbi + pb_eta(vbs, self.phi, self.eta);
        let s = if self.gamma == 0.0 {
            (x / self.eta).sqrt()
        } else {
            ((x * (4.0 * self.eta) + self.gamma * self.gamma).sqrt() - self.gamma)
                / (2.0 * self.eta)
        };
        let vdsat = dmax(s * s - self.phi + vbs, D::constant(0.0));
        let beta = if self.uexp > 0.0 && self.cox > 0.0 {
            let drive = dmax(vgst - vds * self.utra, D::constant(1e-12));
            let f = (D::constant(self.ucrit * EPS_SI / self.cox) / drive).powf(self.uexp);
            if f.v < 1.0 {
                f * self.beta
            } else {
                D::constant(self.beta)
            }
        } else {
            D::constant(self.beta)
        };
        if vds.v <= vdsat.v {
            return self.triode(vgs, vds, vbs, beta);
        }
        let sat = self.triode(vgs, vdsat, vbs, beta);
        let clm = D::constant(1.0) - (vds - vdsat) * self.lambda;
        sat / dmax(clm, D::constant(0.5))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Level3 {
    pub vbi: f64,
    pub beta: f64,
    pub gamma: f64,
    pub phi: f64,
    pub sigma: f64,
    pub fnarrow: f64,
    pub theta: f64,
    pub vmax: f64,
    pub kappa: f64,
    pub xj: f64,
    pub ld: f64,
    pub geo: Geometry,
}

impl Level3 {
    pub fn new(p: &Params, g: &Geometry, vto: f64, beta: f64, gamma: f64, phi: f64) -> Level3 {
        let cox = g.cox.max(1e-30);
        Level3 {
            vbi: vto - gamma * phi.sqrt(),
            beta,
            gamma,
            phi,
            sigma: p.get("eta", 0.0) * 8.15e-22 / (cox * g.leff.powi(3)),
            fnarrow: p.get("delta", 0.0) * std::f64::consts::PI * EPS_SI / (2.0 * cox * g.w),
            theta: p.get("theta", 0.0),
            vmax: p.get("vmax", 0.0),
            kappa: p.get("kappa", 0.2),
            xj: p.get("xj", 0.0),
            ld: p.get("ld", 0.0),
            geo: g.clone(),
        }
    }

    /// The short-channel factor `Fs` (1 without `xj` or `nsub`).
    fn fs(&self, sarg: D) -> D {
        if self.xj <= 0.0 || self.geo.xd <= 0.0 {
            return D::constant(1.0);
        }
        let wp = sarg * self.geo.xd;
        let r = wp / self.xj;
        let wc = (D::constant(0.063_135_3) + r * 0.801_329_2 - r * r * 0.011_107_77) * self.xj;
        let t = wp / (wp + self.xj);
        let root = (D::constant(1.0) - t * t).sqrt();
        D::constant(1.0) - ((wc + self.ld) * root - self.ld) / self.geo.leff
    }

    pub fn ids(&self, vgs: D, vds: D, vbs: D) -> D {
        let vbs = if vbs.v > 0.0 { vbs * 0.0 } else { vbs };
        let pb = -vbs + self.phi;
        let sarg = pb.sqrt();
        let fs = self.fs(sarg);
        let vth = sarg * fs * self.gamma + pb * self.fnarrow + self.vbi - vds * self.sigma;
        let vgst = vgs - vth;
        if vgst.v <= 0.0 {
            return D::constant(0.0);
        }
        let fb = fs * self.gamma / (sarg * 4.0) + self.fnarrow;
        let mob = D::constant(1.0) / (vgst * self.theta + 1.0);
        let vsat = vgst / (fb + 1.0);
        let (vdsat, vc) = if self.vmax > 0.0 {
            let vc = D::constant(self.vmax * self.geo.leff / self.geo.uo) / mob;
            (vsat + vc - (vsat * vsat + vc * vc).sqrt(), Some(vc))
        } else {
            (vsat, None)
        };
        let vde = if vds.v < vdsat.v { vds } else { vdsat };
        let fd = match vc {
            Some(vc) => D::constant(1.0) / (vde / vc + 1.0),
            None => D::constant(1.0),
        };
        let mut i = (vgst - (fb + 1.0) * vde * 0.5) * vde * fd * mob * self.beta;
        if vds.v > vdsat.v && self.geo.xd > 0.0 && self.kappa > 0.0 {
            let dl = ((vds - vdsat) * self.kappa).sqrt() * self.geo.xd;
            let dl = dmin(dl, D::constant(0.5 * self.geo.leff));
            i = i * self.geo.leff / (D::constant(self.geo.leff) - dl);
        }
        i
    }
}
