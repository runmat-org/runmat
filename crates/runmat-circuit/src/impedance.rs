//! Trace impedance from the stackup, by closed forms (IPC-2141A and Hammerstad–Jensen).
//!
//! A track's geometry comes from the stackup around its copper layer: an outer layer is a
//! surface microstrip over the nearest copper layer (its reference plane), an inner layer a
//! stripline between the nearest copper layers above and below. The dielectric between the
//! track and a plane is the stack between them, its permittivity the thickness-weighted mean of
//! their Dk. Solder mask over a microstrip is left out (it lowers the impedance by a few
//! percent).
//!
//! - Microstrip: Hammerstad–Jensen with the effective width for the copper thickness
//!   (accurate to about 1 % for 0.1 ≤ w/h ≤ 10).
//! - Stripline: the IPC-2141A forms, centred (`60/√εr · ln(4b / (0.67π(0.8w + t)))`) and
//!   off-centre (`80/√εr · ln(1.9(2h + t) / (0.8w + t)) · (1 − h / (4(h + h′ + t)))`, `h`
//!   the nearer plane).
//! - Edge-coupled differential pairs: `2·Z0·(1 − 0.48·e^(−0.96 s/h))` on a microstrip and
//!   `2·Z0·(1 − 0.347·e^(−2.9 s/b))` on a stripline (IPC-2141A).
//!
//! Lengths are in metres here; [`Coord`] values are nanometres.

use crate::layer_set::CopperIndex;
use crate::geom::Coord;
use crate::layout::{StackKind, Stackup};
use std::f64::consts::PI;

/// A copper layer's transmission-line geometry, from the stackup (m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TraceGeometry {
    /// An outer layer over its reference plane at `h`.
    Microstrip { h: f64, t: f64, er: f64 },
    /// An inner layer between planes `h1` above and `h2` below.
    Stripline { h1: f64, h2: f64, t: f64, er: f64 },
}

impl TraceGeometry {
    /// The geometry of copper layer `layer` (0 = top), or `None` with no other copper layer.
    pub fn of(stackup: &Stackup, layer: CopperIndex) -> Option<TraceGeometry> {
        let copper: Vec<usize> = stackup
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l.kind, StackKind::Copper { .. }))
            .map(|(i, _)| i)
            .collect();
        let at = *copper.get(layer as usize)?;
        let t = stackup.layers[at].thickness as f64 * 1e-9;
        // The dielectric between stack rows a and b (exclusive): its thickness and mean Dk.
        let gap = |a: usize, b: usize| -> (f64, f64) {
            let (lo, hi) = (a.min(b), a.max(b));
            let mut h = 0.0;
            let mut weighted = 0.0;
            for l in &stackup.layers[lo + 1..hi] {
                if matches!(l.kind, StackKind::Dielectric { .. }) {
                    let d = l.thickness as f64 * 1e-9;
                    h += d;
                    weighted += d * l.material.dk;
                }
            }
            (h, if h > 0.0 { weighted / h } else { 1.0 })
        };
        let above = layer.checked_sub(1).map(|i| copper[i as usize]);
        let below = copper.get(layer as usize + 1).copied();
        match (above, below) {
            (Some(a), Some(b)) => {
                let ((h1, e1), (h2, e2)) = (gap(a, at), gap(at, b));
                let er = (h1 * e1 + h2 * e2) / (h1 + h2);
                Some(TraceGeometry::Stripline { h1, h2, t, er })
            }
            (Some(p), None) | (None, Some(p)) => {
                let (h, er) = gap(p, at);
                Some(TraceGeometry::Microstrip { h, t, er })
            }
            (None, None) => None,
        }
    }

    /// The characteristic impedance of a track `w` wide (m), in ohms.
    pub fn single_ended(&self, w: f64) -> f64 {
        match *self {
            TraceGeometry::Microstrip { h, t, er } => microstrip(w, h, t, er),
            TraceGeometry::Stripline { h1, h2, t, er } => stripline(w, h1, h2, t, er),
        }
    }

    /// The differential impedance of an edge-coupled pair of `w` wide tracks `s` apart (m).
    pub fn differential(&self, w: f64, s: f64) -> f64 {
        let z0 = self.single_ended(w);
        match *self {
            TraceGeometry::Microstrip { h, .. } => 2.0 * z0 * (1.0 - 0.48 * (-0.96 * s / h).exp()),
            TraceGeometry::Stripline { h1, h2, t, .. } => {
                2.0 * z0 * (1.0 - 0.347 * (-2.9 * s / (h1 + h2 + t)).exp())
            }
        }
    }

    /// The track width (nm) whose single-ended impedance is `target` ohms, between 10 µm and
    /// 10 mm, or `None` when no width in that range reaches it.
    pub fn width_for(&self, target: f64) -> Option<Coord> {
        // Impedance falls as the track widens.
        let (mut lo, mut hi) = (10e-6, 10e-3);
        if self.single_ended(lo) < target || self.single_ended(hi) > target {
            return None;
        }
        for _ in 0..80 {
            let mid = (lo * hi).sqrt();
            if self.single_ended(mid) > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(((lo + hi) / 2.0 * 1e9).round() as Coord)
    }
}

/// Hammerstad–Jensen microstrip with the copper thickness's effective width.
fn microstrip(w: f64, h: f64, t: f64, er: f64) -> f64 {
    let u0 = w / h;
    let du = if t > 0.0 {
        let coth = 1.0 / (6.517 * u0).sqrt().tanh();
        let du1 = t / (PI * h) * (1.0 + 4.0 * std::f64::consts::E / (t / h * coth * coth)).ln();
        0.5 * (1.0 + 1.0 / (er - 1.0).max(0.0).sqrt().cosh()) * du1
    } else {
        0.0
    };
    let u = u0 + du;
    let f = 6.0 + (2.0 * PI - 6.0) * (-(30.666 / u).powf(0.7528)).exp();
    let z_air = 60.0 * (f / u + (1.0 + 4.0 / (u * u)).sqrt()).ln();
    let a = 1.0
        + (((u.powi(4) + (u / 52.0).powi(2)) / (u.powi(4) + 0.432)).ln()) / 49.0
        + ((1.0 + (u / 18.1).powi(3)).ln()) / 18.7;
    let b = 0.564 * ((er - 0.9) / (er + 3.0)).powf(0.053);
    let e_eff = (er + 1.0) / 2.0 + (er - 1.0) / 2.0 * (1.0 + 10.0 / u).powf(-a * b);
    z_air / e_eff.sqrt()
}

/// IPC-2141A stripline, centred or off-centre between its planes.
fn stripline(w: f64, h1: f64, h2: f64, t: f64, er: f64) -> f64 {
    let b = h1 + h2 + t;
    if (h1 - h2).abs() <= 1e-3 * b {
        60.0 / er.sqrt() * (4.0 * b / (0.67 * PI * (0.8 * w + t))).ln()
    } else {
        let (h, far) = (h1.min(h2), h1.max(h2));
        80.0 / er.sqrt() * (1.9 * (2.0 * h + t) / (0.8 * w + t)).ln() * (1.0 - h / (4.0 * (h + far + t)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::mm;

    #[test]
    fn closed_forms_match_published_values() {
        // 1.6 mm FR-4 two-layer board (εr 4.3): a 50 Ω microstrip is about 3 mm wide.
        let ms = TraceGeometry::Microstrip { h: 1.5e-3, t: 35e-6, er: 4.3 };
        let w = ms.width_for(50.0).unwrap() as f64 * 1e-9;
        assert!((2.7e-3..3.1e-3).contains(&w), "{w}");
        // Thin dielectric: 0.2 mm prepreg (εr 4.1), 0.35 mm track, 1 oz: about 50 Ω.
        let z = TraceGeometry::Microstrip { h: 0.2e-3, t: 35e-6, er: 4.1 }.single_ended(0.35e-3);
        assert!((45.0..55.0).contains(&z), "{z}");
        // Centred stripline, b = 0.5 mm, εr 4.3, 0.15 mm track, 17 µm copper: 60/√4.3 · ln(2 /
        // (0.67π · 0.137)) = 56 Ω.
        let sl = TraceGeometry::Stripline { h1: 0.2415e-3, h2: 0.2415e-3, t: 17e-6, er: 4.3 };
        let z = sl.single_ended(0.15e-3);
        assert!((55.5..56.5).contains(&z), "{z}");
        // Coupling lowers a pair below twice its single-ended impedance, less as it spreads.
        let near = ms.differential(0.3e-3, 0.15e-3);
        let far = ms.differential(0.3e-3, 1.5e-3);
        let z0 = ms.single_ended(0.3e-3);
        assert!(near < far && far < 2.0 * z0, "{near} {far} {z0}");
    }

    #[test]
    fn a_layers_geometry_reads_the_stackup() {
        let stack = Stackup::standard(4, mm(1.6), 1.0, 0.5);
        assert!(matches!(TraceGeometry::of(&stack, 0), Some(TraceGeometry::Microstrip { .. })));
        assert!(matches!(TraceGeometry::of(&stack, 1), Some(TraceGeometry::Stripline { .. })));
        assert!(matches!(TraceGeometry::of(&stack, 3), Some(TraceGeometry::Microstrip { .. })));
        assert_eq!(TraceGeometry::of(&stack, 4), None);
        let Some(TraceGeometry::Microstrip { h, er, .. }) = TraceGeometry::of(&stack, 0) else { unreachable!() };
        // The top prepreg (εr 4.1) between F.Cu and In1.Cu.
        assert!((er - 4.1).abs() < 1e-9 && h > 0.3e-3 && h < 0.6e-3, "{h} {er}");
    }
}
