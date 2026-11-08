//! Propagation delay from the stackup: each track's delay from its layer's effective
//! permittivity, each via's from the dielectric it crosses.
//!
//! - A track: `length · √εeff / c`, with `εeff` from [`TraceGeometry::effective_dk`] at the
//!   track's width (a stripline's dielectric Dk; a microstrip's Hammerstad–Jensen value, part
//!   air and part dielectric). A layer with no reference plane uses the stack's mean Dk.
//! - A via: its span (centre to centre of the copper it joins) at the thickness-weighted mean
//!   Dk of the dielectric it crosses.
//!
//! Delays are in picoseconds, lengths in nanometres.

use crate::geom::Coord;
use crate::impedance::{TraceGeometry, SPEED_OF_LIGHT};
use crate::layer_set::CopperIndex;
use crate::layout::{Board, NetId, StackKind, Stackup};
use std::collections::BTreeMap;

/// A net's (or a signal's) routed length and delay.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timing {
    /// nm.
    pub length: f64,
    /// ps.
    pub delay: f64,
}

impl std::ops::AddAssign for Timing {
    fn add_assign(&mut self, o: Timing) {
        self.length += o.length;
        self.delay += o.delay;
    }
}

/// ps per nm at relative permittivity `er`.
fn ps_per_nm(er: f64) -> f64 {
    er.sqrt() / SPEED_OF_LIGHT * 1e-9 * 1e12
}

/// The stackup's delay per layer.
pub struct DelayModel {
    lines: Vec<Option<TraceGeometry>>,
    mean_dk: f64,
    stackup: Stackup,
}

impl DelayModel {
    pub fn new(stackup: &Stackup) -> DelayModel {
        let n = stackup.copper_count();
        let (mut h, mut w) = (0.0, 0.0);
        for l in &stackup.layers {
            if matches!(l.kind, StackKind::Dielectric { .. }) {
                h += l.thickness as f64;
                w += l.thickness as f64 * l.material.dk;
            }
        }
        DelayModel {
            lines: (0..n).map(|l| TraceGeometry::of(stackup, l)).collect(),
            mean_dk: if h > 0.0 { w / h } else { 1.0 },
            stackup: stackup.clone(),
        }
    }

    /// Delay per mm of a `width` wide track on `layer`, ps/mm.
    pub fn ps_per_mm(&self, layer: CopperIndex, width: Coord) -> f64 {
        let er = match self.lines.get(layer as usize).copied().flatten() {
            Some(g) => g.effective_dk(width as f64 * 1e-9),
            None => self.mean_dk,
        };
        ps_per_nm(er) * 1e6
    }

    /// Delay of `length` nm of a `width` wide track on `layer`, ps.
    pub fn track(&self, layer: CopperIndex, width: Coord, length: f64) -> f64 {
        self.ps_per_mm(layer, width) * length * 1e-6
    }

    /// The span (nm) and delay (ps) of a via between copper layers `from` and `to`.
    pub fn via(&self, from: CopperIndex, to: CopperIndex) -> Timing {
        let span = self.stackup.span_length(from, to) as f64;
        let (Some(a), Some(b)) = (self.stackup.copper_z(from.min(to)), self.stackup.copper_z(from.max(to))) else {
            return Timing { length: span, delay: span * ps_per_nm(self.mean_dk) };
        };
        // Dielectric rows whose middle lies between the two copper centres.
        let (mut z, mut h, mut w) = (0, 0.0, 0.0);
        for l in &self.stackup.layers {
            let mid = z + l.thickness / 2;
            z += l.thickness;
            if mid > a && mid < b && matches!(l.kind, StackKind::Dielectric { .. }) {
                h += l.thickness as f64;
                w += l.thickness as f64 * l.material.dk;
            }
        }
        let er = if h > 0.0 { w / h } else { self.mean_dk };
        Timing { length: span, delay: span * ps_per_nm(er) }
    }
}

/// Every routed net's length and delay: its tracks and its vias' spans.
pub fn net_timing(board: &Board) -> BTreeMap<NetId, Timing> {
    let model = DelayModel::new(&board.stackup);
    let mut out: BTreeMap<NetId, Timing> = BTreeMap::new();
    for t in board.tracks.values() {
        let Some(n) = t.net else { continue };
        let length = t.length();
        *out.entry(n).or_default() += Timing { length, delay: model.track(t.layer, t.width, length) };
    }
    for v in board.vias.values() {
        let Some(n) = v.net else { continue };
        *out.entry(n).or_default() += model.via(v.from, v.to);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::mm;

    #[test]
    fn delay_follows_the_dielectric() {
        let stack = Stackup::standard(4, mm(1.6), 1.0, 0.5);
        let m = DelayModel::new(&stack);
        // Stripline in FR-4: √Dk / c. The inner layers sit between core (4.5) and prepreg
        // (4.1): about 6.9 ps/mm (175 ps/in).
        let sl = m.ps_per_mm(1, mm(0.13));
        assert!((6.6..7.2).contains(&sl), "{sl}");
        // Microstrip runs faster: part of its field is in air (εeff ≈ 3.1, about 5.9 ps/mm).
        let ms = m.ps_per_mm(0, mm(0.3));
        assert!((5.5..6.3).contains(&ms) && ms < sl, "{ms}");
        // 10 mm of stripline.
        assert!((m.track(1, mm(0.13), 10e6) - 10.0 * sl).abs() < 1e-9);
        // A through via: its span at the stack's Dk.
        let v = m.via(0, 3);
        assert!(v.length > mm(1.4) as f64 && v.delay > 0.0);
        assert!((v.delay / (v.length * 1e-6) - 6.9).abs() < 0.4, "{v:?}");
    }
}
