//! Impedance-driven widths and gaps per copper layer.
//!
//! A net class writes its targets in ohms:
//!
//! ```text
//! net_class usb = { diff_impedance = 90, clearance = 0.15 mm }
//! net_class rf = { impedance = 50 }
//! net_class ddr = { impedance = 40, diff_impedance = 80, diff_gap = 0.12 mm }
//! ```
//!
//! For each copper layer with a reference plane, [`controlled_layers`] solves the closed forms
//! of [`crate::impedance`]:
//!
//! - **Single-ended** (`impedance` only): the width that meets it.
//! - **Differential** (`diff_impedance`): with `diff_gap` written, the width that meets the
//!   target at that gap; with `impedance` also written, the width that meets the single-ended
//!   target and then the gap that meets the differential one; with neither, a gap equal to the
//!   width (and at least the board's minimum clearance), the common edge-coupled starting point.
//!
//! Widths and gaps round to 1 µm. A layer whose width or gap falls below the board's limits
//! keeps its solved value and says so in [`ControlledLayer::note`]; DRC then flags tracks on it
//! by the minimum-width and clearance rules.

use crate::board_rules::{props, RuleValue};
use crate::geom::Coord;
use crate::impedance::TraceGeometry;
use crate::layer_set::CopperIndex;
use crate::layout::{copper_name, FabLimits, Stackup};

/// A net class's impedance targets, in ohms, and a written pair gap.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImpedanceTarget {
    pub single: Option<f64>,
    pub diff: Option<f64>,
    pub gap: Option<Coord>,
}

/// An impedance written as a rule value: a number of ohms (`50` or `50.0`).
pub fn ohms(v: &RuleValue) -> Option<f64> {
    match v {
        RuleValue::Num(x) => Some(*x),
        RuleValue::Int(x) => Some(*x as f64),
        _ => None,
    }
}

impl ImpedanceTarget {
    /// The targets a class's parameters write, or `None` when it writes no impedance.
    pub fn of(decls: &[(String, RuleValue)]) -> Option<ImpedanceTarget> {
        let get = |k: &str| decls.iter().rev().find(|(p, _)| p == k).map(|(_, v)| v);
        let t = ImpedanceTarget {
            single: get(props::IMPEDANCE).and_then(ohms).filter(|z| *z > 0.0),
            diff: get(props::DIFF_IMPEDANCE).and_then(ohms).filter(|z| *z > 0.0),
            gap: get(props::DIFF_GAP).and_then(RuleValue::as_len),
        };
        (t.single.is_some() || t.diff.is_some()).then_some(t)
    }
}

/// One copper layer's solved width and gap for a class.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledLayer {
    pub layer: CopperIndex,
    /// The layer's file name (`F.Cu`, `In1.Cu`, `B.Cu`).
    pub name: String,
    /// `microstrip`, `stripline`, or `none` with no reference plane.
    pub kind: &'static str,
    pub width: Option<Coord>,
    /// The pair gap (differential targets only).
    pub gap: Option<Coord>,
    /// The single-ended impedance at `width`, Ω.
    pub z: Option<f64>,
    /// The differential impedance at `width` and `gap`, Ω.
    pub z_diff: Option<f64>,
    /// Propagation delay at `width`, ps/mm.
    pub delay_ps_per_mm: Option<f64>,
    /// Why the layer can't carry the class, or what it breaks (`below the 0.1 mm minimum
    /// track`).
    pub note: Option<String>,
}

const UM: Coord = 1_000;

fn round_um(v: Coord) -> Coord {
    ((v + UM / 2) / UM) * UM
}

/// The width and gap that meet `target` on each copper layer of `stackup`.
pub fn controlled_layers(stackup: &Stackup, target: &ImpedanceTarget, limits: &FabLimits) -> Vec<ControlledLayer> {
    let n = stackup.copper_count();
    (0..n)
        .map(|layer| {
            let name = copper_name(layer, n);
            let Some(g) = TraceGeometry::of(stackup, layer) else {
                return ControlledLayer {
                    layer,
                    name,
                    kind: "none",
                    width: None,
                    gap: None,
                    z: None,
                    z_diff: None,
                    delay_ps_per_mm: None,
                    note: Some("no reference plane".into()),
                };
            };
            let (width, gap) = solve(&g, target, limits);
            let m = |v: Coord| v as f64 * 1e-9;
            let z = width.map(|w| g.single_ended(m(w)));
            let z_diff = match (width, gap) {
                (Some(w), Some(s)) => Some(g.differential(m(w), m(s))),
                _ => None,
            };
            let mut notes = Vec::new();
            match width {
                None => notes.push("no width between 10 µm and 10 mm meets the target".to_string()),
                Some(w) if w < limits.min_track => {
                    notes.push(format!("below the {:.3} mm minimum track", m(limits.min_track) * 1e3))
                }
                _ => {}
            }
            if target.diff.is_some() {
                match gap {
                    None if width.is_some() => notes.push("no gap meets the differential target".into()),
                    Some(s) if s < limits.min_clearance => {
                        notes.push(format!("gap below the {:.3} mm minimum clearance", m(limits.min_clearance) * 1e3))
                    }
                    _ => {}
                }
            }
            ControlledLayer {
                layer,
                name,
                kind: g.kind(),
                width,
                gap,
                z,
                z_diff,
                delay_ps_per_mm: width.map(|w| g.delay_ps_per_mm(m(w))),
                note: (!notes.is_empty()).then(|| notes.join("; ")),
            }
        })
        .collect()
}

fn solve(g: &TraceGeometry, t: &ImpedanceTarget, limits: &FabLimits) -> (Option<Coord>, Option<Coord>) {
    let m = |v: Coord| v as f64 * 1e-9;
    let Some(d) = t.diff else {
        return (t.single.and_then(|z| g.width_for(z)).map(round_um), None);
    };
    if let Some(s) = t.gap {
        return (g.width_for_diff(d, m(s)).map(round_um), Some(s));
    }
    if let Some(z) = t.single {
        let w = g.width_for(z).map(round_um);
        let s = w.and_then(|w| g.gap_for_diff(m(w), d)).map(round_um);
        return (w, s);
    }
    // Gap = width (at least the minimum clearance): iterate to the fixed point.
    let mut s = limits.min_clearance.max(UM * 50);
    let mut w = None;
    for _ in 0..40 {
        w = g.width_for_diff(d, m(s));
        let Some(wv) = w else { break };
        let next = wv.max(limits.min_clearance);
        if (next - s).abs() < UM / 10 {
            s = next;
            break;
        }
        s = next;
    }
    match w {
        Some(_) => {
            let s = round_um(s);
            (g.width_for_diff(d, m(s)).map(round_um), Some(s))
        }
        None => (None, None),
    }
}

/// The per-layer rules a class with an impedance target compiles to: on each layer with a
/// solved width, `track_width` (and for a pair `diff_width` and `diff_gap`) on
/// `:net-class(<class>):layer(<layer>)`.
pub fn layer_rule_decls(layers: &[ControlledLayer]) -> Vec<(String, Vec<(String, RuleValue)>)> {
    layers
        .iter()
        .filter_map(|l| {
            let w = l.width?;
            let mut decls = vec![(props::TRACK_WIDTH.to_string(), RuleValue::Len(w))];
            if let Some(s) = l.gap {
                decls.push((props::DIFF_WIDTH.to_string(), RuleValue::Len(w)));
                decls.push((props::DIFF_GAP.to_string(), RuleValue::Len(s)));
            }
            Some((l.name.clone(), decls))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::mm;

    fn four_layer() -> Stackup {
        Stackup::standard(4, mm(1.6), 1.0, 0.5)
    }

    #[test]
    fn a_single_ended_target_solves_a_width_per_layer() {
        let t = ImpedanceTarget { single: Some(50.0), ..Default::default() };
        let layers = controlled_layers(&four_layer(), &t, &FabLimits::default());
        assert_eq!(layers.len(), 4);
        for l in &layers {
            let z = l.z.unwrap();
            assert!((z - 50.0).abs() < 0.5, "{} {z}", l.name);
            assert_eq!(l.gap, None);
        }
        // Outer layers are microstrip, inner stripline; the stripline is narrower for 50 Ω.
        assert_eq!((layers[0].kind, layers[1].kind), ("microstrip", "stripline"));
        assert!(layers[1].width.unwrap() < layers[0].width.unwrap());
    }

    #[test]
    fn a_differential_target_solves_width_and_gap() {
        let limits = FabLimits::default();
        // Gap = width.
        let t = ImpedanceTarget { diff: Some(90.0), ..Default::default() };
        for l in controlled_layers(&four_layer(), &t, &limits) {
            let (w, s) = (l.width.unwrap(), l.gap.unwrap());
            assert!((l.z_diff.unwrap() - 90.0).abs() < 1.0, "{} {:?}", l.name, l.z_diff);
            assert!(s == w.max(limits.min_clearance) || (s - w.max(limits.min_clearance)).abs() <= 2 * UM, "{} {w} {s}", l.name);
        }
        // A written gap is kept.
        let t = ImpedanceTarget { diff: Some(100.0), gap: Some(mm(0.2)), ..Default::default() };
        let l = &controlled_layers(&four_layer(), &t, &limits)[0];
        assert_eq!(l.gap, Some(mm(0.2)));
        assert!((l.z_diff.unwrap() - 100.0).abs() < 1.0);
        // Both targets: the width meets 50 Ω, the gap 90 Ω.
        let t = ImpedanceTarget { single: Some(50.0), diff: Some(90.0), gap: None };
        let l = &controlled_layers(&four_layer(), &t, &limits)[1];
        assert!((l.z.unwrap() - 50.0).abs() < 0.5 && (l.z_diff.unwrap() - 90.0).abs() < 1.0, "{l:?}");
    }

    #[test]
    fn a_two_layer_board_reports_widths_below_its_limits() {
        // 1.6 mm two-layer: 90 Ω differential needs a wide track; 50 Ω single-ended about 3 mm.
        let s = Stackup::standard(2, mm(1.6), 1.0, 1.0);
        let t = ImpedanceTarget { single: Some(50.0), ..Default::default() };
        let w = controlled_layers(&s, &t, &FabLimits::default())[0].width.unwrap();
        assert!((mm(2.7)..mm(3.2)).contains(&w), "{w}");
        // An unreachable target says so.
        let t = ImpedanceTarget { single: Some(5.0), ..Default::default() };
        let l = &controlled_layers(&s, &t, &FabLimits::default())[0];
        assert_eq!(l.width, None);
        assert!(l.note.as_deref().unwrap().contains("no width"));
    }

    #[test]
    fn targets_read_from_class_parameters() {
        let decls = vec![
            ("diff_impedance".to_string(), RuleValue::Int(90)),
            ("clearance".to_string(), RuleValue::Len(mm(0.15))),
        ];
        let t = ImpedanceTarget::of(&decls).unwrap();
        assert_eq!((t.single, t.diff, t.gap), (None, Some(90.0), None));
        assert_eq!(ImpedanceTarget::of(&decls[1..]), None);
    }
}
