//! Transmission-line cross-sections for the board study (claim PCB-42).
//!
//! Each controlled-impedance line on the board, a net or a declared pair whose class has an
//! impedance target, gets one cross-section per copper layer and width (and gap) it is routed
//! at. A cross-section holds what a 2D field solver needs: the conductors' width, gap and
//! thickness, the etch factor, the stack between the reference planes with each layer's
//! thickness and Dk, the solder mask over an outer layer, and the closed-form values to
//! compare with. FEA builds the 2D geometry from it and solves Z0, Zdiff, delay and the
//! per-length L and C; the layer-stack view shows the solved value beside the closed form.
//!
//! The region of a cross-section is `pcb/<board>/line/<net>/<layer>` (a pair: `<p>+<n>`), with
//! its canonical ID from [`crate::fea::canonical_id`], so a study names it like any other
//! board region.
//!
//! RF structures (coplanar waveguide with ground, a fenced line) are one more `kind` with
//! their side grounds in `coplanar`; the dielectric stack and the planes stay the same.

use super::ImpedanceTarget;
use crate::board_rules::props;
use crate::fea::{canonical_id, segment};
use crate::geom::Coord;
use crate::ids::Uuid;
use crate::impedance::TraceGeometry;
use crate::layer_set::CopperIndex;
use crate::layout::{copper_name, Board, NetId, StackKind, Stackup};
use std::collections::BTreeMap;

/// One stack row of a cross-section, top to bottom: m and Dk.
#[derive(Clone, Debug, PartialEq)]
pub struct SectionLayer {
    pub name: String,
    /// `copper`, `dielectric` or `mask`.
    pub kind: &'static str,
    /// Depth of the row's top below the board's top surface, m.
    pub z_top: f64,
    pub thickness: f64,
    pub dk: f64,
    pub df: f64,
}

/// A controlled-impedance line's cross-section.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossSection {
    pub region: String,
    pub id: Uuid,
    /// The net, or a pair's two nets (P first).
    pub nets: Vec<String>,
    pub class: String,
    /// The signal layer (`F.Cu`, `In1.Cu`, …).
    pub layer: String,
    /// `microstrip` or `stripline` (RF kinds such as coplanar waveguide join here).
    pub kind: &'static str,
    /// Bottom width of each conductor, m.
    pub width: f64,
    /// Edge-to-edge gap of a pair, m.
    pub gap: Option<f64>,
    /// Copper thickness, m.
    pub thickness: f64,
    /// Etch factor (thickness over the side's undercut); `None` for a rectangular conductor.
    pub etch_factor: Option<f64>,
    /// The reference planes: copper layers above and below (`None` for the open side of a
    /// microstrip).
    pub plane_above: Option<String>,
    pub plane_below: Option<String>,
    /// The stack from the plane above (or the mask over an outer layer) to the plane below.
    pub stack: Vec<SectionLayer>,
    /// Closed-form single-ended and differential impedance, Ω, and delay, ps/mm.
    pub z0: f64,
    pub z_diff: Option<f64>,
    pub delay_ps_per_mm: f64,
    /// Routed length on this layer at this width, m.
    pub length: f64,
}

fn copper_rows(stackup: &Stackup) -> Vec<usize> {
    stackup
        .layers
        .iter()
        .enumerate()
        .filter(|(_, l)| matches!(l.kind, StackKind::Copper { .. }))
        .map(|(i, _)| i)
        .collect()
}

/// The rows of `stackup` a line on copper `layer` sees: from the plane above (or the mask over
/// the top) to the plane below (or the mask under the bottom).
fn section_stack(
    stackup: &Stackup,
    layer: CopperIndex,
) -> (Vec<SectionLayer>, Option<String>, Option<String>) {
    let copper = copper_rows(stackup);
    let n = copper.len() as CopperIndex;
    let at = copper[layer as usize];
    let above = layer.checked_sub(1).map(|i| copper[i as usize]);
    let below = copper.get(layer as usize + 1).copied();
    let start = match above {
        Some(a) => a,
        None => stackup.layers[..at]
            .iter()
            .rposition(|l| l.kind == StackKind::SolderMask)
            .unwrap_or(at),
    };
    let end = match below {
        Some(b) => b,
        None => stackup.layers[at..]
            .iter()
            .position(|l| l.kind == StackKind::SolderMask)
            .map_or(at, |k| at + k),
    };
    let mut z = 0.0;
    let mut rows = Vec::new();
    for (i, l) in stackup.layers.iter().enumerate() {
        let t = l.thickness as f64 * 1e-9;
        if i >= start && i <= end {
            let kind = match l.kind {
                StackKind::Copper { .. } => "copper",
                StackKind::SolderMask => "mask",
                _ => "dielectric",
            };
            if matches!(
                l.kind,
                StackKind::Copper { .. } | StackKind::Dielectric { .. } | StackKind::SolderMask
            ) {
                rows.push(SectionLayer {
                    name: l.name.clone(),
                    kind,
                    z_top: z,
                    thickness: t,
                    dk: l.material.dk,
                    df: l.material.df,
                });
            }
        }
        z += t;
    }
    let name = |i: Option<CopperIndex>| i.map(|i| copper_name(i, n));
    (
        rows,
        name(layer.checked_sub(1)),
        name((layer + 1 < n).then_some(layer + 1)),
    )
}

/// Every controlled-impedance line's cross-sections. `pairs` are the declared pairs' nets.
pub fn cross_sections(board: &Board, pairs: &[(NetId, NetId)]) -> Vec<CrossSection> {
    let n = board.copper_count();
    let targets: BTreeMap<&str, (ImpedanceTarget, Option<f64>)> = board
        .net_class_rules
        .iter()
        .filter_map(|(k, c)| {
            let etch = c
                .decls
                .iter()
                .rev()
                .find(|(p, _)| p == props::ETCH_FACTOR)
                .and_then(|(_, v)| v.num());
            Some((k.as_str(), (ImpedanceTarget::of(&c.decls)?, etch)))
        })
        .collect();
    let class_of = |net: NetId| {
        board.nets[net.0 as usize]
            .class
            .as_deref()
            .filter(|c| targets.contains_key(c))
    };
    let partner: BTreeMap<NetId, NetId> =
        pairs.iter().flat_map(|&(p, q)| [(p, q), (q, p)]).collect();
    // Routed runs: (lead net, layer, width) → length; pairs are led by P.
    let mut runs: BTreeMap<(NetId, CopperIndex, Coord), f64> = BTreeMap::new();
    for t in board.tracks.values() {
        let Some(net) = t.net else { continue };
        if class_of(net).is_none() {
            continue;
        }
        let lead = match partner.get(&net) {
            Some(&q) if pairs.iter().any(|&(_, nn)| nn == net) => q,
            _ => net,
        };
        // A pair's length is its P conductor's.
        *runs.entry((lead, t.layer, t.width)).or_default() +=
            if lead == net { t.length() } else { 0.0 };
    }
    let mut out = Vec::new();
    for ((lead, layer, width), length) in runs {
        let class = class_of(lead).expect("a controlled class");
        let (target, etch) = targets[class];
        let Some(geometry) = TraceGeometry::of(&board.stackup, layer) else {
            continue;
        };
        let pair = partner.get(&lead).copied();
        let w = width as f64 * 1e-9;
        // The pair's gap: the class's solved (or written) gap on this layer.
        let gap = pair.and_then(|_| {
            super::controlled_layers(&board.stackup, &target, &board.limits)
                .get(layer as usize)
                .and_then(|l| l.gap)
        });
        let (stack, plane_above, plane_below) = section_stack(&board.stackup, layer);
        let thickness = stack
            .iter()
            .find(|r| {
                r.kind == "copper"
                    && r.name
                        == board.stackup.layers[copper_rows(&board.stackup)[layer as usize]].name
            })
            .map_or(35e-6, |r| r.thickness);
        let mut nets = vec![board.net_name(lead).to_owned()];
        if let Some(q) = pair {
            nets.push(board.net_name(q).to_owned());
        }
        let line = nets
            .iter()
            .map(|s| segment(s))
            .collect::<Vec<_>>()
            .join("+");
        let region = format!(
            "pcb/{}/line/{line}/{}",
            segment(&board.name),
            copper_name(layer, n)
        );
        out.push(CrossSection {
            id: canonical_id(&region),
            region,
            nets,
            class: class.to_owned(),
            layer: copper_name(layer, n),
            kind: geometry.kind(),
            width: w,
            gap: gap.map(|s| s as f64 * 1e-9),
            thickness,
            etch_factor: etch,
            plane_above,
            plane_below,
            stack,
            z0: geometry.single_ended(w),
            z_diff: gap.map(|s| geometry.differential(w, s as f64 * 1e-9)),
            delay_ps_per_mm: geometry.delay_ps_per_mm(w),
            length: length * 1e-9,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board_rules::RuleValue;
    use crate::geom::{mm, Point, Rect};
    use crate::layout::NetClassRules;

    #[test]
    fn a_pair_on_an_outer_layer_has_its_mask_dielectric_and_plane() {
        let outline = crate::zone::rect_outline(Rect::new(Point::ORIGIN, Point::mm(30.0, 20.0)));
        let mut b = Board::new("hs", outline, Stackup::standard(4, mm(1.6), 1.0, 0.5));
        let p = b.add_net("D_P", Some("usb"));
        let q = b.add_net("D_N", Some("usb"));
        b.net_class_rules.insert(
            "usb".into(),
            NetClassRules {
                decls: vec![
                    ("diff_impedance".into(), RuleValue::Int(90)),
                    ("etch_factor".into(), RuleValue::Num(2.5)),
                ],
                ..Default::default()
            },
        );
        b.add_track(
            Some(p),
            0,
            mm(0.16),
            Point::mm(2.0, 5.0),
            Point::mm(22.0, 5.0),
        );
        b.add_track(
            Some(q),
            0,
            mm(0.16),
            Point::mm(2.0, 5.32),
            Point::mm(22.0, 5.32),
        );
        let xs = cross_sections(&b, &[(p, q)]);
        assert_eq!(xs.len(), 1, "{xs:#?}");
        let x = &xs[0];
        assert_eq!(x.nets, vec!["D_P", "D_N"]);
        assert_eq!(
            (
                x.layer.as_str(),
                x.kind,
                x.plane_above.as_deref(),
                x.plane_below.as_deref()
            ),
            ("F.Cu", "microstrip", None, Some("In1.Cu"))
        );
        assert_eq!(x.region, "pcb/hs/line/D_P+D_N/F.Cu");
        assert_eq!(x.etch_factor, Some(2.5));
        // Mask, signal copper, prepreg, the plane.
        let kinds: Vec<_> = x.stack.iter().map(|r| r.kind).collect();
        assert_eq!(kinds, vec!["mask", "copper", "dielectric", "copper"]);
        assert!((x.length - 0.020).abs() < 1e-9 && x.gap.is_some() && x.z_diff.is_some());
    }
}
