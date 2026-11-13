//! Board projection rows for high-speed design: each impedance class's solved width and gap
//! per layer, each matched-length group's members against its target, and each via stub.

use crate::board_rules::{BoardGraph, BoardRules};
use crate::high_speed::{controlled_layers, groups, ImpedanceTarget};
use crate::layout::Board;
use serde::{Deserialize, Serialize};

fn round(v: f64, places: i32) -> f64 {
    let k = 10f64.powi(places);
    (v * k).round() / k
}

/// A net class with an impedance target and what it solves to on each copper layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlledClassRow {
    pub class: String,
    /// Single-ended target, Ω.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impedance: Option<f64>,
    /// Differential target, Ω.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff_impedance: Option<f64>,
    pub layers: Vec<ControlledLayerRow>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlledLayerRow {
    pub layer: String,
    /// `microstrip`, `stripline` or `none`.
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width_mm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gap_mm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impedance: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff_impedance: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delay_ps_per_mm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Every net class with an impedance target, with its widths per layer.
pub fn controlled_rows(board: &Board) -> Vec<ControlledClassRow> {
    board
        .net_class_rules
        .iter()
        .filter_map(|(class, c)| {
            let t = ImpedanceTarget::of(&c.decls)?;
            let layers = controlled_layers(&board.stackup, &t, &board.limits)
                .into_iter()
                .map(|l| ControlledLayerRow {
                    layer: l.name,
                    kind: l.kind.to_owned(),
                    width_mm: l.width.map(|w| round(w as f64 * 1e-6, 3)),
                    gap_mm: l.gap.map(|s| round(s as f64 * 1e-6, 3)),
                    impedance: l.z.map(|z| round(z, 1)),
                    diff_impedance: l.z_diff.map(|z| round(z, 1)),
                    delay_ps_per_mm: l.delay_ps_per_mm.map(|d| round(d, 2)),
                    note: l.note,
                })
                .collect();
            Some(ControlledClassRow {
                class: class.clone(),
                impedance: t.single,
                diff_impedance: t.diff,
                layers,
            })
        })
        .collect()
}

/// A matched-length group: its target and tolerance in its unit, and each member.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LengthGroupRow {
    pub name: String,
    /// `mm` or `ps`.
    pub unit: String,
    pub target: f64,
    /// Whether the target is written (else it is the longest member's).
    pub written: bool,
    pub tolerance: f64,
    pub members: Vec<LengthMemberRow>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LengthMemberRow {
    pub net: String,
    pub length_mm: f64,
    pub delay_ps: f64,
    /// In the group's unit.
    pub value: f64,
    pub deviation: f64,
    /// `matched`, `short`, `long` or `unrouted`.
    pub status: String,
    /// Series parts the member's signal runs through.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub through: Vec<String>,
}

/// The board's matched-length groups.
pub fn group_rows(board: &Board, rules: &BoardRules, g: &BoardGraph) -> Vec<LengthGroupRow> {
    let timing = crate::high_speed::delay::net_timing(board, rules, g);
    groups::groups(board, rules, g, &timing)
        .into_iter()
        .map(|grp| LengthGroupRow {
            unit: grp.basis.unit().to_owned(),
            target: round(grp.target, 3),
            written: grp.written,
            tolerance: round(grp.tolerance, 3),
            members: grp
                .members
                .into_iter()
                .map(|m| LengthMemberRow {
                    net: m.name,
                    length_mm: round(m.timing.length * 1e-6, 3),
                    delay_ps: round(m.timing.delay, 2),
                    value: round(m.value, 3),
                    deviation: round(m.deviation, 3),
                    status: m.status.word().to_owned(),
                    through: m.through,
                })
                .collect(),
            name: grp.name,
        })
        .collect()
}

/// A via with a stub past the layers its net connects on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ViaStubRow {
    /// The via's item id.
    pub via: u64,
    pub net: String,
    pub x_mm: f64,
    pub y_mm: f64,
    /// The drilled copper span, `F.Cu-B.Cu`.
    pub span: String,
    /// The outermost connected layers, `F.Cu-In1.Cu`.
    pub connected: String,
    /// The longest stub left, after any back-drill.
    pub stub_mm: f64,
    /// The longest stub drilled, before back-drill.
    pub drilled_mm: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_mm: Option<f64>,
    /// The delay the stubs add to the net, ps.
    pub delay_ps: f64,
    /// `ok`, `over` (past `stub_max`) or `none` (no limit).
    pub status: String,
    pub backdrills: Vec<BackdrillRow>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackdrillRow {
    /// `top` or `bottom`.
    pub face: String,
    pub diameter_mm: f64,
    pub depth_mm: f64,
    pub must_not_cut: String,
    pub residual_mm: f64,
}

/// The board's via stubs that matter (see the filter), longest first.
pub fn stub_rows(board: &Board, rules: &BoardRules, g: &BoardGraph) -> Vec<ViaStubRow> {
    let names = board.stackup.copper_names();
    let name =
        |l: crate::layer_set::CopperIndex| names.get(l as usize).cloned().unwrap_or_default();
    let mm = |v: crate::geom::Coord| round(v as f64 * 1e-6, 3);
    let mut rows: Vec<ViaStubRow> = crate::high_speed::stubs::via_stubs(board, rules, g)
        .into_iter()
        // Stubs that matter: a net with a limit, a back-drill or an impedance target. A ground
        // or supply via's stub into its plane is listed nowhere.
        .filter(|st| {
            let net = crate::board_rules::BItem::Net(st.net.0);
            st.max.is_some()
                || st.stubs.iter().any(|s| s.backdrill.is_some())
                || rules
                    .value(g, net, crate::board_rules::props::IMPEDANCE)
                    .is_some()
                || rules
                    .value(g, net, crate::board_rules::props::DIFF_IMPEDANCE)
                    .is_some()
        })
        .filter_map(|st| {
            let v = board.vias.get(&st.via)?;
            Some(ViaStubRow {
                via: st.via.0,
                net: board.net_name(st.net).to_owned(),
                x_mm: mm(v.at.x),
                y_mm: mm(v.at.y),
                span: format!("{}-{}", name(v.from), name(v.to)),
                connected: format!("{}-{}", name(st.connected.0), name(st.connected.1)),
                stub_mm: mm(st.length),
                drilled_mm: mm(st.stubs.iter().map(|s| s.drilled).max().unwrap_or(0)),
                max_mm: st.max.map(mm),
                delay_ps: round(st.delay, 2),
                status: match st.max {
                    Some(m) if st.length > m => "over",
                    Some(_) => "ok",
                    None => "none",
                }
                .to_owned(),
                backdrills: st
                    .stubs
                    .iter()
                    .filter_map(|s| s.backdrill.as_ref())
                    .map(|b| BackdrillRow {
                        face: b.face.word().to_owned(),
                        diameter_mm: mm(b.diameter),
                        depth_mm: mm(b.depth),
                        must_not_cut: name(b.must_not_cut),
                        residual_mm: mm(b.residual),
                    })
                    .collect(),
            })
        })
        .collect();
    rows.sort_by(|a, b| {
        b.stub_mm
            .total_cmp(&a.stub_mm)
            .then(a.net.cmp(&b.net))
            .then(a.via.cmp(&b.via))
    });
    rows
}
