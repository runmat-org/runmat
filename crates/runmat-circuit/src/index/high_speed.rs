//! Board projection rows for high-speed design: each impedance class's solved width and gap
//! per layer, and each matched-length group's members against its target.

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
            Some(ControlledClassRow { class: class.clone(), impedance: t.single, diff_impedance: t.diff, layers })
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
    let timing = crate::high_speed::delay::net_timing(board);
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
