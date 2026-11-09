//! The ECO projection: pending changes by group, each with its cause, impact and placement,
//! and what the last Apply did.

use super::*;
use crate::eco::{Direction, EcoPlan, Origin};
use crate::layout::Board;
use serde::{Deserialize, Serialize};

/// What the last Apply did and what DRC and ERC found after it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EcoApplied {
    /// `forward` or `back`.
    pub direction: String,
    pub applied: u32,
    pub removed_tracks: u32,
    pub removed_vias: u32,
    pub unnetted: u32,
    pub drc_errors: u32,
    pub drc_warnings: u32,
    pub erc_errors: u32,
    pub erc_warnings: u32,
}

impl EcoApplied {
    pub fn forward(report: &crate::eco::ApplyReport, checks: &crate::eco::CheckSummary) -> Self {
        EcoApplied {
            direction: "forward".into(),
            applied: report.applied.len() as u32,
            removed_tracks: report.removed_tracks.len() as u32,
            removed_vias: report.removed_vias.len() as u32,
            unnetted: report.unnetted.len() as u32,
            drc_errors: checks.drc_errors as u32,
            drc_warnings: checks.drc_warnings as u32,
            erc_errors: checks.erc_errors as u32,
            erc_warnings: checks.erc_warnings as u32,
        }
    }
}

/// One direction of the ECO: its changes grouped by type, each validated against the whole list.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EcoIndex {
    /// `forward` (schematic → board) or `back` (board → schematic).
    pub direction: String,
    pub total: u32,
    pub errors: u32,
    pub warnings: u32,
    pub groups: Vec<EcoGroupRow>,
    /// The difference types the board's saved comparator settings ignore (`rooms`, `values`):
    /// the review's starting choices. Rows any of them hides ([`EcoRow::ignored_by`]) are not shown, and a
    /// compare with those settings does not produce them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignored: Vec<String>,
    /// Parameters the saved settings never compare (`datasheet`): the review's starting list.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignored_parameters: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EcoGroupRow {
    /// `components`, `footprints`, `parameters`, `nets`, `connections`, `classes`, `rooms`, `rules`.
    pub group: String,
    pub label: String,
    pub count: u32,
    pub rows: Vec<EcoRow>,
}

/// One ECO line: what changes, from what to what, why, its validation and what it touches.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EcoRow {
    /// Stable across recompiles (`add_component:/ch[2]/U`); the review's selection key.
    pub id: String,
    /// `add_component`, `rename_net`, `change_pin_net`…
    pub kind: String,
    /// `Add component`, `Rename net`…
    pub label: String,
    /// A designator, `designator.pad`, a net or class name: never a raw path.
    pub item: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    pub cause: EcoCause,
    /// `ok`, `warning` or `error`, validated with every line selected.
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Ids of lines this one needs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conflict: Option<String>,
    /// What validation finds whatever else is selected (conflict, library, pads, board edits);
    /// requirements and designator claims depend on the selection and are checked per selection.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<EcoIssue>,
    pub impact: EcoImpact,
    /// The difference types whose ignore setting hides this row (any one of them does).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignored_by: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EcoIssue {
    /// `warning` or `error`.
    pub status: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EcoCause {
    /// `ch[2]`, `main`.
    pub sheet: String,
    /// The sheet instance path, for the hierarchy node (`/ch[2]`).
    pub sheet_path: String,
    /// `io.circuit:120..168`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub reason: String,
    /// `schematic`, `board` or `agent`.
    pub origin: String,
    /// The agent change set's title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EcoImpact {
    /// Canvas region ids to highlight (`component:R1`, `pad:R1.2`, `net:GND`, `track:12`).
    pub regions: Vec<String>,
    pub tracks: u32,
    pub vias: u32,
    pub pours: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<EcoPlacement>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EcoPlacement {
    pub x_mm: f64,
    pub y_mm: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub room: Option<String>,
    /// Off the board, in the staging area.
    pub staged: bool,
}

/// The ECO projection of a plan, validated with every line selected.
pub fn eco_index(plan: &EcoPlan, board: &Board) -> EcoIndex {
    let checks = crate::eco::validate(plan, board, &plan.ids());
    let mut groups: Vec<EcoGroupRow> = Vec::new();
    let (mut errors, mut warnings) = (0, 0);
    for c in &plan.changes {
        let g = c.kind.group();
        if groups.last().map(|x| x.group.as_str()) != Some(g.as_str()) {
            groups.push(EcoGroupRow {
                group: g.as_str().to_owned(),
                label: g.label().to_owned(),
                count: 0,
                rows: Vec::new(),
            });
        }
        let check = &checks[&c.id];
        match check.status {
            crate::eco::Status::Error => errors += 1,
            crate::eco::Status::Warning => warnings += 1,
            crate::eco::Status::Ok => {}
        }
        let (origin, agent) = match &c.origin {
            Origin::Schematic => ("schematic", None),
            Origin::Board => ("board", None),
            Origin::Agent { title } => ("agent", Some(title.clone())),
        };
        let group = groups.last_mut().unwrap();
        group.count += 1;
        group.rows.push(EcoRow {
            id: c.id.clone(),
            kind: c.kind.as_str().to_owned(),
            label: c.kind.label().to_owned(),
            item: c.item.clone(),
            from: c.from.clone(),
            to: c.to.clone(),
            cause: EcoCause {
                sheet: c.cause.sheet.clone(),
                sheet_path: c.cause.sheet_path.clone(),
                source: c.cause.source.clone(),
                reason: c.cause.reason.clone(),
                origin: origin.to_owned(),
                agent,
            },
            status: check.status.as_str().to_owned(),
            reason: check.reason(),
            requires: c.requires.clone(),
            conflict: c.conflict.clone(),
            issues: c
                .conflict
                .iter()
                .map(|r| EcoIssue {
                    status: "error".into(),
                    reason: format!("conflict: {r}"),
                })
                .chain(c.issues.iter().map(|(st, r)| EcoIssue {
                    status: st.as_str().into(),
                    reason: r.clone(),
                }))
                .collect(),
            impact: EcoImpact {
                regions: c.impact.regions(),
                tracks: c.impact.tracks.len() as u32,
                vias: c.impact.vias.len() as u32,
                pours: c.impact.zones.len() as u32,
                placement: c.impact.placement.as_ref().map(|p| EcoPlacement {
                    x_mm: mm(p.at.x),
                    y_mm: mm(p.at.y),
                    room: p.room.clone(),
                    staged: p.staged,
                }),
            },
            ignored_by: c
                .kind
                .ignored_by()
                .iter()
                .map(|d| d.as_str().to_owned())
                .collect(),
        });
    }
    EcoIndex {
        direction: match plan.direction {
            Direction::Forward => "forward",
            Direction::Back => "back",
        }
        .to_owned(),
        total: plan.changes.len() as u32,
        errors,
        warnings,
        groups,
        ignored: board
            .sync
            .options
            .ignored()
            .iter()
            .map(|d| d.as_str().to_owned())
            .collect(),
        ignored_parameters: board
            .sync
            .options
            .ignored_parameters
            .iter()
            .cloned()
            .collect(),
    }
}

/// Adds both ECO directions to the projection.
pub fn add_eco(index: &mut CircuitIndex, forward: &EcoPlan, back: &EcoPlan, board: &Board) {
    index.eco = eco_index(forward, board);
    index.back_annotation = eco_index(back, board);
}
