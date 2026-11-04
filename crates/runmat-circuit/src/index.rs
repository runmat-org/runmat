//! The circuit JSON projection: a deterministic view of the elaborated design (hierarchy,
//! components, nets, net classes, BOM), its checks (ERC, design review, DRC), pending ECO
//! lines and, when there is a board, its layers and copper with the rule each value came from.
//! The Desktop's Outline tree and Properties panel read this; so can agents and the server
//! index. Every list is sorted, so identical inputs give identical bytes.

use crate::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use crate::drc::Drc;
use crate::elab::{FitState, Netlist, Severity};
use crate::eco::{Direction, EcoPlan, Origin};
use crate::ir::{Design, Param, ParamValue, Tolerance};
use crate::layout::{Board, NetId, PadConnect, Side, StackKind};
use crate::library::PadShape;
use crate::library::Pool;
use crate::review::ReviewFinding;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Bumped on any change to the projection's shape.
pub const CIRCUIT_INDEX_FORMAT_VERSION: u32 = 3;

/// The parameters that carry a component's shown value, in order of preference.
const VALUE_PARAMS: [&str; 4] = ["resistance", "capacitance", "inductance", "frequency"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitIndex {
    pub format_version: u32,
    pub design: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    pub variants: Vec<String>,
    pub hierarchy: Vec<SheetRow>,
    pub components: Vec<ComponentRow>,
    pub nets: Vec<NetRow>,
    pub net_classes: Vec<NetClassRow>,
    pub bom: Vec<BomRow>,
    /// Schematic → board changes the board hasn't taken (Update board), grouped by type.
    pub eco: EcoIndex,
    /// Board → schematic changes (back-annotation), grouped the same way.
    #[serde(default)]
    pub back_annotation: EcoIndex,
    /// The last applied ECO and the checks that ran after it, for the status popover.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_eco: Option<EcoApplied>,
    pub erc: Vec<CheckRow>,
    pub review: Vec<CheckRow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board: Option<BoardIndex>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetRow {
    /// Instance path (`main/pwr`), root first.
    pub path: String,
    pub sheet: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComponentRow {
    pub designator: String,
    pub path: String,
    pub sheet: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manufacturer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The value a schematic and BOM show (`10 kΩ ±1%`): the `Value` property, else the first of
    /// resistance, capacitance, inductance and frequency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// Parameter name → display value (`10 kΩ ±1 %`).
    pub params: BTreeMap<String, String>,
    pub fitting: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_w: Option<f64>,
    pub pins: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub renamed_from: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NetRow {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    pub global: bool,
    /// `designator.pin`, sorted.
    pub pins: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NetClassRow {
    pub name: String,
    pub params: BTreeMap<String, String>,
    pub nets: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BomRow {
    pub part: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manufacturer: Option<String>,
    pub designators: Vec<String>,
    pub quantity: u32,
}

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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckRow {
    pub code: String,
    pub severity: String,
    pub message: String,
    pub items: Vec<String>,
    /// A typed repair the editor can offer as one click (design review fixes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoardIndex {
    pub name: String,
    pub layers: Vec<LayerRow>,
    pub footprints: Vec<FootprintRow>,
    pub pads: Vec<PadRow>,
    pub tracks: Vec<CopperRow>,
    pub vias: Vec<CopperRow>,
    pub zones: Vec<CopperRow>,
    /// The design rules in source order: what each selects and sets, and where it is written.
    pub rules: Vec<RuleRow>,
    pub drc: Vec<CheckRow>,
}

/// One design rule (`rule power-width { track:net-class(power) { track_width: 0.4mm } }`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuleRow {
    pub id: String,
    pub selector: String,
    /// The second selector of a pair rule (clearance between two kinds of copper).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub other: Option<String>,
    pub priority: i32,
    /// Property → value as written (`0.40 mm`).
    pub values: BTreeMap<String, String>,
    /// `io.circuit:120..168`.
    pub source: String,
}

/// A placed pad, with its rule-driven clearance and where it came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PadRow {
    /// `R1.2`: footprint reference and pad name.
    pub id: String,
    pub component: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub net: Option<String>,
    pub layer: String,
    /// `circle`, `rect`, `round_rect`, `oval`, `trapezoid` or `custom`.
    pub shape: String,
    pub values: BTreeMap<String, f64>,
    pub explain: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerRow {
    pub name: String,
    /// `copper`, `dielectric`, `soldermask`, `silkscreen` or `paste`.
    pub kind: String,
    pub thickness_mm: f64,
    /// Copper weight, oz/ft² (informative; the thickness is authoritative).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_oz: Option<f64>,
    /// Dielectric: a core rather than prepreg.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub core: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FootprintRow {
    pub reference: String,
    pub value: String,
    pub x_mm: f64,
    pub y_mm: f64,
    pub rotation: f64,
    pub side: String,
    pub locked: bool,
}

/// A track, via or pour, with its rule-driven values and where each came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CopperRow {
    pub id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub net: Option<String>,
    pub layer: String,
    /// Property → value in mm.
    pub values: BTreeMap<String, f64>,
    /// Property → the rule that set its required value (`clearance 0.10 mm from rule power-clear at board.circuit:120..180`).
    pub explain: BTreeMap<String, String>,
    /// Property → a value that isn't a length (a pour's name and pad connection).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub text: BTreeMap<String, String>,
}

fn mm(nm: i64) -> f64 {
    (nm as f64 / 1.0e6 * 1.0e6).round() / 1.0e6
}

fn severity(s: &Severity) -> String {
    match s {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Info => "info",
    }
    .to_owned()
}

/// The value a schematic, BOM and board show: the `Value` property, else the first of
/// resistance, capacitance, inductance and frequency.
pub fn component_value(c: &crate::elab::CompInst) -> Option<String> {
    c.properties.get("Value").cloned().or_else(|| {
        VALUE_PARAMS
            .iter()
            .find_map(|k| c.params.get(*k).map(param_display))
    })
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
    }
}

/// Adds both ECO directions to the projection.
pub fn add_eco(index: &mut CircuitIndex, forward: &EcoPlan, back: &EcoPlan, board: &Board) {
    index.eco = eco_index(forward, board);
    index.back_annotation = eco_index(back, board);
}

/// `10 kΩ ±1%`: the nominal with an SI prefix and its tolerance.
pub fn param_display(p: &Param) -> String {
    match &p.value {
        ParamValue::Text(t) => t.clone(),
        ParamValue::Bool(b) => b.to_string(),
        ParamValue::Quantity { nominal, tol } => {
            let base = quantity_display(nominal.value, nominal.dim);
            match tol {
                Tolerance::Exact => base,
                Tolerance::Relative { minus, plus } if (minus - plus).abs() < 1e-12 => {
                    if *plus == 0.0 {
                        base
                    } else {
                        format!("{base} \u{b1}{}%", trim(plus * 100.0))
                    }
                }
                Tolerance::Relative { minus, plus } => {
                    format!("{base} +{}% -{}%", trim(plus * 100.0), trim(minus * 100.0))
                }
                other => format!("{base} {other:?}"),
            }
        }
    }
}

fn trim(v: f64) -> String {
    let s = format!("{v:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// A value in its unit with an engineering prefix (`100 nF`, `4.7 \u{b5}H`, `16 V`).
pub fn quantity_display(value: f64, dim: runmat_rules::Dim) -> String {
    use runmat_rules::Dim;
    let unit = [
        (Dim::RESISTANCE, "\u{3a9}"),
        (Dim::CAPACITANCE, "F"),
        (Dim::INDUCTANCE, "H"),
        (Dim::VOLTAGE, "V"),
        (Dim::CURRENT, "A"),
        (Dim::POWER, "W"),
        (Dim::FREQUENCY, "Hz"),
        (Dim::TIME, "s"),
        (Dim::LENGTH, "m"),
        (Dim::TEMPERATURE, "K"),
        (Dim::NONE, ""),
    ]
    .iter()
    .find(|(d, _)| *d == dim)
    .map(|(_, u)| *u);
    let Some(unit) = unit else {
        return format!("{value} {dim:?}");
    };
    if value == 0.0 || unit.is_empty() {
        return format!("{} {unit}", trim(value)).trim_end().to_owned();
    }
    let prefixes = [
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "k"),
        (1.0, ""),
        (1e-3, "m"),
        (1e-6, "\u{b5}"),
        (1e-9, "n"),
        (1e-12, "p"),
    ];
    let magnitude = value.abs();
    let (scale, prefix) = prefixes
        .iter()
        .find(|(s, _)| magnitude >= *s * 0.999_999)
        .copied()
        .unwrap_or((1e-12, "p"));
    format!("{} {prefix}{unit}", trim(value / scale))
}

/// Builds the projection of an elaborated design and its schematic checks.
pub fn build_index(
    design: &Design,
    netlist: &Netlist,
    pool: &Pool,
    erc: &[crate::elab::Diagnostic],
    review: &[ReviewFinding],
) -> CircuitIndex {
    let part_of = |c: &crate::elab::CompInst| {
        c.part
            .and_then(|u| pool.parts.values().find(|p| p.uuid == u))
            .map(|p| (p.mpn.clone(), p.manufacturer.clone(), p.description.clone()))
    };
    let mut hierarchy: Vec<SheetRow> = netlist
        .sheets
        .iter()
        .map(|(path, sheet)| {
            let text = path.to_string();
            let parent = text.rsplit_once('/').map(|(p, _)| p.to_owned());
            SheetRow {
                path: text,
                sheet: sheet.clone(),
                parent,
            }
        })
        .collect();
    hierarchy.sort_by(|a, b| a.path.cmp(&b.path));

    let mut components: Vec<ComponentRow> = netlist
        .components
        .iter()
        .map(|c| {
            let part = part_of(c);
            ComponentRow {
                designator: c.designator.clone(),
                path: c.path.to_string(),
                sheet: c.sheet.clone(),
                part: part.as_ref().map(|p| p.0.clone()),
                manufacturer: part.as_ref().map(|p| p.1.clone()).filter(|s| !s.is_empty()),
                description: part.as_ref().map(|p| p.2.clone()).filter(|s| !s.is_empty()),
                value: component_value(c),
                params: c
                    .params
                    .iter()
                    .map(|(k, v)| (k.clone(), param_display(v)))
                    .collect(),
                fitting: match c.fitting {
                    FitState::Fitted => "fitted",
                    FitState::NotFitted => "not_fitted",
                    FitState::Alternate => "alternate",
                }
                .to_owned(),
                power_w: c.power.map(|q| q.value),
                pins: c.pins.len() as u32,
                renamed_from: c.renamed_from.as_ref().map(|p| p.to_string()),
            }
        })
        .collect();
    components.sort_by_key(|c| natural(&c.designator));

    let mut nets: Vec<NetRow> = netlist
        .nets
        .iter()
        .map(|n| {
            let mut pins: Vec<String> = n
                .pins
                .iter()
                .map(|(ci, pi)| {
                    let c = &netlist.components[*ci];
                    format!("{}.{}", c.designator, c.pins[*pi].pin)
                })
                .collect();
            pins.sort_by_key(|p| natural(p));
            NetRow {
                name: n.name.clone(),
                class: n.class.clone(),
                global: n.global,
                pins,
            }
        })
        .collect();
    nets.sort_by(|a, b| a.name.cmp(&b.name));

    let net_classes = design
        .net_classes
        .values()
        .map(|nc| NetClassRow {
            name: nc.name.clone(),
            params: nc
                .params
                .iter()
                .map(|(k, v)| (k.clone(), param_display(v)))
                .collect(),
            nets: nets
                .iter()
                .filter(|n| n.class.as_deref() == Some(nc.name.as_str()))
                .count() as u32,
        })
        .collect();

    let mut bom: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    for c in components.iter().filter(|c| c.fitting != "not_fitted") {
        let key = (
            c.part.clone().unwrap_or_else(|| "(no part)".to_owned()),
            c.manufacturer.clone().unwrap_or_default(),
        );
        bom.entry(key).or_default().push(c.designator.clone());
    }
    let bom = bom
        .into_iter()
        .map(|((part, manufacturer), designators)| BomRow {
            quantity: designators.len() as u32,
            part,
            manufacturer: Some(manufacturer).filter(|m| !m.is_empty()),
            designators,
        })
        .collect();

    let mut erc_rows: Vec<CheckRow> = erc
        .iter()
        .map(|d| {
            let mut items = vec![d.item.clone()];
            items.extend(d.related.iter().cloned());
            CheckRow {
                code: d.code.clone(),
                severity: severity(&d.severity),
                message: d.message.clone(),
                items,
                fix: None,
                source: Some(d.source.to_string()).filter(|s| !s.starts_with(':')),
            }
        })
        .collect();
    erc_rows
        .sort_by(|a, b| (&a.severity, &a.code, &a.items).cmp(&(&b.severity, &b.code, &b.items)));

    let mut review_rows: Vec<CheckRow> = review
        .iter()
        .map(|f| CheckRow {
            code: format!("{}.{}", f.suite, f.rule),
            severity: severity(&f.severity),
            message: f.message.clone(),
            items: f.items.clone(),
            fix: f.fix.as_ref().map(fix_label),
            source: None,
        })
        .collect();
    review_rows.sort_by(|a, b| (&a.code, &a.items).cmp(&(&b.code, &b.items)));

    CircuitIndex {
        format_version: CIRCUIT_INDEX_FORMAT_VERSION,
        design: design.name.clone(),
        variant: netlist.variant.clone(),
        variants: design.variants.keys().cloned().collect(),
        hierarchy,
        components,
        nets,
        net_classes,
        bom,
        eco: EcoIndex::default(),
        back_annotation: EcoIndex::default(),
        last_eco: None,
        erc: erc_rows,
        review: review_rows,
        board: None,
    }
}

/// A one-line description of a review fix, for its one-click button.
pub fn fix_label(edit: &crate::review::ModelEdit) -> String {
    use crate::review::ModelEdit;
    match edit {
        ModelEdit::AddComponent { decl, connect, .. } => {
            let nets: Vec<String> = connect.iter().map(|(pin, _)| pin.clone()).collect();
            let part = match &decl.part {
                crate::ir::PartRef::Part(p) => p.clone(),
                crate::ir::PartRef::Entity { entity, .. } => entity.clone(),
                #[allow(unreachable_patterns)]
                other => format!("{other:?}"),
            };
            format!("Add {part} on pins {}", nets.join(", "))
        }
        ModelEdit::SetParam {
            component,
            key,
            value,
            ..
        } => {
            format!("Set {component} {key} to {}", param_display(value))
        }
        ModelEdit::Batch(edits) => edits.iter().map(fix_label).collect::<Vec<_>>().join("; "),
    }
}

/// Adds the board: layers, footprints, copper with rule explanations, and DRC findings.
pub fn add_board(index: &mut CircuitIndex, board: &Board, rules: &BoardRules, drc: &Drc) {
    let graph = BoardGraph::new(board, &[]);
    let copper = board.stackup.copper_count();
    let net_name = |n: Option<NetId>| n.map(|id| board.nets[id.0 as usize].name.clone());
    let explain = |it: BItem, prop: &str, label: &str, value: Option<f64>| -> Option<String> {
        let (rule, source) = rules.single_origin(&graph, it, prop);
        let value = value.map(|v| format!("{v:.2} mm ")).unwrap_or_default();
        Some(if rule == "default" {
            format!("{label} {value}from the board defaults")
        } else {
            format!("{label} {value}from rule {rule} at {source}")
        })
    };
    let layer_name = |l: u8| crate::layout::copper_name(l, copper);
    let tracks = board
        .tracks
        .values()
        .map(|t| {
            let it = BItem::Track(t.id);
            let required = rules.len(&graph, it, props::TRACK_WIDTH).map(mm);
            let mut values = BTreeMap::new();
            values.insert("width".to_owned(), mm(t.width));
            values.insert(
                "length".to_owned(),
                (t.length() / 1.0e6 * 1.0e4).round() / 1.0e4,
            );
            let mut explain_map = BTreeMap::new();
            if let Some(line) = explain(it, props::TRACK_WIDTH, "width", required) {
                explain_map.insert("width".to_owned(), line);
            }
            CopperRow {
                id: t.id.0,
                net: net_name(t.net),
                layer: layer_name(t.layer),
                values,
                explain: explain_map,
                text: BTreeMap::new(),
            }
        })
        .collect();
    let vias = board
        .vias
        .values()
        .map(|v| {
            let it = BItem::Via(v.id);
            let mut values = BTreeMap::new();
            values.insert("diameter".to_owned(), mm(v.diameter));
            values.insert("drill".to_owned(), mm(v.drill));
            let mut explain_map = BTreeMap::new();
            let required = rules.len(&graph, it, props::VIA_DIAMETER).map(mm);
            if let Some(line) = explain(it, props::VIA_DIAMETER, "diameter", required) {
                explain_map.insert("diameter".to_owned(), line);
            }
            CopperRow {
                id: v.id.0,
                net: net_name(v.net),
                layer: format!("{}-{}", layer_name(v.from), layer_name(v.to)),
                values,
                explain: explain_map,
                text: BTreeMap::new(),
            }
        })
        .collect();
    let zones = board
        .zones
        .values()
        .map(|z| {
            let mut values = BTreeMap::new();
            let required = rules.pair_len(BItem::Zone(z.id), BItem::Edge, props::CLEARANCE);
            values.insert(
                "clearance".to_owned(),
                mm(z.clearance.map_or(required, |c| c.max(required))),
            );
            values.insert("priority".to_owned(), z.priority as f64);
            values.insert("min_width".to_owned(), mm(z.min_width));
            values.insert("thermal_gap".to_owned(), mm(z.thermal_gap));
            values.insert("spoke_width".to_owned(), mm(z.spoke_width));
            let mut text = BTreeMap::new();
            text.insert("name".to_owned(), z.name.clone());
            text.insert(
                "connect".to_owned(),
                match z.connect {
                    PadConnect::Solid => "solid",
                    PadConnect::Thermal => "relief",
                    PadConnect::ThermalThtOnly => "relief_tht",
                    PadConnect::None => "none",
                }
                .to_owned(),
            );
            let (rule, source) =
                rules.pair_origin(BItem::Zone(z.id), BItem::Edge, props::CLEARANCE);
            let mut explain_map = BTreeMap::new();
            explain_map.insert(
                "clearance".to_owned(),
                if rule == "default" {
                    "clearance from the board defaults".to_owned()
                } else {
                    format!("clearance from rule {rule} at {source}")
                },
            );
            CopperRow {
                id: z.id.0,
                net: net_name(z.net),
                layer: z
                    .layers
                    .iter()
                    .map(|l| layer_name(*l))
                    .collect::<Vec<_>>()
                    .join(", "),
                values,
                explain: explain_map,
                text,
            }
        })
        .collect();
    let pads = board
        .footprints
        .values()
        .flat_map(|f| {
            f.pads(copper).into_iter().map(move |p| {
                let it = BItem::Pad(f.id, p.index as u32);
                let mut values = BTreeMap::new();
                values.insert(
                    "clearance".to_owned(),
                    mm(rules.pair_len(it, BItem::Edge, props::CLEARANCE)),
                );
                let (rule, source) = rules.pair_origin(it, BItem::Edge, props::CLEARANCE);
                let mut explain_map = BTreeMap::new();
                explain_map.insert(
                    "clearance".to_owned(),
                    if rule == "default" {
                        "clearance from the board defaults".to_owned()
                    } else {
                        format!("clearance from rule {rule} at {source}")
                    },
                );
                let shape = match f.package.pads.get(p.index).map(|d| &d.shape) {
                    Some(PadShape::Circle { .. }) => "circle",
                    Some(PadShape::Rect { .. }) => "rect",
                    Some(PadShape::RoundRect { .. }) => "round_rect",
                    Some(PadShape::Oval { .. }) => "oval",
                    Some(PadShape::Trapezoid { .. }) => "trapezoid",
                    Some(PadShape::Chamfered { .. }) => "chamfered",
                    _ => "custom",
                };
                PadRow {
                    id: format!("{}.{}", f.reference, p.name),
                    component: f.reference.clone(),
                    net: net_name(p.net),
                    layer: p
                        .layers
                        .iter()
                        .map(|l| layer_name(*l))
                        .collect::<Vec<_>>()
                        .join(", "),
                    shape: shape.to_owned(),
                    values,
                    explain: explain_map,
                }
            })
        })
        .collect();
    let rule_rows = rules
        .specs
        .iter()
        .map(|r| RuleRow {
            id: r.id.clone(),
            selector: r.selector.clone(),
            other: r.other.clone(),
            priority: r.priority,
            values: r
                .decls
                .iter()
                .map(|(k, v)| (k.clone(), rule_value_display(v)))
                .collect(),
            source: r.source.to_string(),
        })
        .collect();
    let footprints = board
        .footprints
        .values()
        .map(|f| FootprintRow {
            reference: f.reference.clone(),
            value: f.value.clone(),
            x_mm: mm(f.at.x),
            y_mm: mm(f.at.y),
            rotation: f.rotation,
            side: match f.side {
                Side::Top => "top",
                Side::Bottom => "bottom",
            }
            .to_owned(),
            locked: f.locked,
        })
        .collect();
    let item_label = |it: &BItem| match it {
        BItem::Edge => "board edge".to_owned(),
        BItem::Net(n) => board
            .nets
            .get(*n as usize)
            .map(|n| n.name.clone())
            .unwrap_or_default(),
        BItem::Footprint(id) => board
            .footprints
            .get(id)
            .map(|f| f.reference.clone())
            .unwrap_or_default(),
        BItem::Pad(id, pad) => format!(
            "{}.{}",
            board
                .footprints
                .get(id)
                .map(|f| f.reference.as_str())
                .unwrap_or("?"),
            pad + 1
        ),
        BItem::Track(id) => format!("track {}", id.0),
        BItem::Via(id) => format!("via {}", id.0),
        BItem::Zone(id) => board
            .zones
            .get(id)
            .map(|z| format!("pour {}", z.name))
            .unwrap_or_default(),
    };
    let drc_rows = drc
        .findings()
        .into_iter()
        .map(|f| CheckRow {
            code: f.code.clone(),
            severity: severity(&f.severity),
            message: f.message.clone(),
            items: f.items.iter().map(item_label).collect(),
            fix: None,
            source: Some(format!("rule {} at {}", f.rule, f.source)),
        })
        .collect();
    index.board = Some(BoardIndex {
        name: board.name.clone(),
        layers: board
            .stackup
            .layers
            .iter()
            .map(|l| {
                let (kind, weight_oz, core) = match l.kind {
                    StackKind::Copper { weight_oz } => ("copper", Some(weight_oz), None),
                    StackKind::Dielectric { core } => ("dielectric", None, Some(core)),
                    StackKind::SolderMask => ("soldermask", None, None),
                    StackKind::Silkscreen => ("silkscreen", None, None),
                    StackKind::Paste => ("paste", None, None),
                };
                LayerRow {
                    name: l.name.clone(),
                    kind: kind.to_owned(),
                    thickness_mm: mm(l.thickness),
                    weight_oz,
                    core,
                }
            })
            .collect(),
        footprints,
        pads,
        tracks,
        vias,
        zones,
        rules: rule_rows,
        drc: drc_rows,
    });
}

/// A rule value as written: lengths in mm, the rest as they are.
fn rule_value_display(v: &RuleValue) -> String {
    match v {
        RuleValue::Len(c) => format!("{} mm", trim(mm(*c))),
        RuleValue::Num(n) => trim(*n),
        RuleValue::Int(i) => i.to_string(),
        RuleValue::Bool(b) => b.to_string(),
        RuleValue::Text(t) => t.clone(),
        RuleValue::ViaKinds(kinds) => kinds
            .iter()
            .map(|k| format!("{k:?}").to_lowercase())
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// Sort key that orders `R2` before `R10`.
fn natural(s: &str) -> (String, u64, String) {
    let digits_at = s.find(|c: char| c.is_ascii_digit()).unwrap_or(s.len());
    let (head, rest) = s.split_at(digits_at);
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    let number = rest[..end].parse().unwrap_or(0);
    (head.to_owned(), number, rest[end..].to_owned())
}

pub fn to_json(index: &CircuitIndex) -> String {
    let mut json = serde_json::to_string_pretty(index).expect("circuit index serializes");
    json.push('\n');
    json
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_rules::Dim;

    #[test]
    fn values_read_like_a_schematic() {
        assert_eq!(quantity_display(1.0e-7, Dim::CAPACITANCE), "100 nF");
        assert_eq!(quantity_display(10_000.0, Dim::RESISTANCE), "10 k\u{3a9}");
        assert_eq!(quantity_display(4.7e-6, Dim::INDUCTANCE), "4.7 \u{b5}H");
        assert_eq!(quantity_display(16.0, Dim::VOLTAGE), "16 V");
        assert!(natural("R2") < natural("R10"));
    }
}
