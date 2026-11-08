//! The circuit JSON projection: a deterministic view of the elaborated design (hierarchy,
//! components, nets, net classes, BOM), its checks (ERC, design review, DRC), pending ECO
//! lines and, when there is a board, its layers and copper with the rule each value came from.
//! The Desktop's Outline tree and Properties panel read this; so can agents and the server
//! index. Every list is sorted, so identical inputs give identical bytes.

use crate::elab::{FitState, Netlist, Severity};
use crate::ir::{Design, Param, ParamValue, Tolerance};
use crate::library::Pool;
use crate::review::ReviewFinding;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

mod board;
mod eco;
mod high_speed;
pub use high_speed::{stub_rows, BackdrillRow, ControlledClassRow, ControlledLayerRow, LengthGroupRow, LengthMemberRow, ViaStubRow};

pub use board::*;
pub use eco::*;

/// The projection's format version. This release's shape is version 1; a later release that
/// changes the shape bumps it.
pub const CIRCUIT_INDEX_FORMAT_VERSION: u32 = 1;

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
    /// The schematic as drawn: a page per sheet instance with placed symbols, wires and labels.
    pub schematic: crate::schematic::SchematicIndex,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board: Option<BoardIndex>,
    /// The design's `simulation` sections and the nets each probes, for the schematic's probe
    /// markers and the simulation pane.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub simulations: Vec<SimulationRow>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimulationRow {
    pub name: String,
    /// The sheet it simulates alone, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sheet: Option<String>,
    /// Probed nets and pins (`OUT`, `RO.2`), in the section's order (their trace colors follow
    /// it).
    pub probes: Vec<String>,
    /// The probes that are pin currents.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub currents: Vec<String>,
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
    /// Parameter name → how a `.m` function computed it (PCB-38), for the params written as a
    /// call (`value = rcResistor(fc = 160 Hz, c = 100 nF) as ohm`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub computed: BTreeMap<String, ComputedRow>,
}

/// A param computed by a `.m` function: the function, the call as written and where, and the
/// `.m` file the function resolves to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComputedRow {
    /// `rcResistor`, or `pkg.fn` for a `+pkg` function.
    pub function: String,
    /// The call as written: `rcResistor(fc = 160 Hz, c = 100 nF) as ohm`.
    pub call: String,
    /// The `.m` file the function resolves to (filled by the compile, which has the files).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// The circuit file and byte range of the param that makes the call.
    pub source: ComputedSource,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComputedSource {
    pub path: String,
    pub start: u32,
    pub end: u32,
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
    /// Where a board finding is, mm: the DRC view zooms there and draws its marker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<Xy>,
    /// The two ends of a clearance finding's gap, mm: the view dimensions it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gap: Option<[Xy; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    /// The scene regions of the objects involved (`track:17`, `pad:R1.2`, `component:R1`,
    /// `net:GND`): the view highlights them and dims the rest.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<String>,
    /// The rule that set the limit (`power-clearance`), when one did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule: Option<String>,
    /// The rule's (or the item's) text in its file, for the link to the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<SourceSpan>,
}

/// A range of a source file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpan {
    pub path: String,
    pub start: u32,
    pub end: u32,
}

impl SourceSpan {
    pub fn of(key: &runmat_rules::SourceKey) -> Option<Self> {
        (!key.file.is_empty() && key.end > key.start).then(|| SourceSpan {
            path: key.file.clone(),
            start: key.start,
            end: key.end,
        })
    }
}

/// A point in millimetres, `[x, y]`.
pub type Xy = [f64; 2];

fn xy(p: crate::geom::Point) -> Xy {
    [mm(p.x), mm(p.y)]
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
                computed: c
                    .params
                    .iter()
                    .filter_map(|(k, v)| {
                        let call = v.call.as_ref()?;
                        Some((
                            k.clone(),
                            ComputedRow {
                                function: call.function.clone(),
                                call: call.text.clone(),
                                file: None,
                                source: ComputedSource { path: v.source.file.clone(), start: v.source.start, end: v.source.end },
                            },
                        ))
                    })
                    .collect(),
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
                regions: items.iter().map(|i| erc_region(i)).collect(),
                items,
                fix: None,
                source: Some(d.source.to_string()).filter(|s| !s.starts_with(':')),
                at: None,
                gap: None,
                layer: None,
                rule: None,
                span: SourceSpan::of(&d.source),
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
            at: None,
            gap: None,
            layer: None,
            regions: Vec::new(),
            rule: None,
            span: None,
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
        schematic: crate::schematic::layout(netlist),
        board: None,
        simulations: design
            .simulations
            .values()
            .map(|s| SimulationRow {
                name: s.name.clone(),
                sheet: s.sheet.clone(),
                probes: s.probes.iter().map(|p| p.name()).collect(),
                currents: s.probes.iter().filter(|p| matches!(p, crate::simulation::SimProbe::Pin { .. })).map(|p| p.name()).collect(),
            })
            .collect(),
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

/// An ERC item's schematic region: a pin (`U1.VCCINT (F7)`, `R1.2`) is that pin, as the
/// schematic scene names it (`pin:U1.VCCINT`), so selecting a finding lights the pin and not
/// every unit of a large part; a net name is its net.
fn erc_region(item: &str) -> String {
    let item = item.split(" (").next().unwrap_or(item);
    match item.split_once('.') {
        Some((component, pin)) if !component.is_empty() && !pin.is_empty() => {
            format!("pin:{component}.{pin}")
        }
        _ => format!("net:{item}"),
    }
}
