//! The board projection: layers, rooms, rules, footprints, pads and copper with the rule each
//! value came from, DRC rows and airwires, and how error lists and scenes name board items.

use super::*;
use crate::board_rules::{props, BItem, BoardGraph, BoardRules, RuleValue};
use crate::drc::Drc;
use crate::layout::{Board, ItemId, NetId, PadConnect, Side, StackKind};
use crate::library::PadShape;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// A board item's scene region, as the board view names it.
pub fn item_region(board: &Board, it: &BItem) -> Option<String> {
    match it {
        BItem::Edge => Some("board:outline".to_owned()),
        BItem::Net(n) => board.nets.get(*n as usize).map(|n| format!("net:{}", n.name)),
        BItem::Footprint(id) => board.footprints.get(id).map(|f| format!("component:{}", f.reference)),
        BItem::Pad(id, pad) => board.footprints.get(id).and_then(|f| {
            f.package
                .pads
                .get(*pad as usize)
                .map(|p| format!("pad:{}.{}", f.reference, p.name))
        }),
        BItem::Track(id) => Some(format!("track:{}", id.0)),
        BItem::Via(id) => Some(format!("via:{}", id.0)),
        BItem::Zone(id) => Some(format!("zone:{}", id.0)),
    }
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
    /// Placement rooms in board order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rooms: Vec<RoomRow>,
    /// The board edge: outer boundary first, then cut-outs.
    pub outline: crate::display::OutlineGeometry,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keepouts: Vec<crate::display::KeepoutRow>,
    /// Board-level texts (footprint texts are on their footprint).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub texts: Vec<crate::display::TextGeometry>,
    /// Board-level graphics (silkscreen art, fabrication and user drawings).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub graphics: Vec<crate::display::GraphicRow>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dimensions: Vec<crate::display::DimensionRow>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fiducials: Vec<crate::display::FiducialRow>,
    /// The ratsnest: links still needed to join each net's copper islands.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unrouted: Vec<crate::display::AirwireRow>,
}

/// A placement room: its members and its outline's bounding box.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoomRow {
    pub name: String,
    pub members: Vec<String>,
    pub x_mm: f64,
    pub y_mm: f64,
    pub width_mm: f64,
    pub height_mm: f64,
    /// The room's outline (closed).
    pub outline: Vec<crate::display::Xy>,
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
    /// Centre in board coordinates.
    pub x_mm: f64,
    pub y_mm: f64,
    /// Absolute rotation: the footprint's and the pad's.
    pub rotation: f64,
    /// Size along the pad's own axes.
    pub size_x_mm: f64,
    pub size_y_mm: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drill_mm: Option<f64>,
    /// A slotted hole's length along the pad's x axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot_mm: Option<f64>,
    /// Rounded rectangles and ovals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corner_radius_mm: Option<f64>,
    /// The exact outline (outer ring, then holes) of shapes the parameters don't describe:
    /// trapezoids, chamfered and custom pads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outline: Option<Vec<Vec<crate::display::Xy>>>,
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
    /// Courtyard rings, closed, in board coordinates.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub courtyard: Vec<Vec<crate::display::Xy>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub silkscreen: Vec<crate::display::Stroke>,
    /// Fabrication-layer outline (the component body as drawn on the assembly drawing).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fab: Vec<crate::display::Stroke>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub texts: Vec<crate::display::TextGeometry>,
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
    /// A track's centreline (arcs flattened within 5 µm).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub points: Vec<crate::display::Xy>,
    /// A track authored as an arc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arc: Option<crate::display::ArcGeometry>,
    /// A via's centre.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_mm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y_mm: Option<f64>,
    /// A pour, per copper layer: its outline and the copper left after clearances.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub polygons: Vec<crate::display::PourLayer>,
}

/// Adds the board: layers, footprints, copper with rule explanations, and DRC findings.
pub fn add_board(index: &mut CircuitIndex, board: &Board, rules: &BoardRules, drc: &Drc) {
    add_board_reusing(index, board, rules, drc, None);
}

/// What a board projection may take from the previous one: its rows for every item not in
/// `items` and its airwires for every net not in `nets`.
pub struct Reuse<'a> {
    pub old: BoardIndex,
    pub items: &'a BTreeSet<ItemId>,
    pub nets: &'a BTreeSet<NetId>,
}

/// [`add_board`], taking unchanged rows from a previous projection of the same board.
pub fn add_board_reusing(
    index: &mut CircuitIndex,
    board: &Board,
    rules: &BoardRules,
    drc: &Drc,
    reuse: Option<Reuse>,
) {
    let (
        mut old_tracks,
        mut old_vias,
        mut old_zones,
        mut old_fps,
        mut old_pads,
        mut old_air,
        changed,
        dirty_nets,
    ) = match reuse {
        Some(r) => {
            let by_id = |rows: Vec<CopperRow>| -> HashMap<u64, CopperRow> {
                rows.into_iter().map(|row| (row.id, row)).collect()
            };
            let mut pads: HashMap<String, Vec<PadRow>> = HashMap::new();
            for p in r.old.pads {
                pads.entry(p.component.clone()).or_default().push(p);
            }
            let mut air: BTreeMap<String, Vec<crate::display::AirwireRow>> = BTreeMap::new();
            for a in r.old.unrouted {
                air.entry(a.net.clone()).or_default().push(a);
            }
            (
                by_id(r.old.tracks),
                by_id(r.old.vias),
                by_id(r.old.zones),
                r.old
                    .footprints
                    .into_iter()
                    .map(|f| (f.reference.clone(), f))
                    .collect::<HashMap<_, _>>(),
                pads,
                Some(air),
                Some(r.items),
                Some(r.nets),
            )
        }
        None => Default::default(),
    };
    let keep = |id: ItemId| changed.is_some_and(|c| !c.contains(&id));
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
            if let Some(row) = keep(t.id).then(|| old_tracks.remove(&t.id.0)).flatten() {
                return row;
            }
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
            let (points, arc) = crate::display::track_points(&t.geom);
            CopperRow {
                id: t.id.0,
                net: net_name(t.net),
                layer: layer_name(t.layer),
                values,
                explain: explain_map,
                text: BTreeMap::new(),
                points,
                arc,
                x_mm: None,
                y_mm: None,
                polygons: Vec::new(),
            }
        })
        .collect();
    let vias = board
        .vias
        .values()
        .map(|v| {
            if let Some(row) = keep(v.id).then(|| old_vias.remove(&v.id.0)).flatten() {
                return row;
            }
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
                points: Vec::new(),
                arc: None,
                x_mm: Some(mm(v.at.x)),
                y_mm: Some(mm(v.at.y)),
                polygons: Vec::new(),
            }
        })
        .collect();
    let zones = board
        .zones
        .values()
        .map(|z| {
            if let Some(row) = keep(z.id).then(|| old_zones.remove(&z.id.0)).flatten() {
                return row;
            }
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
                points: Vec::new(),
                arc: None,
                x_mm: None,
                y_mm: None,
                polygons: crate::display::pour_layers(board, z),
            }
        })
        .collect();
    let pads = board
        .footprints
        .values()
        .flat_map(|f| {
            if keep(f.id) {
                if let Some(rows) = old_pads.remove(&f.reference) {
                    return rows;
                }
            }
            f.pads(copper)
                .into_iter()
                .map(|p| {
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
                    let g = crate::display::pad(f, p.index, copper).unwrap_or_default();
                    PadRow {
                        x_mm: g.x_mm,
                        y_mm: g.y_mm,
                        rotation: g.rotation,
                        size_x_mm: g.size_x_mm,
                        size_y_mm: g.size_y_mm,
                        drill_mm: g.drill_mm,
                        slot_mm: g.slot_mm,
                        corner_radius_mm: g.corner_radius_mm,
                        outline: g.outline,
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
                .collect::<Vec<_>>()
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
    // Designators for parts whose package doesn't name them, placed once for the whole board
    // (the KiCad export writes the same).
    let designators = crate::designators::designators(board);
    let footprint_row = |f: &crate::layout::Footprint| -> FootprintRow {
        let g = crate::display::footprint(f);
        let mut texts = g.texts;
        if let Some(d) = designators.get(&f.id) {
            texts.push(crate::display::designator_text(f, d));
        }
        FootprintRow {
            courtyard: g.courtyard,
            silkscreen: g.silkscreen,
            fab: g.fab,
            texts,
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
        }
    };
    let footprints = board
        .footprints
        .values()
        .map(|f| {
            if keep(f.id) {
                if let Some(row) = old_fps.remove(&f.reference) {
                    return row;
                }
            }
            footprint_row(f)
        })
        .collect();
    let item_label = |it: &BItem| item_label(board, it);
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
            at: Some(xy(f.at)),
            layer: f.layer.map(|l| crate::layout::copper_name(l, board.copper_count())),
            regions: f.items.iter().filter_map(|it| item_region(board, it)).collect(),
            rule: Some(f.rule.clone()).filter(|r| !r.is_empty() && r != "default"),
            span: SourceSpan::of(&f.source),
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
        rooms: board
            .rooms
            .iter()
            .map(|r| {
                let b = r.outline.bbox();
                RoomRow {
                    name: r.name.clone(),
                    members: r.members.clone(),
                    x_mm: mm(b.min.x),
                    y_mm: mm(b.min.y),
                    width_mm: mm(b.max.x - b.min.x),
                    height_mm: mm(b.max.y - b.min.y),
                    outline: r
                        .outline
                        .outer
                        .iter()
                        .map(|p| crate::display::xy(*p))
                        .collect(),
                }
            })
            .collect(),
        outline: crate::display::outline(board),
        keepouts: crate::display::keepouts(board),
        texts: crate::display::board_texts(board),
        graphics: crate::display::board_graphics(board),
        dimensions: crate::display::dimensions(board),
        fiducials: crate::display::fiducials(board),
        unrouted: match (old_air.as_mut(), dirty_nets) {
            (Some(old), Some(nets)) => {
                let fresh = crate::display::unrouted_nets(board, nets);
                let names: BTreeSet<&str> = nets.iter().map(|n| board.net_name(*n)).collect();
                let mut rows: Vec<crate::display::AirwireRow> = old
                    .iter_mut()
                    .filter(|(net, _)| !names.contains(net.as_str()))
                    .flat_map(|(_, rows)| std::mem::take(rows))
                    .chain(fresh)
                    .collect();
                crate::display::sort_airwires(&mut rows);
                rows
            }
            _ => crate::display::unrouted(board),
        },
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

/// How an error list names a board item: `board edge`, a net name, `U3`, `U3.4` (pads count
/// from one), `track 12`, `via 3`, `pour GND`.
pub fn item_label(board: &Board, it: &BItem) -> String {
    match it {
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
    }
}
