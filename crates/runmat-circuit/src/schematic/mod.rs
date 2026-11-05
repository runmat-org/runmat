//! The schematic as drawn: one page per sheet instance, with every symbol placed, its pins,
//! the wires between them, junctions, net labels, power ports, no-connect markers, the child
//! sheets' symbols and a filled title block. The Desktop's schematic canvas, the projection and
//! the KiCad export read this layout.
//!
//! A `.circuit` file describes connectivity, so the drawing is generated from it,
//! deterministically, the way an engineer draws a sheet:
//!
//! - **Pages.** One per sheet instance (`/`, `/ch[0]`), holding the components whose parent is
//!   that instance and a sheet symbol per child instance. The frame is the smallest of A4 to A0
//!   the page fits on, clear of the title block.
//! - **Symbols** ([`symbol`]): supplies on the top edge and grounds on the bottom, inputs left,
//!   outputs right; R, C, L and D with their standard art. A gate with more than 64 pins splits
//!   into units by bank, port prefix and power ([`units`]).
//! - **Placement** ([`place`]): signal flow left to right (connector, front end, converter,
//!   FPGA), shunts upright under their nets, decoupling caps in a row under their IC, feedback
//!   above its amplifier, each net's groups packed in rows.
//! - **Connections** ([`route`]): orthogonal wires with junction dots that never cross a
//!   symbol, power ports (VCC up, GND down) instead of rail wires, labels where a wire would be
//!   long or a net leaves the sheet, no-connect markers on unused pins.
//! - **Places stick.** A symbol whose place is recorded in the lock (`schematic.annotate` and
//!   `schematic.move` record them) stays exactly there; new symbols go to the nearest free
//!   spot, so one edit never reshuffles the sheet.
//!
//! Lengths are millimetres on the page (origin bottom left, y up); every list is sorted by
//! placement, so identical designs give identical bytes.

mod place;
pub mod route;
pub mod symbol;
mod units;

pub use symbol::{Art, PinSide, Place};
pub(crate) use symbol::UnitLayout;
pub use units::{units, Unit, SPLIT_ABOVE, UNIT_PINS};

use crate::display::Xy;
use crate::elab::Netlist;
use crate::geom::{Coord, Point};
use crate::ids::{NamePath, SchPlace};
use crate::library::PinType;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const GRID: Coord = 1_270_000;
pub(crate) const PIN_PITCH: Coord = 2_540_000;
pub(crate) const PIN_LEN: Coord = 2_540_000;
/// Pin name and number text height: 50 mil, half the pin pitch, as KiCad and Altium draw it.
pub const PIN_TEXT: Coord = 1_270_000;
/// The designator's and the value's text heights.
pub const DESIGNATOR_TEXT: Coord = 1_524_000;
pub const VALUE_TEXT: Coord = 1_270_000;
/// Net label and power port text height.
pub const LABEL_TEXT: Coord = 1_270_000;
/// The gap between the body edge and a pin name inside it (KiCad's default name offset).
pub const NAME_OFFSET: Coord = 508_000;
/// Half the size of a no-connect cross drawn at a pin's tip.
pub const NO_CONNECT_HALF: Coord = 800_000;
/// The content's inset from the paper edge.
pub(super) const MARGIN: Coord = 20 * GRID;
/// The title block, bottom right inside the frame (mm).
pub const TITLE_W_MM: f64 = 130.0;
pub const TITLE_ROW_MM: f64 = 6.0;
/// The frame's inset from the paper edge (mm).
pub const FRAME_MM: f64 = 10.0;
/// Sheet sizes tried in order, landscape (mm).
pub(super) const FRAMES: [(&str, f64, f64); 5] = [
    ("A4", 297.0, 210.0),
    ("A3", 420.0, 297.0),
    ("A2", 594.0, 420.0),
    ("A1", 841.0, 594.0),
    ("A0", 1189.0, 841.0),
];

/// How wide a line of text of height `size` is: an upper bound on the advance of the fonts the
/// canvas (Inter) and KiCad (its stroke font) draw, 0.72 of the height per character.
pub fn text_width(text: &str, size: Coord) -> Coord {
    ceil_div(text.chars().count() as Coord * size * 72, 100)
}

/// The box a line of text covers, `[x0, y0, x1, y1]` in nm: the baseline at `y`, descenders
/// 0.2 of the height below it and capitals and ascenders 0.8 above, aligned `start`, `middle`
/// or `end` at `x`.
pub fn text_box(text: &str, size: Coord, x: Coord, y: Coord, anchor: &str) -> [Coord; 4] {
    let w = text_width(text, size);
    let x0 = match anchor {
        "middle" => x - w / 2,
        "end" => x - w,
        _ => x,
    };
    [x0, y - size / 5, x0 + w, y + size * 4 / 5]
}

/// A pin's number as drawn: its pad, or its first pad and how many more (`A1 +7`).
pub fn number_text(pads: &[String]) -> String {
    match pads {
        [] => String::new(),
        [one] => one.clone(),
        [a, b] => format!("{a},{b}"),
        [first, rest @ ..] => format!("{first} +{}", rest.len()),
    }
}

fn ceil_div(a: Coord, b: Coord) -> Coord {
    (a + b - 1).div_euclid(b)
}

fn up_to_grid(v: Coord) -> Coord {
    ceil_div(v, GRID) * GRID
}

/// The drawn schematic: one page per sheet instance, in hierarchy order.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SchematicIndex {
    pub sheets: Vec<SchematicSheet>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SchematicSheet {
    /// The sheet instance path (`/`, `/ch[0]`), as in the hierarchy.
    pub path: String,
    /// The sheet definition it instantiates.
    pub sheet: String,
    /// `A4` to `A0`.
    pub frame: String,
    pub width_mm: f64,
    pub height_mm: f64,
    /// The title block's rows, top to bottom: (field, value).
    #[serde(default)]
    pub title_block: Vec<(String, String)>,
    pub symbols: Vec<SchematicSymbol>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sheet_symbols: Vec<SheetSymbol>,
    pub wires: Vec<SchematicWire>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub junctions: Vec<Xy>,
    pub labels: Vec<NetLabel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub no_connects: Vec<Xy>,
}

/// One placed gate of a component.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SchematicSymbol {
    pub designator: String,
    /// The unit, when the component has more than one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// What a split unit holds (`bank 34`, `power`, `IO0–IO47`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit_label: Option<String>,
    /// The component's instance path.
    pub path: String,
    pub sheet_path: String,
    /// The place key the lock records this symbol's place under.
    pub key: String,
    pub x_mm: f64,
    pub y_mm: f64,
    pub rotation: f64,
    pub mirror: bool,
    /// `box`, `resistor`, `capacitor`, `inductor`, `diode` or `hole`.
    pub art: String,
    /// Outlines: the body box first, then one line per pin from the body to its tip.
    pub body: Vec<Vec<Xy>>,
    /// The art's strokes (open lines) for a part drawn by its symbol, not its box.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub strokes: Vec<Vec<Xy>>,
    pub pins: Vec<SchematicPin>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// Where the designator and value texts sit, and how they align (`start` or `middle`).
    pub designator_at: Xy,
    pub value_at: Xy,
    pub text_anchor: String,
    /// Text heights (mm): pin names and numbers, the designator and the value.
    pub pin_text_mm: f64,
    pub designator_text_mm: f64,
    pub value_text_mm: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SchematicPin {
    pub name: String,
    /// The pad number (several pads joined with `,`).
    pub number: String,
    /// The connection point (the pin's tip).
    pub x_mm: f64,
    pub y_mm: f64,
    /// `left`, `right`, `top` or `bottom` of the body, as placed.
    pub side: String,
    /// The name's baseline point inside the body, and its anchor (`start`, `end`, `middle`).
    pub name_at: Xy,
    pub name_anchor: String,
    /// The number as drawn (`A1`, `A1 +7`) and its baseline centre beside the pin line.
    pub number_text: String,
    pub number_at: Xy,
    /// Small parts (R, C, L, D) hide pin names and numbers.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hidden: bool,
    /// `input`, `output`, `passive`, `power_in`…
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub net: Option<String>,
}

/// A child sheet instance drawn on its parent: a box with a pin per net that crosses it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetSymbol {
    pub path: String,
    pub sheet: String,
    pub key: String,
    /// Bottom-left corner.
    pub x_mm: f64,
    pub y_mm: f64,
    pub w_mm: f64,
    pub h_mm: f64,
    pub pins: Vec<SheetPin>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetPin {
    pub net: String,
    pub x_mm: f64,
    pub y_mm: f64,
    pub side: String,
    /// The port's name inside the child sheet (`IN`), as the entry shows it.
    pub name: String,
    /// `in` or `out`: whether the child drives the net.
    pub direction: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SchematicWire {
    pub net: String,
    pub points: Vec<Xy>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NetLabel {
    pub net: String,
    /// The connection point.
    pub x_mm: f64,
    pub y_mm: f64,
    /// `local`, `global`, `hier` or `power`.
    pub kind: String,
    /// The text shown: the net's name on this sheet (a port's name inside a channel).
    pub text: String,
    /// The way the label points from the connection point: `left` or `right` (text extends
    /// that way), and for a power port also `up` (supplies) or `down` (grounds).
    pub side: String,
}

impl NetLabel {
    /// The box the label covers on the page, `[x0, y0, x1, y1]` in mm.
    pub fn extent(&self) -> [f64; 4] {
        let side = match self.side.as_str() {
            "left" => PinSide::Left,
            "right" => PinSide::Right,
            "up" => PinSide::Top,
            _ => PinSide::Bottom,
        };
        let kind = match self.kind.as_str() {
            "power" => "power",
            "global" => "global",
            "hier" => "hier",
            _ => "local",
        };
        let l = route::Label {
            net: 0,
            at: Point::new(mm_to_nm(self.x_mm), mm_to_nm(self.y_mm)),
            kind,
            side,
        };
        let e = route::label_extent(&l, &self.text);
        [mm(e[0]), mm(e[1]), mm(e[2]), mm(e[3])]
    }
}

fn mm(v: Coord) -> f64 {
    (v as f64 / 1.0e6 * 1.0e6).round() / 1.0e6
}

fn mm_to_nm(v: f64) -> Coord {
    (v * 1.0e6).round() as Coord
}

pub fn pin_kind(k: PinType) -> &'static str {
    match k {
        PinType::Input => "input",
        PinType::Output => "output",
        PinType::Bidirectional => "bidirectional",
        PinType::TriState => "tri_state",
        PinType::Passive => "passive",
        PinType::PowerIn => "power_in",
        PinType::PowerOut => "power_out",
        PinType::OpenCollector => "open_collector",
        PinType::OpenEmitter => "open_emitter",
        PinType::NotConnected => "no_connect",
        PinType::Free => "free",
        PinType::Unspecified => "unspecified",
    }
}

/// Whether a net is a power rail (drawn as a power port): `GND`, `+3V3`, `VCC`, `VDD`…
pub fn is_rail(name: &str, class: Option<&str>) -> bool {
    let leaf = name.rsplit('/').next().unwrap_or(name).to_ascii_uppercase();
    class.is_some_and(|c| c.eq_ignore_ascii_case("power"))
        || leaf.starts_with('+')
        || leaf.starts_with('-')
        || [
            "GND", "VCC", "VDD", "VSS", "VEE", "AGND", "DGND", "PGND", "VBAT", "VIN",
        ]
        .iter()
        .any(|r| leaf == *r || leaf.starts_with(&format!("{r}_")))
}

/// The last path segment of a net name, as a label shows it.
fn short(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

/// A net's name as one sheet instance knows it: the alias declared there (a port's name inside
/// a channel), else its leaf.
pub(super) fn local_name(n: &crate::elab::NetInst, sheet: &NamePath) -> String {
    std::iter::once(&n.name)
        .chain(&n.aliases)
        .find(|a| NamePath::parse(a).parent().as_ref() == Some(sheet))
        .map(|a| short(a).to_owned())
        .unwrap_or_else(|| short(&n.name).to_owned())
}

/// What placement and routing know about a net on one page.
#[derive(Clone, Debug, Default)]
pub(super) struct NetInfo {
    pub rail: bool,
    pub ground: bool,
    pub global: bool,
    /// The net leaves this page's subtree through a port.
    pub external: bool,
    /// Expected to get labels rather than a wire.
    pub labelled: bool,
    pub short: String,
}

pub(super) enum CellKind {
    Unit { comp: usize, gate: usize },
    Sheet { child: usize },
}

/// One placed thing on a page: a component's unit or a child sheet's symbol.
pub(super) struct Cell {
    pub key: String,
    pub kind: CellKind,
    pub lay: UnitLayout,
    /// The net on each layout pin.
    pub nets: Vec<Option<usize>>,
    /// Each layout pin's netlist pin (units only).
    pub pins: Vec<usize>,
    pub designator: String,
    pub connector: bool,
    /// Sheets: whether the child drives each pin's net.
    pub sheet_out: Vec<bool>,
}

/// Orders a sheet symbol's pins by side, top down, and sizes its box.
pub(super) fn restack_sheet(lay: &mut UnitLayout) {
    let mut rows = [0 as Coord; 2];
    let count = |s: PinSide| lay.pins.iter().filter(|p| p.side == s).count() as Coord;
    let (l, r) = (count(PinSide::Left), count(PinSide::Right));
    let most = l.max(r).max(1);
    for p in lay.pins.iter_mut() {
        let k = usize::from(p.side == PinSide::Right);
        let n = if k == 0 { l } else { r };
        let first = (n - 1) * PIN_PITCH / 2;
        p.at = (first - rows[k] * PIN_PITCH).div_euclid(GRID) * GRID;
        rows[k] += 1;
    }
    lay.half_h = most * PIN_PITCH / 2 + 2 * GRID;
    let widest = |s: PinSide| {
        lay.pins
            .iter()
            .filter(|p| p.side == s)
            .map(|p| text_width(short(&p.name), PIN_TEXT))
            .max()
            .unwrap_or(0)
    };
    let names = widest(PinSide::Left) + widest(PinSide::Right) + 4 * GRID;
    lay.half_w = lay.half_w.max(up_to_grid(ceil_div(names, 2)));
}

fn connector(designator: &str) -> bool {
    let p: String = designator
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    matches!(p.as_str(), "J" | "P" | "CN" | "X" | "CON")
}

/// Lays out the whole design.
pub fn layout(nl: &Netlist) -> SchematicIndex {
    let comp_sheet: Vec<NamePath> = nl
        .components
        .iter()
        .map(|c| c.path.parent().unwrap_or_default())
        .collect();
    // Sheets each net touches, through the components on them.
    let net_sheets: Vec<BTreeSet<&NamePath>> = nl
        .nets
        .iter()
        .map(|n| n.pins.iter().map(|(c, _)| &comp_sheet[*c]).collect())
        .collect();
    let mut sheets: Vec<(NamePath, String)> = nl.sheets.clone();
    if !sheets.iter().any(|(p, _)| p.depth() == 0) {
        sheets.insert(0, (NamePath::root(), nl.design.clone()));
    }
    sheets.sort();
    let mut by_sheet: BTreeMap<&NamePath, Vec<usize>> = BTreeMap::new();
    for (c, s) in comp_sheet.iter().enumerate() {
        by_sheet.entry(s).or_default().push(c);
    }
    let mut out = SchematicIndex::default();
    for (path, sheet_name) in &sheets {
        let members = by_sheet.get(path).cloned().unwrap_or_default();
        out.sheets.push(page(nl, path, sheet_name, &sheets, &members, &net_sheets));
    }
    // The hierarchy's order: by path text; then each title block's page count.
    out.sheets.sort_by(|a, b| a.path.cmp(&b.path));
    let count = out.sheets.len();
    for (k, s) in out.sheets.iter_mut().enumerate() {
        s.title_block.insert(3, ("Page".into(), format!("{} of {count}", k + 1)));
    }
    out
}

/// Every symbol's place on the drawn schematic, as the lock records it.
pub fn places(index: &SchematicIndex) -> BTreeMap<String, SchPlace> {
    let mut out = BTreeMap::new();
    for s in &index.sheets {
        let down = |y: f64| mm_to_nm(s.height_mm - y);
        for sym in &s.symbols {
            out.entry(sym.key.clone()).or_insert(SchPlace {
                x: mm_to_nm(sym.x_mm),
                y: down(sym.y_mm),
                rotation: sym.rotation as u16,
                mirror: sym.mirror,
            });
        }
        for sh in &s.sheet_symbols {
            // A sheet symbol's place is its top-left corner.
            out.entry(sh.key.clone()).or_insert(SchPlace {
                x: mm_to_nm(sh.x_mm),
                y: down(sh.y_mm + sh.h_mm),
                rotation: 0,
                mirror: false,
            });
        }
    }
    out
}

fn page(
    nl: &Netlist,
    path: &NamePath,
    sheet_name: &str,
    sheets: &[(NamePath, String)],
    members: &[usize],
    net_sheets: &[BTreeSet<&NamePath>],
) -> SchematicSheet {
    let children: Vec<usize> = (0..sheets.len())
        .filter(|k| sheets[*k].0.parent().as_ref() == Some(path))
        .collect();
    // Net facts for this page.
    let infos: Vec<NetInfo> = nl
        .nets
        .iter()
        .enumerate()
        .map(|(k, n)| {
            let on_page_only = net_sheets[k].iter().all(|s| s.starts_with(path));
            let rail = is_rail(&n.name, n.class.as_deref()) && (n.global || on_page_only);
            let external = !n.global && net_sheets[k].iter().any(|s| !s.starts_with(path));
            NetInfo {
                rail,
                ground: route::text_is_ground(short(&n.name)),
                global: n.global && !rail,
                external,
                labelled: !rail && (n.global || external),
                short: local_name(n, path),
            }
        })
        .collect();
    let mut cells: Vec<Cell> = Vec::new();
    for &ci in members {
        let c = &nl.components[ci];
        let us = units(c);
        for (g, u) in us.iter().enumerate() {
            let lay = UnitLayout::of(c, u, us.len());
            let mut unused: Vec<usize> = u.pins.clone();
            let pins: Vec<usize> = lay
                .pins
                .iter()
                .map(|lp| {
                    let at = unused
                        .iter()
                        .position(|i| c.pins[*i].pin == lp.name)
                        .unwrap_or(0);
                    unused.remove(at)
                })
                .collect();
            let key = if us.len() > 1 {
                format!("{}/{}:{}", c.sheet, c.path.leaf(), u.name)
            } else {
                format!("{}/{}", c.sheet, c.path.leaf())
            };
            cells.push(Cell {
                key,
                kind: CellKind::Unit { comp: ci, gate: g },
                nets: pins.iter().map(|p| c.pins[*p].net).collect(),
                pins,
                lay,
                designator: c.designator.clone(),
                connector: connector(&c.designator),
                sheet_out: Vec::new(),
            });
        }
    }
    let drives = |k: PinType| {
        matches!(
            k,
            PinType::Output
                | PinType::PowerOut
                | PinType::TriState
                | PinType::OpenCollector
                | PinType::OpenEmitter
        )
    };
    for &k in &children {
        let child = &sheets[k].0;
        let crossing: Vec<usize> = (0..nl.nets.len())
            .filter(|n| {
                !nl.nets[*n].global
                    && net_sheets[*n].iter().any(|s| s.starts_with(child))
                    && net_sheets[*n].iter().any(|s| !s.starts_with(child))
            })
            .collect();
        let sheet_out: Vec<bool> = crossing
            .iter()
            .map(|&n| {
                nl.nets[n].pins.iter().any(|(c, p)| {
                    nl.components[*c].path.starts_with(child)
                        && drives(nl.components[*c].pins[*p].kind)
                })
            })
            .collect();
        let name = child.to_string();
        let mut lay = UnitLayout {
            pins: crossing
                .iter()
                .map(|&n| symbol::LayoutPin {
                    name: local_name(&nl.nets[n], child),
                    pads: Vec::new(),
                    kind: PinType::Passive,
                    side: PinSide::Left,
                    at: 0,
                })
                .collect(),
            half_w: up_to_grid((text_width(&name, DESIGNATOR_TEXT) + 4 * GRID) / 2).max(8 * GRID),
            half_h: 4 * GRID,
            pin_len: 0,
            art: Art::Box,
        };
        restack_sheet(&mut lay);
        let parent_sheet = sheet_name;
        cells.push(Cell {
            key: format!("{parent_sheet}/{}", child.leaf()),
            kind: CellKind::Sheet { child: k },
            nets: crossing.iter().map(|n| Some(*n)).collect(),
            pins: Vec::new(),
            lay,
            designator: name,
            connector: false,
            sheet_out,
        });
    }
    let mut places = place::place(&mut cells, &infos);
    let recorded = cells.iter().any(|c| nl.sch_places.contains_key(&c.key));
    stick(&cells, &mut places, &nl.sch_places);
    draw::page(nl, path, sheet_name, sheets, &cells, &places, &infos, !recorded)
}

/// Recorded places override the automatic ones (stored in paper mm, y down from the top; the
/// content's top-left corner sits [`MARGIN`] in). Unrecorded symbols on a page with recorded
/// ones move to the nearest spot clear of them.
fn stick(cells: &[Cell], places: &mut [Place], stored: &BTreeMap<String, SchPlace>) {
    let mut fixed: Vec<[Coord; 4]> = Vec::new();
    let rect = |c: &Cell, p: &Place| {
        let e = p.rect(c.lay.extent());
        [e[0] - 3 * GRID, e[1] - 3 * GRID, e[2] + 3 * GRID, e[3] + 3 * GRID]
    };
    let mut free: Vec<usize> = Vec::new();
    for (i, c) in cells.iter().enumerate() {
        match stored.get(&c.key) {
            Some(s) => {
                let o = Point::new(s.x - MARGIN, MARGIN - s.y);
                places[i] = match c.kind {
                    // A sheet symbol is recorded by its top-left corner.
                    CellKind::Sheet { .. } => Place::at(Point::new(
                        o.x + c.lay.half_w,
                        o.y - c.lay.half_h,
                    )),
                    _ => Place {
                        origin: o,
                        rotation: s.rotation,
                        mirror: s.mirror,
                    },
                };
                fixed.push(rect(c, &places[i]));
            }
            None => free.push(i),
        }
    }
    if fixed.is_empty() {
        return;
    }
    let hits = |r: &[Coord; 4], all: &[[Coord; 4]]| {
        all.iter()
            .any(|f| r[0] < f[2] && f[0] < r[2] && r[1] < f[3] && f[1] < r[3])
    };
    for i in free {
        let base = places[i];
        let mut best = base;
        'search: for ring in 0..400 {
            let d = ring as Coord * 2 * GRID;
            let offsets = [(d, 0), (0, -d), (-d, 0), (0, d), (d, -d), (-d, -d), (d, d), (-d, d)];
            for (dx, dy) in offsets {
                let p = Place {
                    origin: Point::new(base.origin.x + dx, base.origin.y + dy),
                    ..base
                };
                if !hits(&rect(&cells[i], &p), &fixed) {
                    best = p;
                    break 'search;
                }
            }
        }
        places[i] = best;
        fixed.push(rect(&cells[i], &best));
    }
}

mod draw;
pub use draw::{designator_text, page_boxes, symbol_texts};
