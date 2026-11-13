//! The schematic as drawn: one page per sheet instance, with every symbol placed, its pins,
//! the wires between them, junctions, net labels, power ports, no-connect markers, the child
//! sheets' symbols and a filled title block. The Desktop's schematic canvas, the projection and
//! the KiCad export read this layout.
//!
//! A `.circuit` file describes connectivity, so the drawing is generated from it,
//! deterministically, the way an engineer draws a sheet:
//!
//! - **Pages.** One per sheet instance (`/`, `/ch[0]`), holding the components whose parent is
//!   that instance and a sheet symbol per child instance, with an entry per port. The frame is
//!   the smallest of A4 to A0 the page fits on, clear of the title block; a page whose places
//!   are recorded keeps its size while its content strays into the margin.
//! - **Notes** ([`notes`]): unplaced ones beside their subject or in a notes block inside the
//!   sheet. The layout check ([`check`]) reports what lands off the sheet or over other text.
//! - **Symbols** ([`symbol`]): supplies on the top edge and grounds on the bottom, inputs left,
//!   outputs right; R, C, L and D with their standard art; a part with a declared library
//!   `symbol` with that symbol's art. A gate with more than 64 pins splits
//!   into units by bank, port prefix and power ([`units`]).
//! - **Placement** ([`place`]): signal flow left to right (connector, front end, converter,
//!   FPGA), shunts upright under their nets, decoupling caps in a row under their IC, feedback
//!   above its amplifier, each net's groups packed in rows.
//! - **Connections** ([`route`]): orthogonal wires with junction dots that never cross a
//!   symbol, power ports (VCC up, GND down) instead of rail wires, labels where a wire would be
//!   long or a net leaves the sheet, no-connect crosses on pins marked no-connect (a `no_erc`
//!   mark or a not-connected pin type); other pins on no net are drawn open for ERC to flag.
//! - **Places stick** ([`stick`]). A symbol whose place is recorded in the lock
//!   (`schematic.annotate` and `schematic.move` record them) stays exactly there; new symbols go
//!   to the nearest free spot inside the frame and off the title block, a new IC with its new
//!   caps as one block, so one edit never reshuffles the sheet.
//!
//! Lengths are millimetres on the page (origin bottom left, y up); every list is sorted by
//! placement, so identical designs give identical bytes.

pub mod check;
pub(crate) mod hand;
mod notes;
mod pages;
mod place;
pub mod route;
mod stick;
pub mod symbol;
mod units;

pub use pages::{merge_pages, SPARE_TITLE};
pub(crate) use symbol::UnitLayout;
pub use symbol::{Art, PinSide, Place};
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
/// Pin name and number text height: 50 mil, half the pin pitch, the schematic convention.
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
/// The title block, bottom right inside the frame (mm): a compact identification block in
/// ISO 7200's proportions, narrow beside the sheet's parts, its rows one text line tall.
pub const TITLE_W_MM: f64 = 100.0;
pub const TITLE_ROW_MM: f64 = 5.0;
/// The title row (the design's name, in larger type) is this many rows tall.
pub const TITLE_FIRST_ROW_SCALE: f64 = 1.6;

/// The title block's height (mm) for `rows` rows: the title row [`TITLE_FIRST_ROW_SCALE`]
/// rows tall, the others one row each. The canvas, the PDF and the frame's reservation all
/// draw this height.
pub fn title_height_mm(rows: usize) -> f64 {
    match rows {
        0 => 0.0,
        n => TITLE_ROW_MM * (TITLE_FIRST_ROW_SCALE + (n - 1) as f64),
    }
}
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
    /// The instance's page: 1, or 2 for its large parts' unused units ([`pages`]).
    #[serde(default = "pages::one", skip_serializing_if = "pages::is_one")]
    pub page: u32,
    /// What a page after the first holds ([`SPARE_TITLE`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holds: Option<String>,
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
    /// The sheet's notes (`text` statements), at their recorded places or, unrecorded, beside
    /// their subject or in a notes block inside the sheet.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub texts: Vec<SheetText>,
    /// The sheet's blankets: a dashed box round the wires, labels and pins of the nets each
    /// covers, its caption above the box's top-left corner.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blankets: Vec<SheetBlanket>,
}

/// A blanket drawn on a page: the box `[x0, y0, x1, y1]` (mm) round what it covers, and its
/// caption (`blanket rails · class power · max_length = 50 mm`) at `caption_at`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetBlanket {
    pub name: String,
    pub caption: String,
    pub caption_at: Xy,
    pub caption_mm: f64,
    pub rect: [f64; 4],
    /// The nets it covers, as the page's wires name them.
    pub nets: Vec<String>,
}

/// A blanket caption's text height (mm): 50 mil.
pub const BLANKET_TEXT: Coord = 1_270_000;

/// A note drawn on a sheet: its baseline's start (mm) and text height.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetText {
    pub name: String,
    pub text: String,
    pub x_mm: f64,
    pub y_mm: f64,
    pub size_mm: f64,
}

/// A note's text height (mm): 70 mil, a schematic's usual note size.
pub const NOTE_TEXT: Coord = 1_778_000;

/// The lock key of a note's place (`main/text:note1`).
pub fn text_key(sheet: &str, name: &str) -> String {
    format!("{sheet}/text:{name}")
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
    /// Texts in a declared symbol's art (`kWh` in a meter), each centred on its point.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub art_texts: Vec<ArtLabel>,
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

/// A text in a symbol's art, centred on `at` (mm).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtLabel {
    pub text: String,
    pub at: Xy,
    pub size_mm: f64,
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
    /// Marked no-connect on purpose: a `no_erc` mark or a not-connected pin type. Such a pin on
    /// no net gets a no-connect cross at its tip; any other pin on no net is drawn open, and
    /// ERC warns about it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_connect: bool,
}

/// Whether a component's pin is marked no-connect on purpose (see [`SchematicPin::no_connect`]).
pub(super) fn marked_no_connect(nl: &Netlist, comp: usize, pin: usize) -> bool {
    nl.no_erc.contains(&(comp, pin)) || nl.components[comp].pins[pin].kind == PinType::NotConnected
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
    /// The net is on one of the root sheet's own ports (its IN and OUT).
    pub own_port: bool,
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
    /// The widest of the designator and the value as drawn (0 for a sheet symbol, whose name
    /// sits in its box): an upright part's texts reach this far right of its body.
    pub text_w: Coord,
    pub connector: bool,
    /// Sheets: whether the child drives each pin's net.
    pub sheet_out: Vec<bool>,
}

/// A sheet symbol's least half width and half height: 20 mm by 15 mm.
const SHEET_HALF_W: Coord = 8 * GRID;
const SHEET_HALF_H: Coord = 6 * GRID;

/// Orders a sheet symbol's pins by side, top down, and sizes its box.
pub(super) fn restack_sheet(lay: &mut UnitLayout) {
    let mut rows = [0 as Coord; 2];
    let count = |s: PinSide| lay.pins.iter().filter(|p| p.side == s).count() as Coord;
    let (l, r) = (count(PinSide::Left), count(PinSide::Right));
    let most = l.max(r).max(1);
    // A sheet symbol reads as a block of the design: at least 15 mm tall, with a pitch of
    // room above and below its pins. Both sides' pins start from the top row, as a sheet's
    // entries are drawn.
    lay.half_h = (most * PIN_PITCH / 2 + 2 * GRID).max(SHEET_HALF_H);
    let first = lay.half_h - 2 * GRID - PIN_PITCH / 2;
    for p in lay.pins.iter_mut() {
        let k = usize::from(p.side == PinSide::Right);
        p.at = (first - rows[k] * PIN_PITCH).div_euclid(GRID) * GRID;
        rows[k] += 1;
    }
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
        out.sheets
            .extend(page(nl, path, sheet_name, &sheets, &members, &net_sheets));
    }
    // The hierarchy's order: by path text, an instance's pages together; then each title
    // block's page count.
    out.sheets
        .sort_by(|a, b| (&a.path, a.page).cmp(&(&b.path, b.page)));
    let count = out.sheets.len();
    for (k, s) in out.sheets.iter_mut().enumerate() {
        s.title_block
            .insert(3, ("Page".into(), format!("{} of {count}", k + 1)));
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
                page: s.page as u16,
            });
        }
        for t in &s.texts {
            out.entry(text_key(&s.sheet, &t.name)).or_insert(SchPlace {
                x: mm_to_nm(t.x_mm),
                y: down(t.y_mm),
                rotation: 0,
                mirror: false,
                page: s.page as u16,
            });
        }
        for sh in &s.sheet_symbols {
            // A sheet symbol's place is its top-left corner.
            out.entry(sh.key.clone()).or_insert(SchPlace {
                x: mm_to_nm(sh.x_mm),
                y: down(sh.y_mm + sh.h_mm),
                rotation: 0,
                mirror: false,
                page: s.page as u16,
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
) -> Vec<SchematicSheet> {
    let children: Vec<usize> = (0..sheets.len())
        .filter(|k| sheets[*k].0.parent().as_ref() == Some(path))
        .collect();
    let ports: Vec<String> = nl
        .sheet_ports
        .get(sheet_name)
        .map(|ps| {
            ps.iter()
                .map(|p| crate::elab::port_net_name(path, p))
                .collect()
        })
        .unwrap_or_default();
    // Net facts for this page.
    let infos: Vec<NetInfo> = nl
        .nets
        .iter()
        .enumerate()
        .map(|(k, n)| {
            // A rail by name (GND, +5V, VBAT) is drawn with ports on a page that holds all of
            // its pins too; one that runs into child sheets (VBAT_SENSE between the power and
            // MCU sheets) is a signal between them. The power net class alone makes a rail only
            // of a global net, so a local power node (a regulator's switch node) is drawn as the
            // wire it is.
            let here_only = net_sheets[k].iter().all(|s| *s == path);
            let rail = if is_rail(&n.name, None) {
                n.global || here_only
            } else {
                is_rail(&n.name, n.class.as_deref()) && n.global
            };
            // A net leaving this page is labelled as a port, and so is a net on one of the
            // sheet's own ports: the root's (a one-sheet filter's IN and OUT) leave the design,
            // and a child's leave its page even when only this page's parts are on the net (a
            // branch circuit fed from the parent's power-flagged net).
            let bears_port = ports.iter().any(|p| n.name == *p || n.aliases.contains(p));
            let on_port = path.depth() == 0 && bears_port;
            let external =
                !n.global && (bears_port || net_sheets[k].iter().any(|s| !s.starts_with(path)));
            NetInfo {
                rail,
                ground: route::text_is_ground(short(&n.name)),
                global: n.global && !rail,
                external,
                own_port: on_port,
                labelled: !rail && (n.global || external),
                short: local_name(n, path),
            }
        })
        .collect();
    let mut cells: Vec<Cell> = Vec::new();
    // A large part's unconnected units, for page 2.
    let mut spare: Vec<Cell> = Vec::new();
    for &ci in members {
        let c = &nl.components[ci];
        let us = units(c);
        for (g, u) in us.iter().enumerate() {
            let lay = UnitLayout::of(c, u, us.len(), nl.symbol_of(c));
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
            let to = if us.len() > 1 && pages::unused(nl, ci, u) {
                &mut spare
            } else {
                &mut cells
            };
            to.push(Cell {
                key,
                kind: CellKind::Unit { comp: ci, gate: g },
                nets: pins.iter().map(|p| c.pins[*p].net).collect(),
                pins,
                lay,
                designator: c.designator.clone(),
                text_w: text_width(&c.designator, DESIGNATOR_TEXT).max(
                    crate::index::component_value(c).map_or(0, |v| text_width(&v, VALUE_TEXT)),
                ),
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
        // The child's own ports: each is an entry, whether or not the parent's net reaches a
        // part on this page (power ports fed from a power-flagged net reach none).
        let child_ports: Vec<String> = nl
            .sheet_ports
            .get(&sheets[k].1)
            .map(|ps| {
                ps.iter()
                    .map(|p| crate::elab::port_net_name(child, p))
                    .collect()
            })
            .unwrap_or_default();
        let crossing: Vec<usize> = (0..nl.nets.len())
            .filter(|n| {
                let net = &nl.nets[*n];
                let on_port = child_ports
                    .iter()
                    .any(|p| net.name == *p || net.aliases.contains(p));
                !net.global
                    && net_sheets[*n].iter().any(|s| s.starts_with(child))
                    && (on_port || net_sheets[*n].iter().any(|s| !s.starts_with(child)))
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
            half_w: up_to_grid((text_width(&name, DESIGNATOR_TEXT) + 4 * GRID) / 2)
                .max(SHEET_HALF_W),
            half_h: 4 * GRID,
            pin_len: 0,
            art: Art::Box,
            declared: None,
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
            text_w: 0,
            connector: false,
            sheet_out,
        });
    }
    if cells.is_empty() {
        cells = std::mem::take(&mut spare);
    }
    let mut out = Vec::new();
    for (k, mut cells) in [cells, spare].into_iter().enumerate() {
        if k > 0 && cells.is_empty() {
            continue;
        }
        let page_no = k as u16 + 1;
        // A place recorded on the other page is in that page's coordinates: it doesn't apply.
        let stored: BTreeMap<String, SchPlace> = nl
            .sch_places
            .iter()
            .filter(|(_, p)| p.page == page_no)
            .map(|(key, p)| (key.clone(), *p))
            .collect();
        let mut placed = place::place(&mut cells, &infos);
        let recorded = cells.iter().any(|c| stored.contains_key(&c.key));
        let title_rows = 5.0 + if nl.variant.is_some() { 1.0 } else { 0.0 };
        stick::stick(&cells, &infos, &mut placed, &stored, title_rows);
        let places = placed.places;
        let mut drawn = draw::page(
            nl,
            path,
            sheet_name,
            sheets,
            &infos,
            draw::PageLayout {
                cells: &cells,
                places: &places,
                centre: !recorded,
            },
        );
        if k > 0 {
            drawn.page = 2;
            drawn.holds = Some(SPARE_TITLE.to_owned());
            if let Some(row) = drawn.title_block.iter_mut().find(|(f, _)| f == "Sheet") {
                row.1 = format!("{sheet_name}, page 2 ({SPARE_TITLE})");
            }
        }
        out.push(drawn);
    }
    out
}

mod draw;
use draw::shift;
pub use draw::{designator_text, page_boxes, symbol_texts};
