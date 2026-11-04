//! The schematic as drawn: one page per sheet instance, with every symbol placed, its pins,
//! the wires between them, junctions, net labels, power ports, no-connect markers and the
//! child sheets' symbols. The Desktop's schematic canvas and the projection read this layout.
//!
//! A `.circuit` file describes connectivity, so the drawing is generated from it,
//! deterministically:
//!
//! - **Pages.** One per sheet instance (`/`, `/ch[0]`, `/power`), holding the components whose
//!   parent is that instance and a sheet symbol per child instance. The frame is the smallest
//!   of A3, A2, A1 and A0 the page fits on (A0 grows taller when even that is full).
//! - **Order.** Components follow connectivity: a breadth-first walk over shared signal nets
//!   (rails and nets with more than eight pins don't order), starting from the most connected
//!   component, so connected parts sit side by side.
//! - **Placement.** Shelf packing in rows from the top left, on the 1.27 mm grid; each cell
//!   reserves room for the symbol, its pins and its labels.
//! - **Symbols.** A box per unit: inputs, power inputs and the first half of passive pins on the
//!   left, the rest on the right; pins numbered by pad. A unit is a library gate, and a gate with
//!   more than 64 pins (an FPGA, a large MCU) splits into units as KiCad and Altium draw them:
//!   its signal pins by bank (a `_34` suffix or `BANK3` token in the pin name), else by port
//!   prefix (`PA`, `PB`), in even units of at most 48 pins; its power pins in units of their own,
//!   last. [`units`] is the one split; the canvas and the KiCad export both draw it.
//! - **Connections.** A two-pin net between facing pins of neighbouring symbols in a row is
//!   drawn as a wire. Every other connected pin gets a 2.54 mm stub and a label: a power port
//!   for a power rail, a global label for other global nets, a hierarchical label for a net that
//!   leaves the sheet through a port (it reaches a sheet outside this page's subtree), and a
//!   local label otherwise (which also joins the pins of child sheets drawn on the page). Unconnected pins get a
//!   no-connect marker. Junctions mark where three or more wire ends meet or a wire ends on
//!   another wire.
//!
//! Lengths are millimetres on the page (origin bottom left, y up); every list is sorted by
//! placement, so identical designs give identical bytes.

use crate::display::Xy;
use crate::elab::Netlist;
use crate::geom::{Coord, Point};
use crate::ids::NamePath;
use crate::library::PinType;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub(crate) const GRID: Coord = 1_270_000;
pub(crate) const PIN_PITCH: Coord = 2_540_000;
pub(crate) const PIN_LEN: Coord = 2_540_000;
const STUB: Coord = 2_540_000;
const MARGIN: Coord = 20 * GRID;
/// Sheet sizes tried in order, landscape (mm).
const FRAMES: [(&str, f64, f64); 4] = [
    ("A3", 420.0, 297.0),
    ("A2", 594.0, 420.0),
    ("A1", 841.0, 594.0),
    ("A0", 1189.0, 841.0),
];

/// One generated unit: pins on the left and right, in library coordinates (y up).
#[derive(Clone, Debug)]
pub(crate) struct UnitLayout {
    /// (pin name, pad numbers, type, left side, library y of the pin row).
    pub(crate) pins: Vec<(String, Vec<String>, PinType, bool, Coord)>,
    pub(crate) half_w: Coord,
    pub(crate) half_h: Coord,
}

impl UnitLayout {
    pub(crate) fn new(pins: &[(String, Vec<String>, PinType)]) -> UnitLayout {
        let passive: Vec<usize> = (0..pins.len())
            .filter(|&i| pins[i].2 == PinType::Passive)
            .collect();
        let left_of = |i: usize| match pins[i].2 {
            PinType::Input | PinType::PowerIn => true,
            PinType::Passive => {
                passive.iter().position(|&p| p == i).unwrap() < passive.len().div_ceil(2)
            }
            _ => false,
        };
        let (mut left, mut right): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        for i in 0..pins.len() {
            if left_of(i) {
                left.push(i)
            } else {
                right.push(i)
            }
        }
        if right.is_empty() && left.len() > 1 {
            right = left.split_off(left.len().div_ceil(2));
        }
        // A large part keeps both sides in use: the longer side gives half its excess to the
        // other, from its end, so the box is half as tall.
        if right.len() > left.len() + 8 {
            let n = (right.len() - left.len()) / 2;
            left.extend(right.split_off(right.len() - n));
        } else if left.len() > right.len() + 8 {
            let n = (left.len() - right.len()) / 2;
            right.extend(left.split_off(left.len() - n));
        }
        let rows = left.len().max(right.len()).max(1) as Coord;
        let half_h = (rows * PIN_PITCH) / 2 + GRID;
        let name_len = pins.iter().map(|p| p.0.chars().count()).max().unwrap_or(1) as Coord;
        let half_w =
            ((((name_len * GRID + 5 * GRID) / 2) + GRID - 1) / GRID * GRID).max(2 * PIN_PITCH);
        let mut out = Vec::new();
        for (side, list) in [(true, &left), (false, &right)] {
            let top = (list.len() as Coord - 1) * PIN_PITCH / 2;
            for (k, &i) in list.iter().enumerate() {
                // Rows on the 2.54 mm grid from the top, centred.
                let y = (top - k as Coord * PIN_PITCH).div_euclid(GRID) * GRID;
                out.push((pins[i].0.clone(), pins[i].1.clone(), pins[i].2, side, y));
            }
        }
        UnitLayout {
            pins: out,
            half_w,
            half_h,
        }
    }
    /// Pin tip in library coordinates.
    pub(crate) fn tip(&self, left: bool, y: Coord) -> Point {
        let x = self.half_w + PIN_LEN;
        Point::new(if left { -x } else { x }, y)
    }
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
    /// `A3`, `A2`, `A1` or `A0`.
    pub frame: String,
    pub width_mm: f64,
    pub height_mm: f64,
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
    pub x_mm: f64,
    pub y_mm: f64,
    pub rotation: f64,
    pub mirror: bool,
    /// Outlines: the body box first, then one line per pin from the body to its tip.
    pub body: Vec<Vec<Xy>>,
    pub pins: Vec<SchematicPin>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// Where the designator and value texts sit.
    pub designator_at: Xy,
    pub value_at: Xy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SchematicPin {
    pub name: String,
    /// The pad number (several pads joined with `,`).
    pub number: String,
    /// The connection point (the pin's tip).
    pub x_mm: f64,
    pub y_mm: f64,
    /// `left` or `right` of the body.
    pub side: String,
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
    /// The side the text extends to from the connection point: `left` or `right`.
    pub side: String,
}

fn mm(v: Coord) -> f64 {
    (v as f64 / 1.0e6 * 1.0e6).round() / 1.0e6
}

fn xy(p: Point) -> Xy {
    [mm(p.x), mm(p.y)]
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

/// Whether a global net is a power rail (drawn as a power port): `GND`, `+3V3`, `VCC`, `VDD`…
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

/// A gate with more pins than this splits into units.
pub const SPLIT_ABOVE: usize = 64;
/// The most pins a split unit holds.
pub const UNIT_PINS: usize = 48;

/// One schematic unit of a component: its name (`A`, `B`… for a split gate, else the gate's
/// name), what it holds when split, and its pins (indices into the component's pins).
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    pub name: String,
    pub label: Option<String>,
    pub pins: Vec<usize>,
}

/// The bank a pin name declares: a trailing `_34` (`IO_L1P_T0_34`) or a `BANK3` token.
fn bank_of(name: &str) -> Option<String> {
    let upper = name.to_ascii_uppercase();
    for token in upper.split(['_', '.', '-']) {
        if let Some(n) = token.strip_prefix("BANK") {
            if !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()) {
                return Some(n.to_owned());
            }
        }
    }
    let (head, tail) = upper.rsplit_once('_')?;
    (!head.is_empty() && !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()))
        .then(|| tail.to_owned())
}

/// A pin name's leading letters (`PA` of `PA12`, `IO` of `IO7`).
fn prefix_of(name: &str) -> String {
    name.chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect()
}

fn letters(mut k: usize) -> String {
    let mut out = Vec::new();
    loop {
        out.push((b'A' + (k % 26) as u8) as char);
        if k < 26 {
            break;
        }
        k = k / 26 - 1;
    }
    out.iter().rev().collect()
}

/// A component's units, in drawing order: one per gate, a large gate split as the module
/// docs describe. Deterministic: the same pins give the same units.
pub fn units(c: &crate::elab::CompInst) -> Vec<Unit> {
    let mut gates: Vec<&str> = Vec::new();
    for p in &c.pins {
        if !gates.contains(&p.gate.as_str()) {
            gates.push(&p.gate);
        }
    }
    let single = gates.len() == 1;
    let mut out = Vec::new();
    for g in &gates {
        let idx: Vec<usize> = (0..c.pins.len())
            .filter(|i| c.pins[*i].gate == *g)
            .collect();
        if idx.len() <= SPLIT_ABOVE {
            out.push(Unit {
                name: (*g).to_owned(),
                label: None,
                pins: idx,
            });
            continue;
        }
        let power = |i: &usize| matches!(c.pins[*i].kind, PinType::PowerIn | PinType::PowerOut);
        let (pwr, signal): (Vec<usize>, Vec<usize>) = idx.iter().partition(|i| power(i));
        // Groups of signal pins: by bank when every pin names one, else by port prefix.
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        let banks: Vec<Option<String>> = signal.iter().map(|i| bank_of(&c.pins[*i].pin)).collect();
        let by_bank = banks.iter().all(Option::is_some);
        for (k, &pin) in signal.iter().enumerate() {
            let label = match &banks[k] {
                Some(b) if by_bank => format!("bank {b}"),
                _ => prefix_of(&c.pins[pin].pin),
            };
            match groups.iter_mut().find(|(l, _)| *l == label) {
                Some((_, v)) => v.push(pin),
                None => groups.push((label, vec![pin])),
            }
        }
        if !pwr.is_empty() {
            groups.push(("power".to_owned(), pwr));
        }
        let mut n = 0;
        for (label, pins) in groups {
            // Even units: 200 pins make five of 40, not four of 48 and one of 8.
            let chunks = pins.len().div_ceil(UNIT_PINS);
            let size = pins.len().div_ceil(chunks.max(1));
            for (k, chunk) in pins.chunks(size.max(1)).enumerate() {
                let label = if chunks == 1 {
                    label.clone()
                } else {
                    let first = &c.pins[chunk[0]].pin;
                    let last = &c.pins[chunk[chunk.len() - 1]].pin;
                    if label == "power" || label.starts_with("bank ") {
                        format!("{label} ({}/{chunks})", k + 1)
                    } else {
                        format!("{first}–{last}")
                    }
                };
                let name = if single {
                    letters(n)
                } else {
                    format!("{g}{}", letters(n))
                };
                out.push(Unit {
                    name,
                    label: Some(label),
                    pins: chunk.to_vec(),
                });
                n += 1;
            }
        }
    }
    out
}

/// A unit's pins as the symbol layout reads them: (name, pads, type).
pub(crate) fn unit_pins(
    c: &crate::elab::CompInst,
    unit: &Unit,
) -> Vec<(String, Vec<String>, PinType)> {
    unit.pins
        .iter()
        .map(|i| {
            let p = &c.pins[*i];
            let pads = if p.pads.is_empty() {
                vec![p.pin.clone()]
            } else {
                p.pads.clone()
            };
            (p.pin.clone(), pads, p.kind)
        })
        .collect()
}

/// One placed cell: a component's unit or a child sheet.
enum Cell {
    Unit { comp: usize, gate: usize },
    Sheet { child: usize },
}

/// Lays out the whole design.
pub fn layout(nl: &Netlist) -> SchematicIndex {
    // Units per component: (unit, symbol layout, pin indices).
    let gates: Vec<Vec<(Unit, UnitLayout, Vec<usize>)>> = nl
        .components
        .iter()
        .map(|c| {
            units(c)
                .into_iter()
                .map(|u| {
                    let lay = UnitLayout::new(&unit_pins(c, &u));
                    let idx = u.pins.clone();
                    (u, lay, idx)
                })
                .collect()
        })
        .collect();
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
    // Signal neighbours for ordering.
    let mut neighbours: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nl.components.len()];
    for n in &nl.nets {
        if n.global || n.pins.len() > 8 {
            continue;
        }
        let comps: BTreeSet<usize> = n.pins.iter().map(|p| p.0).collect();
        for &a in &comps {
            for &b in &comps {
                if a != b {
                    neighbours[a].insert(b);
                }
            }
        }
    }
    let mut out = SchematicIndex::default();
    for (path, sheet_name) in &sheets {
        let members: Vec<usize> = (0..nl.components.len())
            .filter(|c| &comp_sheet[*c] == path)
            .collect();
        let children: Vec<usize> = (0..sheets.len())
            .filter(|k| sheets[*k].0.parent().as_ref() == Some(path))
            .collect();
        // Nets crossing each child's boundary (not global ones: those connect by name).
        let crossing = |child: &NamePath| -> Vec<usize> {
            (0..nl.nets.len())
                .filter(|n| {
                    !nl.nets[*n].global
                        && net_sheets[*n].iter().any(|s| s.starts_with(child))
                        && net_sheets[*n].iter().any(|s| !s.starts_with(child))
                })
                .collect()
        };
        let order = connectivity_order(&members, &neighbours);
        let mut cells: Vec<Cell> = Vec::new();
        for c in order {
            for g in 0..gates[c].len() {
                cells.push(Cell::Unit { comp: c, gate: g });
            }
        }
        cells.extend(children.iter().map(|k| Cell::Sheet { child: *k }));
        let sheet_pins: BTreeMap<usize, Vec<usize>> = children
            .iter()
            .map(|k| (*k, crossing(&sheets[*k].0)))
            .collect();
        let label_room = |nets: &mut dyn Iterator<Item = Option<usize>>| -> Coord {
            let longest = nets
                .filter_map(|n| n.map(|n| short(&nl.nets[n].name).chars().count()))
                .max()
                .unwrap_or(4) as Coord;
            STUB + (longest + 4) * GRID
        };
        // Cell sizes (w, h) and, for sheets, the box size.
        let sizes: Vec<(Coord, Coord)> = cells
            .iter()
            .map(|cell| match cell {
                Cell::Unit { comp, gate } => {
                    let (_, lay, idx) = &gates[*comp][*gate];
                    let room =
                        label_room(&mut idx.iter().map(|i| nl.components[*comp].pins[*i].net));
                    (2 * (lay.half_w + PIN_LEN + room), 2 * lay.half_h + 8 * GRID)
                }
                Cell::Sheet { child } => {
                    let pins = &sheet_pins[child];
                    let room = label_room(&mut pins.iter().map(|n| Some(*n)));
                    let (bw, bh) = sheet_box(&sheets[*child].0, pins.len());
                    (bw + 2 * room, bh + 6 * GRID)
                }
            })
            .collect();
        // The smallest frame the shelves fit in.
        let mut placed: Vec<Point> = Vec::new();
        let mut rows: Vec<usize> = Vec::new();
        let mut frame = FRAMES[FRAMES.len() - 1];
        let mut height = mm_to_nm(frame.2);
        for f in FRAMES {
            let (w, h) = (mm_to_nm(f.1), mm_to_nm(f.2));
            let (pos, row, used) = shelves(&sizes, w - 2 * MARGIN);
            frame = f;
            placed = pos;
            rows = row;
            height = h.max(used + 2 * MARGIN);
            if used + 2 * MARGIN <= h {
                break;
            }
        }
        let top = height - MARGIN;
        // Cell centres on the page (shelves come back as offsets from the top left).
        let centres: Vec<Point> = placed
            .iter()
            .zip(&sizes)
            .map(|(p, (w, h))| {
                Point::new(
                    (MARGIN + p.x + w / 2).div_euclid(GRID) * GRID,
                    (top - p.y - h / 2).div_euclid(GRID) * GRID,
                )
            })
            .collect();
        let mut page = SchematicSheet {
            path: path.to_string(),
            sheet: sheet_name.clone(),
            frame: frame.0.to_owned(),
            width_mm: frame.1,
            height_mm: mm(height),
            ..Default::default()
        };
        // The top of each cell's body, for wires drawn over it.
        let cell_tops: Vec<Coord> = cells
            .iter()
            .zip(&centres)
            .map(|(cell, o)| match cell {
                Cell::Unit { comp, gate } => o.y + gates[*comp][*gate].1.half_h,
                Cell::Sheet { child } => {
                    o.y + sheet_box(&sheets[*child].0, sheet_pins[child].len()).1 / 2
                }
            })
            .collect();
        // Pin tips: (net, tip, left side, cell index).
        let mut tips: Vec<(Option<usize>, Point, bool, usize)> = Vec::new();
        for (ci, cell) in cells.iter().enumerate() {
            let o = centres[ci];
            match cell {
                Cell::Unit { comp, gate } => {
                    let c = &nl.components[*comp];
                    let (unit, lay, idx) = &gates[*comp][*gate];
                    let (hw, hh) = (lay.half_w, lay.half_h);
                    let corner = |dx: Coord, dy: Coord| xy(Point::new(o.x + dx, o.y + dy));
                    let mut body = vec![vec![
                        corner(-hw, -hh),
                        corner(hw, -hh),
                        corner(hw, hh),
                        corner(-hw, hh),
                        corner(-hw, -hh),
                    ]];
                    let mut pins = Vec::new();
                    // The layout orders pins by side; each finds its netlist pin by name.
                    let mut unused: Vec<usize> = idx.clone();
                    for (name, pads, kind, left, ly) in &lay.pins {
                        let at = unused
                            .iter()
                            .position(|i| &c.pins[*i].pin == name)
                            .unwrap_or(0);
                        let pin_index = &unused.remove(at);
                        let tip = lay.tip(*left, *ly);
                        let tip = Point::new(o.x + tip.x, o.y + tip.y);
                        let edge = Point::new(o.x + if *left { -hw } else { hw }, tip.y);
                        body.push(vec![xy(edge), xy(tip)]);
                        let net = c.pins[*pin_index].net;
                        pins.push(SchematicPin {
                            name: name.clone(),
                            number: pads.join(","),
                            x_mm: mm(tip.x),
                            y_mm: mm(tip.y),
                            side: side(*left),
                            kind: pin_kind(*kind).to_owned(),
                            net: net.map(|n| nl.nets[n].name.clone()),
                        });
                        tips.push((net, tip, *left, ci));
                        if net.is_none() {
                            page.no_connects.push(xy(tip));
                        }
                    }
                    page.symbols.push(SchematicSymbol {
                        designator: c.designator.clone(),
                        unit: (gates[*comp].len() > 1).then(|| unit.name.clone()),
                        unit_label: unit.label.clone(),
                        path: c.path.to_string(),
                        sheet_path: path.to_string(),
                        x_mm: mm(o.x),
                        y_mm: mm(o.y),
                        rotation: 0.0,
                        mirror: false,
                        body,
                        pins,
                        value: crate::index::component_value(c),
                        designator_at: xy(Point::new(o.x, o.y + hh + GRID)),
                        value_at: xy(Point::new(o.x, o.y - hh - 2 * GRID)),
                    });
                }
                Cell::Sheet { child } => {
                    let nets = &sheet_pins[child];
                    let (bw, bh) = sheet_box(&sheets[*child].0, nets.len());
                    let (x0, y0) = (o.x - bw / 2, o.y - bh / 2);
                    let pins = nets
                        .iter()
                        .enumerate()
                        .map(|(k, n)| {
                            // Alternate sides, top down.
                            let left = k % 2 == 0;
                            let y = y0 + bh - 2 * GRID - (k as Coord / 2) * PIN_PITCH;
                            let tip = Point::new(if left { x0 } else { x0 + bw }, y);
                            tips.push((Some(*n), tip, left, ci));
                            SheetPin {
                                net: nl.nets[*n].name.clone(),
                                x_mm: mm(tip.x),
                                y_mm: mm(tip.y),
                                side: side(left),
                            }
                        })
                        .collect();
                    page.sheet_symbols.push(SheetSymbol {
                        path: sheets[*child].0.to_string(),
                        sheet: sheets[*child].1.clone(),
                        x_mm: mm(x0),
                        y_mm: mm(y0),
                        w_mm: mm(bw),
                        h_mm: mm(bh),
                        pins,
                    });
                }
            }
        }
        // Two-pin nets between facing pins of neighbouring cells in a row: a wire.
        let mut on_page: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (t, (net, ..)) in tips.iter().enumerate() {
            if let Some(n) = net {
                on_page.entry(*n).or_default().push(t);
            }
        }
        let mut wired: BTreeSet<usize> = BTreeSet::new();
        let mut loops: BTreeMap<usize, (Coord, Coord)> = BTreeMap::new();
        for (n, ts) in &on_page {
            let net = &nl.nets[*n];
            let local = net_sheets[*n].len() == 1 && !net.global;
            if !local || ts.len() != 2 {
                continue;
            }
            let (a, b) = {
                let (x, y) = (&tips[ts[0]], &tips[ts[1]]);
                if x.1.x <= y.1.x {
                    (x, y)
                } else {
                    (y, x)
                }
            };
            // A loop on one symbol (an amplifier's feedback): over the top of the body. Its
            // sides run clear of the label points at the stub ends (a wire through a label
            // would join its net): the first loop between the pin tips and the stub ends, later
            // ones outside them.
            if a.3 == b.3 && a.2 != b.2 {
                let (top, count) = loops.entry(a.3).or_insert((cell_tops[a.3] + 3 * GRID, 0));
                let y = *top;
                let out = if *count == 0 {
                    GRID
                } else {
                    STUB + *count * GRID
                };
                *top += GRID;
                *count += 1;
                let (l, r) = if a.2 { (a.1, b.1) } else { (b.1, a.1) };
                let pts = [
                    r,
                    Point::new(r.x + out, r.y),
                    Point::new(r.x + out, y),
                    Point::new(l.x - out, y),
                    Point::new(l.x - out, l.y),
                    l,
                ];
                page.wires.push(SchematicWire {
                    net: net.name.clone(),
                    points: pts.into_iter().map(xy).collect(),
                });
                wired.extend(ts.iter().copied());
                continue;
            }
            let neighbours = rows[a.3] == rows[b.3] && a.3.abs_diff(b.3) == 1;
            if neighbours && !a.2 && b.2 && a.1.x < b.1.x {
                let mx = ((a.1.x + b.1.x) / 2).div_euclid(GRID) * GRID;
                let mut pts = vec![a.1];
                if a.1.y != b.1.y {
                    pts.push(Point::new(mx, a.1.y));
                    pts.push(Point::new(mx, b.1.y));
                }
                pts.push(b.1);
                page.wires.push(SchematicWire {
                    net: net.name.clone(),
                    points: pts.into_iter().map(xy).collect(),
                });
                wired.extend(ts.iter().copied());
            }
        }
        // Every other connected pin: a stub and a label.
        for (t, (net, tip, left, _)) in tips.iter().enumerate() {
            let Some(n) = net else { continue };
            if wired.contains(&t) {
                continue;
            }
            let ni = &nl.nets[*n];
            let end = Point::new(tip.x + if *left { -STUB } else { STUB }, tip.y);
            page.wires.push(SchematicWire {
                net: ni.name.clone(),
                points: vec![xy(*tip), xy(end)],
            });
            let kind = if ni.global && is_rail(&ni.name, ni.class.as_deref()) {
                "power"
            } else if ni.global {
                "global"
            } else if net_sheets[*n].iter().any(|s| !s.starts_with(path)) {
                "hier"
            } else {
                "local"
            };
            page.labels.push(NetLabel {
                net: ni.name.clone(),
                x_mm: mm(end.x),
                y_mm: mm(end.y),
                kind: kind.to_owned(),
                side: side(*left),
            });
        }
        page.junctions = junctions(&page.wires);
        out.sheets.push(page);
    }
    // The hierarchy's order: by path text.
    out.sheets.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

fn side(left: bool) -> String {
    (if left { "left" } else { "right" }).to_owned()
}

/// The last path segment of a net name, as a label shows it.
fn short(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

fn mm_to_nm(v: f64) -> Coord {
    (v * 1.0e6).round() as Coord
}

/// A child sheet's box: wide enough for its path, tall enough for its pins on two sides.
fn sheet_box(path: &NamePath, pins: usize) -> (Coord, Coord) {
    let name = path.to_string().chars().count() as Coord;
    let w = ((name + 6) * GRID).max(16 * GRID);
    let h = ((pins.div_ceil(2) as Coord) * PIN_PITCH + 4 * GRID).max(10 * GRID);
    (w.div_euclid(GRID) * GRID, h.div_euclid(GRID) * GRID)
}

/// Breadth-first order over signal neighbours, most connected first.
fn connectivity_order(members: &[usize], neighbours: &[BTreeSet<usize>]) -> Vec<usize> {
    let set: BTreeSet<usize> = members.iter().copied().collect();
    let mut rest = members.to_vec();
    rest.sort_by_key(|k| {
        (
            std::cmp::Reverse(neighbours[*k].intersection(&set).count()),
            *k,
        )
    });
    let (mut order, mut seen) = (Vec::new(), BTreeSet::new());
    for &start in &rest {
        if !seen.insert(start) {
            continue;
        }
        let mut q = VecDeque::from([start]);
        while let Some(k) = q.pop_front() {
            order.push(k);
            for &n in &neighbours[k] {
                if set.contains(&n) && seen.insert(n) {
                    q.push_back(n);
                }
            }
        }
    }
    order
}

/// Shelf packing in rows of `width`: each cell's offset from the top left, its row, and the
/// height used.
fn shelves(sizes: &[(Coord, Coord)], width: Coord) -> (Vec<Point>, Vec<usize>, Coord) {
    let (mut x, mut y, mut row_h, mut row) = (0, 0, 0, 0);
    let mut pos = Vec::with_capacity(sizes.len());
    let mut rows = Vec::with_capacity(sizes.len());
    for (w, h) in sizes {
        if x + w > width && x > 0 {
            x = 0;
            y += row_h;
            row_h = 0;
            row += 1;
        }
        pos.push(Point::new(x, y));
        rows.push(row);
        x += w;
        row_h = row_h.max(*h);
    }
    (pos, rows, y + row_h)
}

/// Where three or more wire ends meet, or a wire ends inside another wire's segment.
fn junctions(wires: &[SchematicWire]) -> Vec<Xy> {
    let mut ends: BTreeMap<(i64, i64), usize> = BTreeMap::new();
    let key = |p: &Xy| ((p[0] * 1.0e4).round() as i64, (p[1] * 1.0e4).round() as i64);
    for w in wires {
        for p in [w.points.first(), w.points.last()].into_iter().flatten() {
            *ends.entry(key(p)).or_default() += 1;
        }
    }
    let mut out: BTreeSet<(i64, i64)> = ends
        .iter()
        .filter(|(_, n)| **n >= 3)
        .map(|(k, _)| *k)
        .collect();
    for k in ends.keys() {
        let p = [k.0 as f64 / 1.0e4, k.1 as f64 / 1.0e4];
        let inside = wires.iter().any(|w| {
            w.points.windows(2).any(|s| {
                let (a, b) = (s[0], s[1]);
                let on_x = (a[0] - b[0]).abs() < 1e-6
                    && (p[0] - a[0]).abs() < 1e-6
                    && p[1] > a[1].min(b[1]) + 1e-6
                    && p[1] < a[1].max(b[1]) - 1e-6;
                let on_y = (a[1] - b[1]).abs() < 1e-6
                    && (p[1] - a[1]).abs() < 1e-6
                    && p[0] > a[0].min(b[0]) + 1e-6
                    && p[0] < a[0].max(b[0]) - 1e-6;
                on_x || on_y
            })
        });
        if inside {
            out.insert(*k);
        }
    }
    out.into_iter()
        .map(|(x, y)| [x as f64 / 1.0e4, y as f64 / 1.0e4])
        .collect()
}

#[cfg(test)]
mod unit_split_tests {
    use super::*;

    #[test]
    fn a_large_gate_splits_into_even_bank_and_power_units() {
        let (_, _, nl, _) = crate::fixture::large(&crate::fixture::LargeParams {
            channels: 2,
            copper_layers: 4,
            route_every: 0,
            pours: false,
        });
        let fpga = nl
            .components
            .iter()
            .find(|c| c.pins.len() > SPLIT_ABOVE)
            .expect("an FPGA");
        let units = units(fpga);
        let sizes: Vec<usize> = units.iter().map(|u| u.pins.len()).collect();
        // 200 IO pins in five units of 40, then the power unit.
        assert_eq!(sizes, vec![40, 40, 40, 40, 40, 3], "{units:?}");
        assert_eq!(units[0].name, "A");
        assert_eq!(units[0].label.as_deref(), Some("IO0–IO39"));
        assert_eq!(units[5].label.as_deref(), Some("power"));
        let all: std::collections::BTreeSet<usize> =
            units.iter().flat_map(|u| u.pins.clone()).collect();
        assert_eq!(all.len(), fpga.pins.len(), "every pin in exactly one unit");
    }

    #[test]
    fn banks_come_from_the_pin_name() {
        assert_eq!(bank_of("IO_L1P_T0_34").as_deref(), Some("34"));
        assert_eq!(bank_of("BANK3_IO7").as_deref(), Some("3"));
        assert_eq!(bank_of("IO12"), None);
        assert_eq!(bank_of("VCC_"), None);
        assert_eq!(prefix_of("PA12"), "PA");
        assert_eq!(letters(0), "A");
        assert_eq!(letters(25), "Z");
        assert_eq!(letters(26), "AA");
    }
}
