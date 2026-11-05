//! The `.circuit.layout` board front end on the shared design grammar.
//!
//! A layout file holds one board: its stackups, outline, rules, nets, placement, copper and
//! annotations, the packages its footprints use, and the `sync { }` section
//! ([`super::layout`]). Every item is one statement, and every statement but the stackup and
//! package blocks is one line, so a board with 10k tracks diffs line by line.
//!
//! ```text
//! runmat 0.7
//! std 1
//!
//! use circuit ctl from "./ctl.circuit"
//!
//! layout ctl {
//!     stackup main {
//!         finish = ENIG
//!         layer "F.Cu" = { kind = copper, thickness = 0.035, weight = 1, material = copper }
//!         layer core = { kind = core, thickness = 1.51, material = fr4 }
//!         layer "B.Cu" = { kind = copper, thickness = 0.035, weight = 1, material = copper }
//!     }
//!     region flex = { stackup = flex2, outline = [[0, 0], [10, 0], [10, 5]], bends = [{ from = [2, 0], to = [2, 5], radius = 1, angle = 90 }] }
//!     outline = [[0, 0], [100, 0], [100, 80], [0, 80]]
//!     cutout = [[40, 30], [40, 50], [60, 50], [60, 30]]
//!     limits = { clearance = 0.15, track = 0.15, via_diameter = 0.45, drill = 0.2, … }
//!     net_class power = { track_width = 0.5 mm, clearance = 0.2 mm }
//!     rule hv = { select = ":net-class(hv)", clearance = 0.8 mm, priority = 2 }
//!     rule hv_lv = { between = [":net-class(hv)", ":net-class(lv)"], clearance = 1.5 mm }
//!     net GND : power
//!     net "+3V3" : power
//!     net SW
//!     room ch0 = { outline = [[0, 0], [20, 0], [20, 20], [0, 20]], members = [R1, C1] }
//!     footprint R1 = { package = R0402, at = [10.5, 20], rotation = 90, side = bottom, locked = true, pads = { "1" = GND, "2" = SW } }
//!     fiducial FID1 = { at = [3, 3], diameter = 1, opening = 2 }
//!     track GND = { layer = F.Cu, width = 0.25, from = [10, 20], to = [30, 20] }
//!     arc SW = { layer = F.Cu, width = 0.25, from = [0, 0], mid = [1, 1], to = [2, 0] }
//!     track = { layer = B.Cu, width = 0.2, from = [1, 1], to = [2, 2] }
//!     via GND = { at = [5, 5], size = 0.6, drill = 0.3, layers = [F.Cu, In1.Cu] }
//!     pour GND = { name = gnd, layers = [F.Cu, B.Cu], outline = [[0, 0], [100, 0], [100, 80]], priority = 1 }
//!     keepout antenna = { outline = [[…]], layers = [F.Cu], tracks = true, vias = true, pours = true }
//!     text "REV A" = { layer = F.SilkS, at = [90, 5], size = 1, thickness = 0.15 }
//!     graphic line = { layer = F.SilkS, points = [[0, 0], [5, 0]], width = 0.12 }
//!     dimension = { layer = "User.1", from = [0, 0], to = [100, 0], offset = 5, precision = 2 }
//!     package R0402 {
//!         uuid = "…"
//!         smd = true
//!         pad "1" = { shape = roundrect, size = [0.5, 0.6], ratio = 0.25, at = [-0.5, 0] }
//!         courtyard = [[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5]]
//!         silk line = { points = [[-0.2, 0.4], [0.2, 0.4]], width = 0.12 }
//!     }
//!     sync { … }
//! }
//! ```
//!
//! - **Lengths** are millimetres, exact to the nanometre; any length may carry a unit
//!   (`50 mil`). **Angles** are degrees.
//! - **Nets** are named where they are used (`track GND = …`); a statement with no net leaves
//!   out the name (`track = …`). Every net is declared with `net <name> [: <class>]`.
//! - **Footprints** name their package; the package is defined once in the file, so a layout
//!   is self-contained. The link from a footprint to its schematic component lives only in
//!   `sync { }`.
//! - A field at its default is left out (`rotation = 0`, `side = top`, `locked = false`).
//! - **Sites** key each statement (`footprint.R1`, `track[17]`, `net.GND`, `package.R0402`)
//!   and map to its span, for diagnostics, selection and structured edits ([`ParsedBoard`]).
//!
//! The reader assigns item ids in source order; [`canonical`] renumbers a board the same way,
//! so `parse(print(b)) == canonical(b)` and printing is a fixed point.

use super::values::*;
use crate::geom::poly::Polygon;
use crate::layout::*;
use crate::library::{
    Package,
};
use runmat_design_syntax::ast::Headed;
use runmat_design_syntax::document::UseItem;
use runmat_design_syntax::{parse_source, Dialect, FeaSyntaxDiagnostic};
use runmat_types::Span;
use std::collections::{BTreeMap, HashMap};

mod read;
mod items;
mod limits;
mod stackup;
mod package;
mod graphic;
mod print;

pub use read::*;
use items::*;
use limits::*;
use stackup::*;
use package::*;
use graphic::*;
pub use print::*;


/// The `.circuit.layout` statement words: one registry for the board and its sync section.
pub use super::layout::words;

mod reparse;
mod repackage;
pub(crate) use reparse::{item_of, read_window, window_items, window_replaced};
pub use reparse::{reparse_board, reparse_board_keeping, BoardDelta, BoardItem};

/// The words of item statements: each reads into one board item with its own id.
const ITEM_WORDS: [&str; 10] = [
    words::FOOTPRINT,
    words::FIDUCIAL,
    words::TRACK,
    words::ARC,
    words::VIA,
    words::POUR,
    words::KEEPOUT,
    words::TEXT,
    words::GRAPHIC,
    words::DIMENSION,
];

/// The section a statement belongs to: statements print grouped, in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Group {
    Stackup,
    Region,
    Outline,
    Limits,
    NetClass,
    Rule,
    Net,
    Room,
    Footprint,
    Fiducial,
    Copper,
    Via,
    Pour,
    Keepout,
    Text,
    Graphic,
    Dimension,
    Package,
    Sync,
}

/// One statement as the printer writes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Printed {
    pub group: Group,
    pub site: String,
    /// The statement, its first line unindented and the rest indented for a layout block.
    pub text: String,
    pub item: Option<ItemId>,
}

/// One statement as the reader found it.
#[derive(Clone, Debug, PartialEq)]
pub struct BoardStatement {
    pub group: Group,
    pub site: String,
    /// The statement without surrounding trivia.
    pub span: Span,
    pub item: Option<ItemId>,
}

/// A parsed `.circuit.layout` file.
#[derive(Clone, Debug)]
pub struct ParsedBoard {
    /// The board, when the source is free of errors.
    pub board: Option<Board>,
    pub name: Option<String>,
    /// The file's `use` lines.
    pub uses: Vec<UseItem>,
    /// Every statement the reader took, in source order.
    pub statements: Vec<BoardStatement>,
    /// Site → statement span.
    pub spans: BTreeMap<String, Span>,
    /// Item → its statement's site.
    pub sites: BTreeMap<ItemId, String>,
    /// The layout declaration's block, `{` to `}`.
    pub block: Option<Span>,
    pub diagnostics: Vec<FeaSyntaxDiagnostic>,
    /// The packages the file defines, by the name footprints give them.
    pub packages: HashMap<String, Package>,
}

/* ------------------------------------------------------------------------------------------------
 * Sites
 * --------------------------------------------------------------------------------------------- */

/// Site keys, issued in statement order by the reader and the printer alike.
#[derive(Default)]
struct Sites {
    counts: HashMap<&'static str, usize>,
    named: HashMap<String, usize>,
}

impl Sites {
    fn indexed(&mut self, word: &'static str) -> String {
        let n = self.counts.entry(word).or_default();
        *n += 1;
        format!("{word}[{}]", *n - 1)
    }
    fn named(&mut self, word: &str, name: &str) -> String {
        let site = format!("{word}.{name}");
        let n = self.named.entry(site.clone()).or_default();
        *n += 1;
        if *n == 1 {
            site
        } else {
            format!("{site}~{n}")
        }
    }
}

/* ------------------------------------------------------------------------------------------------
 * Layers and enums
 * --------------------------------------------------------------------------------------------- */

fn side_name(s: Side) -> &'static str {
    match s {
        Side::Top => "top",
        Side::Bottom => "bottom",
    }
}

fn side_of(v: &Val) -> Read<Side> {
    match text(v)?.as_str() {
        "top" => Ok(Side::Top),
        "bottom" => Ok(Side::Bottom),
        other => Err(diagnostic(
            v.span,
            format!("`{other}` is not a side; expected `top` or `bottom`"),
        )),
    }
}

fn fixed_layer_name(l: Layer) -> Option<String> {
    let f = |s: Side, top: &str, bottom: &str| {
        Some(match s {
            Side::Top => top.to_owned(),
            Side::Bottom => bottom.to_owned(),
        })
    };
    match l {
        Layer::Copper(_) => None,
        Layer::Mask(s) => f(s, "F.Mask", "B.Mask"),
        Layer::Paste(s) => f(s, "F.Paste", "B.Paste"),
        Layer::Silk(s) => f(s, "F.SilkS", "B.SilkS"),
        Layer::Fab(s) => f(s, "F.Fab", "B.Fab"),
        Layer::Courtyard(s) => f(s, "F.CrtYd", "B.CrtYd"),
        Layer::Edge => Some("Edge.Cuts".into()),
        Layer::User(n) => Some(format!("User.{n}")),
    }
}

/// Layer names of one board: its copper layers by stackup name, and the fixed layers.
struct Layers {
    copper: Vec<String>,
    texts: Vec<String>,
    by_name: HashMap<String, u8>,
}

impl Layers {
    fn of(stackup: &Stackup) -> Layers {
        let copper = stackup.copper_names();
        let n = copper.len() as u8;
        let mut by_name = HashMap::new();
        // The conventional names first, so a stackup's own names win.
        for i in 0..n {
            by_name.insert(copper_name(i, n), i);
        }
        for (i, name) in copper.iter().enumerate() {
            by_name.insert(name.clone(), i as u8);
        }
        Layers {
            texts: copper.iter().map(|c| dotted_text(c)).collect(),
            copper,
            by_name,
        }
    }

    fn copper_text(&self, i: u8) -> String {
        self.texts
            .get(i as usize)
            .cloned()
            .unwrap_or_else(|| dotted_text(&format!("Cu{i}")))
    }

    fn layer_text(&self, l: Layer) -> String {
        match l {
            Layer::Copper(i) => self.copper_text(i),
            other => dotted_text(&fixed_layer_name(other).unwrap_or_default()),
        }
    }

    fn copper(&self, v: &Val) -> Read<u8> {
        let name = text(v)?;
        self.by_name.get(&name).copied().ok_or_else(|| {
            diagnostic(
                v.span,
                format!(
                    "`{name}` is not a copper layer of this stackup; it has {}",
                    self.copper.join(", ")
                ),
            )
        })
    }

    fn layer(&self, v: &Val) -> Read<Layer> {
        let name = text(v)?;
        if let Some(i) = self.by_name.get(&name) {
            return Ok(Layer::Copper(*i));
        }
        let fixed = [
            Layer::Mask(Side::Top),
            Layer::Mask(Side::Bottom),
            Layer::Paste(Side::Top),
            Layer::Paste(Side::Bottom),
            Layer::Silk(Side::Top),
            Layer::Silk(Side::Bottom),
            Layer::Fab(Side::Top),
            Layer::Fab(Side::Bottom),
            Layer::Courtyard(Side::Top),
            Layer::Courtyard(Side::Bottom),
            Layer::Edge,
        ];
        if let Some(l) = fixed
            .into_iter()
            .find(|l| fixed_layer_name(*l).as_deref() == Some(name.as_str()))
        {
            return Ok(l);
        }
        if let Some(n) = name
            .strip_prefix("User.")
            .and_then(|n| n.parse::<u8>().ok())
        {
            return Ok(Layer::User(n));
        }
        Err(diagnostic(
            v.span,
            format!("`{name}` is not a layer of this board"),
        ))
    }
}

fn connect_name(c: PadConnect) -> &'static str {
    match c {
        PadConnect::Solid => "solid",
        PadConnect::Thermal => "thermal",
        PadConnect::ThermalThtOnly => "thermal_tht",
        PadConnect::None => "no_connection",
    }
}

fn connect_of(v: &Val) -> Read<PadConnect> {
    Ok(match text(v)?.as_str() {
        "solid" => PadConnect::Solid,
        "thermal" => PadConnect::Thermal,
        "thermal_tht" => PadConnect::ThermalThtOnly,
        "no_connection" => PadConnect::None,
        other => {
            return Err(diagnostic(
                v.span,
                format!(
                    "`{other}` is not a pad connection; expected solid, thermal, thermal_tht \
                     or no_connection"
                ),
            ))
        }
    })
}

const NM2_PER_MM2: f64 = 1e12;

fn islands_text(i: IslandPolicy) -> String {
    match i {
        IslandPolicy::Remove => "remove".into(),
        IslandPolicy::Keep => "keep".into(),
        IslandPolicy::RemoveBelow(area) => {
            let mm2 = area / NM2_PER_MM2;
            if mm2 * NM2_PER_MM2 == area {
                format!("{} mm2", f64_text(mm2))
            } else {
                format!("{} nm2", f64_text(area))
            }
        }
    }
}

fn islands_of(v: &Val) -> Read<IslandPolicy> {
    match &v.v {
        V::Word(w) if w == "remove" => Ok(IslandPolicy::Remove),
        V::Word(w) if w == "keep" => Ok(IslandPolicy::Keep),
        V::Qty(t, unit) if unit == "mm2" || unit == "nm2" => {
            let a: f64 = t
                .parse()
                .map_err(|_| diagnostic(v.span, "expected an area"))?;
            Ok(IslandPolicy::RemoveBelow(if unit == "mm2" {
                a * NM2_PER_MM2
            } else {
                a
            }))
        }
        _ => Err(diagnostic(
            v.span,
            "islands are `remove`, `keep` or an area to remove below (`0.5 mm2`)",
        )),
    }
}
