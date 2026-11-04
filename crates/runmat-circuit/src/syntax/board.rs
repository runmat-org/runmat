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

use super::layout::{print_sync, read_sync_section, SyncSection};
use super::values::*;
use crate::board_rules::{RuleSpec, RuleValue};
use crate::geom::poly::Polygon;
use crate::geom::{mm, Coord, Point};
use crate::ids::Uuid;
use crate::layout::*;
use crate::library::{
    Body, CopperSpan, Drill, Graphic, Model3d, Package, PadDef, PadKind, PadShape,
};
use runmat_design_syntax::ast::{self, Expr, Headed, Item, Statement};
use runmat_design_syntax::document::UseItem;
use runmat_design_syntax::json::INDENT;
use runmat_design_syntax::{parse_source, Dialect, FeaSyntaxDiagnostic};
use runmat_rules::{SourceKey, SourceOrder};
use runmat_types::Span;
use std::collections::{BTreeMap, HashMap};

/// The `.circuit.layout` statement words: the board front end's registry.
pub mod words {
    pub const LAYOUT: &str = "layout";
    pub const STACKUP: &str = "stackup";
    pub const LAYER: &str = "layer";
    pub const FINISH: &str = "finish";
    pub const REGION: &str = "region";
    pub const OUTLINE: &str = "outline";
    pub const CUTOUT: &str = "cutout";
    pub const LIMITS: &str = "limits";
    pub const NET_CLASS: &str = "net_class";
    pub const RULE: &str = "rule";
    pub const NET: &str = "net";
    pub const ROOM: &str = "room";
    pub const FOOTPRINT: &str = "footprint";
    pub const FIDUCIAL: &str = "fiducial";
    pub const TRACK: &str = "track";
    pub const ARC: &str = "arc";
    pub const VIA: &str = "via";
    pub const POUR: &str = "pour";
    pub const KEEPOUT: &str = "keepout";
    pub const TEXT: &str = "text";
    pub const GRAPHIC: &str = "graphic";
    pub const DIMENSION: &str = "dimension";
    pub const PACKAGE: &str = "package";
    pub const NAME: &str = "name";
    pub const UUID: &str = "uuid";
    pub const DESCRIPTION: &str = "description";
    pub const SMD: &str = "smd";
    pub const PAD: &str = "pad";
    pub const COURTYARD: &str = "courtyard";
    pub const SILK: &str = "silk";
    pub const FAB: &str = "fab";
    pub const MODEL: &str = "model";
    pub const BODY: &str = "body";
    pub const SYNC: &str = super::super::layout::words::SYNC;
    /// Every statement word (board, stackup and package statements), for completion and the
    /// highlighter.
    pub const ALL: &[&str] = &[
        LAYOUT,
        STACKUP,
        LAYER,
        FINISH,
        REGION,
        OUTLINE,
        CUTOUT,
        LIMITS,
        NET_CLASS,
        RULE,
        NET,
        ROOM,
        FOOTPRINT,
        FIDUCIAL,
        TRACK,
        ARC,
        VIA,
        POUR,
        KEEPOUT,
        TEXT,
        GRAPHIC,
        DIMENSION,
        PACKAGE,
        NAME,
        UUID,
        DESCRIPTION,
        SMD,
        PAD,
        COURTYARD,
        SILK,
        FAB,
        MODEL,
        BODY,
        SYNC,
    ];
}

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

/* ------------------------------------------------------------------------------------------------
 * Reading
 * --------------------------------------------------------------------------------------------- */

struct Reader<'a> {
    path: &'a str,
    out: ParsedBoard,
    sites: Sites,
}

/// A statement's head: its word, name and value.
struct Head {
    word: String,
    name: Option<(String, Span)>,
    type_text: Option<String>,
    value: Option<Val>,
    span: Span,
}

fn head(statement: &Statement) -> Read<Option<Head>> {
    let span = statement.span();
    let value = |e: Option<Expr>| e.as_ref().map(val).transpose();
    Ok(Some(match statement {
        Statement::Value(v) => Head {
            word: v.keyword(),
            name: v.name().map(|n| (n.text(), n.span())),
            type_text: v.type_text(),
            value: value(v.value())?,
            span,
        },
        Statement::Assignment(a) => {
            let target = a.target();
            if target.len() != 1 {
                return Err(diagnostic(
                    a.target_span(),
                    "expected a statement word such as `outline` or `track`",
                ));
            }
            Head {
                word: target[0].clone(),
                name: None,
                type_text: None,
                value: value(a.value())?,
                span,
            }
        }
        Statement::Command(c) => {
            // `net GND` / `net "+3V3"`: a name and nothing else.
            let name = match c.argument() {
                None => None,
                Some(Expr::Reference(r)) => Some((r.segments().join("."), r.span())),
                Some(Expr::String(s)) => Some((s.value(), s.span())),
                Some(other) => {
                    return Err(diagnostic(
                        other.span(),
                        format!("expected `{} <name> = <value>`", c.word()),
                    ))
                }
            };
            Head {
                word: c.word(),
                name,
                type_text: None,
                value: None,
                span,
            }
        }
        _ => return Ok(None),
    }))
}

fn unquote_type(t: &str) -> String {
    match t.strip_prefix('"').and_then(|t| t.strip_suffix('"')) {
        Some(inner) => inner.replace("\\\"", "\"").replace("\\\\", "\\"),
        None => t.to_owned(),
    }
}

fn need_value(h: &Head) -> Read<&Val> {
    h.value
        .as_ref()
        .ok_or_else(|| diagnostic(h.span, format!("`{}` needs `= <value>`", h.word)))
}

fn need_name(h: &Head) -> Read<String> {
    h.name
        .as_ref()
        .map(|(n, _)| n.clone())
        .ok_or_else(|| diagnostic(h.span, format!("`{}` needs a name", h.word)))
}

fn polygon(f: &mut Fields) -> Read<Polygon> {
    let outer = ring(f.need(words::OUTLINE)?)?;
    let holes = f.or("holes", Vec::new(), rings)?;
    Ok(Polygon { outer, holes })
}

/// Parses a `.circuit.layout` source.
pub fn parse_board(path: &str, source: &str) -> ParsedBoard {
    let syntax = parse_source(Dialect::CircuitLayout, source);
    let mut reader = Reader {
        path,
        out: ParsedBoard {
            board: None,
            name: None,
            uses: Vec::new(),
            statements: Vec::new(),
            spans: BTreeMap::new(),
            sites: BTreeMap::new(),
            block: None,
            diagnostics: syntax.diagnostics.clone(),
        },
        sites: Sites::default(),
    };
    let mut board = None;
    for item in syntax.file().items() {
        match item {
            Item::Use(u) => match use_item(&u) {
                Ok(item) => reader.out.uses.push(item),
                Err(e) => reader.out.diagnostics.push(e),
            },
            Item::Import(i) => reader.out.diagnostics.push(diagnostic(
                i.span(),
                "a layout reads other files with `use`",
            )),
            Item::Declaration(d) if d.keyword() == words::LAYOUT => {
                if board.is_some() {
                    reader
                        .out
                        .diagnostics
                        .push(diagnostic(d.keyword_span(), "a file holds one `layout`"));
                    continue;
                }
                board = Some(reader.layout(&d));
            }
            Item::Declaration(_) | Item::Error(_) => {}
        }
    }
    if board.is_none() {
        reader.out.diagnostics.push(diagnostic(
            Span::default(),
            "expected a `layout <name> { … }` declaration",
        ));
    }
    let mut out = reader.out;
    for d in &mut out.diagnostics {
        d.span = trim_span(source, d.span);
    }
    if out.diagnostics.is_empty() {
        out.board = board;
    }
    out
}

pub(crate) fn use_item(u: &ast::UseDeclaration) -> Read<UseItem> {
    let from = match u.source() {
        Some(ast::UseSource::Path(path)) => path,
        Some(ast::UseSource::Package(segments)) => segments.join("."),
        None => return Err(diagnostic(u.span(), "`use` needs a source")),
    };
    if u.overrides().is_some() {
        return Err(diagnostic(
            u.span(),
            "a `use` here takes no `{ … }` overrides",
        ));
    }
    Ok(UseItem {
        symbol: u.symbol().map(|(kind, name)| (kind.text(), name.text())),
        from,
        alias: u.alias().map(|a| a.text()),
        revision: u.pin(),
        span: u.span(),
    })
}

impl Reader<'_> {
    fn note<T>(&mut self, r: Read<T>) -> Option<T> {
        match r {
            Ok(v) => Some(v),
            Err(e) => {
                self.out.diagnostics.push(e);
                None
            }
        }
    }

    fn record(&mut self, group: Group, site: String, span: Span, item: Option<ItemId>) {
        if let Some(id) = item {
            self.out.sites.insert(id, site.clone());
        }
        self.out.spans.insert(site.clone(), span);
        self.out.statements.push(BoardStatement {
            group,
            site,
            span,
            item,
        });
    }

    fn layout(&mut self, d: &ast::Declaration) -> Board {
        let name = d.name().map(|n| n.text()).unwrap_or_default();
        let stem = Dialect::CircuitLayout.stem(self.path);
        if !self.path.is_empty() && name != stem {
            self.out.diagnostics.push(diagnostic(
                d.name().map_or(d.span(), |n| n.span()),
                format!("this layout is named `{name}`; its file is `{stem}`"),
            ));
        }
        self.out.name = Some(name.clone());
        let statements = match d.block() {
            Some(block) => {
                self.out.block = Some(block.span());
                block.statements()
            }
            None => Vec::new(),
        };
        // Pass 1: what the rest refers to — stackups, packages and nets.
        let mut stackups: Vec<Stackup> = Vec::new();
        let mut packages: HashMap<String, Package> = HashMap::new();
        let mut nets: Vec<BoardNet> = Vec::new();
        let mut net_ids: HashMap<String, NetId> = HashMap::new();
        for statement in &statements {
            match statement {
                Statement::Block(b) if b.keyword() == words::STACKUP => {
                    let name = b.name().map(|n| n.text()).unwrap_or_default();
                    let site = self.sites.named(words::STACKUP, &name);
                    self.record(Group::Stackup, site, statement.span(), None);
                    if let Some(s) = self.note(read_stackup(b, name)) {
                        if stackups.iter().any(|x| x.name == s.name) {
                            self.out.diagnostics.push(diagnostic(
                                b.keyword_span(),
                                format!("stackup `{}` is given twice", s.name),
                            ));
                        }
                        stackups.push(s);
                    }
                }
                Statement::Block(b) if b.keyword() == words::PACKAGE => {
                    let key = b.name().map(|n| n.text()).unwrap_or_default();
                    let site = self.sites.named(words::PACKAGE, &key);
                    self.record(Group::Package, site, statement.span(), None);
                    if let Some(p) = self.note(read_package(b, &key)) {
                        if packages.insert(key.clone(), p).is_some() {
                            self.out.diagnostics.push(diagnostic(
                                b.keyword_span(),
                                format!("package `{key}` is given twice"),
                            ));
                        }
                    }
                }
                _ => {
                    let is_net = match statement {
                        Statement::Value(v) => v.keyword() == words::NET,
                        Statement::Command(c) => c.word() == words::NET,
                        _ => false,
                    };
                    if !is_net {
                        continue;
                    }
                    let Ok(Some(h)) = head(statement) else {
                        continue;
                    };
                    if h.word != words::NET {
                        continue;
                    }
                    let Some(name) = self.note(need_name(&h)) else {
                        continue;
                    };
                    let site = self.sites.named(words::NET, &name);
                    self.record(Group::Net, site, h.span, None);
                    if h.value.is_some() {
                        self.out.diagnostics.push(diagnostic(
                            h.span,
                            "a net is `net <name>` or `net <name> : <class>`",
                        ));
                    }
                    if net_ids.contains_key(&name) {
                        self.out
                            .diagnostics
                            .push(diagnostic(h.span, format!("net `{name}` is given twice")));
                        continue;
                    }
                    net_ids.insert(name.clone(), NetId(nets.len() as u32));
                    nets.push(BoardNet {
                        name,
                        class: h.type_text.as_deref().map(unquote_type),
                    });
                }
            }
        }
        let mut stackups = stackups.into_iter();
        let main = stackups
            .next()
            .unwrap_or_else(|| Stackup::standard(2, mm(1.6), 1.0, 0.5));
        let mut board = Board::new(&name, Polygon::default(), main);
        board.stackups = stackups.map(|s| (s.name.clone(), s)).collect();
        board.nets = nets;
        let layers = Layers::of(&board.stackup);
        let ctx = Ctx {
            layers: &layers,
            nets: &net_ids,
            packages: &packages,
        };
        // Pass 2: everything else, in source order.
        let mut sync: Option<(SyncSection, Span)> = None;
        let mut outline_seen = false;
        let mut limits_seen = false;
        for statement in &statements {
            let at = statement.span();
            if let Statement::Block(b) = statement {
                match b.keyword().as_str() {
                    words::STACKUP | words::PACKAGE => {}
                    words::SYNC => {
                        self.record(Group::Sync, "sync".into(), at, None);
                        if sync.is_some() {
                            self.out.diagnostics.push(diagnostic(
                                b.keyword_span(),
                                "a layout holds one `sync` section",
                            ));
                            continue;
                        }
                        if let Some(name) = b.name() {
                            self.out
                                .diagnostics
                                .push(diagnostic(name.span(), "the `sync` section takes no name"));
                        }
                        let inner = b.block().map(|b| b.statements()).unwrap_or_default();
                        let (section, errors) = read_sync_section(&inner, at);
                        self.out.diagnostics.extend(errors);
                        if let Some(section) = section {
                            sync = Some((section, at));
                        }
                    }
                    other => self.out.diagnostics.push(diagnostic(
                        b.keyword_span(),
                        format!("`{other} {{ … }}` is not a layout section"),
                    )),
                }
                continue;
            }
            let h = match head(statement) {
                Ok(Some(h)) => h,
                Ok(None) => {
                    self.out.diagnostics.push(diagnostic(
                        at,
                        "expected a board statement such as `footprint`, `track` or `via`",
                    ));
                    continue;
                }
                Err(e) => {
                    self.out.diagnostics.push(e);
                    continue;
                }
            };
            match h.word.as_str() {
                words::NET => {}
                words::OUTLINE => {
                    self.record(Group::Outline, "outline".into(), at, None);
                    if outline_seen {
                        self.out
                            .diagnostics
                            .push(diagnostic(at, "`outline` is given twice"));
                    }
                    outline_seen = true;
                    if let Some(r) = self.note(need_value(&h).and_then(ring)) {
                        board.outline.outer = r;
                    }
                }
                words::CUTOUT => {
                    let site = self.sites.indexed(words::CUTOUT);
                    self.record(Group::Outline, site, at, None);
                    if let Some(r) = self.note(need_value(&h).and_then(ring)) {
                        board.outline.holes.push(r);
                    }
                }
                words::LIMITS => {
                    self.record(Group::Limits, "limits".into(), at, None);
                    if limits_seen {
                        self.out
                            .diagnostics
                            .push(diagnostic(at, "`limits` is given twice"));
                    }
                    limits_seen = true;
                    if let Some(l) = self.note(need_value(&h).and_then(read_limits)) {
                        board.limits = l;
                    }
                }
                words::REGION => {
                    let r = need_name(&h).and_then(|name| {
                        let mut f = Fields::of(need_value(&h)?, "region")?;
                        let stackup = text(f.need(words::STACKUP)?)?;
                        let outline = polygon(&mut f)?;
                        let bends = f.or("bends", Vec::new(), |v| {
                            list(v)?.iter().map(read_bend).collect()
                        })?;
                        f.finish()?;
                        Ok(StackRegion {
                            name,
                            outline,
                            stackup,
                            bends,
                        })
                    });
                    let site = self
                        .sites
                        .named(words::REGION, &h.name.clone().unwrap_or_default().0);
                    self.record(Group::Region, site, at, None);
                    if let Some(r) = self.note(r) {
                        board.regions.push(r);
                    }
                }
                words::NET_CLASS => {
                    let name = h.name.clone().unwrap_or_default().0;
                    let site = self.sites.named(words::NET_CLASS, &name);
                    self.record(Group::NetClass, site, at, None);
                    let r = need_name(&h).and_then(|_| {
                        let mut f = Fields::of(need_value(&h)?, "net class")?;
                        f.rest()
                            .into_iter()
                            .map(|(k, v)| Ok((k.to_owned(), rule_value(v)?)))
                            .collect::<Read<Vec<_>>>()
                    });
                    if let Some(decls) = self.note(r) {
                        if board.net_class_rules.insert(name.clone(), decls).is_some() {
                            self.out
                                .diagnostics
                                .push(diagnostic(at, format!("net class `{name}` is given twice")));
                        }
                    }
                }
                words::RULE => {
                    let name = h.name.clone().unwrap_or_default().0;
                    let site = self.sites.named(words::RULE, &name);
                    self.record(Group::Rule, site, at, None);
                    let order = SourceOrder {
                        file_rank: 0,
                        offset: at.start as u32,
                    };
                    let key = SourceKey::new(self.path, at.start as u32, at.end as u32);
                    let r = need_name(&h)
                        .and_then(|id| read_rule(&id, need_value(&h)?))
                        .map(|r| r.at(key, order));
                    if let Some(r) = self.note(r) {
                        if board.rules.iter().any(|x| x.id == r.id) {
                            self.out
                                .diagnostics
                                .push(diagnostic(at, format!("rule `{}` is given twice", r.id)));
                        }
                        board.rules.push(r);
                    }
                }
                words::ROOM => {
                    let name = h.name.clone().unwrap_or_default().0;
                    let site = self.sites.named(words::ROOM, &name);
                    self.record(Group::Room, site, at, None);
                    let r = need_name(&h).and_then(|name| {
                        let mut f = Fields::of(need_value(&h)?, "room")?;
                        let outline = polygon(&mut f)?;
                        let members = f.or("members", Vec::new(), texts)?;
                        f.finish()?;
                        Ok(Room {
                            name,
                            outline,
                            members,
                        })
                    });
                    if let Some(r) = self.note(r) {
                        board.rooms.push(r);
                    }
                }
                words::FOOTPRINT => {
                    let name = h.name.clone().unwrap_or_default().0;
                    let site = self.sites.named(words::FOOTPRINT, &name);
                    let id = board.alloc();
                    self.record(Group::Footprint, site, at, Some(id));
                    let r = need_name(&h).and_then(|r| ctx.footprint(id, r, need_value(&h)?));
                    if let Some(f) = self.note(r) {
                        board.footprints.insert(id, f);
                    }
                }
                words::FIDUCIAL => {
                    let name = h.name.clone().unwrap_or_default().0;
                    let site = self.sites.named(words::FIDUCIAL, &name);
                    let id = board.alloc();
                    self.record(Group::Fiducial, site, at, Some(id));
                    let r = need_name(&h).and_then(|name| {
                        let mut f = Fields::of(need_value(&h)?, "fiducial")?;
                        let fid = Fiducial {
                            id,
                            name,
                            at: point(f.need("at")?)?,
                            side: f.or("side", Side::Top, side_of)?,
                            diameter: len(f.need("diameter")?)?,
                            opening: len(f.need("opening")?)?,
                            locked: f.or("locked", false, boolean)?,
                        };
                        f.finish()?;
                        Ok(fid)
                    });
                    if let Some(f) = self.note(r) {
                        board.fiducials.insert(id, f);
                    }
                }
                words::TRACK | words::ARC => {
                    let word = if h.word == words::TRACK {
                        words::TRACK
                    } else {
                        words::ARC
                    };
                    let site = self.sites.indexed(word);
                    let id = board.alloc();
                    self.record(Group::Copper, site, at, Some(id));
                    let r = ctx.track(id, word, &h);
                    if let Some(t) = self.note(r) {
                        board.tracks.insert(id, t);
                    }
                }
                words::VIA => {
                    let site = self.sites.indexed(words::VIA);
                    let id = board.alloc();
                    self.record(Group::Via, site, at, Some(id));
                    let n = board.copper_count();
                    let r = ctx.via(id, &h, n);
                    if let Some(v) = self.note(r) {
                        board.vias.insert(id, v);
                    }
                }
                words::POUR => {
                    let site = self.sites.indexed(words::POUR);
                    let id = board.alloc();
                    self.record(Group::Pour, site, at, Some(id));
                    let r = ctx.pour(id, &h);
                    if let Some(z) = self.note(r) {
                        board.zones.insert(id, z);
                    }
                }
                words::KEEPOUT => {
                    let site = self.sites.indexed(words::KEEPOUT);
                    let id = board.alloc();
                    self.record(Group::Keepout, site, at, Some(id));
                    let r = ctx.keepout(id, &h);
                    if let Some(k) = self.note(r) {
                        board.keepouts.insert(id, k);
                    }
                }
                words::TEXT => {
                    let site = self.sites.indexed(words::TEXT);
                    let id = board.alloc();
                    self.record(Group::Text, site, at, Some(id));
                    let r = need_name(&h).and_then(|text| {
                        let mut f = Fields::of(need_value(&h)?, "text")?;
                        let t = BoardText {
                            id,
                            layer: layers.layer(f.need("layer")?)?,
                            at: point(f.need("at")?)?,
                            text,
                            size: len(f.need("size")?)?,
                            thickness: len(f.need("thickness")?)?,
                            angle: f.or("angle", 0.0, angle)?,
                            mirrored: f.or("mirrored", false, boolean)?,
                        };
                        f.finish()?;
                        Ok(t)
                    });
                    if let Some(t) = self.note(r) {
                        board.texts.insert(id, t);
                    }
                }
                words::GRAPHIC => {
                    let site = self.sites.indexed(words::GRAPHIC);
                    let id = board.alloc();
                    self.record(Group::Graphic, site, at, Some(id));
                    let r = need_name(&h).and_then(|kind| {
                        let mut f = Fields::of(need_value(&h)?, "graphic")?;
                        let layer = layers.layer(f.need("layer")?)?;
                        let graphic = read_graphic(&kind, &mut f, h.span)?;
                        f.finish()?;
                        Ok(BoardGraphic { id, layer, graphic })
                    });
                    if let Some(g) = self.note(r) {
                        board.graphics.insert(id, g);
                    }
                }
                words::DIMENSION => {
                    let site = self.sites.indexed(words::DIMENSION);
                    let id = board.alloc();
                    self.record(Group::Dimension, site, at, Some(id));
                    let r = need_value(&h).and_then(|v| {
                        let mut f = Fields::of(v, "dimension")?;
                        let d = Dimension {
                            id,
                            layer: layers.layer(f.need("layer")?)?,
                            a: point(f.need("from")?)?,
                            b: point(f.need("to")?)?,
                            offset: f.or("offset", 0, len)?,
                            precision: f.or("precision", 2, |v| {
                                u8::try_from(whole(v)?)
                                    .map_err(|_| diagnostic(v.span, "precision is 0 to 255"))
                            })?,
                        };
                        f.finish()?;
                        Ok(d)
                    });
                    if let Some(d) = self.note(r) {
                        board.dimensions.insert(id, d);
                    }
                }
                other => self.out.diagnostics.push(diagnostic(
                    at,
                    format!(
                        "`{other}` does not start a board statement; expected stackup, region, \
                         outline, cutout, limits, net_class, rule, net, room, footprint, \
                         fiducial, track, arc, via, pour, keepout, text, graphic, dimension, \
                         package or sync"
                    ),
                )),
            }
        }
        self.out.statements.sort_by_key(|s| s.span.start);
        if let Some((section, at)) = sync {
            board.sync = section.sync;
            let by_ref: HashMap<&str, ItemId> = board
                .footprints
                .values()
                .map(|f| (f.reference.as_str(), f.id))
                .collect();
            let mut links = Vec::new();
            for (reference, path) in &section.links {
                match by_ref.get(reference.as_str()) {
                    Some(id) => links.push((*id, path.clone())),
                    None => self.out.diagnostics.push(diagnostic(
                        at,
                        format!("the sync section links footprint `{reference}`, which the board does not have"),
                    )),
                }
            }
            for (id, path) in links {
                if let Some(f) = board.footprints.get_mut(&id) {
                    f.component = Some(path);
                }
            }
        }
        board
    }
}

struct Ctx<'a> {
    layers: &'a Layers,
    nets: &'a HashMap<String, NetId>,
    packages: &'a HashMap<String, Package>,
}

impl Ctx<'_> {
    fn net(&self, name: &str, span: Span) -> Read<NetId> {
        self.nets.get(name).copied().ok_or_else(|| {
            diagnostic(
                span,
                format!("`{name}` is not a net of this board; declare it with `net {name}`"),
            )
        })
    }

    fn head_net(&self, h: &Head) -> Read<Option<NetId>> {
        h.name
            .as_ref()
            .map(|(n, span)| self.net(n, *span))
            .transpose()
    }

    fn footprint(&self, id: ItemId, reference: String, v: &Val) -> Read<Footprint> {
        let mut f = Fields::of(v, "footprint")?;
        let pv = f.need(words::PACKAGE)?;
        let key = text(pv)?;
        let package = self.packages.get(&key).cloned().ok_or_else(|| {
            diagnostic(
                pv.span,
                format!("package `{key}` is not defined in this layout"),
            )
        })?;
        let mut pad_nets = BTreeMap::new();
        if let Some(pads) = f.get("pads") {
            let V::Rec(entries) = &pads.v else {
                return Err(diagnostic(pads.span, "pads are `{ \"1\" = <net>, … }`"));
            };
            for (pad, net) in entries {
                pad_nets.insert(pad.clone(), self.net(&text(net)?, net.span)?);
            }
        }
        let properties = match f.get("properties") {
            Some(p) => {
                let V::Rec(entries) = &p.v else {
                    return Err(diagnostic(p.span, "properties are `{ <name> = \"…\", … }`"));
                };
                entries
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), text(v)?)))
                    .collect::<Read<_>>()?
            }
            None => BTreeMap::new(),
        };
        let fp = Footprint {
            id,
            reference,
            component: None,
            value: f.or("value", String::new(), text)?,
            package,
            at: point(f.need("at")?)?,
            rotation: f.or("rotation", 0.0, angle)?,
            side: f.or("side", Side::Top, side_of)?,
            locked: f.or("locked", false, boolean)?,
            pad_nets,
            fitted: f.or("fitted", true, boolean)?,
            properties,
        };
        f.finish()?;
        Ok(fp)
    }

    fn track(&self, id: ItemId, word: &'static str, h: &Head) -> Read<Track> {
        let net = self.head_net(h)?;
        let mut f = Fields::of(need_value(h)?, word)?;
        let layer = self.layers.copper(f.need("layer")?)?;
        let width = len(f.need("width")?)?;
        let geom = if word == words::ARC {
            TrackGeom::Arc {
                start: point(f.need("from")?)?,
                mid: point(f.need("mid")?)?,
                end: point(f.need("to")?)?,
            }
        } else {
            TrackGeom::Segment {
                a: point(f.need("from")?)?,
                b: point(f.need("to")?)?,
            }
        };
        let t = Track {
            id,
            net,
            layer,
            width,
            geom,
            locked: f.or("locked", false, boolean)?,
        };
        f.finish()?;
        Ok(t)
    }

    fn via(&self, id: ItemId, h: &Head, copper: u8) -> Read<Via> {
        let net = self.head_net(h)?;
        let mut f = Fields::of(need_value(h)?, "via")?;
        let (from, to) = match f.get("layers") {
            None => (0, copper.saturating_sub(1)),
            Some(v) => match list(v)? {
                [a, b] => {
                    let (a, b) = (self.layers.copper(a)?, self.layers.copper(b)?);
                    (a.min(b), a.max(b))
                }
                _ => {
                    return Err(diagnostic(
                        v.span,
                        "a via's layers are its first and last copper layer: `[F.Cu, In2.Cu]`",
                    ))
                }
            },
        };
        let v = Via {
            id,
            net,
            at: point(f.need("at")?)?,
            diameter: len(f.need("size")?)?,
            drill: len(f.need("drill")?)?,
            from,
            to,
            micro: f.or("micro", false, boolean)?,
            tented: f.or("tented", true, boolean)?,
            locked: f.or("locked", false, boolean)?,
        };
        f.finish()?;
        Ok(v)
    }

    fn copper_list(&self, v: &Val) -> Read<Vec<u8>> {
        list(v)?.iter().map(|l| self.layers.copper(l)).collect()
    }

    fn pour(&self, id: ItemId, h: &Head) -> Read<Zone> {
        let net = self.head_net(h)?;
        let mut f = Fields::of(need_value(h)?, "pour")?;
        let name = f.or("name", String::new(), text)?;
        let layers = self.copper_list(f.need("layers")?)?;
        let outline = polygon(&mut f)?;
        let mut z = Zone::new(id, &name, net, layers, outline);
        z.priority = f.or("priority", 0, whole)? as i32;
        z.clearance = f.opt("clearance", len)?;
        z.min_width = f.or("min_width", z.min_width, len)?;
        z.connect = f.or("connect", z.connect, connect_of)?;
        z.thermal_gap = f.or("thermal_gap", z.thermal_gap, len)?;
        z.spoke_width = f.or("spoke_width", z.spoke_width, len)?;
        z.islands = f.or("islands", z.islands, islands_of)?;
        z.locked = f.or("locked", false, boolean)?;
        f.finish()?;
        Ok(z)
    }

    fn keepout(&self, id: ItemId, h: &Head) -> Read<Keepout> {
        let name = h.name.clone().map(|n| n.0).unwrap_or_default();
        let mut f = Fields::of(need_value(h)?, "keepout")?;
        let outline = polygon(&mut f)?;
        let layers = f.or("layers", Vec::new(), |v| self.copper_list(v))?;
        let rules = KeepoutRules {
            tracks: f.or("tracks", false, boolean)?,
            vias: f.or("vias", false, boolean)?,
            pads: f.or("pads", false, boolean)?,
            pours: f.or("pours", false, boolean)?,
            footprints: f.or("footprints", false, boolean)?,
            max_height: f.opt("max_height", len)?,
        };
        f.finish()?;
        Ok(Keepout {
            id,
            name,
            outline,
            layers,
            rules,
        })
    }
}

fn read_bend(v: &Val) -> Read<Bend> {
    let mut f = Fields::of(v, "bend")?;
    let b = Bend {
        a: point(f.need("from")?)?,
        b: point(f.need("to")?)?,
        radius: len(f.need("radius")?)?,
        angle: angle(f.need("angle")?)?,
        keepout: f.or("keepout", 0, len)?,
    };
    f.finish()?;
    Ok(b)
}

const LIMIT_KEYS: [&str; 9] = [
    "clearance",
    "track",
    "via_diameter",
    "drill",
    "annular",
    "hole_to_hole",
    "copper_to_edge",
    "silk_to_pad",
    "microvia_drill",
];

fn limit_fields(l: &mut FabLimits) -> [&mut Coord; 9] {
    [
        &mut l.min_clearance,
        &mut l.min_track,
        &mut l.min_via_diameter,
        &mut l.min_drill,
        &mut l.min_annular,
        &mut l.min_hole_to_hole,
        &mut l.min_copper_to_edge,
        &mut l.min_silk_to_pad,
        &mut l.min_microvia_drill,
    ]
}

fn read_limits(v: &Val) -> Read<FabLimits> {
    let mut f = Fields::of(v, "limits")?;
    let mut l = FabLimits::default();
    for (key, field) in LIMIT_KEYS.iter().zip(limit_fields(&mut l)) {
        if let Some(v) = f.opt(key, len)? {
            *field = v;
        }
    }
    f.finish()?;
    Ok(l)
}

const VIA_KINDS: [(ViaKind, &str); 4] = [
    (ViaKind::Through, "through"),
    (ViaKind::Blind, "blind"),
    (ViaKind::Buried, "buried"),
    (ViaKind::Micro, "micro"),
];

/// A rule value: `0.2 mm` is a length, `2` a whole number, `2.5` a number, `true` a flag,
/// `[through, micro]` via kinds, anything else text.
fn rule_value(v: &Val) -> Read<RuleValue> {
    Ok(match &v.v {
        V::Qty(..) => RuleValue::Len(len(v)?),
        V::Num(t) if t.contains(['.', 'e', 'E']) => RuleValue::Num(num(v)?),
        V::Num(_) => RuleValue::Int(whole(v)?),
        V::Word(w) if w == "true" || w == "false" => RuleValue::Bool(boolean(v)?),
        V::Word(w) | V::Str(w) => RuleValue::Text(w.clone()),
        V::List(items) => RuleValue::ViaKinds(
            items
                .iter()
                .map(|i| {
                    let t = text(i)?;
                    VIA_KINDS
                        .iter()
                        .find(|(_, n)| *n == t)
                        .map(|(k, _)| *k)
                        .ok_or_else(|| {
                            diagnostic(
                                i.span,
                                format!("`{t}` is not a via kind; expected through, blind, buried or micro"),
                            )
                        })
                })
                .collect::<Read<_>>()?,
        ),
        V::Rec(_) => return Err(diagnostic(v.span, "a rule value is a length, number, flag or text")),
    })
}

fn rule_value_text(v: &RuleValue) -> String {
    match v {
        RuleValue::Len(l) => format!("{} mm", mm_text(*l)),
        RuleValue::Num(n) => {
            let t = f64_text(*n);
            if t.contains(['.', 'e', 'E', 'N', 'i']) {
                t
            } else {
                format!("{t}.0")
            }
        }
        RuleValue::Int(i) => i.to_string(),
        RuleValue::Bool(b) => b.to_string(),
        RuleValue::Text(t) => name_text(t),
        RuleValue::ViaKinds(kinds) => {
            let names: Vec<&str> = kinds
                .iter()
                .map(|k| VIA_KINDS.iter().find(|(x, _)| x == k).unwrap().1)
                .collect();
            format!("[{}]", names.join(", "))
        }
    }
}

fn read_rule(id: &str, v: &Val) -> Read<RuleSpec> {
    let mut f = Fields::of(v, "rule")?;
    let mut spec = match (f.get("select"), f.get("between")) {
        (Some(s), None) => RuleSpec::single(id, &text(s)?),
        (None, Some(b)) => match list(b)? {
            [a, c] => RuleSpec::pair(id, &text(a)?, &text(c)?),
            _ => return Err(diagnostic(b.span, "`between` takes two selectors")),
        },
        _ => {
            return Err(diagnostic(
                v.span,
                "a rule has `select = \"<selector>\"` or `between = [\"<a>\", \"<b>\"]`",
            ))
        }
    };
    spec.priority = f.or("priority", 0, whole)? as i32;
    for (k, v) in f.rest() {
        spec.decls.push((k.to_owned(), rule_value(v)?));
    }
    Ok(spec)
}

/* --- stackups ------------------------------------------------------------------------------ */

const MATERIALS: [(&str, fn() -> Material); 4] = [
    ("copper", Material::copper),
    ("fr4", Material::fr4),
    ("polyimide", Material::polyimide),
    ("solder_mask", Material::solder_mask),
];

const MATERIAL_KEYS: [&str; 8] = [
    "dk",
    "df",
    "conductivity",
    "k_inplane",
    "k_through",
    "density",
    "cp",
    "cte_ppm",
];

fn material_fields(m: &mut Material) -> [&mut f64; 8] {
    [
        &mut m.dk,
        &mut m.df,
        &mut m.conductivity,
        &mut m.k_inplane,
        &mut m.k_through,
        &mut m.density,
        &mut m.cp,
        &mut m.cte_ppm,
    ]
}

fn read_material(v: &Val) -> Read<Material> {
    if let V::Word(w) = &v.v {
        return MATERIALS
            .iter()
            .find(|(n, _)| n == w)
            .map(|(_, f)| f())
            .ok_or_else(|| {
                diagnostic(
                    v.span,
                    format!(
                        "`{w}` is not a built-in material; write `{{ name = \"…\", dk = …, … }}`"
                    ),
                )
            });
    }
    let mut f = Fields::of(v, "material")?;
    let mut m = Material {
        name: text(f.need("name")?)?,
        ..Material::fr4()
    };
    for (key, field) in MATERIAL_KEYS.iter().zip(material_fields(&mut m)) {
        *field = num(f.need(key)?)?;
    }
    f.finish()?;
    Ok(m)
}

fn material_text(m: &Material) -> String {
    if let Some((n, _)) = MATERIALS.iter().find(|(_, f)| f() == *m) {
        return (*n).to_owned();
    }
    let mut r = Rec::new();
    r.put("name", name_text(&m.name));
    let mut m = m.clone();
    for (key, field) in MATERIAL_KEYS.iter().zip(material_fields(&mut m)) {
        r.put(key, f64_text(*field));
    }
    r.text()
}

fn read_stackup(b: &ast::BlockStatement, name: String) -> Read<Stackup> {
    let mut s = Stackup {
        name,
        layers: Vec::new(),
        finish: String::new(),
    };
    for statement in b.block().map(|b| b.statements()).unwrap_or_default() {
        let Some(h) = head(&statement)? else {
            return Err(diagnostic(statement.span(), "expected `layer` or `finish`"));
        };
        match h.word.as_str() {
            words::FINISH if h.name.is_none() => s.finish = text(need_value(&h)?)?,
            words::LAYER => {
                let name = need_name(&h)?;
                let mut f = Fields::of(need_value(&h)?, "stackup layer")?;
                let kv = f.need("kind")?;
                let kind = match text(kv)?.as_str() {
                    "copper" => StackKind::Copper {
                        weight_oz: f.or("weight", 1.0, num)?,
                    },
                    "core" => StackKind::Dielectric { core: true },
                    "prepreg" => StackKind::Dielectric { core: false },
                    "mask" => StackKind::SolderMask,
                    "silk" => StackKind::Silkscreen,
                    "paste" => StackKind::Paste,
                    other => {
                        return Err(diagnostic(
                            kv.span,
                            format!("`{other}` is not a layer kind; expected copper, core, prepreg, mask, silk or paste"),
                        ))
                    }
                };
                let thickness = len(f.need("thickness")?)?;
                let material = read_material(f.need("material")?)?;
                f.finish()?;
                s.layers.push(StackLayer {
                    name,
                    kind,
                    thickness,
                    material,
                });
            }
            other => {
                return Err(diagnostic(
                    h.span,
                    format!("`{other}` does not belong in a stackup; expected `layer` or `finish`"),
                ))
            }
        }
    }
    Ok(s)
}

/* --- packages ------------------------------------------------------------------------------ */

fn pad_kind_name(k: PadKind) -> &'static str {
    match k {
        PadKind::Smd => "smd",
        PadKind::Tht => "tht",
        PadKind::Npth => "npth",
        PadKind::Connector => "edge",
    }
}

fn copper_span_name(c: CopperSpan) -> &'static str {
    match c {
        CopperSpan::Front => "front",
        CopperSpan::Back => "back",
        CopperSpan::All => "all",
    }
}

/// The pad a kind starts from; a pad statement writes only what differs.
fn pad_default(kind: PadKind, name: &str, at: Point, shape: PadShape, drill: Coord) -> PadDef {
    match kind {
        PadKind::Smd => PadDef::smd(name, at, shape),
        PadKind::Connector => PadDef {
            kind: PadKind::Connector,
            paste: false,
            ..PadDef::smd(name, at, shape)
        },
        PadKind::Tht => PadDef::tht(name, at, shape, drill),
        PadKind::Npth => PadDef {
            shape,
            ..PadDef::npth(name, at, drill)
        },
    }
}

fn read_shape(f: &mut Fields, span: Span) -> Read<PadShape> {
    let size = |f: &mut Fields| -> Read<(Coord, Coord)> {
        let p = point(f.need("size")?)?;
        Ok((p.x, p.y))
    };
    let sv = f.need("shape")?;
    Ok(match text(sv)?.as_str() {
        "circle" => PadShape::Circle {
            d: len(f.need("d")?)?,
        },
        "rect" => {
            let (w, h) = size(f)?;
            PadShape::Rect { w, h }
        }
        "roundrect" => {
            let (w, h) = size(f)?;
            PadShape::RoundRect {
                w,
                h,
                ratio: num(f.need("ratio")?)?,
            }
        }
        "oval" => {
            let (w, h) = size(f)?;
            PadShape::Oval { w, h }
        }
        "trapezoid" => {
            let (w, h) = size(f)?;
            PadShape::Trapezoid {
                w,
                h,
                delta: len(f.need("delta")?)?,
            }
        }
        "chamfered" => {
            let (w, h) = size(f)?;
            PadShape::Chamfered {
                w,
                h,
                chamfer: len(f.need("chamfer")?)?,
                corners: u8::try_from(whole(f.need("corners")?)?)
                    .map_err(|_| diagnostic(span, "corners is a bit mask 0 to 15"))?,
            }
        }
        "custom" => PadShape::Custom {
            points: ring(f.need("points")?)?,
            holes: f.or("holes", Vec::new(), rings)?,
        },
        other => {
            return Err(diagnostic(
                sv.span,
                format!("`{other}` is not a pad shape; expected circle, rect, roundrect, oval, trapezoid, chamfered or custom"),
            ))
        }
    })
}

fn shape_fields(s: &PadShape, r: &mut Rec) {
    let size = |r: &mut Rec, w: Coord, h: Coord| {
        r.put("size", pt_text(Point::new(w, h)));
    };
    match s {
        PadShape::Circle { d } => {
            r.put("shape", "circle").len("d", *d);
        }
        PadShape::Rect { w, h } => {
            r.put("shape", "rect");
            size(r, *w, *h);
        }
        PadShape::RoundRect { w, h, ratio } => {
            r.put("shape", "roundrect");
            size(r, *w, *h);
            r.put("ratio", f64_text(*ratio));
        }
        PadShape::Oval { w, h } => {
            r.put("shape", "oval");
            size(r, *w, *h);
        }
        PadShape::Trapezoid { w, h, delta } => {
            r.put("shape", "trapezoid");
            size(r, *w, *h);
            r.len("delta", *delta);
        }
        PadShape::Chamfered {
            w,
            h,
            chamfer,
            corners,
        } => {
            r.put("shape", "chamfered");
            size(r, *w, *h);
            r.len("chamfer", *chamfer)
                .put("corners", corners.to_string());
        }
        PadShape::Custom { points, holes } => {
            r.put("shape", "custom").put("points", ring_text(points));
            if !holes.is_empty() {
                r.put("holes", rings_text(holes));
            }
        }
    }
}

fn read_pad(name: String, v: &Val) -> Read<PadDef> {
    let mut f = Fields::of(v, "pad")?;
    let kv = f.get("kind");
    let kind = match kv.map(text).transpose()?.as_deref() {
        None | Some("smd") => PadKind::Smd,
        Some("tht") => PadKind::Tht,
        Some("npth") => PadKind::Npth,
        Some("edge") => PadKind::Connector,
        Some(other) => {
            return Err(diagnostic(
                kv.unwrap().span,
                format!("`{other}` is not a pad kind; expected smd, tht, npth or edge"),
            ))
        }
    };
    let shape = read_shape(&mut f, v.span)?;
    let at = f.or("at", Point::ORIGIN, point)?;
    let drill = match f.get("drill") {
        None => None,
        Some(d) if matches!(&d.v, V::Word(w) if w == "none") => Some(None),
        Some(d) => Some(Some(len(d)?)),
    };
    let mut p = pad_default(kind, &name, at, shape, drill.flatten().unwrap_or(0));
    match drill {
        Some(None) => p.drill = None,
        Some(Some(d)) => {
            let plated = p.drill.as_ref().is_none_or(|x| x.plated);
            p.drill = Some(Drill {
                diameter: d,
                slot: None,
                plated,
            });
        }
        None if p.drill.is_some() => {
            return Err(diagnostic(v.span, "a through-hole pad needs `drill`"))
        }
        None => {}
    }
    if let Some(drill) = p.drill.as_mut() {
        drill.slot = f.opt("slot", len)?;
        drill.plated = f.or("plated", drill.plated, boolean)?;
    }
    p.rotation = f.or("rotation", 0.0, angle)?;
    p.copper = match f.get("copper") {
        None => p.copper,
        Some(c) => match text(c)?.as_str() {
            "front" => CopperSpan::Front,
            "back" => CopperSpan::Back,
            "all" => CopperSpan::All,
            other => {
                return Err(diagnostic(
                    c.span,
                    format!("`{other}` is not a copper span; expected front, back or all"),
                ))
            }
        },
    };
    p.mask = f.or("mask", p.mask, boolean)?;
    p.paste = f.or("paste", p.paste, boolean)?;
    p.mask_margin = f.opt("mask_margin", len)?;
    p.paste_margin = f.opt("paste_margin", len)?;
    p.thermal_pad = f.or("thermal", false, boolean)?;
    f.finish()?;
    Ok(p)
}

fn pad_text(p: &PadDef) -> String {
    let d = pad_default(
        p.kind,
        &p.name,
        p.at,
        p.shape.clone(),
        p.drill.as_ref().map_or(0, |d| d.diameter),
    );
    let mut r = Rec::new();
    if p.kind != PadKind::Smd {
        r.put("kind", pad_kind_name(p.kind));
    }
    shape_fields(&p.shape, &mut r);
    if p.at != Point::ORIGIN {
        r.pt("at", p.at);
    }
    if p.rotation != 0.0 {
        r.put("rotation", f64_text(p.rotation));
    }
    match (&p.drill, &d.drill) {
        (None, Some(_)) => {
            r.put("drill", "none");
        }
        (None, None) => {}
        (Some(x), base) => {
            r.len("drill", x.diameter);
            if let Some(s) = x.slot {
                r.len("slot", s);
            }
            if x.plated != base.as_ref().is_none_or(|b| b.plated) {
                r.put("plated", x.plated.to_string());
            }
        }
    }
    if p.copper != d.copper {
        r.put("copper", copper_span_name(p.copper));
    }
    if p.mask != d.mask {
        r.put("mask", p.mask.to_string());
    }
    if p.paste != d.paste {
        r.put("paste", p.paste.to_string());
    }
    if let Some(m) = p.mask_margin {
        r.len("mask_margin", m);
    }
    if let Some(m) = p.paste_margin {
        r.len("paste_margin", m);
    }
    r.flag("thermal", p.thermal_pad);
    r.text()
}

fn floats3(v: &Val) -> Read<(f64, f64, f64)> {
    match list(v)? {
        [a, b, c] => Ok((num(a)?, num(b)?, num(c)?)),
        _ => Err(diagnostic(v.span, "expected `[x, y, z]`")),
    }
}

fn floats3_text(t: (f64, f64, f64)) -> String {
    format!("[{}, {}, {}]", f64_text(t.0), f64_text(t.1), f64_text(t.2))
}

fn read_package(b: &ast::BlockStatement, key: &str) -> Read<Package> {
    let mut p = Package {
        uuid: Uuid::NIL,
        name: key.to_owned(),
        description: String::new(),
        pads: Vec::new(),
        courtyard: Vec::new(),
        silk: Vec::new(),
        fab: Vec::new(),
        model: None,
        body: None,
        smd: false,
    };
    for statement in b.block().map(|b| b.statements()).unwrap_or_default() {
        let Some(h) = head(&statement)? else {
            return Err(diagnostic(
                statement.span(),
                "expected a package statement such as `pad`",
            ));
        };
        let value = need_value(&h)?;
        match (h.word.as_str(), h.name.as_ref()) {
            (words::NAME, None) => p.name = text(value)?,
            (words::UUID, None) => {
                let t = text(value)?;
                p.uuid = Uuid::parse(&t)
                    .ok_or_else(|| diagnostic(value.span, format!("`{t}` is not a UUID")))?;
            }
            (words::DESCRIPTION, None) => p.description = text(value)?,
            (words::SMD, None) => p.smd = boolean(value)?,
            (words::PAD, Some((name, _))) => p.pads.push(read_pad(name.clone(), value)?),
            (words::COURTYARD, None) => p.courtyard.push(ring(value)?),
            (words::SILK | words::FAB, Some((kind, _))) => {
                let mut f = Fields::of(value, "graphic")?;
                let g = read_graphic(kind, &mut f, h.span)?;
                f.finish()?;
                if h.word == words::SILK {
                    p.silk.push(g);
                } else {
                    p.fab.push(g);
                }
            }
            (words::MODEL, None) => {
                let mut f = Fields::of(value, "3D model")?;
                p.model = Some(Model3d {
                    path: text(f.need("path")?)?,
                    offset: f.or("offset", (0.0, 0.0, 0.0), floats3)?,
                    rotation: f.or("rotation", (0.0, 0.0, 0.0), floats3)?,
                    scale: f.or("scale", (1.0, 1.0, 1.0), floats3)?,
                    origin: f.or("origin", "footprint".to_owned(), text)?,
                });
                f.finish()?;
            }
            (words::BODY, None) => {
                let mut f = Fields::of(value, "body")?;
                p.body = Some(Body {
                    width: len(f.need("width")?)?,
                    length: len(f.need("length")?)?,
                    height: len(f.need("height")?)?,
                });
                f.finish()?;
            }
            (word, _) => {
                return Err(diagnostic(
                    h.span,
                    format!("`{word}` does not belong in a package; expected name, uuid, description, smd, pad, courtyard, silk, fab, model or body"),
                ))
            }
        }
    }
    Ok(p)
}

/* --- graphics ------------------------------------------------------------------------------ */

fn read_graphic(kind: &str, f: &mut Fields, span: Span) -> Read<Graphic> {
    let width = |f: &mut Fields| f.or("width", 0, len);
    let fill = |f: &mut Fields| f.or("fill", false, boolean);
    Ok(match kind {
        "line" => Graphic::Line {
            points: ring(f.need("points")?)?,
            width: width(f)?,
        },
        "rect" => Graphic::Rect {
            a: point(f.need("from")?)?,
            b: point(f.need("to")?)?,
            width: width(f)?,
            fill: fill(f)?,
        },
        "circle" => Graphic::Circle {
            center: point(f.need("center")?)?,
            radius: len(f.need("radius")?)?,
            width: width(f)?,
            fill: fill(f)?,
        },
        "arc" => Graphic::Arc {
            start: point(f.need("from")?)?,
            mid: point(f.need("mid")?)?,
            end: point(f.need("to")?)?,
            width: width(f)?,
        },
        "polygon" => Graphic::Polygon {
            points: ring(f.need("points")?)?,
            width: width(f)?,
            fill: fill(f)?,
        },
        "text" => Graphic::Text {
            at: point(f.need("at")?)?,
            text: text(f.need("text")?)?,
            size: len(f.need("size")?)?,
            angle: f.or("angle", 0.0, angle)?,
        },
        other => {
            return Err(diagnostic(
                span,
                format!(
                    "`{other}` is not a graphic; expected line, rect, circle, arc, polygon or text"
                ),
            ))
        }
    })
}

/// The graphic's kind word and its fields after `prefix` (a board graphic's layer).
fn graphic_text(g: &Graphic, mut r: Rec) -> (&'static str, String) {
    let kind = match g {
        Graphic::Line { points, width } => {
            r.put("points", ring_text(points)).len("width", *width);
            "line"
        }
        Graphic::Rect { a, b, width, fill } => {
            r.pt("from", *a)
                .pt("to", *b)
                .len("width", *width)
                .flag("fill", *fill);
            "rect"
        }
        Graphic::Circle {
            center,
            radius,
            width,
            fill,
        } => {
            r.pt("center", *center)
                .len("radius", *radius)
                .len("width", *width)
                .flag("fill", *fill);
            "circle"
        }
        Graphic::Arc {
            start,
            mid,
            end,
            width,
        } => {
            r.pt("from", *start)
                .pt("mid", *mid)
                .pt("to", *end)
                .len("width", *width);
            "arc"
        }
        Graphic::Polygon {
            points,
            width,
            fill,
        } => {
            r.put("points", ring_text(points))
                .len("width", *width)
                .flag("fill", *fill);
            "polygon"
        }
        Graphic::Text {
            at,
            text,
            size,
            angle,
        } => {
            r.pt("at", *at)
                .put("text", name_text(text))
                .len("size", *size);
            if *angle != 0.0 {
                r.put("angle", f64_text(*angle));
            }
            "text"
        }
    };
    (kind, r.text())
}

/* ------------------------------------------------------------------------------------------------
 * Printing
 * --------------------------------------------------------------------------------------------- */

fn net_head(board: &Board, word: &str, net: Option<NetId>) -> String {
    match net {
        Some(n) => format!("{word} {} =", name_text(board.net_name(n))),
        None => format!("{word} ="),
    }
}

fn polygon_fields(r: &mut Rec, p: &Polygon) {
    r.put(words::OUTLINE, ring_text(&p.outer));
    if !p.holes.is_empty() {
        r.put("holes", rings_text(&p.holes));
    }
}

/// The package key each footprint prints: its package name, or `<name>#<n>` when another
/// footprint uses a different package of the same name.
fn package_keys(board: &Board) -> (BTreeMap<ItemId, String>, Vec<(String, Package)>) {
    let mut keys = BTreeMap::new();
    let mut defined: Vec<(String, Package)> = Vec::new();
    for f in board.footprints.values() {
        let p = &f.package;
        let key = match defined.iter().find(|(_, x)| x == p) {
            Some((k, _)) => k.clone(),
            None => {
                let taken = |k: &str| defined.iter().any(|(x, _)| x == k);
                let mut key = p.name.clone();
                let mut n = 1;
                while taken(&key) {
                    n += 1;
                    key = format!("{}#{n}", p.name);
                }
                defined.push((key.clone(), p.clone()));
                key
            }
        };
        keys.insert(f.id, key);
    }
    (keys, defined)
}

fn stackup_text(s: &Stackup) -> String {
    let inner = INDENT.repeat(2);
    let mut out = format!("{} {} {{\n", words::STACKUP, name_text(&s.name));
    if !s.finish.is_empty() {
        out.push_str(&format!(
            "{inner}{} = {}\n",
            words::FINISH,
            name_text(&s.finish)
        ));
    }
    for l in &s.layers {
        let mut r = Rec::new();
        match l.kind {
            StackKind::Copper { weight_oz } => {
                r.put("kind", "copper");
                r.len("thickness", l.thickness);
                r.put("weight", f64_text(weight_oz));
            }
            StackKind::Dielectric { core } => {
                r.put("kind", if core { "core" } else { "prepreg" });
                r.len("thickness", l.thickness);
            }
            StackKind::SolderMask => {
                r.put("kind", "mask").len("thickness", l.thickness);
            }
            StackKind::Silkscreen => {
                r.put("kind", "silk").len("thickness", l.thickness);
            }
            StackKind::Paste => {
                r.put("kind", "paste").len("thickness", l.thickness);
            }
        }
        r.put("material", material_text(&l.material));
        out.push_str(&format!(
            "{inner}{} {} = {}\n",
            words::LAYER,
            name_text(&l.name),
            r.text()
        ));
    }
    out.push_str(INDENT);
    out.push('}');
    out
}

fn package_text(key: &str, p: &Package) -> String {
    let inner = INDENT.repeat(2);
    let mut out = format!("{} {} {{\n", words::PACKAGE, name_text(key));
    let mut line = |head: String, value: String| {
        out.push_str(&inner);
        out.push_str(&head);
        out.push_str(" = ");
        out.push_str(&value);
        out.push('\n');
    };
    if p.name != key {
        line(words::NAME.into(), name_text(&p.name));
    }
    if p.uuid != Uuid::NIL {
        line(
            words::UUID.into(),
            crate::syntax::values::name_text(&p.uuid.to_string()),
        );
    }
    if !p.description.is_empty() {
        line(words::DESCRIPTION.into(), name_text(&p.description));
    }
    if p.smd {
        line(words::SMD.into(), "true".into());
    }
    for pad in &p.pads {
        line(
            format!("{} {}", words::PAD, name_text(&pad.name)),
            pad_text(pad),
        );
    }
    for ring in &p.courtyard {
        line(words::COURTYARD.into(), ring_text(ring));
    }
    for (word, graphics) in [(words::SILK, &p.silk), (words::FAB, &p.fab)] {
        for g in graphics {
            let (kind, text) = graphic_text(g, Rec::new());
            line(format!("{word} {kind}"), text);
        }
    }
    if let Some(m) = &p.model {
        let mut r = Rec::new();
        r.put("path", name_text(&m.path));
        if m.offset != (0.0, 0.0, 0.0) {
            r.put("offset", floats3_text(m.offset));
        }
        if m.rotation != (0.0, 0.0, 0.0) {
            r.put("rotation", floats3_text(m.rotation));
        }
        if m.scale != (1.0, 1.0, 1.0) {
            r.put("scale", floats3_text(m.scale));
        }
        if m.origin != "footprint" {
            r.put("origin", name_text(&m.origin));
        }
        line(words::MODEL.into(), r.text());
    }
    if let Some(b) = &p.body {
        let mut r = Rec::new();
        r.len("width", b.width)
            .len("length", b.length)
            .len("height", b.height);
        line(words::BODY.into(), r.text());
    }
    out.push_str(INDENT);
    out.push('}');
    out
}

/// Every statement of a board, in print order, with its site.
pub fn statements(board: &Board) -> Vec<Printed> {
    let mut out: Vec<Printed> = Vec::new();
    let mut sites = Sites::default();
    let layers = Layers::of(&board.stackup);
    let mut push = |group: Group, site: String, text: String, item: Option<ItemId>| {
        out.push(Printed {
            group,
            site,
            text,
            item,
        })
    };
    for s in std::iter::once(&board.stackup).chain(board.stackups.values()) {
        push(
            Group::Stackup,
            sites.named(words::STACKUP, &s.name),
            stackup_text(s),
            None,
        );
    }
    for r in &board.regions {
        let mut rec = Rec::new();
        rec.put(words::STACKUP, name_text(&r.stackup));
        polygon_fields(&mut rec, &r.outline);
        if !r.bends.is_empty() {
            let bends: Vec<String> = r
                .bends
                .iter()
                .map(|b| {
                    let mut x = Rec::new();
                    x.pt("from", b.a)
                        .pt("to", b.b)
                        .len("radius", b.radius)
                        .put("angle", f64_text(b.angle));
                    if b.keepout != 0 {
                        x.len("keepout", b.keepout);
                    }
                    x.text()
                })
                .collect();
            rec.put("bends", format!("[{}]", bends.join(", ")));
        }
        push(
            Group::Region,
            sites.named(words::REGION, &r.name),
            format!("{} {} = {}", words::REGION, name_text(&r.name), rec.text()),
            None,
        );
    }
    if !board.outline.outer.is_empty() {
        push(
            Group::Outline,
            "outline".into(),
            format!("{} = {}", words::OUTLINE, ring_text(&board.outline.outer)),
            None,
        );
    }
    for hole in &board.outline.holes {
        push(
            Group::Outline,
            sites.indexed(words::CUTOUT),
            format!("{} = {}", words::CUTOUT, ring_text(hole)),
            None,
        );
    }
    {
        let mut l = board.limits.clone();
        let mut r = Rec::new();
        for (key, field) in LIMIT_KEYS.iter().zip(limit_fields(&mut l)) {
            r.len(key, *field);
        }
        push(
            Group::Limits,
            "limits".into(),
            format!("{} = {}", words::LIMITS, r.text()),
            None,
        );
    }
    for (class, decls) in &board.net_class_rules {
        let mut r = Rec::new();
        for (k, v) in decls {
            r.put(k, rule_value_text(v));
        }
        push(
            Group::NetClass,
            sites.named(words::NET_CLASS, class),
            format!("{} {} = {}", words::NET_CLASS, name_text(class), r.text()),
            None,
        );
    }
    for rule in &board.rules {
        let mut r = Rec::new();
        match &rule.other {
            None => r.put("select", name_text(&rule.selector)),
            Some(o) => r.put(
                "between",
                format!("[{}, {}]", name_text(&rule.selector), name_text(o)),
            ),
        };
        if rule.priority != 0 {
            r.put("priority", rule.priority.to_string());
        }
        for (k, v) in &rule.decls {
            r.put(k, rule_value_text(v));
        }
        push(
            Group::Rule,
            sites.named(words::RULE, &rule.id),
            format!("{} {} = {}", words::RULE, name_text(&rule.id), r.text()),
            None,
        );
    }
    for n in &board.nets {
        let text = match &n.class {
            Some(c) => format!("{} {} : {}", words::NET, name_text(&n.name), name_text(c)),
            None => format!("{} {}", words::NET, name_text(&n.name)),
        };
        push(Group::Net, sites.named(words::NET, &n.name), text, None);
    }
    for room in &board.rooms {
        let mut r = Rec::new();
        polygon_fields(&mut r, &room.outline);
        if !room.members.is_empty() {
            let m: Vec<String> = room.members.iter().map(|m| name_text(m)).collect();
            r.put("members", format!("[{}]", m.join(", ")));
        }
        push(
            Group::Room,
            sites.named(words::ROOM, &room.name),
            format!("{} {} = {}", words::ROOM, name_text(&room.name), r.text()),
            None,
        );
    }
    let (keys, packages) = package_keys(board);
    for f in board.footprints.values() {
        let mut r = Rec::new();
        r.put(words::PACKAGE, name_text(&keys[&f.id]))
            .pt("at", f.at);
        if f.rotation != 0.0 {
            r.put("rotation", f64_text(f.rotation));
        }
        if f.side != Side::Top {
            r.put("side", side_name(f.side));
        }
        r.flag("locked", f.locked);
        if !f.value.is_empty() {
            r.put("value", name_text(&f.value));
        }
        if !f.fitted {
            r.put("fitted", "false");
        }
        if !f.pad_nets.is_empty() {
            let mut pads = Rec::new();
            for (pad, net) in &f.pad_nets {
                pads.put(pad, name_text(board.net_name(*net)));
            }
            r.put("pads", pads.text());
        }
        if !f.properties.is_empty() {
            let mut props = Rec::new();
            for (k, v) in &f.properties {
                props.put(k, name_text(v));
            }
            r.put("properties", props.text());
        }
        push(
            Group::Footprint,
            sites.named(words::FOOTPRINT, &f.reference),
            format!(
                "{} {} = {}",
                words::FOOTPRINT,
                name_text(&f.reference),
                r.text()
            ),
            Some(f.id),
        );
    }
    for fid in board.fiducials.values() {
        let mut r = Rec::new();
        r.pt("at", fid.at);
        if fid.side != Side::Top {
            r.put("side", side_name(fid.side));
        }
        r.len("diameter", fid.diameter)
            .len("opening", fid.opening)
            .flag("locked", fid.locked);
        push(
            Group::Fiducial,
            sites.named(words::FIDUCIAL, &fid.name),
            format!(
                "{} {} = {}",
                words::FIDUCIAL,
                name_text(&fid.name),
                r.text()
            ),
            Some(fid.id),
        );
    }
    for t in board.tracks.values() {
        let mut r = Rec::new();
        r.put("layer", layers.copper_text(t.layer))
            .len("width", t.width);
        let word = match t.geom {
            TrackGeom::Segment { a, b } => {
                r.pt("from", a).pt("to", b);
                words::TRACK
            }
            TrackGeom::Arc { start, mid, end } => {
                r.pt("from", start).pt("mid", mid).pt("to", end);
                words::ARC
            }
        };
        r.flag("locked", t.locked);
        push(
            Group::Copper,
            sites.indexed(word),
            format!("{} {}", net_head(board, word, t.net), r.text()),
            Some(t.id),
        );
    }
    let n = board.copper_count();
    for v in board.vias.values() {
        let mut r = Rec::new();
        r.pt("at", v.at)
            .len("size", v.diameter)
            .len("drill", v.drill);
        if v.from != 0 || v.to + 1 != n {
            r.put(
                "layers",
                format!(
                    "[{}, {}]",
                    layers.copper_text(v.from),
                    layers.copper_text(v.to)
                ),
            );
        }
        r.flag("micro", v.micro);
        if !v.tented {
            r.put("tented", "false");
        }
        r.flag("locked", v.locked);
        push(
            Group::Via,
            sites.indexed(words::VIA),
            format!("{} {}", net_head(board, words::VIA, v.net), r.text()),
            Some(v.id),
        );
    }
    for z in board.zones.values() {
        let d = Zone::new(z.id, "", None, Vec::new(), Polygon::default());
        let mut r = Rec::new();
        if !z.name.is_empty() {
            r.put("name", name_text(&z.name));
        }
        let l: Vec<String> = z.layers.iter().map(|l| layers.copper_text(*l)).collect();
        r.put("layers", format!("[{}]", l.join(", ")));
        polygon_fields(&mut r, &z.outline);
        if z.priority != 0 {
            r.put("priority", z.priority.to_string());
        }
        if let Some(c) = z.clearance {
            r.len("clearance", c);
        }
        if z.min_width != d.min_width {
            r.len("min_width", z.min_width);
        }
        if z.connect != d.connect {
            r.put("connect", connect_name(z.connect));
        }
        if z.thermal_gap != d.thermal_gap {
            r.len("thermal_gap", z.thermal_gap);
        }
        if z.spoke_width != d.spoke_width {
            r.len("spoke_width", z.spoke_width);
        }
        if z.islands != d.islands {
            r.put("islands", islands_text(z.islands));
        }
        r.flag("locked", z.locked);
        push(
            Group::Pour,
            sites.indexed(words::POUR),
            format!("{} {}", net_head(board, words::POUR, z.net), r.text()),
            Some(z.id),
        );
    }
    for k in board.keepouts.values() {
        let mut r = Rec::new();
        polygon_fields(&mut r, &k.outline);
        if !k.layers.is_empty() {
            let l: Vec<String> = k.layers.iter().map(|l| layers.copper_text(*l)).collect();
            r.put("layers", format!("[{}]", l.join(", ")));
        }
        r.flag("tracks", k.rules.tracks)
            .flag("vias", k.rules.vias)
            .flag("pads", k.rules.pads)
            .flag("pours", k.rules.pours)
            .flag("footprints", k.rules.footprints);
        if let Some(h) = k.rules.max_height {
            r.len("max_height", h);
        }
        let head = if k.name.is_empty() {
            format!("{} =", words::KEEPOUT)
        } else {
            format!("{} {} =", words::KEEPOUT, name_text(&k.name))
        };
        push(
            Group::Keepout,
            sites.indexed(words::KEEPOUT),
            format!("{head} {}", r.text()),
            Some(k.id),
        );
    }
    for t in board.texts.values() {
        let mut r = Rec::new();
        r.put("layer", layers.layer_text(t.layer))
            .pt("at", t.at)
            .len("size", t.size)
            .len("thickness", t.thickness);
        if t.angle != 0.0 {
            r.put("angle", f64_text(t.angle));
        }
        r.flag("mirrored", t.mirrored);
        push(
            Group::Text,
            sites.indexed(words::TEXT),
            format!(
                "{} {} = {}",
                words::TEXT,
                runmat_design_syntax::json::quote(&t.text),
                r.text()
            ),
            Some(t.id),
        );
    }
    for g in board.graphics.values() {
        let mut r = Rec::new();
        r.put("layer", layers.layer_text(g.layer));
        let (kind, text) = graphic_text(&g.graphic, r);
        push(
            Group::Graphic,
            sites.indexed(words::GRAPHIC),
            format!("{} {kind} = {text}", words::GRAPHIC),
            Some(g.id),
        );
    }
    for d in board.dimensions.values() {
        let mut r = Rec::new();
        r.put("layer", layers.layer_text(d.layer))
            .pt("from", d.a)
            .pt("to", d.b);
        if d.offset != 0 {
            r.len("offset", d.offset);
        }
        if d.precision != 2 {
            r.put("precision", d.precision.to_string());
        }
        push(
            Group::Dimension,
            sites.indexed(words::DIMENSION),
            format!("{} = {}", words::DIMENSION, r.text()),
            Some(d.id),
        );
    }
    for (key, p) in &packages {
        push(
            Group::Package,
            sites.named(words::PACKAGE, key),
            package_text(key, p),
            None,
        );
    }
    let section = SyncSection::of(board);
    if section != SyncSection::default() {
        push(Group::Sync, "sync".into(), print_sync(&section, 1), None);
    }
    out
}

/// The file's head: the version header and the `use` lines.
pub fn print_head(uses: &[UseItem]) -> String {
    let mut out = String::from("runmat 0.7\nstd 1\n\n");
    for u in uses {
        out.push_str(&u.print());
        out.push('\n');
    }
    if !uses.is_empty() {
        out.push('\n');
    }
    out
}

/// A board as a whole `.circuit.layout` file.
pub fn print_board(board: &Board, uses: &[UseItem]) -> String {
    let printed = statements(board);
    let size: usize = printed.iter().map(|p| p.text.len() + 6).sum();
    let mut out = String::with_capacity(size + 64);
    out.push_str(&print_head(uses));
    out.push_str(&format!(
        "{} {} {{\n",
        words::LAYOUT,
        name_text(&board.name)
    ));
    for p in &printed {
        out.push_str(INDENT);
        out.push_str(&p.text);
        out.push('\n');
    }
    out.push_str("}\n");
    out
}

/// `board` with its items renumbered in print order, derived data (pour fills) dropped and
/// rule source positions cleared: the board its printed text reads back as.
pub fn canonical(board: &Board) -> Board {
    let mut b = board.clone();
    let mut next = 1u64;
    let mut renumber = |id: &mut ItemId| {
        *id = ItemId(next);
        next += 1;
    };
    macro_rules! renumber_map {
        ($field:ident) => {{
            let items = std::mem::take(&mut b.$field);
            b.$field = items
                .into_values()
                .map(|mut item| {
                    renumber(&mut item.id);
                    (item.id, item)
                })
                .collect();
        }};
    }
    renumber_map!(footprints);
    renumber_map!(fiducials);
    renumber_map!(tracks);
    renumber_map!(vias);
    renumber_map!(zones);
    renumber_map!(keepouts);
    renumber_map!(texts);
    renumber_map!(graphics);
    renumber_map!(dimensions);
    b.next_id = next;
    for z in b.zones.values_mut() {
        z.fill.clear();
    }
    for r in &mut b.rules {
        r.source = SourceKey::default();
        r.order = SourceOrder::default();
    }
    b
}

/// Whether two boards are the same model: equal once both are [`canonical`].
pub fn same_board(a: &Board, b: &Board) -> bool {
    canonical(a) == canonical(b)
}
