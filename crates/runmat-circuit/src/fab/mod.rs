//! Fabrication and assembly outputs.
//!
//! - [`gerber`]: Gerber X2 (RS-274X with `%TF`/`%TA`/`%TO` attributes) per layer: copper
//!   (pads flashed or as regions, tracks drawn, vias flashed, zone fills as regions with holes
//!   cleared, every object carrying `.N` net / `.P` pin / `.C` component attributes), solder
//!   mask (negative polarity, openings grown by the mask margin, vias tented), paste, legend
//!   (silk graphics) and profile; [`job::gerber_job`] is the X2 job file with the stackup.
//! - [`excellon`]: Excellon drill files: plated holes split by span (through, blind, buried
//!   per layer pair, with X2-style `TF.FileFunction` comments) and non-plated holes; slots as
//!   `G85`.
//! - [`drawing`]: the drill map and the fabrication drawing (drill table, layer and stackup
//!   table, notes) as vector drawings, written as Gerber or laid out on a sheet.
//! - [`odb`]: the ODB++ job tree, drawn by the same layer images as the Gerber files.
//! - [`ipc356`]: an IPC-D-356A netlist (fixed-column `317`/`327` test records, metric units).
//! - [`bom`]: grouped, variant-aware bill of materials (CSV).
//! - [`pick_place`]: pick-and-place (CSV).
//! - [`set`]: the whole fabrication set a request names, by file name.
//!
//! Outputs are deterministic (sorted, fixed precision, no dates unless the caller passes one).
//! Text (board text, footprint fields with `${REFERENCE}`/`${VALUE}` substituted) is drawn as
//! strokes of the [`crate::font`] stroke font. Zone fills are single regions whose holes are
//! joined by cut-ins, so overlapping zones on one layer never erase each other.

pub mod assembly;
pub mod drawing;
pub mod drill;
pub mod job;
pub mod odb;
pub mod plan;
pub mod preview;
pub mod set;
pub mod settings;
pub mod assembly_drawing;

pub use assembly::{bom, bom_csv, ipc356, natural_key, pick_place, BomLine};
pub use drill::excellon;

use crate::geom::poly::Ring;
use crate::geom::shape::{Core, Prim};
use crate::geom::{to_mm, Coord, Point};
use crate::layout::{Board, Footprint, Layer, Side, TrackGeom};
use crate::library::{Graphic, PadKind, PadShape};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// The length unit of a Gerber file or a pick-and-place list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LengthUnit {
    #[default]
    Mm,
    Inch,
}

impl LengthUnit {
    pub fn word(self) -> &'static str {
        match self {
            LengthUnit::Mm => "mm",
            LengthUnit::Inch => "inch",
        }
    }
    pub fn from_word(word: &str) -> Option<LengthUnit> {
        match word {
            "mm" => Some(LengthUnit::Mm),
            "inch" | "in" => Some(LengthUnit::Inch),
            _ => None,
        }
    }
    /// `v` in this unit.
    pub fn of(self, v: Coord) -> f64 {
        match self {
            LengthUnit::Mm => to_mm(v),
            LengthUnit::Inch => to_mm(v) / 25.4,
        }
    }
}

/// A Gerber file's coordinate format: the unit and the decimals kept. Millimetres use 4 integer
/// digits and inches 2 (`%FSLAX46Y46*%`, `%FSLAX26Y26*%`), as the Gerber specification
/// recommends; 6 decimals is 1 nm in millimetres, 5 is 10 nm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct GerberFormat {
    pub units: LengthUnit,
    /// 5 or 6.
    pub decimals: u8,
}

impl Default for GerberFormat {
    fn default() -> Self {
        GerberFormat { units: LengthUnit::Mm, decimals: 6 }
    }
}

impl GerberFormat {
    /// The decimals clamped to the two the format offers.
    fn decimals(self) -> u8 {
        self.decimals.clamp(5, 6)
    }
    /// A coordinate in the file's integer units.
    fn coord(self, v: Coord) -> String {
        if self == GerberFormat::default() {
            return v.to_string();
        }
        let scale = 10f64.powi(self.decimals() as i32);
        ((self.units.of(v) * scale).round() as i64).to_string()
    }
    /// An aperture size in the file's unit.
    fn size(self, v: Coord) -> String {
        format!("{:.6}", self.units.of(v))
    }
    fn statement(self) -> String {
        let d = self.decimals();
        match self.units {
            LengthUnit::Mm => format!("%FSLAX4{d}Y4{d}*%\n%MOMM*%\n"),
            LengthUnit::Inch => format!("%FSLAX2{d}Y2{d}*%\n%MOIN*%\n"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub(crate) enum Aperture {
    Circle(Coord),
    Rect(Coord, Coord),
    Obround(Coord, Coord),
}

impl Aperture {
    fn def(&self, format: GerberFormat) -> String {
        let f = |v: Coord| format.size(v);
        match self {
            Aperture::Circle(d) => format!("C,{}", f(*d)),
            Aperture::Rect(w, h) => format!("R,{}X{}", f(*w), f(*h)),
            Aperture::Obround(w, h) => format!("O,{}X{}", f(*w), f(*h)),
        }
    }
}

pub(crate) struct Writer {
    body: String,
    apertures: Vec<(Aperture, String)>,
    current: Option<usize>,
    attrs: Vec<String>,
    format: GerberFormat,
}

/// Where a layer's image goes: a Gerber file, or an ODB++ features file.
pub(crate) trait Sink {
    /// The index of aperture `a` with aperture function `function`, defined on first use.
    fn aperture(&mut self, a: Aperture, function: &str) -> usize;
    /// The object attributes (`N` net, `P` pin, `C` component) of the objects that follow.
    fn object(&mut self, attrs: &[(&str, String)]);
    fn flash(&mut self, ap: usize, p: Point);
    /// A stroked polyline.
    fn draw(&mut self, ap: usize, pts: &[Point]);
    /// A filled region (one ring; holes joined by cut-ins).
    fn region(&mut self, ring: &Ring, function: &str);
}

impl Sink for Writer {
    fn aperture(&mut self, a: Aperture, function: &str) -> usize {
        if let Some(i) = self
            .apertures
            .iter()
            .position(|(x, f)| *x == a && f == function)
        {
            return i;
        }
        self.apertures.push((a, function.to_string()));
        self.apertures.len() - 1
    }
    fn object(&mut self, attrs: &[(&str, String)]) {
        let new: Vec<String> = attrs
            .iter()
            .map(|(k, v)| format!("%TO.{k},{v}*%"))
            .collect();
        if new != self.attrs {
            if !self.attrs.is_empty() {
                self.body.push_str("%TD*%\n");
            }
            for a in &new {
                self.body.push_str(a);
                self.body.push('\n');
            }
            self.attrs = new;
        }
    }
    fn flash(&mut self, ap: usize, p: Point) {
        self.select(ap);
        let _ = writeln!(self.body, "X{}Y{}D03*", self.format.coord(p.x), self.format.coord(p.y));
    }
    fn draw(&mut self, ap: usize, pts: &[Point]) {
        self.select(ap);
        for (i, p) in pts.iter().enumerate() {
            let _ = writeln!(
                self.body,
                "X{}Y{}D0{}*",
                self.format.coord(p.x),
                self.format.coord(p.y),
                if i == 0 { 2 } else { 1 }
            );
        }
    }
    fn region(&mut self, ring: &Ring, function: &str) {
        if ring.len() < 3 {
            return;
        }
        let _ = writeln!(self.body, "%TA.AperFunction,{function}*%");
        self.body.push_str("G36*\n");
        for (i, p) in ring.iter().chain(std::iter::once(&ring[0])).enumerate() {
            let _ = writeln!(
                self.body,
                "X{}Y{}D0{}*",
                self.format.coord(p.x),
                self.format.coord(p.y),
                if i == 0 { 2 } else { 1 }
            );
        }
        self.body.push_str("G37*\n%TD.AperFunction*%\n");
    }
}

impl Writer {
    pub(crate) fn new(format: GerberFormat) -> Self {
        Writer {
            body: String::new(),
            apertures: Vec::new(),
            current: None,
            attrs: Vec::new(),
            format,
        }
    }
    fn select(&mut self, i: usize) {
        if self.current != Some(i) {
            let _ = writeln!(self.body, "D{}*", i + 10);
            self.current = Some(i);
        }
    }
    fn finish(self, header: &str) -> String {
        let mut s = String::from(header);
        s.push_str(&self.format.statement());
        s.push_str("%LPD*%\nG01*\n");
        for (i, (a, f)) in self.apertures.iter().enumerate() {
            let _ = writeln!(
                s,
                "%TA.AperFunction,{f}*%\n%ADD{}{}*%\n%TD*%",
                i + 10,
                a.def(self.format)
            );
        }
        s.push_str(&self.body);
        if !self.attrs.is_empty() {
            s.push_str("%TD*%\n");
        }
        s.push_str("M02*\n");
        s
    }
}

/// Which Gerber layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GerberLayer {
    Copper(u8),
    Mask(Side),
    Paste(Side),
    Legend(Side),
    /// Footprint courtyards, for review (not part of the default set).
    Courtyard(Side),
    Profile,
}

/// The X2 file attributes; the project's revision is `rev?` when none is given, as the
/// Gerber specification asks.
pub(crate) fn header(board: &Board, function: &str, polarity: &str, revision: Option<&str>) -> String {
    format!(
        "%TF.GenerationSoftware,RunMat,runmat-circuit,{}*%\n%TF.ProjectId,{},{},{}*%\n%TF.SameCoordinates,Original*%\n%TF.Part,Single*%\n%TF.FileFunction,{function}*%\n%TF.FilePolarity,{polarity}*%\n",
        env!("CARGO_PKG_VERSION"),
        board.name,
        crate::ids::Uuid(crate::ids::fnv128(board.name.as_bytes())).with_version(8),
        revision.map(project_field).unwrap_or_else(|| "rev?".into())
    )
}

/// A project field of an X2 attribute: the characters an attribute value can't hold (`,` and
/// `*`, `%`) become `_`.
fn project_field(text: &str) -> String {
    text.chars().map(|c| if matches!(c, ',' | '*' | '%') { '_' } else { c }).collect()
}

pub(crate) fn pad_aperture(shape: &PadShape, rot: f64) -> Option<Aperture> {
    let axis = (rot / 90.0 - (rot / 90.0).round()).abs() < 1e-9;
    let swap = ((rot / 90.0).round() as i64).rem_euclid(2) == 1;
    let (w, h) = shape.size();
    let (w, h) = if swap { (h, w) } else { (w, h) };
    match shape {
        PadShape::Circle { d } => Some(Aperture::Circle(*d)),
        PadShape::Rect { .. } if axis => Some(Aperture::Rect(w, h)),
        PadShape::Oval { .. } if axis => Some(Aperture::Obround(w, h)),
        _ => None,
    }
}

fn prim_rings(p: &Prim, grow: Coord) -> Vec<Ring> {
    p.outline(grow, crate::geom::mm(0.005))
}

/// One Gerber file, in millimetres with 6 decimals.
pub fn gerber(board: &Board, layer: GerberLayer) -> String {
    gerber_in(board, layer, GerberFormat::default(), None)
}

/// One Gerber file in `format`; `revision` names the board revision in its project attribute.
pub fn gerber_in(board: &Board, layer: GerberLayer, format: GerberFormat, revision: Option<&str>) -> String {
    let mut w = Writer::new(format);
    image(board, layer, &mut w);
    let (function, polarity) = file_function(board, layer);
    w.finish(&header(board, &function, polarity, revision))
}

/// The layer's X2 file function and polarity.
pub(crate) fn file_function(board: &Board, layer: GerberLayer) -> (String, &'static str) {
    let n = board.copper_count();
    let function = match layer {
        GerberLayer::Copper(i) => {
            let pos = if i == 0 {
                "Top"
            } else if i + 1 == n {
                "Bot"
            } else {
                "Inr"
            };
            format!("Copper,L{},{pos}", i + 1)
        }
        GerberLayer::Mask(s) => {
            format!("Soldermask,{}", if s == Side::Top { "Top" } else { "Bot" })
        }
        GerberLayer::Paste(s) => format!("Paste,{}", if s == Side::Top { "Top" } else { "Bot" }),
        GerberLayer::Legend(s) => format!("Legend,{}", if s == Side::Top { "Top" } else { "Bot" }),
        GerberLayer::Courtyard(s) => format!("Other,Courtyard,{}", if s == Side::Top { "Top" } else { "Bot" }),
        GerberLayer::Profile => "Profile,NP".into(),
    };
    let polarity = if matches!(layer, GerberLayer::Mask(_)) {
        "Negative"
    } else {
        "Positive"
    };
    (function, polarity)
}

/// The layer's image, drawn into `w`.
pub(crate) fn image<S: Sink>(board: &Board, layer: GerberLayer, w: &mut S) {
    let n = board.copper_count();
    let side_layer = |s: Side| if s == Side::Top { 0 } else { n - 1 };
    match layer {
        GerberLayer::Copper(l) => {
            // Zone fills first; their holes cleared before pads and tracks go on top.
            for z in board.zones.values() {
                if let Some(f) = z.fill.get(&l) {
                    let net = z
                        .net
                        .map(|x| board.net_name(x).to_string())
                        .unwrap_or_default();
                    w.object(&[("N", net)]);
                    // One region per polygon, holes joined by cut-ins: clear polarity would
                    // also erase other zones' copper drawn earlier on the layer.
                    for p in &f.polys {
                        w.region(&crate::geom::poly::cut_in_contour(p), "Conductor");
                    }
                }
            }
            for t in board.texts.values().filter(|t| t.layer == Layer::Copper(l)) {
                w.object(&[]);
                text_strokes(w, t, "NonConductor");
            }
            for t in board.tracks.values().filter(|t| t.layer == l) {
                let ap = w.aperture(Aperture::Circle(t.width), "Conductor");
                let net = t
                    .net
                    .map(|x| board.net_name(x).to_string())
                    .unwrap_or_default();
                w.object(&[("N", net)]);
                let pts: Vec<Point> = match t.geom {
                    TrackGeom::Segment { a, b } => vec![a, b],
                    TrackGeom::Arc { .. } => {
                        let ps = t.prims();
                        let mut v: Vec<Point> = ps
                            .iter()
                            .map(|p| match p.core {
                                Core::Segment(a, _) => a,
                                _ => Point::ORIGIN,
                            })
                            .collect();
                        if let Some(Core::Segment(_, b)) = ps.last().map(|p| &p.core) {
                            v.push(*b);
                        }
                        v
                    }
                };
                w.draw(ap, &pts);
            }
            for v in board.vias.values().filter(|v| v.layers().any(|x| x == l)) {
                let ap = w.aperture(Aperture::Circle(v.diameter), "ViaPad");
                let net = v
                    .net
                    .map(|x| board.net_name(x).to_string())
                    .unwrap_or_default();
                w.object(&[("N", net)]);
                w.flash(ap, v.at);
            }
            for f in board.footprints.values() {
                for (pd, pp) in f.package.pads.iter().zip(f.pads(n)) {
                    if !pp.layers.contains(&l) {
                        continue;
                    }
                    let func = match pd.kind {
                        PadKind::Smd => "SMDPad,CuDef",
                        PadKind::Tht => "ComponentPad",
                        PadKind::Connector => "ConnectorPad",
                        PadKind::Npth => continue,
                    };
                    let net = pp
                        .net
                        .map(|x| board.net_name(x).to_string())
                        .unwrap_or_default();
                    w.object(&[
                        ("N", net),
                        ("P", format!("{},{}", f.reference, pd.name)),
                        ("C", f.reference.clone()),
                    ]);
                    let rot = f.transform().apply_angle(pd.rotation);
                    match pad_aperture(&pd.shape, rot) {
                        Some(a) => {
                            let ap = w.aperture(a, func);
                            w.flash(ap, pp.center);
                        }
                        None => {
                            for r in prim_rings(&pp.prim, 0) {
                                w.region(&r, func);
                            }
                        }
                    }
                }
            }
        }
        GerberLayer::Mask(side) | GerberLayer::Paste(side) => {
            let mask = matches!(layer, GerberLayer::Mask(_));
            let cl = side_layer(side);
            for f in board.footprints.values() {
                for (pd, pp) in f.package.pads.iter().zip(f.pads(n)) {
                    let on_side = pp.layers.contains(&cl) || (pd.kind == PadKind::Npth && mask);
                    if !on_side
                        || (mask && !pd.mask)
                        || (!mask && (!pd.paste || pd.kind != PadKind::Smd))
                    {
                        continue;
                    }
                    let grow = if mask {
                        pd.mask_margin.unwrap_or(crate::geom::mm(0.05))
                    } else {
                        pd.paste_margin.unwrap_or(0)
                    };
                    w.object(&[
                        ("P", format!("{},{}", f.reference, pd.name)),
                        ("C", f.reference.clone()),
                    ]);
                    let rings = if grow < 0 {
                        let set = crate::geom::poly::PolySet::from_rings(prim_rings(&pp.prim, 0));
                        crate::geom::poly::deflate(&set, -grow, crate::geom::mm(0.005)).rings()
                    } else {
                        prim_rings(&pp.prim, grow)
                    };
                    for r in rings {
                        w.region(&r, "SMDPad,CuDef");
                    }
                }
            }
            if mask {
                for v in board.vias.values().filter(|v| !v.tented) {
                    let ap = w.aperture(
                        Aperture::Circle(v.diameter + crate::geom::mm(0.1)),
                        "ViaPad",
                    );
                    w.flash(ap, v.at);
                }
            }
        }
        GerberLayer::Legend(side) => {
            for f in board.footprints.values().filter(|f| f.side == side) {
                let t = f.transform();
                w.object(&[("C", f.reference.clone())]);
                for g in &f.package.silk {
                    legend_graphic(w, g, &t, Some(f));
                }
            }
            for g in board.graphics.values() {
                if g.layer == Layer::Silk(side) {
                    legend_graphic(w, &g.graphic, &crate::geom::Transform::IDENTITY, None);
                }
            }
            for t in board
                .texts
                .values()
                .filter(|t| t.layer == Layer::Silk(side))
            {
                w.object(&[]);
                text_strokes(w, t, "Other,Legend");
            }
        }
        GerberLayer::Courtyard(side) => {
            let ap = w.aperture(Aperture::Circle(crate::geom::mm(0.05)), "Other,Courtyard");
            for f in board.footprints.values().filter(|f| f.side == side) {
                w.object(&[("C", f.reference.clone())]);
                for r in f.courtyard() {
                    let mut pts = r.clone();
                    pts.push(r[0]);
                    w.draw(ap, &pts);
                }
            }
        }
        GerberLayer::Profile => {
            let ap = w.aperture(Aperture::Circle(crate::geom::mm(0.05)), "Profile");
            for r in board.outline.rings() {
                let mut pts = r.clone();
                pts.push(r[0]);
                w.draw(ap, &pts);
            }
        }
    }
}

/// Board text as strokes of its own thickness.
fn text_strokes<S: Sink>(w: &mut S, t: &crate::layout::BoardText, function: &str) {
    let th = if t.thickness > 0 {
        t.thickness
    } else {
        crate::font::default_thickness(t.size)
    };
    let ap = w.aperture(Aperture::Circle(th), function);
    for pl in crate::font::strokes(&t.text, t.at, t.size, t.angle, t.mirrored) {
        if pl.len() == 1 {
            w.flash(ap, pl[0]);
        } else {
            w.draw(ap, &pl);
        }
    }
}

/// `${REFERENCE}` / `${VALUE}` (and other footprint properties) substituted.
fn expand_fields(text: &str, f: Option<&Footprint>) -> String {
    let Some(f) = f else {
        return text.to_string();
    };
    let mut s = text
        .replace("${REFERENCE}", &f.reference)
        .replace("${VALUE}", &f.value);
    for (k, v) in &f.properties {
        s = s.replace(&format!("${{{}}}", k.to_uppercase()), v);
    }
    s
}

fn legend_graphic<S: Sink>(
    w: &mut S,
    g: &Graphic,
    t: &crate::geom::Transform,
    fp: Option<&Footprint>,
) {
    match g {
        Graphic::Line { points, width } => {
            let ap = w.aperture(Aperture::Circle((*width).max(1)), "Other,Legend");
            let pts: Vec<Point> = points.iter().map(|p| t.apply(*p)).collect();
            w.draw(ap, &pts);
        }
        Graphic::Rect { a, b, width, .. } => {
            let ap = w.aperture(Aperture::Circle((*width).max(1)), "Other,Legend");
            let c = [*a, Point::new(b.x, a.y), *b, Point::new(a.x, b.y), *a];
            let pts: Vec<Point> = c.iter().map(|p| t.apply(*p)).collect();
            w.draw(ap, &pts);
        }
        Graphic::Circle {
            center,
            radius,
            width,
            fill,
        } => {
            if *fill {
                let ap = w.aperture(Aperture::Circle(2 * radius + width), "Other,Legend");
                w.flash(ap, t.apply(*center));
            } else {
                let ap = w.aperture(Aperture::Circle((*width).max(1)), "Other,Legend");
                let mut ring: Vec<Point> =
                    crate::geom::poly::circle_ring(*center, *radius, crate::geom::mm(0.01))
                        .iter()
                        .map(|p| t.apply(*p))
                        .collect();
                ring.push(ring[0]);
                w.draw(ap, &ring);
            }
        }
        Graphic::Arc {
            start,
            mid,
            end,
            width,
        } => {
            let ap = w.aperture(Aperture::Circle((*width).max(1)), "Other,Legend");
            if let Some((c, sw)) = crate::geom::shape::arc_from_three(*start, *mid, *end) {
                let pts: Vec<Point> =
                    crate::geom::shape::arc_points(c, *start, sw, crate::geom::mm(0.01))
                        .iter()
                        .map(|p| t.apply(*p))
                        .collect();
                w.draw(ap, &pts);
            }
        }
        Graphic::Polygon { points, width, .. } => {
            let ap = w.aperture(Aperture::Circle((*width).max(1)), "Other,Legend");
            let mut pts: Vec<Point> = points.iter().map(|p| t.apply(*p)).collect();
            if let Some(f) = pts.first().copied() {
                pts.push(f);
            }
            w.draw(ap, &pts);
        }
        Graphic::Text {
            at,
            text,
            size,
            angle,
        } => {
            let ap = w.aperture(
                Aperture::Circle(crate::font::default_thickness(*size)),
                "Other,Legend",
            );
            // Placed text reads upright: rotation follows the footprint; the bottom side
            // mirrors it.
            let txt = expand_fields(text, fp);
            for pl in crate::font::strokes(
                &txt,
                t.apply(*at),
                *size,
                t.apply_angle(*angle) - if t.mirror { 180.0 } else { 0.0 },
                t.mirror,
            ) {
                if pl.len() == 1 {
                    w.flash(ap, pl[0]);
                } else {
                    w.draw(ap, &pl);
                }
            }
        }
    }
}

/// The board's Gerber layers with their conventional file names: copper top to bottom, then
/// mask, paste and legend of each side, then the profile.
pub fn gerber_layers(board: &Board) -> Vec<(String, GerberLayer)> {
    let n = board.copper_count();
    let mut out = Vec::new();
    for i in 0..n {
        let name = crate::layout::copper_name(i, n).replace('.', "_");
        out.push((format!("{}-{name}.gbr", board.name), GerberLayer::Copper(i)));
    }
    for s in [Side::Top, Side::Bottom] {
        let p = if s == Side::Top { "F" } else { "B" };
        out.push((format!("{}-{p}_Mask.gbr", board.name), GerberLayer::Mask(s)));
        out.push((
            format!("{}-{p}_Paste.gbr", board.name),
            GerberLayer::Paste(s),
        ));
        out.push((
            format!("{}-{p}_Silkscreen.gbr", board.name),
            GerberLayer::Legend(s),
        ));
    }
    out.push((
        format!("{}-Edge_Cuts.gbr", board.name),
        GerberLayer::Profile,
    ));
    out
}

/// Every Gerber of the board, by conventional file name.
pub fn gerber_set(board: &Board) -> BTreeMap<String, String> {
    gerber_layers(board)
        .into_iter()
        .map(|(name, l)| (name, gerber(board, l)))
        .collect()
}
