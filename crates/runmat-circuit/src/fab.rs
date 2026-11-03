//! Fabrication and assembly outputs.
//!
//! - [`gerber`]: Gerber X2 (RS-274X with `%TF`/`%TA`/`%TO` attributes) per layer: copper
//!   (pads flashed or as regions, tracks drawn, vias flashed, zone fills as regions with holes
//!   cleared, every object carrying `.N` net / `.P` pin / `.C` component attributes), solder
//!   mask (negative polarity, openings grown by the mask margin, vias tented), paste, legend
//!   (silk graphics) and profile.
//! - [`excellon`]: Excellon drill files: plated holes split by span (through, blind, buried
//!   per layer pair, with X2-style `TF.FileFunction` comments) and non-plated holes; slots as
//!   `G85`.
//! - [`ipc356`]: an IPC-D-356A netlist (fixed-column `317`/`327` test records, metric units).
//! - [`bom`]: grouped, variant-aware bill of materials (CSV).
//! - [`pick_place`]: pick-and-place (CSV).
//!
//! Outputs are deterministic (sorted, fixed precision, no dates unless the caller passes one).
//! Text (board text, footprint fields with `${REFERENCE}`/`${VALUE}` substituted) is drawn as
//! strokes of the [`crate::font`] stroke font. Zone fills are single regions whose holes are
//! joined by cut-ins, so overlapping zones on one layer never erase each other.

use crate::geom::poly::Ring;
use crate::geom::shape::{Core, Prim};
use crate::geom::{to_mm, Coord, Point};
use crate::layout::{Board, Footprint, Layer, Side, TrackGeom, ViaKind};
use crate::library::{Graphic, PadKind, PadShape};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Gerber coordinate: 4.6 format in mm (nanometre resolution).
fn gc(v: Coord) -> String {
    v.to_string()
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
enum Aperture {
    Circle(Coord),
    Rect(Coord, Coord),
    Obround(Coord, Coord),
}

impl Aperture {
    fn def(&self) -> String {
        let f = |v: Coord| format!("{:.6}", to_mm(v));
        match self {
            Aperture::Circle(d) => format!("C,{}", f(*d)),
            Aperture::Rect(w, h) => format!("R,{}X{}", f(*w), f(*h)),
            Aperture::Obround(w, h) => format!("O,{}X{}", f(*w), f(*h)),
        }
    }
}

struct Writer {
    body: String,
    apertures: Vec<(Aperture, String)>,
    current: Option<usize>,
    attrs: Vec<String>,
}

impl Writer {
    fn new() -> Self {
        Writer {
            body: String::new(),
            apertures: Vec::new(),
            current: None,
            attrs: Vec::new(),
        }
    }
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
    fn select(&mut self, i: usize) {
        if self.current != Some(i) {
            let _ = writeln!(self.body, "D{}*", i + 10);
            self.current = Some(i);
        }
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
        let _ = writeln!(self.body, "X{}Y{}D03*", gc(p.x), gc(p.y));
    }
    fn draw(&mut self, ap: usize, pts: &[Point]) {
        self.select(ap);
        for (i, p) in pts.iter().enumerate() {
            let _ = writeln!(
                self.body,
                "X{}Y{}D0{}*",
                gc(p.x),
                gc(p.y),
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
                gc(p.x),
                gc(p.y),
                if i == 0 { 2 } else { 1 }
            );
        }
        self.body.push_str("G37*\n%TD.AperFunction*%\n");
    }
    fn finish(self, header: &str) -> String {
        let mut s = String::from(header);
        s.push_str("%FSLAX46Y46*%\n%MOMM*%\n%LPD*%\nG01*\n");
        for (i, (a, f)) in self.apertures.iter().enumerate() {
            let _ = writeln!(
                s,
                "%TA.AperFunction,{f}*%\n%ADD{}{}*%\n%TD*%",
                i + 10,
                a.def()
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
    Profile,
}

fn header(board: &Board, function: &str, polarity: &str) -> String {
    format!(
        "%TF.GenerationSoftware,RunMat,runmat-circuit,{}*%\n%TF.ProjectId,{},{},rev?*%\n%TF.SameCoordinates,Original*%\n%TF.FileFunction,{function}*%\n%TF.FilePolarity,{polarity}*%\n",
        env!("CARGO_PKG_VERSION"),
        board.name,
        crate::ids::Uuid(crate::ids::fnv128(board.name.as_bytes())).with_version(8)
    )
}

fn pad_aperture(shape: &PadShape, rot: f64) -> Option<Aperture> {
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

/// One Gerber file.
pub fn gerber(board: &Board, layer: GerberLayer) -> String {
    let n = board.copper_count();
    let mut w = Writer::new();
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
        GerberLayer::Profile => "Profile,NP".into(),
    };
    let polarity = if matches!(layer, GerberLayer::Mask(_)) {
        "Negative"
    } else {
        "Positive"
    };
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
                text_strokes(&mut w, t, "NonConductor");
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
                    legend_graphic(&mut w, g, &t, Some(f));
                }
            }
            for g in board.graphics.values() {
                if g.layer == Layer::Silk(side) {
                    legend_graphic(&mut w, &g.graphic, &crate::geom::Transform::IDENTITY, None);
                }
            }
            for t in board
                .texts
                .values()
                .filter(|t| t.layer == Layer::Silk(side))
            {
                w.object(&[]);
                text_strokes(&mut w, t, "Other,Legend");
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
    w.finish(&header(board, &function, polarity))
}

/// Board text as strokes of its own thickness.
fn text_strokes(w: &mut Writer, t: &crate::layout::BoardText, function: &str) {
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

fn legend_graphic(w: &mut Writer, g: &Graphic, t: &crate::geom::Transform, fp: Option<&Footprint>) {
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

/// Every Gerber of the board, by conventional file name.
pub fn gerber_set(board: &Board) -> BTreeMap<String, String> {
    let n = board.copper_count();
    let mut out = BTreeMap::new();
    for i in 0..n {
        out.insert(
            format!(
                "{}-{}.gbr",
                board.name,
                crate::layout::copper_name(i, n).replace('.', "_")
            ),
            gerber(board, GerberLayer::Copper(i)),
        );
    }
    for s in [Side::Top, Side::Bottom] {
        let p = if s == Side::Top { "F" } else { "B" };
        out.insert(
            format!("{}-{p}_Mask.gbr", board.name),
            gerber(board, GerberLayer::Mask(s)),
        );
        out.insert(
            format!("{}-{p}_Paste.gbr", board.name),
            gerber(board, GerberLayer::Paste(s)),
        );
        out.insert(
            format!("{}-{p}_Silkscreen.gbr", board.name),
            gerber(board, GerberLayer::Legend(s)),
        );
    }
    out.insert(
        format!("{}-Edge_Cuts.gbr", board.name),
        gerber(board, GerberLayer::Profile),
    );
    out
}

#[derive(Clone, Debug, PartialEq)]
struct Hole {
    at: Point,
    dia: Coord,
    slot: Option<(Point, Point)>,
}

/// Excellon drill files: one per plated span (`PTH`, then blind/buried spans) and one for
/// non-plated holes, keyed by file name.
pub fn excellon(board: &Board) -> BTreeMap<String, String> {
    let n = board.copper_count();
    let mut spans: BTreeMap<(u8, u8, bool), Vec<Hole>> = BTreeMap::new();
    for v in board.vias.values() {
        spans.entry((v.from, v.to, true)).or_default().push(Hole {
            at: v.at,
            dia: v.drill,
            slot: None,
        });
    }
    for f in board.footprints.values() {
        let t = f.transform();
        for (pd, pp) in f.package.pads.iter().zip(f.pads(n)) {
            let Some(d) = &pd.drill else { continue };
            let slot = d.slot.map(|len| {
                let half = (len - d.diameter).max(0) / 2;
                let local = crate::geom::Transform::new(pd.at, pd.rotation, false);
                let tt = t.compose(&local);
                (
                    tt.apply(Point::new(-half, 0)),
                    tt.apply(Point::new(half, 0)),
                )
            });
            spans.entry((0, n - 1, d.plated)).or_default().push(Hole {
                at: pp.center,
                dia: d.diameter,
                slot,
            });
        }
    }
    let mut out = BTreeMap::new();
    for ((from, to, plated), mut holes) in spans {
        holes.sort_by_key(|h| (h.dia, h.at));
        let mut tools: Vec<Coord> = holes.iter().map(|h| h.dia).collect();
        tools.sort();
        tools.dedup();
        let kind = if !plated {
            "NPTH"
        } else {
            let k = crate::layout::Via {
                id: Default::default(),
                net: None,
                at: Point::ORIGIN,
                diameter: 0,
                drill: 0,
                from,
                to,
                micro: false,
                tented: true,
                locked: false,
            }
            .kind(n);
            match k {
                ViaKind::Through => "PTH",
                ViaKind::Buried => "Buried",
                _ => "Blind",
            }
        };
        let mut s = String::new();
        let _ = writeln!(
            s,
            "M48\n; DRILL file {{RunMat runmat-circuit {}}}",
            env!("CARGO_PKG_VERSION")
        );
        let _ = writeln!(s, "; FORMAT={{-:-/ absolute / metric / decimal}}");
        let _ = writeln!(
            s,
            "; #@! TF.FileFunction,{},{},{},{kind}",
            if plated { "Plated" } else { "NonPlated" },
            from + 1,
            to + 1
        );
        let _ = writeln!(s, "FMAT,2\nMETRIC");
        for (i, d) in tools.iter().enumerate() {
            let _ = writeln!(
                s,
                "; #@! TA.AperFunction,{},{}\nT{}C{:.3}",
                if plated { "Plated" } else { "NonPlated" },
                kind,
                i + 1,
                to_mm(*d)
            );
        }
        s.push_str("%\nG90\nG05\n");
        for (i, d) in tools.iter().enumerate() {
            let _ = writeln!(s, "T{}", i + 1);
            for h in holes.iter().filter(|h| h.dia == *d) {
                match h.slot {
                    Some((a, b)) => {
                        let _ = writeln!(
                            s,
                            "X{:.4}Y{:.4}G85X{:.4}Y{:.4}",
                            to_mm(a.x),
                            to_mm(a.y),
                            to_mm(b.x),
                            to_mm(b.y)
                        );
                    }
                    None => {
                        let _ = writeln!(s, "X{:.4}Y{:.4}", to_mm(h.at.x), to_mm(h.at.y));
                    }
                }
            }
        }
        s.push_str("M30\n");
        let name = if plated {
            format!("{}-{}-{}-{}.drl", board.name, kind, from + 1, to + 1)
        } else {
            format!("{}-NPTH.drl", board.name)
        };
        out.insert(name, s);
    }
    out
}

/// IPC-D-356A netlist (metric, 0.001 mm units).
pub fn ipc356(board: &Board) -> String {
    let n = board.copper_count();
    let mut s = String::new();
    let _ = writeln!(s, "C  IPC-D-356A netlist generated by RunMat");
    let _ = writeln!(s, "P  JOB   {}", board.name);
    let _ = writeln!(s, "P  UNITS CUST 1");
    let _ = writeln!(s, "P  DIM   N");
    let mut recs: Vec<String> = Vec::new();
    let num = |v: Coord| {
        let u = (v as f64 / 1000.0).round() as i64;
        format!(
            "{}{:06}",
            if u < 0 { '-' } else { '+' },
            u.abs().min(999_999)
        )
    };
    let sz = |v: Coord| format!("{:04}", ((v as f64 / 1000.0).round() as i64).clamp(0, 9999));
    for f in board.footprints.values() {
        for (pd, pp) in f.package.pads.iter().zip(f.pads(n)) {
            let Some(net) = pp.net else { continue };
            let name: String = board.net_name(net).chars().take(14).collect();
            let thru = pd.kind == PadKind::Tht;
            let code = if thru { "317" } else { "327" };
            let access: String = if thru {
                "A00".into()
            } else if f.side == Side::Top {
                "A01".into()
            } else {
                format!("A{n:02}")
            };
            let b = pp.prim.bbox();
            let drill = match pd.drill.as_ref() {
                Some(d) => format!("D{}{}", sz(d.diameter), if d.plated { 'P' } else { 'U' }),
                None => "      ".into(),
            };
            let rot = (f.transform().apply_angle(pd.rotation).round() as i64).rem_euclid(360);
            recs.push(format!(
                "{code}{name:<14}   {:<6}-{:<4} {drill}{access}X{}Y{}X{}Y{}R{:03} S{}",
                f.reference.chars().take(6).collect::<String>(),
                pd.name.chars().take(4).collect::<String>(),
                num(pp.center.x),
                num(pp.center.y),
                sz(b.width()),
                sz(b.height()),
                rot,
                if pd.mask { 0 } else { 3 }
            ));
        }
    }
    for v in board.vias.values() {
        let Some(net) = v.net else { continue };
        let name: String = board.net_name(net).chars().take(14).collect();
        recs.push(format!(
            "317{name:<14}   VIA   -    D{}PA00X{}Y{}X{}Y{}R000 S{}",
            sz(v.drill),
            num(v.at.x),
            num(v.at.y),
            sz(v.diameter),
            sz(v.diameter),
            if v.tented { 3 } else { 0 }
        ));
    }
    recs.sort();
    for r in recs {
        s.push_str(&r);
        s.push('\n');
    }
    s.push_str("999\n");
    s
}

/// Natural sort key: `R2` < `R10`.
pub fn natural_key(s: &str) -> (String, u64, String) {
    let p: String = s.chars().take_while(|c| !c.is_ascii_digit()).collect();
    let rest = &s[p.len()..];
    let d: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    (p, d.parse().unwrap_or(0), rest[d.len()..].to_string())
}

fn csv(fields: &[String]) -> String {
    fields
        .iter()
        .map(|f| {
            if f.contains([',', '"', '\n']) {
                format!("\"{}\"", f.replace('"', "\"\""))
            } else {
                f.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// A BOM line.
#[derive(Clone, Debug, PartialEq)]
pub struct BomLine {
    pub designators: Vec<String>,
    pub quantity: usize,
    pub value: String,
    pub package: String,
    pub mpn: String,
    pub manufacturer: String,
}

/// Grouped, variant-aware BOM from an elaborated netlist (not-fitted parts are left out;
/// alternates appear with their own part).
pub fn bom(nl: &crate::elab::Netlist, pool: &crate::library::Pool) -> Vec<BomLine> {
    let mut groups: BTreeMap<(String, String, String, String), Vec<String>> = BTreeMap::new();
    for c in nl.components.iter().filter(|c| c.fitted()) {
        let part = c.part.and_then(|u| pool.parts.get(&u));
        let pkg = c
            .package
            .and_then(|u| pool.packages.get(&u))
            .map(|p| p.name.clone())
            .unwrap_or_default();
        let value = c
            .properties
            .get("Value")
            .cloned()
            .or_else(|| {
                ["resistance", "capacitance", "inductance", "frequency"]
                    .iter()
                    .find_map(|k| {
                        c.params.get(*k).map(|p| match &p.value {
                            crate::ir::ParamValue::Quantity { nominal, .. } => {
                                format!("{}", nominal.value)
                            }
                            crate::ir::ParamValue::Text(t) => t.clone(),
                            crate::ir::ParamValue::Bool(b) => b.to_string(),
                        })
                    })
            })
            .unwrap_or_default();
        let key = (
            part.map(|p| p.mpn.clone()).unwrap_or_default(),
            value,
            pkg,
            part.map(|p| p.manufacturer.clone()).unwrap_or_default(),
        );
        groups.entry(key).or_default().push(c.designator.clone());
    }
    let mut out: Vec<BomLine> = groups
        .into_iter()
        .map(|((mpn, value, package, manufacturer), mut d)| {
            d.sort_by_key(|x| natural_key(x));
            BomLine {
                quantity: d.len(),
                designators: d,
                value,
                package,
                mpn,
                manufacturer,
            }
        })
        .collect();
    out.sort_by_key(|l| natural_key(&l.designators[0]));
    out
}

pub fn bom_csv(lines: &[BomLine]) -> String {
    let mut s = String::from("Designators,Quantity,Value,Package,MPN,Manufacturer\n");
    for l in lines {
        s.push_str(&csv(&[
            l.designators.join(" "),
            l.quantity.to_string(),
            l.value.clone(),
            l.package.clone(),
            l.mpn.clone(),
            l.manufacturer.clone(),
        ]));
        s.push('\n');
    }
    s
}

/// Pick-and-place CSV for fitted footprints (SMD only unless `include_tht`), origin at the
/// outline's lower-left corner, rotation counter-clockwise as seen from the top.
pub fn pick_place(board: &Board, include_tht: bool) -> String {
    let origin = board.outline.bbox().min;
    let mut rows: Vec<(Footprint, String)> = Vec::new();
    for f in board
        .footprints
        .values()
        .filter(|f| f.fitted && (f.package.smd || include_tht))
    {
        let p = f.at - origin;
        rows.push((
            f.clone(),
            csv(&[
                f.reference.clone(),
                f.value.clone(),
                f.package.name.clone(),
                format!("{:.4}", to_mm(p.x)),
                format!("{:.4}", to_mm(p.y)),
                format!("{:.2}", f.rotation),
                if f.side == Side::Top {
                    "top".into()
                } else {
                    "bottom".into()
                },
            ]),
        ));
    }
    rows.sort_by_key(|(f, _)| natural_key(&f.reference));
    let mut s = String::from("Designator,Value,Package,X_mm,Y_mm,Rotation,Side\n");
    for (_, r) in rows {
        s.push_str(&r);
        s.push('\n');
    }
    s
}
