//! ODB++ (v8.1 structure): the job tree as path → text, ready to archive as `<job>.tgz`.
//!
//! - `matrix/matrix`: one step (`pcb`) and the layer rows, top to bottom: legend, paste and
//!   mask of the top, the copper layers (signal), mask, paste and legend of the bottom, one
//!   drill layer per span (`START_NAME`/`END_NAME` name its copper), the top and bottom
//!   component layers.
//! - `misc/info`: job name, units and the writer.
//! - `steps/pcb/stephdr`, `steps/pcb/profile`: the step's origin and its outline as a surface.
//! - `steps/pcb/layers/<layer>/features`: each layer's image, drawn by the same code as the
//!   Gerber file of that layer ([`super::image`]): flashed pads as `P` records on standard
//!   symbols (`r`, `rect`, `oval`, sizes in microns), strokes as `L`, regions as `S` surfaces.
//!   Drill layers flash the finished hole size; slots are `L` records.
//!   Text (legend and copper) is carried only as those stroke-font `L` records, the same
//!   strokes the Gerber files hold and board houses fabricate from. No `T` records or
//!   `fonts/standard` are written, so no reader draws a text twice.
//! - `steps/pcb/layers/comp_+_top|comp_+_bot/components`: `CMP` per fitted footprint
//!   (position, rotation clockwise as the format counts it, mirror for the bottom), with its
//!   value and part number as `PRP` and a `TOP` record per pad.
//! - `steps/pcb/eda/data`: the layer list, every net with its pads (`SNT TOP`) and vias
//!   (`SNT VIA`) and the copper features that carry it (`FID`), and the packages with their pins.
//! - `steps/pcb/netlists/cadnet/netlist`: net names and the points of each net.
//!
//! Units are millimetres throughout (`UNITS=MM`). Output is deterministic.

use crate::layer_set::CopperIndex;
use super::drill::{holes, Span};
use super::{image, Aperture, GerberLayer, Sink};
use crate::geom::poly::{area2, Ring};
use crate::geom::{to_mm, Coord, Point};
use crate::layout::{Board, Side};
use std::collections::BTreeMap;
use std::fmt::Write as _;

fn n(v: Coord) -> String {
    let s = format!("{:.6}", to_mm(v));
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s == "-0" {
        "0".into()
    } else {
        s
    }
}

/// Microns, as symbol names carry sizes when `UNITS=MM`.
fn um(v: Coord) -> String {
    let s = format!("{:.3}", v as f64 / 1000.0);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn symbol_name(a: &Aperture) -> String {
    match a {
        Aperture::Circle(d) => format!("r{}", um(*d)),
        Aperture::Rect(w, h) => format!("rect{}x{}", um(*w), um(*h)),
        Aperture::Obround(w, h) => format!("oval{}x{}", um(*w), um(*h)),
    }
}

/// An ODB++ features file under construction; it records which net each feature carries.
#[derive(Default)]
struct Features {
    symbols: Vec<String>,
    /// Aperture index (as handed out) → symbol index.
    apertures: Vec<usize>,
    body: String,
    count: usize,
    net: String,
    /// Net name → feature indices on this layer.
    nets: BTreeMap<String, Vec<usize>>,
}

impl Features {
    fn symbol(&mut self, name: String) -> usize {
        match self.symbols.iter().position(|s| *s == name) {
            Some(i) => i,
            None => {
                self.symbols.push(name);
                self.symbols.len() - 1
            }
        }
    }
    fn feature(&mut self) {
        if !self.net.is_empty() {
            self.nets
                .entry(self.net.clone())
                .or_default()
                .push(self.count);
        }
        self.count += 1;
    }
    fn surface(&mut self, rings: &[Ring]) {
        let mut s = String::from("S P 0\n");
        for (k, r) in rings.iter().enumerate() {
            if r.len() < 3 {
                continue;
            }
            // Islands clockwise, holes counter-clockwise; the first ring is the island.
            let island = k == 0;
            let mut r = r.clone();
            let ccw = area2(&r) > 0;
            if island == ccw {
                r.reverse();
            }
            let _ = writeln!(
                s,
                "OB {} {} {}",
                n(r[0].x),
                n(r[0].y),
                if island { 'I' } else { 'H' }
            );
            for p in r.iter().skip(1).chain(std::iter::once(&r[0])) {
                let _ = writeln!(s, "OS {} {}", n(p.x), n(p.y));
            }
            s.push_str("OE\n");
        }
        s.push_str("SE\n");
        self.body.push_str(&s);
        self.feature();
    }
    fn line(&mut self, sym: usize, a: Point, b: Point) {
        let _ = writeln!(
            self.body,
            "L {} {} {} {} {sym} P 0",
            n(a.x),
            n(a.y),
            n(b.x),
            n(b.y)
        );
        self.feature();
    }
    fn pad(&mut self, sym: usize, p: Point) {
        let _ = writeln!(self.body, "P {} {} {sym} P 0 0", n(p.x), n(p.y));
        self.feature();
    }
    fn text(&self) -> String {
        let mut s = String::from("UNITS=MM\n#\n#Feature symbol names\n#\n");
        for (i, sym) in self.symbols.iter().enumerate() {
            let _ = writeln!(s, "${i} {sym}");
        }
        s.push_str("#\n#Layer features\n#\n");
        s.push_str(&self.body);
        s
    }
}

impl Sink for Features {
    fn aperture(&mut self, a: Aperture, _function: &str) -> usize {
        let sym = self.symbol(symbol_name(&a));
        self.apertures.push(sym);
        self.apertures.len() - 1
    }
    fn object(&mut self, attrs: &[(&str, String)]) {
        self.net = attrs
            .iter()
            .find(|(k, _)| *k == "N")
            .map(|(_, v)| v.clone())
            .unwrap_or_default();
    }
    fn flash(&mut self, ap: usize, p: Point) {
        let sym = self.apertures[ap];
        self.pad(sym, p);
    }
    fn draw(&mut self, ap: usize, pts: &[Point]) {
        let sym = self.apertures[ap];
        if pts.len() == 1 {
            self.pad(sym, pts[0]);
        }
        for w in pts.windows(2) {
            self.line(sym, w[0], w[1]);
        }
    }
    fn region(&mut self, ring: &Ring, _function: &str) {
        self.surface(std::slice::from_ref(ring));
    }
}

/// One matrix row: name, type, and for drill layers the copper span.
struct Row {
    name: String,
    kind: &'static str,
    span: Option<(String, String)>,
    image: Option<GerberLayer>,
    drill: Option<Span>,
}

fn rows(board: &Board) -> Vec<Row> {
    let c = board.copper_count();
    let layer = |name: &str, kind, image| Row {
        name: name.into(),
        kind,
        span: None,
        image: Some(image),
        drill: None,
    };
    let mut v = vec![
        layer("top_legend", "SILK_SCREEN", GerberLayer::Legend(Side::Top)),
        layer("top_paste", "SOLDER_PASTE", GerberLayer::Paste(Side::Top)),
        layer("top_mask", "SOLDER_MASK", GerberLayer::Mask(Side::Top)),
    ];
    for i in 0..c {
        v.push(layer(&odb_copper(i, c), "SIGNAL", GerberLayer::Copper(i)));
    }
    v.push(layer(
        "bottom_mask",
        "SOLDER_MASK",
        GerberLayer::Mask(Side::Bottom),
    ));
    v.push(layer(
        "bottom_paste",
        "SOLDER_PASTE",
        GerberLayer::Paste(Side::Bottom),
    ));
    v.push(layer(
        "bottom_legend",
        "SILK_SCREEN",
        GerberLayer::Legend(Side::Bottom),
    ));
    for span in holes(board).keys() {
        let name = if span.plated {
            format!("drill_{}-{}", span.from + 1, span.to + 1)
        } else {
            "drill_npth".into()
        };
        v.push(Row {
            name,
            kind: "DRILL",
            span: Some((odb_copper(span.from, c), odb_copper(span.to, c))),
            image: None,
            drill: Some(*span),
        });
    }
    v
}

/// Copper layer names: `top`, `inner1` …, `bottom`.
fn odb_copper(i: CopperIndex, count: CopperIndex) -> String {
    if i == 0 {
        "top".into()
    } else if i + 1 == count {
        "bottom".into()
    } else {
        format!("inner{i}")
    }
}

/// The ODB++ job of the board. `parts` maps a reference designator to its part number.
/// Paths are relative to the archive root (`<job>/matrix/matrix` …).
pub fn odb(board: &Board, parts: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let job: String = board
        .name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let root = format!("{job}/");
    let step = format!("{root}steps/pcb/");
    let mut out = BTreeMap::new();
    let rows = rows(board);

    // Matrix.
    let mut m = String::from("STEP {\n   COL=1\n   NAME=PCB\n}\n\n");
    let comp_rows = [("comp_+_top", Side::Top), ("comp_+_bot", Side::Bottom)];
    let mut row_no = 0;
    let mut matrix_row = |m: &mut String,
                          name: &str,
                          kind: &str,
                          span: Option<&(String, String)>| {
        row_no += 1;
        let _ = writeln!(m, "LAYER {{\n   ROW={row_no}\n   CONTEXT=BOARD\n   TYPE={kind}\n   NAME={}\n   POLARITY=POSITIVE", name.to_uppercase());
        if let Some((a, b)) = span {
            let _ = writeln!(
                m,
                "   START_NAME={}\n   END_NAME={}",
                a.to_uppercase(),
                b.to_uppercase()
            );
        }
        m.push_str("}\n\n");
    };
    matrix_row(&mut m, comp_rows[0].0, "COMPONENT", None);
    for r in &rows {
        matrix_row(&mut m, &r.name, r.kind, r.span.as_ref());
    }
    matrix_row(&mut m, comp_rows[1].0, "COMPONENT", None);
    out.insert(format!("{root}matrix/matrix"), m);
    out.insert(
        format!("{root}misc/info"),
        format!(
            "JOB_NAME={job}\nODB_VERSION_MAJOR=8\nODB_VERSION_MINOR=1\nODB_SOURCE=RunMat runmat-circuit {}\nCREATION_DATE=\nSAVE_DATE=\nSAVE_APP=RunMat\nUNITS=MM\n",
            env!("CARGO_PKG_VERSION")
        ),
    );
    out.insert(
        format!("{step}stephdr"),
        "UNITS=MM\nX_DATUM=0\nY_DATUM=0\nX_ORIGIN=0\nY_ORIGIN=0\n".into(),
    );
    let mut profile = Features::default();
    let outline: Vec<Ring> = board.outline.rings().cloned().collect();
    profile.surface(&outline);
    out.insert(format!("{step}profile"), profile.text());

    // Layer features, and which features carry each net.
    let mut net_features: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    let all_holes = holes(board);
    for (li, r) in rows.iter().enumerate() {
        let mut f = Features::default();
        if let Some(img) = r.image {
            image(board, img, &mut f);
        }
        if let Some(span) = r.drill {
            for h in all_holes.get(&span).into_iter().flatten() {
                let sym = f.symbol(format!("r{}", um(h.dia)));
                match h.slot {
                    Some((a, b)) => f.line(sym, a, b),
                    None => f.pad(sym, h.at),
                }
            }
        }
        if r.kind == "SIGNAL" {
            for (net, ids) in &f.nets {
                net_features
                    .entry(net.clone())
                    .or_default()
                    .extend(ids.iter().map(|i| (li, *i)));
            }
        }
        out.insert(format!("{step}layers/{}/features", r.name), f.text());
    }

    // Components, packages and nets.
    let c = board.copper_count();
    let mut packages: Vec<String> = Vec::new();
    let mut pkg_of: BTreeMap<String, &crate::library::Package> = BTreeMap::new();
    for fp in board.footprints.values() {
        if !pkg_of.contains_key(&fp.package.name) {
            packages.push(fp.package.name.clone());
            pkg_of.insert(fp.package.name.clone(), &fp.package);
        }
    }
    let mut net_names: Vec<&str> = board
        .nets
        .iter()
        .map(|x| x.name.as_str())
        .filter(|x| !x.is_empty())
        .collect();
    net_names.sort();
    net_names.dedup();
    let net_no = |name: &str| net_names.iter().position(|x| *x == name);
    // Net name → (side, component number on that side, pin number).
    let mut net_pins: BTreeMap<String, Vec<(char, usize, usize)>> = BTreeMap::new();
    for (layer, side) in comp_rows {
        let mut s = String::from("UNITS=MM\n");
        let mut fps: Vec<&crate::layout::Footprint> = board
            .footprints
            .values()
            .filter(|f| f.side == side && f.fitted)
            .collect();
        fps.sort_by_key(|f| super::natural_key(&f.reference));
        for (ci, fp) in fps.iter().enumerate() {
            let pkg = packages
                .iter()
                .position(|p| *p == fp.package.name)
                .unwrap_or(0);
            let mirror = if side == Side::Bottom { 'M' } else { 'N' };
            let rot = (360.0 - fp.rotation).rem_euclid(360.0);
            let part = parts
                .get(&fp.reference)
                .cloned()
                .unwrap_or_else(|| fp.value.clone());
            let _ = writeln!(
                s,
                "# CMP {ci}\nCMP {pkg} {} {} {} {mirror} {} {}",
                n(fp.at.x),
                n(fp.at.y),
                fmt_angle(rot),
                fp.reference,
                token(&part)
            );
            let _ = writeln!(s, "PRP VALUE '{}'", fp.value.replace('\'', ""));
            for (pi, pp) in fp.pads(c).iter().enumerate() {
                let rot = (360.0 - fp.transform().apply_angle(fp.package.pads[pi].rotation))
                    .rem_euclid(360.0);
                let net_k = pp
                    .net
                    .and_then(|x| net_no(board.net_name(x)))
                    .map_or(-1, |k| k as i64);
                let _ = writeln!(
                    s,
                    "TOP {pi} {} {} {} {mirror} {net_k} 0 {}",
                    n(pp.center.x),
                    n(pp.center.y),
                    fmt_angle(rot),
                    token(&pp.name)
                );
                if let Some(net) = pp.net {
                    let side_ch = if side == Side::Top { 'T' } else { 'B' };
                    net_pins
                        .entry(board.net_name(net).to_string())
                        .or_default()
                        .push((side_ch, ci, pi));
                }
            }
        }
        out.insert(format!("{step}layers/{layer}/components"), s);
        out.insert(
            format!("{step}layers/{layer}/features"),
            "UNITS=MM\n#\n#Layer features\n#\n".into(),
        );
    }

    let mut eda = String::from("HDR RunMat\nUNITS=MM\nLYR");
    for r in rows.iter() {
        let _ = write!(eda, " {}", r.name.to_uppercase());
    }
    eda.push('\n');
    let mut net_vias: BTreeMap<String, usize> = BTreeMap::new();
    for v in board.vias.values() {
        if let Some(net) = v.net {
            *net_vias.entry(board.net_name(net).to_string()).or_default() += 1;
        }
    }
    let mut netlist = String::from("H optimize n staggered n\nUNITS=MM\n");
    for (k, name) in net_names.iter().enumerate() {
        let _ = writeln!(netlist, "${k} {}", token(name));
    }
    for (k, name) in net_names.iter().enumerate() {
        let _ = writeln!(eda, "# NET {k}\nNET {}", token(name));
        for (side, ci, pi) in net_pins.get(*name).into_iter().flatten() {
            let _ = writeln!(eda, "SNT TOP {side} {ci} {pi}");
        }
        if net_vias.get(*name).copied().unwrap_or(0) > 0 {
            eda.push_str("SNT VIA\n");
        }
        if let Some(fids) = net_features.get(*name) {
            if net_pins.get(*name).is_none() && net_vias.get(*name).is_none() {
                eda.push_str("SNT TRC\n");
            }
            for (layer, feat) in fids {
                let _ = writeln!(eda, "FID C {layer} {feat}");
            }
        }
    }
    for v in board.vias.values() {
        let k = v.net.and_then(|x| net_no(board.net_name(x)));
        if let Some(k) = k {
            let _ = writeln!(
                netlist,
                "{k} {} {} {} B",
                n(v.drill / 2),
                n(v.at.x),
                n(v.at.y)
            );
        }
    }
    for (pi, name) in packages.iter().enumerate() {
        let pkg = pkg_of[name];
        let mut bb: Option<crate::geom::Rect> = None;
        for p in &pkg.pads {
            let r = crate::layout::pad_prim(p, &crate::geom::Transform::IDENTITY).bbox();
            bb = Some(bb.map_or(r, |b| b.union(r)));
        }
        let bb = bb.unwrap_or(crate::geom::Rect::new(Point::ORIGIN, Point::ORIGIN));
        let pitch = pitch(pkg);
        let _ = writeln!(
            eda,
            "# PKG {pi}\nPKG {} {} {} {} {} {}",
            token(name),
            n(pitch),
            n(bb.min.x),
            n(bb.min.y),
            n(bb.max.x),
            n(bb.max.y)
        );
        let _ = writeln!(
            eda,
            "RC {} {} {} {}",
            n(bb.min.x),
            n(bb.min.y),
            n(bb.width()),
            n(bb.height())
        );
        for p in &pkg.pads {
            let r = crate::layout::pad_prim(p, &crate::geom::Transform::IDENTITY).bbox();
            let kind = if p.drill.is_some() { 'T' } else { 'S' };
            let _ = writeln!(
                eda,
                "PIN {} {kind} {} {} 0 E U",
                token(&p.name),
                n(p.at.x),
                n(p.at.y)
            );
            let _ = writeln!(
                eda,
                "RC {} {} {} {}",
                n(r.min.x),
                n(r.min.y),
                n(r.width()),
                n(r.height())
            );
        }
    }
    out.insert(format!("{step}eda/data"), eda);
    out.insert(format!("{step}netlists/cadnet/netlist"), netlist);
    out
}

fn fmt_angle(a: f64) -> String {
    let s = format!("{:.3}", if a >= 359.9995 { 0.0 } else { a });
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// A name as one ODB++ token: spaces and semicolons become underscores.
fn token(s: &str) -> String {
    let t: String = s
        .chars()
        .map(|c| {
            if c.is_whitespace() || c == ';' {
                '_'
            } else {
                c
            }
        })
        .collect();
    if t.is_empty() {
        "_".into()
    } else {
        t
    }
}

/// The smallest distance between two pad centres of the package (0 with fewer than two pads).
fn pitch(pkg: &crate::library::Package) -> Coord {
    let mut best: Option<f64> = None;
    for (i, a) in pkg.pads.iter().enumerate() {
        for b in &pkg.pads[i + 1..] {
            let d = a.at.dist(b.at);
            if d > 0.0 && best.is_none_or(|x| d < x) {
                best = Some(d);
            }
        }
    }
    best.unwrap_or(0.0).round() as Coord
}
