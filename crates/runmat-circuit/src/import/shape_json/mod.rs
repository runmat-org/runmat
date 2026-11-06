//! Shape-string JSON documents (standard edition), from the format's public description: a
//! schematic (`docType` 1) or board (`docType` 3) is a JSON object whose `shape` array holds one
//! string per item, fields separated by `~` (`TRACK~1~1~GND~4000 3000 4100 3000~gge5~0`). A
//! placed part (`LIB~…`) carries its own items after `#@$`, already placed on the canvas; a
//! schematic pin (`P~…`) carries its parts after `^^`.
//!
//! - Canvas units are 10 mil (0.254 mm), with y growing down; the import flips y and moves the
//!   board's outline (or the schematic's items) to start at the origin.
//! - The import writes `<stem>.circuit` (a `part` declaration per part number, pins numbered
//!   as their pads), the board's footprints as `<name>-footprints/<package>.kicad_mod`, and with
//!   a board `<stem>.circuit.layout` using the circuit.

pub mod board;
pub mod schematic;

use super::{Imported, Report};
use crate::geom::{mm, Coord, Point};
use crate::runtime::{CIRCUIT_SUFFIX, LAYOUT_SUFFIX};
use serde_json::Value;
use std::collections::BTreeMap;

/// The format's name in reports.
pub const FORMAT: &str = "shape-string JSON";

/// One canvas unit (10 mil) in millimetres.
pub const UNIT_MM: f64 = 0.254;

/// A length in canvas units, in board units.
pub fn length(v: f64) -> Coord {
    mm(v * UNIT_MM)
}

/// Canvas coordinates to RunMat ones: y flipped, relative to `origin` (canvas units).
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub origin: (f64, f64),
}

impl Frame {
    pub fn point(&self, x: f64, y: f64) -> Point {
        Point::new(length(x - self.origin.0), length(self.origin.1 - y))
    }
    /// The frame whose origin is the lower-left corner of `points` (canvas units).
    pub fn around(points: impl IntoIterator<Item = (f64, f64)>) -> Frame {
        let (mut x0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY);
        for (x, y) in points {
            x0 = x0.min(x);
            y1 = y1.max(y);
        }
        Frame { origin: if x0.is_finite() { (x0, y1) } else { (0.0, 0.0) } }
    }
}

/// A shape string's fields.
pub struct Fields<'a>(pub Vec<&'a str>);

impl<'a> Fields<'a> {
    pub fn of(s: &'a str) -> Self {
        Fields(s.split('~').collect())
    }
    pub fn kind(&self) -> &'a str {
        self.0.first().copied().unwrap_or("")
    }
    pub fn str(&self, i: usize) -> &'a str {
        self.0.get(i).copied().unwrap_or("")
    }
    pub fn num(&self, i: usize) -> f64 {
        self.str(i).trim().parse().unwrap_or(0.0)
    }
}

/// Space- or comma-separated numbers (`4000 3000 4100 3000`) as pairs.
pub fn pairs(s: &str) -> Vec<(f64, f64)> {
    let n: Vec<f64> = s.split(|c: char| c == ' ' || c == ',').filter(|t| !t.is_empty()).filter_map(|t| t.parse().ok()).collect();
    n.chunks_exact(2).map(|c| (c[0], c[1])).collect()
}

/// One step of an SVG path, absolute coordinates in canvas units.
#[derive(Clone, Debug, PartialEq)]
pub enum PathStep {
    Move(f64, f64),
    Line(f64, f64),
    /// Radius, large-arc and sweep flags, end point.
    Arc { r: f64, large: bool, sweep: bool, to: (f64, f64) },
    Close,
}

/// The `M`, `L`, `A` and `Z` steps of an SVG path (what the format's areas and arcs use).
/// Returns `None` for a relative or curve command, which the import reports.
pub fn path(d: &str) -> Option<Vec<PathStep>> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for c in d.chars() {
        if c.is_ascii_alphabetic() {
            if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
            tokens.push(c.to_string());
        } else if c == ' ' || c == ',' {
            if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    let mut out = Vec::new();
    let mut i = 0;
    let mut cmd = "";
    let num = |t: Option<&String>| t.and_then(|t| t.parse::<f64>().ok());
    while i < tokens.len() {
        if tokens[i].chars().all(|c| c.is_ascii_alphabetic()) {
            cmd = tokens[i].as_str();
            i += 1;
            if cmd == "Z" || cmd == "z" {
                out.push(PathStep::Close);
            }
            continue;
        }
        match cmd {
            "M" | "L" => {
                let (x, y) = (num(tokens.get(i))?, num(tokens.get(i + 1))?);
                out.push(if cmd == "M" { PathStep::Move(x, y) } else { PathStep::Line(x, y) });
                // Coordinates after a move continue as lines.
                cmd = "L";
                i += 2;
            }
            "A" => {
                let r = num(tokens.get(i))?;
                let (large, sweep) = (num(tokens.get(i + 3))? != 0.0, num(tokens.get(i + 4))? != 0.0);
                let to = (num(tokens.get(i + 5))?, num(tokens.get(i + 6))?);
                out.push(PathStep::Arc { r, large, sweep, to });
                i += 7;
            }
            _ => return None,
        }
    }
    Some(out)
}

/// The mid point of an SVG arc from `a` to `b` (canvas units), or `None` when it has no radius.
pub fn arc_mid(a: (f64, f64), b: (f64, f64), r: f64, large: bool, sweep: bool) -> Option<(f64, f64)> {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let d = dx.hypot(dy);
    if d < 1e-12 || r <= 0.0 {
        return None;
    }
    let h = (r * r - d * d / 4.0).max(0.0).sqrt();
    let r = r.max(d / 2.0);
    let m = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
    let n = (-dy / d, dx / d);
    for s in [1.0, -1.0] {
        let c = (m.0 + s * h * n.0, m.1 + s * h * n.1);
        let (a0, a1) = ((a.1 - c.1).atan2(a.0 - c.0), (b.1 - c.1).atan2(b.0 - c.0));
        let ccw = (a1 - a0).rem_euclid(std::f64::consts::TAU);
        let span = if sweep { ccw } else { std::f64::consts::TAU - ccw };
        if (span > std::f64::consts::PI) == large || h < 1e-9 {
            let mid = a0 + if sweep { span / 2.0 } else { -span / 2.0 };
            return Some((c.0 + r * mid.cos(), c.1 + r * mid.sin()));
        }
    }
    None
}

/// A placed part's attributes (`package`0603`Value`10k`): key → value.
pub fn attributes(s: &str) -> BTreeMap<String, String> {
    let parts: Vec<&str> = s.split('`').collect();
    parts.chunks(2).filter(|c| c.len() == 2 && !c[0].is_empty()).map(|c| (c[0].to_owned(), c[1].to_owned())).collect()
}

/// A document: its type and shape strings.
pub struct Document {
    pub doc_type: String,
    pub shapes: Vec<String>,
    pub json: Value,
}

/// Reads `text` as a shape-string JSON document.
pub fn document(name: &str, text: &str) -> Result<Document, String> {
    let json: Value = serde_json::from_str(text).map_err(|e| format!("{name} isn't JSON: {e}"))?;
    let doc_type = match &json["head"]["docType"] {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => return Err(format!("{name} has no head.docType, so it isn't a shape-string document")),
    };
    let shapes = json["shape"]
        .as_array()
        .ok_or_else(|| format!("{name} has no shape array"))?
        .iter()
        .filter_map(|s| s.as_str().map(str::to_owned))
        .collect();
    Ok(Document { doc_type, shapes, json })
}

/// Imports the schematic and board documents of `files` (file name → JSON text) to
/// `<target_stem>.circuit` and `.circuit.layout`.
pub fn import(files: &BTreeMap<String, String>, target_stem: &str) -> Result<Imported, String> {
    let mut sch = None;
    let mut brd = None;
    for (name, text) in files {
        let doc = document(name, text)?;
        match doc.doc_type.as_str() {
            "1" => sch = Some(doc),
            "3" => brd = Some(doc),
            other => return Err(format!("{name} is a document of type {other}; give a schematic (1) or a board (3)")),
        }
    }
    if sch.is_none() && brd.is_none() {
        return Err("give a schematic or board document".into());
    }
    let name = target_stem.rsplit('/').next().unwrap_or(target_stem).to_owned();
    let folder = target_stem.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();
    let footprint_dir = format!("{name}-footprints");
    let mut report = Report::new(FORMAT);
    let mut out = BTreeMap::new();
    let board = match &brd {
        Some(doc) => Some(board::read(doc, &name, &mut report)?),
        None => None,
    };
    if let Some(doc) = &sch {
        let (packages, footprints) = board.as_ref().map(|b| (b.packages.clone(), b.footprints.clone())).unwrap_or_default();
        let design = schematic::read(doc, &name, &footprint_dir, &packages, &footprints, &mut report)?;
        out.insert(format!("{target_stem}{CIRCUIT_SUFFIX}"), crate::syntax::print_circuit(&design).map_err(|e| format!("{e:?}"))?);
    } else {
        report.not_carried("the schematic: only the board was given, so the circuit has no parts");
    }
    if let Some(b) = board {
        let uses = [runmat_design_syntax::document::UseItem {
            symbol: Some(("circuit".into(), name.clone())),
            from: String::new(),
            alias: None,
            revision: None,
            span: Default::default(),
        }];
        out.insert(format!("{target_stem}{LAYOUT_SUFFIX}"), crate::syntax::board::print_board(&b.board, &uses));
        for (key, pkg) in &b.packages {
            out.insert(format!("{folder}{footprint_dir}/{key}.kicad_mod"), crate::kicad::footprint::write_mod(pkg));
        }
    }
    Ok(Imported { files: out, report })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_and_arcs_read_as_the_format_writes_them() {
        let steps = path("M4000,3000 L 4100 3000 4100 3100Z").unwrap();
        assert_eq!(steps, [PathStep::Move(4000.0, 3000.0), PathStep::Line(4100.0, 3000.0), PathStep::Line(4100.0, 3100.0), PathStep::Close]);
        // A half circle of radius 10 from (0,0) to (20,0). The positive sweep turns clockwise
        // on the y-down canvas, through (10,-10); the other way passes (10,10).
        let m = arc_mid((0.0, 0.0), (20.0, 0.0), 10.0, false, true).unwrap();
        assert!((m.0 - 10.0).abs() < 1e-9 && (m.1 + 10.0).abs() < 1e-9, "{m:?}");
        let m = arc_mid((0.0, 0.0), (20.0, 0.0), 10.0, false, false).unwrap();
        assert!((m.1 - 10.0).abs() < 1e-9, "{m:?}");
        assert!(path("M0 0 c 1 1 2 2 3 3").is_none());
        assert_eq!(attributes("package`0603`Value`10k`")["Value"], "10k");
    }
}
