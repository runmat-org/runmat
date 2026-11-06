//! KiCad interchange, written independently from KiCad's documented behavior: the published
//! file-format documentation (dev-docs.kicad.org, "File Formats"), the user manuals, and the
//! outputs KiCad writes for sample files (netlists and boards). No KiCad source code is
//! included.
//!
//! - [`sexpr`]: lossless S-expressions (atoms keep their source text).
//! - [`pcb`]: `.kicad_pcb` ⇄ [`crate::layout::Board`], with edit-in-place writing: unchanged
//!   items are written back verbatim, edited items keep every child token we do not model,
//!   deleted items are dropped and new items appended, so a read–write cycle is lossless.
//! - [`footprint`]: footprints (`.kicad_mod` and footprints embedded in boards) ⇄
//!   [`crate::library::Package`].
//! - [`sch`]: `.kicad_sch` hierarchies ⇄ [`crate::ir::Design`] (symbols, wires, junctions,
//!   labels, global and hierarchical labels, power symbols, buses, sheets and sheet pins,
//!   no-connects) and `.kicad_sym` libraries ⇄ pool units and entities.
//! - [`sch_export`]: a fresh, automatically placed `.kicad_sch` hierarchy from an elaborated
//!   design (pages by hierarchy, components ordered by connectivity, labels for nets).
//! - [`dru`]: `.kicad_dru` custom rules ⇄ [`crate::board_rules::RuleSpec`].
//! - [`project`]: `.kicad_pro` net classes (patterns and assignments) ⇄ design net classes
//!   and board rules, edited losslessly.
//!
//! Coordinates: KiCad files are millimetres with y pointing down; RunMat is nanometres with y
//! up. Angles keep their value (counter-clockwise as seen on screen in both).

pub mod dru;
pub mod footprint;
pub mod pcb;
pub mod project;
pub mod sch;
mod sch_connect;
pub mod sch_export;
pub mod sexpr;

use crate::geom::{Coord, Point};
use sexpr::SExpr;

/// KiCad millimetres (y down) to a RunMat point.
pub fn pt(x: f64, y: f64) -> Point {
    Point::new((x * 1e6).round() as Coord, (-y * 1e6).round() as Coord)
}

/// `(at x y [angle])`-style node to a point and angle.
pub fn at_of(node: Option<&SExpr>) -> (Point, f64) {
    match node {
        Some(n) => (
            pt(n.arg_f(0).unwrap_or(0.0), n.arg_f(1).unwrap_or(0.0)),
            n.arg_f(2).unwrap_or(0.0),
        ),
        None => (Point::ORIGIN, 0.0),
    }
}

/// Millimetre text of a length.
pub fn len(v: Coord) -> SExpr {
    SExpr::num(v as f64 * 1e-6)
}

pub fn xy(head: &str, p: Point) -> SExpr {
    SExpr::list(head, vec![len(p.x), len(-p.y)])
}

pub fn mm_of(node: Option<&SExpr>) -> Option<Coord> {
    node.and_then(|n| n.arg_f(0))
        .map(|v| (v * 1e6).round() as Coord)
}

/// Points of a `(pts (xy ..) (arc (start)(mid)(end)) ..)` list; arcs are flattened.
pub fn pts_of(node: Option<&SExpr>) -> Vec<Point> {
    let mut out = Vec::new();
    let Some(n) = node else { return out };
    for c in n.items().iter().skip(1) {
        match c.head() {
            Some("xy") => out.push(pt(c.arg_f(0).unwrap_or(0.0), c.arg_f(1).unwrap_or(0.0))),
            Some("arc") => {
                let (s, _) = at_of(c.child("start"));
                let (m, _) = at_of(c.child("mid"));
                let (e, _) = at_of(c.child("end"));
                match crate::geom::shape::arc_from_three(s, m, e) {
                    Some((ctr, sw)) => {
                        out.extend(crate::geom::shape::arc_points(
                            ctr,
                            s,
                            sw,
                            crate::geom::ARC_TOL,
                        ));
                    }
                    None => {
                        out.push(s);
                        out.push(e);
                    }
                }
            }
            _ => {}
        }
    }
    out.dedup();
    if out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    out
}

pub fn pts_node(points: &[Point]) -> SExpr {
    SExpr::list("pts", points.iter().map(|p| xy("xy", *p)).collect())
}

/// A fresh, deterministic KiCad UUID for generated items.
pub fn uuid_node(seed: &str) -> SExpr {
    let u = crate::ids::Uuid(crate::ids::fnv128(seed.as_bytes())).with_version(8);
    SExpr::list("uuid", vec![SExpr::str(&u.to_string())])
}

/// Copper layer name for index `i` of `n`.
pub fn copper_layer(i: u8, n: u8) -> String {
    crate::layout::copper_name(i, n)
}

/// Copper layer index from a KiCad name, for a board with `n` copper layers.
pub fn copper_index(name: &str, n: u8) -> Option<u8> {
    match name {
        "F.Cu" => Some(0),
        "B.Cu" => Some(n.saturating_sub(1)),
        _ => name
            .strip_prefix("In")
            .and_then(|r| r.strip_suffix(".Cu"))
            .and_then(|k| k.parse::<u8>().ok())
            .filter(|k| *k > 0 && *k + 1 < n),
    }
}
