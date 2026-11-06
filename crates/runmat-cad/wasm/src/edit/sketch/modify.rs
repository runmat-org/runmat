//! Trim and Extend from the canvas, each taking the picked
//! curve and the click on it (`at`, sketch millimetres).
//!
//! - **Trim** removes the piece of the curve between the intersections either side of the
//!   click. A line or arc with no intersection inside it is deleted; a piece at one end moves
//!   that end to the intersection; a piece in the middle splits the curve in two. A circle
//!   needs two intersections and becomes the arc that remains.
//! - **Extend** moves the line end nearer the click along the line to the first curve it
//!   meets.
//!
//! The moved ends are written as coordinates. Relations and dimensions that named the removed
//! piece are deleted with it; a point other curves shared stays where it
//! was, written out for them.

use super::{coords, mm, model, picks, target, EditFailure};
use runmat_cad::ids::SketchEntityId;
use runmat_cad::model::sketch::{
    PointExpr, PointRole, Sketch, SketchEntity, SketchGeometry as G, SketchTarget,
};
use serde_json::{Map, Value};
use std::f64::consts::TAU;

pub(super) type P = (f64, f64);

/// Where a point sits, in millimetres: written coordinates, followed through shared points.
pub(super) fn position(sketch: &Sketch, p: &PointExpr, depth: u32) -> Option<P> {
    match p {
        PointExpr::Coords { .. } => coords(p),
        PointExpr::Of { entity, role } if depth < 64 => {
            let e = sketch.entities.iter().find(|e| e.id == *entity)?;
            position(sketch, point_expr(&e.geometry, *role)?, depth + 1)
        }
        PointExpr::Of { .. } => None,
    }
}

pub(super) fn point_expr(g: &G, role: PointRole) -> Option<&PointExpr> {
    match (g, role) {
        (G::Point { at }, PointRole::At) => Some(at),
        (G::Line { start, .. } | G::Arc { start, .. }, PointRole::Start) => Some(start),
        (G::Line { end, .. } | G::Arc { end, .. }, PointRole::End) => Some(end),
        (G::Arc { center, .. } | G::Circle { center, .. }, PointRole::Center) => Some(center),
        (G::Spline { .. }, role) => g.point(role),
        _ => None,
    }
}

/// A curve's resolved shape, in millimetres.
#[derive(Clone, Copy, Debug)]
pub(super) enum Shape {
    Segment(P, P),
    /// Centre, radius, start angle, counter-clockwise sweep (TAU for a circle).
    Round(P, f64, f64, f64),
}

pub(super) fn shape(sketch: &Sketch, e: &SketchEntity) -> Option<Shape> {
    let at = |p: &PointExpr| position(sketch, p, 0);
    Some(match &e.geometry {
        G::Line { start, end } => Shape::Segment(at(start)?, at(end)?),
        G::Arc { center, start, end } => {
            let (c, a, b) = (at(center)?, at(start)?, at(end)?);
            let r = (a.0 - c.0).hypot(a.1 - c.1);
            let t0 = (a.1 - c.1).atan2(a.0 - c.0);
            let t1 = (b.1 - c.1).atan2(b.0 - c.0);
            Shape::Round(c, r, t0, (t1 - t0).rem_euclid(TAU))
        }
        G::Circle { center, radius } => {
            let r = match radius {
                runmat_cad::value::ValueExpr::Quantity { value, .. } => *value,
                _ => return None,
            };
            Shape::Round(at(center)?, r, 0.0, TAU)
        }
        G::Point { .. } | G::Spline { .. } => return None,
    })
}

/// The points where two shapes cross (both as bounded curves), in millimetres.
fn crossings(a: Shape, b: Shape) -> Vec<P> {
    let on_round = |c: P, t0: f64, sweep: f64, p: P| sweep >= TAU - 1e-9 || ((p.1 - c.1).atan2(p.0 - c.0) - t0).rem_euclid(TAU) <= sweep + 1e-9;
    let on_segment = |a: P, b: P, p: P| {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let l2 = dx * dx + dy * dy;
        let t = ((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2.max(1e-18);
        (-1e-9..=1.0 + 1e-9).contains(&t)
    };
    match (a, b) {
        (Shape::Segment(p, q), Shape::Segment(r, s)) => {
            let d = (q.0 - p.0) * (s.1 - r.1) - (q.1 - p.1) * (s.0 - r.0);
            if d.abs() < 1e-12 {
                return Vec::new();
            }
            let t = ((r.0 - p.0) * (s.1 - r.1) - (r.1 - p.1) * (s.0 - r.0)) / d;
            let u = ((r.0 - p.0) * (q.1 - p.1) - (r.1 - p.1) * (q.0 - p.0)) / d;
            if (-1e-9..=1.0 + 1e-9).contains(&t) && (-1e-9..=1.0 + 1e-9).contains(&u) {
                vec![(p.0 + t * (q.0 - p.0), p.1 + t * (q.1 - p.1))]
            } else {
                Vec::new()
            }
        }
        (Shape::Segment(p, q), Shape::Round(c, r, t0, sw)) | (Shape::Round(c, r, t0, sw), Shape::Segment(p, q)) => {
            let (dx, dy) = (q.0 - p.0, q.1 - p.1);
            let (fx, fy) = (p.0 - c.0, p.1 - c.1);
            let (qa, qb, qc) = (dx * dx + dy * dy, 2.0 * (fx * dx + fy * dy), fx * fx + fy * fy - r * r);
            let disc = qb * qb - 4.0 * qa * qc;
            if disc < 0.0 || qa < 1e-18 {
                return Vec::new();
            }
            let root = disc.sqrt();
            [(-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa)]
                .into_iter()
                .filter(|t| (-1e-9..=1.0 + 1e-9).contains(t))
                .map(|t| (p.0 + t * dx, p.1 + t * dy))
                .filter(|pt| on_round(c, t0, sw, *pt) && on_segment(p, q, *pt))
                .collect()
        }
        (Shape::Round(c0, r0, a0, s0), Shape::Round(c1, r1, a1, s1)) => {
            let d = (c1.0 - c0.0).hypot(c1.1 - c0.1);
            if d < 1e-12 || d > r0 + r1 || d < (r0 - r1).abs() {
                return Vec::new();
            }
            let a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d);
            let h = (r0 * r0 - a * a).max(0.0).sqrt();
            let m = (c0.0 + a * (c1.0 - c0.0) / d, c0.1 + a * (c1.1 - c0.1) / d);
            let (ox, oy) = (h * (c1.1 - c0.1) / d, h * (c1.0 - c0.0) / d);
            [(m.0 + ox, m.1 - oy), (m.0 - ox, m.1 + oy)]
                .into_iter()
                .filter(|p| on_round(c0, a0, s0, *p) && on_round(c1, a1, s1, *p))
                .collect()
        }
    }
}

/// A point's parameter along a shape: 0..1 along a segment, 0..sweep (radians) round an arc.
pub(super) fn param(s: Shape, p: P) -> f64 {
    match s {
        Shape::Segment(a, b) => {
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            ((p.0 - a.0) * dx + (p.1 - a.1) * dy) / (dx * dx + dy * dy).max(1e-18)
        }
        Shape::Round(c, _, t0, _) => ((p.1 - c.1).atan2(p.0 - c.0) - t0).rem_euclid(TAU),
    }
}

fn picked(values: &Map<String, Value>, sketch_id: &str) -> Result<SketchEntityId, EditFailure> {
    let pick = picks(values, "entities")
        .into_iter()
        .next()
        .or_else(|| values.get("entity").and_then(Value::as_str).map(str::to_owned))
        .ok_or_else(|| model("pick the curve on the sketch"))?;
    match target(&pick, sketch_id)? {
        SketchTarget::Entity { entity } => Ok(entity),
        SketchTarget::Point { entity, .. } => Ok(entity),
    }
}

fn click(values: &Map<String, Value>) -> Result<P, EditFailure> {
    super::xy(values, "at").ok_or_else(|| model("click the piece of the curve to change"))
}

/// Every other curve's crossings with `id`'s shape.
fn cuts(sketch: &Sketch, id: &SketchEntityId, own: Shape) -> Vec<P> {
    sketch
        .entities
        .iter()
        .filter(|e| e.id != *id && !e.construction)
        .filter_map(|e| shape(sketch, e))
        .flat_map(|s| crossings(own, s))
        .collect()
}

/// Writes out, as coordinates, every point other curves took from `id` (`start = l1.end`), so
/// they stay put when `id` changes or goes.
pub(super) fn detach(sketch: &mut Sketch, id: &SketchEntityId) {
    let snapshot = sketch.clone();
    for e in sketch.entities.iter_mut().filter(|e| e.id != *id) {
        let fix = |p: &mut PointExpr| {
            if matches!(p, PointExpr::Of { entity, .. } if entity == id) {
                if let Some((x, y)) = position(&snapshot, p, 0) {
                    *p = PointExpr::Coords { x: mm(x), y: mm(y) };
                }
            }
        };
        match &mut e.geometry {
            G::Point { at } => fix(at),
            G::Line { start, end } => {
                fix(start);
                fix(end);
            }
            G::Arc { center, start, end } => {
                fix(center);
                fix(start);
                fix(end);
            }
            G::Circle { center, .. } => fix(center),
            G::Spline { points, .. } => points.iter_mut().for_each(fix),
        }
    }
}

/// Drops relations and dimensions that name `id` (its curve or its points).
pub(super) fn drop_constraints_on(sketch: &mut Sketch, id: &SketchEntityId) {
    let needle = format!("\"entity\":\"{}\"", id.0);
    sketch.constraints.retain(|c| !serde_json::to_string(c).unwrap_or_default().contains(&needle));
}

pub(super) fn at(p: P) -> PointExpr {
    PointExpr::Coords { x: mm(p.0), y: mm(p.1) }
}

pub(super) fn fresh_like(sketch: &Sketch, stem: &str) -> SketchEntityId {
    SketchEntityId::new(super::fresh(sketch, &[], stem))
}

/// `sketch.trim`: the piece of the picked curve around the click goes. Returns a summary.
pub(super) fn trim(values: &Map<String, Value>, sketch: &mut Sketch, sketch_id: &str) -> Result<String, EditFailure> {
    let id = picked(values, sketch_id)?;
    let p = click(values)?;
    let entity = sketch.entities.iter().find(|e| e.id == id).cloned().ok_or_else(|| model(format!("the sketch has no curve `{}`", id.0)))?;
    let own = shape(sketch, &entity).ok_or_else(|| model(format!("`{}` has no position to trim yet", id.0)))?;
    let (span, closed) = match own {
        Shape::Segment(..) => (1.0, false),
        Shape::Round(_, _, _, sw) => (sw, sw >= TAU - 1e-9),
    };
    let mut ts: Vec<f64> = cuts(sketch, &id, own).into_iter().map(|q| param(own, q)).filter(|t| *t > 1e-6 && *t < span - 1e-6).collect();
    ts.sort_by(f64::total_cmp);
    ts.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let t = param(own, p).clamp(0.0, span);
    detach(sketch, &id);
    let point_at = |t: f64| match own {
        Shape::Segment(a, b) => (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)),
        Shape::Round(c, r, t0, _) => (c.0 + r * (t0 + t).cos(), c.1 + r * (t0 + t).sin()),
    };
    let index = sketch.entities.iter().position(|e| e.id == id).expect("found above");
    if closed {
        // A circle keeps the arc from the crossing after the click round to the one before it.
        if ts.len() < 2 {
            return Err(model(format!("`{}` needs two crossings to trim; delete it instead", id.0)));
        }
        let after = ts.iter().copied().find(|x| *x > t).unwrap_or(ts[0]);
        let before = ts.iter().rev().copied().find(|x| *x < t).unwrap_or(*ts.last().expect("two"));
        let Shape::Round(c, ..) = own else { unreachable!() };
        drop_constraints_on(sketch, &id);
        sketch.entities[index].geometry = G::Arc { center: at(c), start: at(point_at(after)), end: at(point_at(before)) };
        return Ok(format!("Trimmed {} to an arc", id.0));
    }
    let lo = ts.iter().rev().copied().find(|x| *x < t);
    let hi = ts.iter().copied().find(|x| *x > t);
    let keep_orientation = matches!(own, Shape::Segment(..));
    match (lo, hi) {
        (None, None) => {
            drop_constraints_on(sketch, &id);
            sketch.entities.remove(index);
            Ok(format!("Deleted {}", id.0))
        }
        (Some(lo), None) => {
            set_end(&mut sketch.entities[index].geometry, PointRole::End, at(point_at(lo)));
            drop_size_constraints(sketch, &id, keep_orientation);
            Ok(format!("Trimmed {}", id.0))
        }
        (None, Some(hi)) => {
            set_end(&mut sketch.entities[index].geometry, PointRole::Start, at(point_at(hi)));
            drop_size_constraints(sketch, &id, keep_orientation);
            Ok(format!("Trimmed {}", id.0))
        }
        (Some(lo), Some(hi)) => {
            // The middle goes: this curve ends at `lo`, a new one runs from `hi` to the old end.
            let mut rest = sketch.entities[index].clone();
            let stem = if keep_orientation { "l" } else { "a" };
            rest.id = fresh_like(sketch, stem);
            set_end(&mut rest.geometry, PointRole::Start, at(point_at(hi)));
            if let Some(end) = point_expr(&entity.geometry, PointRole::End) {
                if let Some(q) = position(sketch, end, 0) {
                    set_end(&mut rest.geometry, PointRole::End, at(q));
                }
            }
            set_end(&mut sketch.entities[index].geometry, PointRole::End, at(point_at(lo)));
            drop_size_constraints(sketch, &id, keep_orientation);
            let new_id = rest.id.0.clone();
            sketch.entities.insert(index + 1, rest);
            Ok(format!("Split {} and trimmed it; added {new_id}", id.0))
        }
    }
}

/// Relations that a shorter curve can't keep: its dimensions and point relations (a line's
/// horizontal, vertical, parallel and perpendicular relations stay).
pub(super) fn drop_size_constraints(sketch: &mut Sketch, id: &SketchEntityId, keep_orientation: bool) {
    let needle = format!("\"entity\":\"{}\"", id.0);
    sketch.constraints.retain(|c| {
        let text = serde_json::to_string(c).unwrap_or_default();
        if !text.contains(&needle) {
            return true;
        }
        keep_orientation && ["\"horizontal\"", "\"vertical\"", "\"parallel\"", "\"perpendicular\""].iter().any(|k| text.contains(k))
    });
}

pub(super) fn set_end(g: &mut G, role: PointRole, p: PointExpr) {
    match (g, role) {
        (G::Line { start, .. } | G::Arc { start, .. }, PointRole::Start) => *start = p,
        (G::Line { end, .. } | G::Arc { end, .. }, PointRole::End) => *end = p,
        _ => {}
    }
}

/// `sketch.extend`: the line end nearer the click runs on to the first curve it meets.
pub(super) fn extend(values: &Map<String, Value>, sketch: &mut Sketch, sketch_id: &str) -> Result<String, EditFailure> {
    let id = picked(values, sketch_id)?;
    let p = click(values)?;
    let entity = sketch.entities.iter().find(|e| e.id == id).cloned().ok_or_else(|| model(format!("the sketch has no curve `{}`", id.0)))?;
    let Some(Shape::Segment(a, b)) = shape(sketch, &entity) else {
        return Err(model(format!("extend works on lines; `{}` isn't one", id.0)));
    };
    // From the end nearer the click, outward: a long ray as a segment.
    let (from, toward, role) = if (p.0 - a.0).hypot(p.1 - a.1) <= (p.0 - b.0).hypot(p.1 - b.1) { (a, b, PointRole::Start) } else { (b, a, PointRole::End) };
    let len = (from.0 - toward.0).hypot(from.1 - toward.1).max(1e-9);
    let dir = ((from.0 - toward.0) / len, (from.1 - toward.1) / len);
    let reach = 1.0e6;
    let ray = Shape::Segment((from.0 + dir.0 * 1e-6, from.1 + dir.1 * 1e-6), (from.0 + dir.0 * reach, from.1 + dir.1 * reach));
    let hit = cuts(sketch, &id, ray)
        .into_iter()
        .min_by(|x, y| (x.0 - from.0).hypot(x.1 - from.1).total_cmp(&(y.0 - from.0).hypot(y.1 - from.1)))
        .ok_or_else(|| model(format!("`{}` meets nothing beyond that end", id.0)))?;
    detach(sketch, &id);
    let index = sketch.entities.iter().position(|e| e.id == id).expect("found above");
    set_end(&mut sketch.entities[index].geometry, role, at(hit));
    drop_size_constraints(sketch, &id, true);
    Ok(format!("Extended {}", id.0))
}
