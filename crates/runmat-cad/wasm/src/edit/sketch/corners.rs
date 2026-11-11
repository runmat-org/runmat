//! Sketch fillet and chamfer at the corners where two lines meet.
//!
//! A corner is picked as a point (`sketch.s1.l1.end`), or as the two lines that meet there.
//! Each line is shortened back from the corner, and a tangent arc (fillet) or a straight
//! bevel (chamfer) joins the two new ends; the new curve takes its ends from the lines
//! (`start = l1.end`). A fillet adds tangent relations to both lines and its radius
//! dimension. The shortened lines drop their size dimensions and point relations, as Trim
//! does.
//!
//! A chamfer takes its distance along the first line and its angle from that line; at 45° on
//! a right-angle corner both legs are the distance.

use super::modify::{
    at, detach, drop_size_constraints, fresh_like, point_expr, position, set_end, P,
};
use super::{constraint, model, picks, target, EditFailure, K};
use runmat_cad::ids::SketchEntityId;
use runmat_cad::model::sketch::{
    PointExpr, PointRole, Sketch, SketchEntity, SketchGeometry as G, SketchTarget,
};
use serde_json::{Map, Value};

const TOUCH: f64 = 1e-6;

/// A line end at a corner: the line, which end, and the unit direction away from the corner
/// along the line, with the line's length.
struct Leg {
    line: SketchEntityId,
    role: PointRole,
    dir: P,
    len: f64,
}

fn leg(sketch: &Sketch, e: &SketchEntity, corner: P) -> Option<Leg> {
    let G::Line { start, end } = &e.geometry else {
        return None;
    };
    let (s, t) = (position(sketch, start, 0)?, position(sketch, end, 0)?);
    let near = |q: P| (q.0 - corner.0).hypot(q.1 - corner.1) < TOUCH;
    let (role, from, to) = if near(s) {
        (PointRole::Start, s, t)
    } else if near(t) {
        (PointRole::End, t, s)
    } else {
        return None;
    };
    let len = (to.0 - from.0).hypot(to.1 - from.1);
    (len > TOUCH).then(|| Leg {
        line: e.id.clone(),
        role,
        dir: ((to.0 - from.0) / len, (to.1 - from.1) / len),
        len,
    })
}

/// The two profile lines meeting at `corner`.
fn legs_at(sketch: &Sketch, corner: P) -> Result<(Leg, Leg), EditFailure> {
    let mut found: Vec<Leg> = sketch
        .entities
        .iter()
        .filter(|e| !e.construction)
        .filter_map(|e| leg(sketch, e, corner))
        .collect();
    if found.len() != 2 {
        return Err(model(format!(
            "a corner joins two lines; {} meet at ({:.3}, {:.3})",
            found.len(),
            corner.0,
            corner.1
        )));
    }
    let b = found.pop().expect("two");
    let a = found.pop().expect("two");
    Ok((a, b))
}

/// The corners the picks name: a point each, or a pair of lines.
fn corners(
    values: &Map<String, Value>,
    sketch: &Sketch,
    sketch_id: &str,
) -> Result<Vec<P>, EditFailure> {
    let mut out: Vec<P> = Vec::new();
    let mut lines: Vec<SketchEntityId> = Vec::new();
    for pick in picks(values, "corners") {
        match target(&pick, sketch_id)? {
            SketchTarget::Point { entity, role } => {
                let e = sketch
                    .entities
                    .iter()
                    .find(|e| e.id == entity)
                    .ok_or_else(|| model(format!("the sketch has no entity `{}`", entity.0)))?;
                let p = point_expr(&e.geometry, role)
                    .and_then(|p| position(sketch, p, 0))
                    .ok_or_else(|| model(format!("`{pick}` has no position yet")))?;
                out.push(p);
            }
            SketchTarget::Entity { entity } => lines.push(entity),
        }
    }
    for pair in lines.chunks(2) {
        let [a, b] = pair else {
            return Err(model("pick lines in pairs, or pick the corner point"));
        };
        let ends = |id: &SketchEntityId| -> Vec<P> {
            sketch
                .entities
                .iter()
                .find(|e| e.id == *id)
                .map_or(Vec::new(), |e| match &e.geometry {
                    G::Line { start, end } => [start, end]
                        .into_iter()
                        .filter_map(|p| position(sketch, p, 0))
                        .collect(),
                    _ => Vec::new(),
                })
        };
        let shared = ends(a)
            .into_iter()
            .find(|p| ends(b).iter().any(|q| (q.0 - p.0).hypot(q.1 - p.1) < TOUCH));
        out.push(
            shared
                .ok_or_else(|| model(format!("`{}` and `{}` don't meet at a corner", a.0, b.0)))?,
        );
    }
    if out.is_empty() {
        return Err(model("pick the corners to change"));
    }
    Ok(out)
}

fn num(values: &Map<String, Value>, key: &str, default: f64) -> f64 {
    values.get(key).and_then(super::num).unwrap_or(default)
}

/// Shortens `leg` to `p`, after writing out the points others took from it.
fn shorten(sketch: &mut Sketch, leg: &Leg, p: P) {
    detach(sketch, &leg.line);
    if let Some(e) = sketch.entities.iter_mut().find(|e| e.id == leg.line) {
        set_end(&mut e.geometry, leg.role, at(p));
    }
    drop_size_constraints(sketch, &leg.line, true);
}

fn end_of(leg: &Leg) -> PointExpr {
    PointExpr::of(&leg.line.0, leg.role)
}

/// `sketch.fillet`: a tangent arc of the radius at each picked corner.
pub(super) fn fillet(
    values: &Map<String, Value>,
    sketch: &mut Sketch,
    sketch_id: &str,
) -> Result<String, EditFailure> {
    let r = num(values, "radius", 2.0);
    if r <= 0.0 {
        return Err(model("the fillet radius must be above zero"));
    }
    let mut added = Vec::new();
    for corner in corners(values, sketch, sketch_id)? {
        let (a, b) = legs_at(sketch, corner)?;
        let cos = (a.dir.0 * b.dir.0 + a.dir.1 * b.dir.1).clamp(-1.0, 1.0);
        let half = cos.acos() / 2.0;
        if half < 1e-6 || (std::f64::consts::FRAC_PI_2 - half) < 1e-6 {
            return Err(model(format!(
                "the lines at ({:.3}, {:.3}) are in line; there's no corner to round",
                corner.0, corner.1
            )));
        }
        let d = r / half.tan();
        if d >= a.len || d >= b.len {
            return Err(model(format!(
                "a {r} mm fillet needs {d:.3} mm of each line; `{}` or `{}` is shorter",
                a.line.0, b.line.0
            )));
        }
        let ta = (corner.0 + a.dir.0 * d, corner.1 + a.dir.1 * d);
        let tb = (corner.0 + b.dir.0 * d, corner.1 + b.dir.1 * d);
        let bis = ((a.dir.0 + b.dir.0), (a.dir.1 + b.dir.1));
        let bl = bis.0.hypot(bis.1);
        let c = (
            corner.0 + bis.0 / bl * r / half.sin(),
            corner.1 + bis.1 / bl * r / half.sin(),
        );
        shorten(sketch, &a, ta);
        shorten(sketch, &b, tb);
        // Counter-clockwise from start to end about the centre: the short way round.
        let cross = (ta.0 - c.0) * (tb.1 - c.1) - (ta.1 - c.1) * (tb.0 - c.0);
        let (start, end) = if cross > 0.0 {
            (end_of(&a), end_of(&b))
        } else {
            (end_of(&b), end_of(&a))
        };
        let id = fresh_like(sketch, "a");
        sketch.entities.push(SketchEntity {
            id: id.clone(),
            geometry: G::Arc {
                center: at(c),
                start,
                end,
            },
            construction: false,
        });
        for line in [&a.line, &b.line] {
            sketch.constraints.push(constraint(K::Tangent {
                a: SketchTarget::Entity { entity: id.clone() },
                b: SketchTarget::Entity {
                    entity: line.clone(),
                },
            }));
        }
        sketch.constraints.push(constraint(K::Radius {
            curve: SketchTarget::Entity { entity: id.clone() },
            value: super::mm(r),
        }));
        added.push(id.0);
    }
    Ok(format!(
        "Rounded {} corner(s) at {r} mm; added {}",
        added.len(),
        added.join(", ")
    ))
}

/// `sketch.chamfer`: a bevel at each picked corner, by distance and angle.
pub(super) fn chamfer(
    values: &Map<String, Value>,
    sketch: &mut Sketch,
    sketch_id: &str,
) -> Result<String, EditFailure> {
    let d1 = num(values, "distance", 1.0);
    let alpha = num(values, "angle", 45.0).to_radians();
    if d1 <= 0.0 || alpha <= 0.0 || alpha >= std::f64::consts::PI {
        return Err(model(
            "a chamfer takes a distance above zero and an angle between 0° and 180°",
        ));
    }
    let mut added = Vec::new();
    for corner in corners(values, sketch, sketch_id)? {
        let (a, b) = legs_at(sketch, corner)?;
        let phi = (a.dir.0 * b.dir.0 + a.dir.1 * b.dir.1)
            .clamp(-1.0, 1.0)
            .acos();
        let rest = std::f64::consts::PI - alpha - phi;
        if rest <= 1e-9 {
            return Err(model(format!(
                "a {}° chamfer doesn't reach the other line at that corner",
                alpha.to_degrees()
            )));
        }
        let d2 = d1 * alpha.sin() / rest.sin();
        if d1 >= a.len || d2 >= b.len {
            return Err(model(format!(
                "the chamfer needs {d1:.3} mm of `{}` and {d2:.3} mm of `{}`",
                a.line.0, b.line.0
            )));
        }
        shorten(
            sketch,
            &a,
            (corner.0 + a.dir.0 * d1, corner.1 + a.dir.1 * d1),
        );
        shorten(
            sketch,
            &b,
            (corner.0 + b.dir.0 * d2, corner.1 + b.dir.1 * d2),
        );
        let id = fresh_like(sketch, "l");
        sketch.entities.push(SketchEntity {
            id: id.clone(),
            geometry: G::Line {
                start: end_of(&a),
                end: end_of(&b),
            },
            construction: false,
        });
        added.push(id.0);
    }
    Ok(format!(
        "Chamfered {} corner(s); added {}",
        added.len(),
        added.join(", ")
    ))
}
