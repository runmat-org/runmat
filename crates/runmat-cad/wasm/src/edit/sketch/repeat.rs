//! Sketch tools that copy or move picked entities, from the canvas:
//!
//! - **Construction** toggles each picked entity between construction and profile geometry.
//! - **Mirror** reflects the picked entities about a picked line. With "Keep originals" the
//!   images are new entities; without it the originals move.
//! - **Pattern** repeats the picked entities along a direction (linear: spacing and direction
//!   angle) or round a centre (circular: total angle; 360° spaces the instances evenly). The
//!   instance count includes the originals.
//! - **Transform** moves, copies, rotates (about the base point) or scales (about the base
//!   point) the picked entities.
//! - **Split** cuts a line or an arc in two at the picked point; the second piece starts where
//!   the first ends.
//!
//! New and moved points are written as coordinates (sketch millimetres). A moved entity drops
//! the relations and dimensions that named it, as Trim does; points other entities took from
//! it stay where they were.

use super::modify::{
    at, detach, drop_constraints_on, fresh_like, param, point_expr, position, set_end, shape,
    Shape, P,
};
use super::{model, picks, target, xy, EditFailure};
use runmat_cad::ids::SketchEntityId;
use runmat_cad::model::sketch::{
    PointExpr, PointRole, Sketch, SketchEntity, SketchGeometry as G, SketchTarget,
};
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

/// The picked entities (points pick their entity), in pick order, without repeats.
fn picked(
    values: &Map<String, Value>,
    key: &str,
    sketch_id: &str,
) -> Result<Vec<SketchEntityId>, EditFailure> {
    let mut out: Vec<SketchEntityId> = Vec::new();
    for pick in picks(values, key) {
        let id = match target(&pick, sketch_id)? {
            SketchTarget::Entity { entity } | SketchTarget::Point { entity, .. } => entity,
        };
        if !out.contains(&id) {
            out.push(id);
        }
    }
    Ok(out)
}

fn entity<'a>(sketch: &'a Sketch, id: &SketchEntityId) -> Result<&'a SketchEntity, EditFailure> {
    sketch
        .entities
        .iter()
        .find(|e| e.id == *id)
        .ok_or_else(|| model(format!("the sketch has no entity `{}`", id.0)))
}

fn number(values: &Map<String, Value>, key: &str) -> Option<f64> {
    super::num(values.get(key)?)
}

fn radius_mm(r: &ValueExpr) -> Option<f64> {
    match r {
        ValueExpr::Quantity { value, .. } => Some(*value),
        _ => None,
    }
}

/// A rigid or scaling map of the plane: `f` moves points, `scale` multiplies radii, and `flip`
/// says whether it reverses orientation (an arc's start and end swap so it stays
/// counter-clockwise).
struct Map2<'f> {
    f: &'f dyn Fn(P) -> P,
    scale: f64,
    flip: bool,
}

/// `e` mapped by `m`, every point written as coordinates.
fn mapped(sketch: &Sketch, e: &SketchEntity, m: &Map2<'_>) -> Result<G, EditFailure> {
    let place = |p: &PointExpr| -> Result<PointExpr, EditFailure> {
        let q = position(sketch, p, 0)
            .ok_or_else(|| model(format!("`{}` has a point with no position yet", e.id.0)))?;
        Ok(at((m.f)(q)))
    };
    Ok(match &e.geometry {
        G::Point { at: p } => G::Point { at: place(p)? },
        G::Line { start, end } => G::Line {
            start: place(start)?,
            end: place(end)?,
        },
        G::Arc { center, start, end } => {
            let (s, t) = (place(start)?, place(end)?);
            let (start, end) = if m.flip { (t, s) } else { (s, t) };
            G::Arc {
                center: place(center)?,
                start,
                end,
            }
        }
        G::Circle { center, radius } => {
            let r = radius_mm(radius).ok_or_else(|| {
                model(format!(
                    "`{}` has a radius from an expression; change it in the source",
                    e.id.0
                ))
            })?;
            G::Circle {
                center: place(center)?,
                radius: super::mm(r * m.scale),
            }
        }
        G::Spline { points, closed } => {
            let mut points = points.iter().map(place).collect::<Result<Vec<_>, _>>()?;
            if m.flip && !*closed {
                points.reverse();
            }
            G::Spline {
                points,
                closed: *closed,
            }
        }
    })
}

/// The stem a copy of `e` is named from.
fn stem(e: &SketchEntity) -> &'static str {
    match e.geometry {
        G::Point { .. } => "p",
        G::Line { .. } => "l",
        G::Arc { .. } => "a",
        G::Circle { .. } => "c",
        G::Spline { .. } => "sp",
    }
}

/// Adds the image of each of `ids` under `m` as a new entity; returns the new names.
fn copy(
    sketch: &mut Sketch,
    ids: &[SketchEntityId],
    m: &Map2<'_>,
) -> Result<Vec<String>, EditFailure> {
    let mut added = Vec::new();
    for id in ids {
        let e = entity(sketch, id)?.clone();
        let geometry = mapped(sketch, &e, m)?;
        let new = SketchEntity {
            id: fresh_like(sketch, stem(&e)),
            geometry,
            construction: e.construction,
        };
        added.push(new.id.0.clone());
        sketch.entities.push(new);
    }
    Ok(added)
}

/// Moves each of `ids` by `m` in place.
fn relocate(sketch: &mut Sketch, ids: &[SketchEntityId], m: &Map2<'_>) -> Result<(), EditFailure> {
    let snapshot = sketch.clone();
    let geometries = ids
        .iter()
        .map(|id| mapped(&snapshot, entity(&snapshot, id)?, m))
        .collect::<Result<Vec<_>, _>>()?;
    for (id, geometry) in ids.iter().zip(geometries) {
        detach(sketch, id);
        drop_constraints_on(sketch, id);
        if let Some(e) = sketch.entities.iter_mut().find(|e| e.id == *id) {
            e.geometry = geometry;
        }
    }
    Ok(())
}

fn need(ids: &[SketchEntityId], what: &str) -> Result<(), EditFailure> {
    if ids.is_empty() {
        return Err(model(format!("pick the entities to {what}")));
    }
    Ok(())
}

fn names(ids: &[SketchEntityId]) -> String {
    ids.iter()
        .map(|i| i.0.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// `sketch.construction`: toggles each picked entity.
pub(super) fn construction(
    values: &Map<String, Value>,
    sketch: &mut Sketch,
    sketch_id: &str,
) -> Result<String, EditFailure> {
    let mut ids = picked(values, "entities", sketch_id)?;
    if ids.is_empty() {
        ids = picked(values, "selection", sketch_id)?;
    }
    need(&ids, "toggle")?;
    let (mut to, mut from) = (Vec::new(), Vec::new());
    for id in &ids {
        entity(sketch, id)?;
        let e = sketch
            .entities
            .iter_mut()
            .find(|e| e.id == *id)
            .expect("found");
        e.construction = !e.construction;
        if e.construction {
            to.push(id.clone())
        } else {
            from.push(id.clone())
        }
    }
    Ok(match (to.is_empty(), from.is_empty()) {
        (false, true) => format!("Made {} construction", names(&to)),
        (true, false) => format!("Made {} profile geometry", names(&from)),
        _ => format!(
            "Made {} construction and {} profile geometry",
            names(&to),
            names(&from)
        ),
    })
}

/// `sketch.mirror`: reflects the picked entities about the picked line.
pub(super) fn mirror(
    values: &Map<String, Value>,
    sketch: &mut Sketch,
    sketch_id: &str,
) -> Result<String, EditFailure> {
    let ids = picked(values, "entities", sketch_id)?;
    need(&ids, "mirror")?;
    let about = picked(values, "about", sketch_id)?
        .into_iter()
        .next()
        .ok_or_else(|| model("pick the line to mirror about"))?;
    let Some(Shape::Segment(a, b)) = shape(sketch, entity(sketch, &about)?) else {
        return Err(model(format!(
            "mirror about a line; `{}` isn't one",
            about.0
        )));
    };
    let ids: Vec<_> = ids.into_iter().filter(|i| *i != about).collect();
    need(&ids, "mirror")?;
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = (dx * dx + dy * dy).max(1e-18);
    let f = move |p: P| {
        let t = ((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2;
        let foot = (a.0 + t * dx, a.1 + t * dy);
        (2.0 * foot.0 - p.0, 2.0 * foot.1 - p.1)
    };
    let m = Map2 {
        f: &f,
        scale: 1.0,
        flip: true,
    };
    if values.get("copy").and_then(Value::as_bool).unwrap_or(true) {
        let added = copy(sketch, &ids, &m)?;
        Ok(format!(
            "Mirrored {} about {}; added {}",
            names(&ids),
            about.0,
            added.join(", ")
        ))
    } else {
        relocate(sketch, &ids, &m)?;
        Ok(format!("Mirrored {} about {}", names(&ids), about.0))
    }
}

fn rotation(c: P, angle: f64) -> impl Fn(P) -> P {
    let (s, k) = angle.sin_cos();
    move |p: P| {
        let (x, y) = (p.0 - c.0, p.1 - c.1);
        (c.0 + k * x - s * y, c.1 + s * x + k * y)
    }
}

/// `sketch.pattern`: copies of the picked entities in a line or round a centre.
pub(super) fn pattern(
    values: &Map<String, Value>,
    sketch: &mut Sketch,
    sketch_id: &str,
) -> Result<String, EditFailure> {
    let ids = picked(values, "entities", sketch_id)?;
    need(&ids, "pattern")?;
    let n = number(values, "instances").unwrap_or(4.0).round() as i64;
    if n < 2 {
        return Err(model("a pattern needs at least two instances"));
    }
    let mut added = Vec::new();
    match values
        .get("layout")
        .and_then(Value::as_str)
        .unwrap_or("linear")
    {
        "linear" => {
            let spacing = number(values, "spacing").unwrap_or(10.0);
            let dir = number(values, "direction").unwrap_or(0.0).to_radians();
            for i in 1..n {
                let (ox, oy) = (
                    spacing * i as f64 * dir.cos(),
                    spacing * i as f64 * dir.sin(),
                );
                let f = move |p: P| (p.0 + ox, p.1 + oy);
                added.extend(copy(
                    sketch,
                    &ids,
                    &Map2 {
                        f: &f,
                        scale: 1.0,
                        flip: false,
                    },
                )?);
            }
        }
        "circular" => {
            let c = xy(values, "center").ok_or_else(|| model("pick the pattern's center"))?;
            let total = number(values, "angle").unwrap_or(360.0);
            let step = if (total.abs() - 360.0).abs() < 1e-9 {
                total / n as f64
            } else {
                total / (n - 1) as f64
            };
            for i in 1..n {
                let f = rotation(c, (step * i as f64).to_radians());
                added.extend(copy(
                    sketch,
                    &ids,
                    &Map2 {
                        f: &f,
                        scale: 1.0,
                        flip: false,
                    },
                )?);
            }
        }
        other => {
            return Err(model(format!(
                "there is no `{other}` sketch pattern (linear or circular)"
            )))
        }
    }
    Ok(format!(
        "Patterned {} ({n} instances); added {}",
        names(&ids),
        added.join(", ")
    ))
}

/// `sketch.transform`: move, copy, rotate or scale the picked entities.
pub(super) fn transform(
    values: &Map<String, Value>,
    sketch: &mut Sketch,
    sketch_id: &str,
) -> Result<String, EditFailure> {
    let ids = picked(values, "entities", sketch_id)?;
    need(&ids, "transform")?;
    let base = xy(values, "base").unwrap_or((0.0, 0.0));
    let (dx, dy) = (
        number(values, "dx").unwrap_or(0.0),
        number(values, "dy").unwrap_or(0.0),
    );
    let kind = values.get("type").and_then(Value::as_str).unwrap_or("move");
    let shift = move |p: P| (p.0 + dx, p.1 + dy);
    match kind {
        "move" => {
            relocate(
                sketch,
                &ids,
                &Map2 {
                    f: &shift,
                    scale: 1.0,
                    flip: false,
                },
            )?;
            Ok(format!("Moved {}", names(&ids)))
        }
        "copy" => {
            let added = copy(
                sketch,
                &ids,
                &Map2 {
                    f: &shift,
                    scale: 1.0,
                    flip: false,
                },
            )?;
            Ok(format!(
                "Copied {}; added {}",
                names(&ids),
                added.join(", ")
            ))
        }
        "rotate" => {
            let f = rotation(base, number(values, "angle").unwrap_or(0.0).to_radians());
            relocate(
                sketch,
                &ids,
                &Map2 {
                    f: &f,
                    scale: 1.0,
                    flip: false,
                },
            )?;
            Ok(format!("Rotated {}", names(&ids)))
        }
        "scale" => {
            let k = number(values, "factor").unwrap_or(1.0);
            if k <= 0.0 {
                return Err(model("the scale factor must be above zero"));
            }
            let f = move |p: P| (base.0 + k * (p.0 - base.0), base.1 + k * (p.1 - base.1));
            relocate(
                sketch,
                &ids,
                &Map2 {
                    f: &f,
                    scale: k,
                    flip: false,
                },
            )?;
            Ok(format!("Scaled {} by {k}", names(&ids)))
        }
        other => Err(model(format!(
            "there is no `{other}` transform (move, copy, rotate or scale)"
        ))),
    }
}

/// `sketch.split`: the picked line or arc becomes two at the picked point.
pub(super) fn split(
    values: &Map<String, Value>,
    sketch: &mut Sketch,
    sketch_id: &str,
) -> Result<String, EditFailure> {
    let mut ids = picked(values, "entity", sketch_id)?;
    if ids.is_empty() {
        ids = picked(values, "entities", sketch_id)?;
    }
    let id = ids
        .into_iter()
        .next()
        .ok_or_else(|| model("pick the line or arc to split"))?;
    let click = xy(values, "at").ok_or_else(|| model("pick where to split it"))?;
    let e = entity(sketch, &id)?.clone();
    let own = shape(sketch, &e)
        .ok_or_else(|| model(format!("`{}` has no position to split yet", id.0)))?;
    let (span, point_at): (f64, Box<dyn Fn(f64) -> P>) = match own {
        Shape::Segment(a, b) => (
            1.0,
            Box::new(move |t| (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1))),
        ),
        Shape::Round(_, _, _, sw) if sw >= std::f64::consts::TAU - 1e-9 => {
            return Err(model(format!(
                "a circle has no ends to split between; trim `{}` instead",
                id.0
            )))
        }
        Shape::Round(c, r, t0, sw) => (
            sw,
            Box::new(move |t| (c.0 + r * (t0 + t).cos(), c.1 + r * (t0 + t).sin())),
        ),
    };
    let t = param(own, click);
    if t <= 1e-6 * span || t >= span * (1.0 - 1e-6) {
        return Err(model(format!(
            "pick a point inside `{}`, away from its ends",
            id.0
        )));
    }
    let cut = point_at(t);
    let end = point_expr(&e.geometry, PointRole::End)
        .cloned()
        .ok_or_else(|| model(format!("`{}` has no end", id.0)))?;
    let index = sketch
        .entities
        .iter()
        .position(|x| x.id == id)
        .expect("found");
    let mut rest = e.clone();
    rest.id = fresh_like(sketch, stem(&e));
    // Whatever shared the old end shares the new piece's end.
    for other in sketch.entities.iter_mut() {
        let follow = |p: &mut PointExpr| {
            if matches!(p, PointExpr::Of { entity, role: PointRole::End } if *entity == id) {
                *p = PointExpr::of(&rest.id.0, PointRole::End);
            }
        };
        match &mut other.geometry {
            G::Line { start, end } | G::Arc { start, end, .. } => {
                follow(start);
                follow(end);
            }
            G::Point { at } => follow(at),
            _ => {}
        }
    }
    set_end(
        &mut rest.geometry,
        PointRole::Start,
        PointExpr::of(&id.0, PointRole::End),
    );
    set_end(&mut rest.geometry, PointRole::End, end);
    set_end(
        &mut sketch.entities[index].geometry,
        PointRole::End,
        at(cut),
    );
    super::modify::drop_size_constraints(sketch, &id, true);
    let new = rest.id.0.clone();
    sketch.entities.insert(index + 1, rest);
    Ok(format!(
        "Split {} at ({:.3}, {:.3}); added {new}",
        id.0, cut.0, cut.1
    ))
}
