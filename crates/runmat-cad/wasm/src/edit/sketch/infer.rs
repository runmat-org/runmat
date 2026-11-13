//! `sketch.auto-constrain`: the relations the drawn geometry already shows, added to the
//! sketch.
//!
//! Over the picked entities (all profile and construction entities when none are picked):
//!
//! - a line within the angle tolerance (0.5°) of the sketch's x or y axis gets a horizontal or
//!   vertical relation;
//! - two points written as separate coordinates closer than the length tolerance (mm) get a
//!   coincident relation;
//! - two circles or arcs with centres within the tolerance get a coincident relation between
//!   the centres, and circles of radii within the tolerance an equal relation.
//!
//! A relation the sketch already has (either way round) isn't added again. Points that one
//! entity takes from another (`start = l1.end`) are already one point.

use super::modify::{position, P};
use super::{constraint, model, picks, target, EditFailure, K};
use runmat_cad::ids::SketchEntityId;
use runmat_cad::model::sketch::{
    PointExpr, Sketch, SketchConstraint, SketchGeometry as G, SketchTarget,
};
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

const ANGLE: f64 = 0.5;

fn same(a: &SketchTarget, b: &SketchTarget, c: &SketchTarget, d: &SketchTarget) -> bool {
    (a == c && b == d) || (a == d && b == c)
}

fn has(existing: &[SketchConstraint], k: &K) -> bool {
    existing.iter().any(|c| match (&c.kind, k) {
        (K::Horizontal { line: a }, K::Horizontal { line: b })
        | (K::Vertical { line: a }, K::Vertical { line: b }) => a == b,
        (K::Horizontal { line: a }, K::Vertical { line: b })
        | (K::Vertical { line: a }, K::Horizontal { line: b }) => a == b,
        (K::Coincident { a, b }, K::Coincident { a: c, b: d })
        | (K::Equal { a, b }, K::Equal { a: c, b: d }) => same(a, b, c, d),
        _ => false,
    })
}

/// `sketch.auto-constrain`. Returns the summary.
pub(super) fn auto_constrain(
    values: &Map<String, Value>,
    sketch: &mut Sketch,
    sketch_id: &str,
) -> Result<String, EditFailure> {
    let tol = values
        .get("tolerance")
        .and_then(super::num)
        .unwrap_or(0.01)
        .max(1e-9);
    let mut scope: Vec<SketchEntityId> = Vec::new();
    for pick in picks(values, "entities") {
        let id = match target(&pick, sketch_id)? {
            SketchTarget::Entity { entity } | SketchTarget::Point { entity, .. } => entity,
        };
        if !scope.contains(&id) {
            scope.push(id);
        }
    }
    let within = |id: &SketchEntityId| scope.is_empty() || scope.contains(id);
    let mut found: Vec<K> = Vec::new();
    let add = |k: K, found: &mut Vec<K>, existing: &[SketchConstraint]| {
        if !has(existing, &k)
            && !has(
                &found.iter().cloned().map(constraint).collect::<Vec<_>>(),
                &k,
            )
        {
            found.push(k);
        }
    };
    // Written points, by entity point.
    let mut points: Vec<(SketchTarget, P)> = Vec::new();
    for e in sketch.entities.iter().filter(|e| within(&e.id)) {
        if let G::Line { start, end } = &e.geometry {
            if let (Some(a), Some(b)) = (position(sketch, start, 0), position(sketch, end, 0)) {
                let deg = (b.1 - a.1).atan2(b.0 - a.0).to_degrees().rem_euclid(180.0);
                let line = SketchTarget::entity(&e.id.0);
                if deg.min(180.0 - deg) < ANGLE {
                    add(K::Horizontal { line }, &mut found, &sketch.constraints);
                } else if (deg - 90.0).abs() < ANGLE {
                    add(K::Vertical { line }, &mut found, &sketch.constraints);
                }
            }
        }
        for (role, p) in e.geometry.points() {
            if let (PointExpr::Coords { .. }, Some(q)) = (p, position(sketch, p, 0)) {
                points.push((SketchTarget::point(&e.id.0, role), q));
            }
        }
    }
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            let ((a, p), (b, q)) = (&points[i], &points[j]);
            let entity = |t: &SketchTarget| match t {
                SketchTarget::Point { entity, .. } | SketchTarget::Entity { entity } => {
                    entity.clone()
                }
            };
            if entity(a) != entity(b) && (p.0 - q.0).hypot(p.1 - q.1) < tol {
                add(
                    K::Coincident {
                        a: a.clone(),
                        b: b.clone(),
                    },
                    &mut found,
                    &sketch.constraints,
                );
            }
        }
    }
    let circles: Vec<(SketchEntityId, f64)> = sketch
        .entities
        .iter()
        .filter(|e| within(&e.id))
        .filter_map(|e| match &e.geometry {
            G::Circle {
                radius: ValueExpr::Quantity { value, .. },
                ..
            } => Some((e.id.clone(), *value)),
            _ => None,
        })
        .collect();
    for i in 0..circles.len() {
        for j in i + 1..circles.len() {
            if (circles[i].1 - circles[j].1).abs() < tol {
                add(
                    K::Equal {
                        a: SketchTarget::entity(&circles[i].0 .0),
                        b: SketchTarget::entity(&circles[j].0 .0),
                    },
                    &mut found,
                    &sketch.constraints,
                );
            }
        }
    }
    if found.is_empty() {
        return Err(model(
            "the sketch shows no relations it doesn't already have",
        ));
    }
    let n = found.len();
    sketch.constraints.extend(found.into_iter().map(constraint));
    Ok(format!("Added {n} relation(s) the geometry shows"))
}
