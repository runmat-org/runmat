//! Sketch commands from the canvas, each written as the smallest text change to the sketch's
//! block: `part.sketch` starts a sketch on a picked plane or planar face, and the sketch tools
//! add entities, relations and dimensions to the open sketch.
//!
//! Values come from the Desktop's sketch overlay, in sketch-plane millimetres:
//!
//! - `sketch`: the open sketch's feature ID.
//! - A point is `{ x, y }`, or `{ x, y, of: "l1.end" }` when it snapped to an existing point;
//!   a snapped point is written as that point (`start = l1.end`), which makes the two one point.
//! - Relations and dimensions name what they act on as canvas picks: `sketch.<sketch>.<entity>`
//!   for a curve, `sketch.<sketch>.<entity>.<start|end|center|at>` for a point.
//!
//! Lines, rectangles (four lines sharing their corners, two horizontal, two vertical), circles,
//! centre-point arcs, straight slots (two lines and two arcs, tangent), points, offsets of lines
//! and circles, trim and extend (`modify`), mirror, pattern, transform, split and the
//! construction toggle (`repeat`), fillet and chamfer (`corners`), auto-constrain (`infer`), and
//! the relations and dimensions the sketch model has. A tool the sketch model lacks (splines, ellipses, text) says so.

use super::{model, EditFailure, SourceChange};
use runmat_cad::ids::SketchEntityId;
use runmat_cad::model::datum::PlaneRef;
use runmat_cad::model::sketch::{
    PointExpr, PointRole, Sketch, SketchConstraint, SketchConstraintKind as K, SketchEntity,
    SketchGeometry as G, SketchTarget,
};
use runmat_cad::model::{Feature, FeatureKind, PartDocument};
use runmat_cad::syntax::edit::edit_part;
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

fn num(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| {
        v.as_str()
            .and_then(|s| s.trim().trim_end_matches("mm").trim().parse().ok())
    })
}

fn mm(v: f64) -> ValueExpr {
    // Tenths of a micrometre: what a click on the canvas can mean.
    ValueExpr::mm((v * 1.0e4).round() / 1.0e4)
}

fn role(word: &str) -> Option<PointRole> {
    Some(match word {
        "start" => PointRole::Start,
        "end" => PointRole::End,
        "center" => PointRole::Center,
        "at" => PointRole::At,
        _ => return None,
    })
}

/// A point value: coordinates, or the existing point it snapped to.
fn point(values: &Map<String, Value>, id: &str) -> Result<PointExpr, EditFailure> {
    let v = values
        .get(id)
        .filter(|v| v.is_object())
        .ok_or_else(|| model(format!("pick the {id} point on the sketch")))?;
    if let Some(of) = v["of"].as_str() {
        if let Some((entity, word)) = of.rsplit_once('.') {
            if let Some(role) = role(word) {
                return Ok(PointExpr::Of {
                    entity: SketchEntityId::new(entity),
                    role,
                });
            }
        }
    }
    let (Some(x), Some(y)) = (num(&v["x"]), num(&v["y"])) else {
        return Err(model(format!("the {id} point has no coordinates")));
    };
    Ok(PointExpr::Coords { x: mm(x), y: mm(y) })
}

fn xy(values: &Map<String, Value>, id: &str) -> Option<(f64, f64)> {
    let v = values.get(id)?;
    Some((num(&v["x"])?, num(&v["y"])?))
}

/// What a relation or dimension pick names: a curve or one of its points.
fn target(pick: &str, sketch: &str) -> Result<SketchTarget, EditFailure> {
    let rest = pick
        .strip_prefix(&format!("sketch.{sketch}."))
        .unwrap_or(pick);
    match rest.rsplit_once('.') {
        Some((entity, word)) if role(word).is_some() => Ok(SketchTarget::Point {
            entity: SketchEntityId::new(entity),
            role: role(word).expect("checked"),
        }),
        _ if !rest.is_empty() && !rest.contains('.') => Ok(SketchTarget::entity(rest)),
        _ => Err(model(format!(
            "`{pick}` isn't a curve or point of sketch {sketch}"
        ))),
    }
}

fn picks(values: &Map<String, Value>, id: &str) -> Vec<String> {
    match values.get(id) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        _ => Vec::new(),
    }
}

fn sketch_mut<'a>(document: &'a mut PartDocument, id: &str) -> Result<&'a mut Sketch, EditFailure> {
    match document.feature_mut(id).map(|f| &mut f.kind) {
        Some(FeatureKind::Sketch(s)) => Ok(s),
        Some(_) => Err(model(format!("`{id}` isn't a sketch"))),
        None => Err(model(format!("the part has no sketch `{id}`"))),
    }
}

/// The next free entity name with `stem` (`l5`, `c2`).
fn fresh(sketch: &Sketch, taken: &[String], stem: &str) -> String {
    (1..)
        .map(|n| format!("{stem}{n}"))
        .find(|id| !sketch.entities.iter().any(|e| e.id.0 == *id) && !taken.contains(id))
        .expect("an unused name")
}

fn of(entity: &str, role: PointRole) -> PointExpr {
    PointExpr::Of {
        entity: SketchEntityId::new(entity),
        role,
    }
}

fn constraint(kind: K) -> SketchConstraint {
    SketchConstraint { id: None, kind }
}

/// Where a point written as coordinates sits (snapped points are resolved by the caller).
fn coords(p: &PointExpr) -> Option<(f64, f64)> {
    match p {
        PointExpr::Coords {
            x: ValueExpr::Quantity { value: x, .. },
            y: ValueExpr::Quantity { value: y, .. },
        } => Some((*x, *y)),
        _ => None,
    }
}

/// The entities (and relations) one drawing tool adds.
fn draw(
    command: &str,
    values: &Map<String, Value>,
    sketch: &Sketch,
) -> Result<(Vec<SketchEntity>, Vec<SketchConstraint>), EditFailure> {
    let mut taken: Vec<String> = Vec::new();
    let mut name = |stem: &str| {
        let id = fresh(sketch, &taken, stem);
        taken.push(id.clone());
        id
    };
    let entity = |id: &str, geometry: G, construction: bool| SketchEntity {
        id: SketchEntityId::new(id),
        geometry,
        construction,
    };
    let kind = values.get("type").and_then(Value::as_str).unwrap_or("");
    match command {
        "sketch.line" => {
            let (start, end) = (point(values, "start")?, point(values, "end")?);
            let id = name("l");
            let mut relations = Vec::new();
            // A line drawn along an axis keeps it: the inference the canvas showed.
            if let Some(axis) = values.get("align").and_then(Value::as_str) {
                let line = SketchTarget::entity(&id);
                relations.push(constraint(match axis {
                    "horizontal" => K::Horizontal { line },
                    "vertical" => K::Vertical { line },
                    other => return Err(model(format!("a line can't align `{other}`"))),
                }));
            }
            Ok((
                vec![entity(&id, G::Line { start, end }, kind == "centerline")],
                relations,
            ))
        }
        "sketch.rectangle" => {
            let (a, b) = (xy(values, "first"), xy(values, "second"));
            let (Some((x0, y0)), Some((x1, y1))) = (a, b) else {
                return Err(model("pick the rectangle's two corners on the sketch"));
            };
            // Centre rectangles: the first point is the centre, the second a corner.
            let (x0, y0, x1, y1) = if kind == "center" {
                (2.0 * x0 - x1, 2.0 * y0 - y1, x1, y1)
            } else {
                (x0, y0, x1, y1)
            };
            if (x1 - x0).abs() < 1e-6 || (y1 - y0).abs() < 1e-6 {
                return Err(model(
                    "a rectangle needs two corners apart in both directions",
                ));
            }
            let ids: Vec<String> = (0..4).map(|_| name("l")).collect();
            let first = PointExpr::Coords {
                x: mm(x0),
                y: mm(y0),
            };
            let corners = [
                first,
                PointExpr::Coords {
                    x: mm(x1),
                    y: mm(y0),
                },
                PointExpr::Coords {
                    x: mm(x1),
                    y: mm(y1),
                },
                PointExpr::Coords {
                    x: mm(x0),
                    y: mm(y1),
                },
            ];
            let entities = (0..4)
                .map(|k| {
                    let start = if k == 0 {
                        corners[0].clone()
                    } else {
                        of(&ids[k - 1], PointRole::End)
                    };
                    let end = if k == 3 {
                        of(&ids[0], PointRole::Start)
                    } else {
                        corners[k + 1].clone()
                    };
                    entity(&ids[k], G::Line { start, end }, false)
                })
                .collect();
            let relations = vec![
                constraint(K::Horizontal {
                    line: SketchTarget::entity(&ids[0]),
                }),
                constraint(K::Vertical {
                    line: SketchTarget::entity(&ids[1]),
                }),
                constraint(K::Horizontal {
                    line: SketchTarget::entity(&ids[2]),
                }),
                constraint(K::Vertical {
                    line: SketchTarget::entity(&ids[3]),
                }),
            ];
            Ok((entities, relations))
        }
        "sketch.circle" => {
            if kind == "three_point" {
                return Err(model(
                    "3-point circles aren't in the sketch model yet; draw it by centre and radius",
                ));
            }
            let center = point(values, "center")?;
            let radius = values
                .get("radius")
                .and_then(num)
                .filter(|r| *r > 0.0)
                .ok_or_else(|| model("give the circle a radius"))?;
            Ok((
                vec![entity(
                    &name("c"),
                    G::Circle {
                        center,
                        radius: mm(radius),
                    },
                    false,
                )],
                Vec::new(),
            ))
        }
        "sketch.arc" => {
            if kind == "tangent" || kind == "three_point" {
                return Err(model(format!(
                    "{kind} arcs aren't in the sketch model yet; draw a centre-point arc"
                )));
            }
            let (center, start, end) = (
                point(values, "center")?,
                point(values, "start")?,
                point(values, "end")?,
            );
            Ok((
                vec![entity(&name("a"), G::Arc { center, start, end }, false)],
                Vec::new(),
            ))
        }
        "sketch.point" => Ok((
            vec![entity(
                &name("p"),
                G::Point {
                    at: point(values, "at")?,
                },
                false,
            )],
            Vec::new(),
        )),
        "sketch.slot" => {
            if kind == "arc" {
                return Err(model(
                    "arc slots aren't in the sketch model yet; draw a straight slot",
                ));
            }
            let (Some((x0, y0)), Some((x1, y1))) = (xy(values, "start"), xy(values, "end")) else {
                return Err(model("pick the slot's two centres on the sketch"));
            };
            let w = values
                .get("width")
                .and_then(num)
                .filter(|w| *w > 0.0)
                .ok_or_else(|| model("give the slot a width"))?;
            let len = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
            if len < 1e-6 {
                return Err(model("a slot needs two different centres"));
            }
            let (nx, ny) = (-(y1 - y0) / len * w / 2.0, (x1 - x0) / len * w / 2.0);
            let p = |x: f64, y: f64| PointExpr::Coords { x: mm(x), y: mm(y) };
            let (top, bottom, a_end, a_start) = (name("l"), name("l"), name("a"), name("a"));
            // Counter-clockwise: the bottom line from start to end, the end cap, the top line
            // back, the start cap.
            let entities = vec![
                entity(
                    &bottom,
                    G::Line {
                        start: p(x0 - nx, y0 - ny),
                        end: p(x1 - nx, y1 - ny),
                    },
                    false,
                ),
                entity(
                    &a_end,
                    G::Arc {
                        center: p(x1, y1),
                        start: of(&bottom, PointRole::End),
                        end: p(x1 + nx, y1 + ny),
                    },
                    false,
                ),
                entity(
                    &top,
                    G::Line {
                        start: of(&a_end, PointRole::End),
                        end: p(x0 + nx, y0 + ny),
                    },
                    false,
                ),
                entity(
                    &a_start,
                    G::Arc {
                        center: p(x0, y0),
                        start: of(&top, PointRole::End),
                        end: of(&bottom, PointRole::Start),
                    },
                    false,
                ),
            ];
            let t = SketchTarget::entity;
            let relations = vec![
                constraint(K::Tangent {
                    a: t(&bottom),
                    b: t(&a_end),
                }),
                constraint(K::Tangent {
                    a: t(&a_end),
                    b: t(&top),
                }),
                constraint(K::Tangent {
                    a: t(&top),
                    b: t(&a_start),
                }),
                constraint(K::Tangent {
                    a: t(&a_start),
                    b: t(&bottom),
                }),
                constraint(K::Equal {
                    a: t(&a_start),
                    b: t(&a_end),
                }),
            ];
            Ok((entities, relations))
        }
        "sketch.offset" => {
            let d = values
                .get("distance")
                .and_then(num)
                .ok_or_else(|| model("give the offset a distance"))?;
            let mut out = Vec::new();
            for pick in picks(values, "entities") {
                let SketchTarget::Entity { entity: id } = target(&pick, "")? else {
                    continue;
                };
                let rest = id.0.rsplit('.').next().unwrap_or(&id.0).to_owned();
                let Some(source) = sketch.entities.iter().find(|e| e.id.0 == rest) else {
                    return Err(model(format!("the sketch has no curve `{rest}`")));
                };
                let geometry = match &source.geometry {
                    G::Line { start, end } => {
                        let (Some(a), Some(b)) = (
                            coords(start).or_else(|| solved(sketch, &rest, PointRole::Start)),
                            coords(end).or_else(|| solved(sketch, &rest, PointRole::End)),
                        ) else {
                            return Err(model(format!(
                                "`{rest}` has no position to offset from yet"
                            )));
                        };
                        let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt().max(1e-9);
                        let (nx, ny) = (-(b.1 - a.1) / len * d, (b.0 - a.0) / len * d);
                        G::Line {
                            start: PointExpr::Coords {
                                x: mm(a.0 + nx),
                                y: mm(a.1 + ny),
                            },
                            end: PointExpr::Coords {
                                x: mm(b.0 + nx),
                                y: mm(b.1 + ny),
                            },
                        }
                    }
                    G::Circle {
                        center,
                        radius: ValueExpr::Quantity { value, unit },
                    } if unit == "mm" => G::Circle {
                        center: center.clone(),
                        radius: mm((value + d).max(1e-3)),
                    },
                    _ => {
                        return Err(model(format!(
                            "Offset takes lines and circles, not `{rest}`"
                        )))
                    }
                };
                let stem = if matches!(geometry, G::Line { .. }) {
                    "l"
                } else {
                    "c"
                };
                out.push(entity(&name(stem), geometry, source.construction));
            }
            if out.is_empty() {
                return Err(model("pick the curves to offset"));
            }
            Ok((out, Vec::new()))
        }
        other => Err(super::unhandled(other)),
    }
}

/// A point's position from the sketch's `solved { }` cache, in millimetres.
fn solved(sketch: &Sketch, entity: &str, role: PointRole) -> Option<(f64, f64)> {
    sketch.solved.points.iter().find_map(|(k, p)| {
        let text = serde_json::to_value(k).ok()?;
        let matches =
            text["entity"] == entity && text["role"] == serde_json::to_value(role).ok()?;
        matches.then_some((p[0] * 1000.0, p[1] * 1000.0))
    })
}

/// A relation between the picked curves and points.
fn relation(values: &Map<String, Value>, sketch: &str) -> Result<SketchConstraint, EditFailure> {
    let relation = values
        .get("relation")
        .and_then(Value::as_str)
        .ok_or_else(|| model("choose a relation"))?;
    let targets = picks(values, "entities")
        .iter()
        .map(|p| target(p, sketch))
        .collect::<Result<Vec<_>, _>>()?;
    let one = || {
        targets
            .first()
            .cloned()
            .ok_or_else(|| model(format!("pick what {relation} acts on")))
    };
    let two = || match targets.as_slice() {
        [a, b, ..] => Ok((a.clone(), b.clone())),
        _ => Err(model(format!("{relation} takes two picks"))),
    };
    Ok(constraint(match relation {
        "horizontal" => K::Horizontal { line: one()? },
        "vertical" => K::Vertical { line: one()? },
        "fix" => K::Fix { target: one()? },
        "coincident" => two().map(|(a, b)| K::Coincident { a, b })?,
        "parallel" => two().map(|(a, b)| K::Parallel { a, b })?,
        "perpendicular" => two().map(|(a, b)| K::Perpendicular { a, b })?,
        "tangent" => two().map(|(a, b)| K::Tangent { a, b })?,
        "equal" => two().map(|(a, b)| K::Equal { a, b })?,
        // A point and a line, picked in either order.
        "midpoint" => match two()? {
            (point @ SketchTarget::Point { .. }, line @ SketchTarget::Entity { .. })
            | (line @ SketchTarget::Entity { .. }, point @ SketchTarget::Point { .. }) => {
                K::Midpoint { point, line }
            }
            _ => return Err(model("Midpoint takes a point and a line")),
        },
        "concentric" => two().map(|(a, b)| K::Concentric { a, b })?,
        "coradial" => two().map(|(a, b)| K::Coradial { a, b })?,
        "collinear" => two().map(|(a, b)| K::Collinear { a, b })?,
        // Two points and the line they mirror in; the line may come first or last.
        "symmetric" => {
            let mut points = targets
                .iter()
                .filter(|t| matches!(t, SketchTarget::Point { .. }))
                .cloned();
            let axis = targets
                .iter()
                .find(|t| matches!(t, SketchTarget::Entity { .. }))
                .cloned();
            match (points.next(), points.next(), axis, targets.len()) {
                (Some(a), Some(b), Some(axis), 3) => K::Symmetric { a, b, axis },
                _ => {
                    return Err(model(
                        "Symmetric takes two points and the line between them",
                    ))
                }
            }
        }
        other => return Err(model(format!("Unknown sketch relation `{other}`"))),
    }))
}

/// A dimension on the picks: a line's length, a circle's or arc's radius, the distance between
/// two points (or a point and a line), or the angle between two lines.
fn dimension(
    values: &Map<String, Value>,
    sketch: &Sketch,
    id: &str,
) -> Result<SketchConstraint, EditFailure> {
    let targets = picks(values, "entities")
        .iter()
        .map(|p| target(p, id))
        .collect::<Result<Vec<_>, _>>()?;
    let value = values
        .get("value")
        .and_then(num)
        .ok_or_else(|| model("type the dimension's value"))?;
    let is = |t: &SketchTarget, f: fn(&G) -> bool| match t {
        SketchTarget::Entity { entity } => sketch
            .entities
            .iter()
            .any(|e| e.id == *entity && f(&e.geometry)),
        SketchTarget::Point { .. } => false,
    };
    let line = |g: &G| matches!(g, G::Line { .. });
    let circle = |g: &G| matches!(g, G::Circle { .. });
    let arc = |g: &G| matches!(g, G::Arc { .. });
    Ok(constraint(match targets.as_slice() {
        [t] if is(t, line) => K::Length {
            line: t.clone(),
            value: mm(value),
        },
        // Full circles take a diameter and arcs a radius, as drawings dimension them.
        [t] if is(t, circle) => K::Diameter {
            curve: t.clone(),
            value: mm(value),
        },
        [t] if is(t, arc) => K::Radius {
            curve: t.clone(),
            value: mm(value),
        },
        [a, b]
            if is(a, line)
                && is(b, line)
                && values.get("type").and_then(Value::as_str) == Some("angle") =>
        {
            K::Angle {
                a: a.clone(),
                b: b.clone(),
                value: ValueExpr::deg(value),
            }
        }
        [a, b] => K::Distance {
            a: a.clone(),
            b: b.clone(),
            value: mm(value),
        },
        _ => return Err(model("pick a line, a circle, or two points to dimension")),
    }))
}

/// Runs a sketch command; `None` when `command` isn't one.
pub fn sketch_edit(
    path: &str,
    text: &str,
    command: &str,
    values: &Map<String, Value>,
    picks_of: &crate::picks::Picks<'_>,
) -> Option<Result<(SourceChange, String), EditFailure>> {
    if command == "part.sketch" {
        return Some(new_sketch(path, text, values, picks_of));
    }
    if !command.starts_with("sketch.") {
        return None;
    }
    let Some(id) = values
        .get("sketch")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return Some(Err(model("open a sketch first")));
    };
    let mut summary = String::new();
    let result = edit_part(text, path, |document| {
        let sketch = sketch_mut(document, &id)?;
        match command {
            "sketch.constraint" => {
                let c = relation(values, &id)?;
                summary = format!("Added a relation to {id}");
                sketch.constraints.push(c);
            }
            "sketch.dimension" => {
                let c = dimension(values, sketch, &id)?;
                summary = format!("Dimensioned {id}");
                sketch.constraints.push(c);
            }
            "sketch.trim" => summary = modify::trim(values, sketch, &id)?,
            "sketch.extend" => summary = modify::extend(values, sketch, &id)?,
            "sketch.construction" => summary = repeat::construction(values, sketch, &id)?,
            "sketch.mirror" => summary = repeat::mirror(values, sketch, &id)?,
            "sketch.pattern" => summary = repeat::pattern(values, sketch, &id)?,
            "sketch.transform" => summary = repeat::transform(values, sketch, &id)?,
            "sketch.split" => summary = repeat::split(values, sketch, &id)?,
            "sketch.fillet" => summary = corners::fillet(values, sketch, &id)?,
            "sketch.chamfer" => summary = corners::chamfer(values, sketch, &id)?,
            "sketch.auto-constrain" => summary = infer::auto_constrain(values, sketch, &id)?,
            _ => {
                let (entities, relations) = draw(command, values, sketch)?;
                summary = format!(
                    "Added {} to {id}",
                    entities
                        .iter()
                        .map(|e| e.id.0.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                sketch.entities.extend(entities);
                sketch.constraints.extend(relations);
            }
        }
        Ok(())
    });
    Some(result.map(|change| (change, summary)))
}

/// `part.sketch`: an empty sketch on the picked plane or planar face, after the last feature.
fn new_sketch(
    path: &str,
    text: &str,
    values: &Map<String, Value>,
    picks_of: &crate::picks::Picks<'_>,
) -> Result<(SourceChange, String), EditFailure> {
    let pick = picks(values, "plane")
        .into_iter()
        .next()
        .ok_or_else(|| {
            model(format!(
                "pick a plane or a planar face to sketch on: `part.sketch` reads it from `values.plane`. {}",
                crate::picks::PLANE_PICK_FORMS
            ))
        })?;
    let plane: PlaneRef = serde_json::from_value(picks_of.plane(&pick)?)
        .map_err(|e| model(format!("the sketch plane: {e}")))?;
    let mut added = String::new();
    let change = edit_part(text, path, |document| {
        added = (1..)
            .map(|n| format!("sketch{n}"))
            .find(|id| !document.features.iter().any(|f| f.id.0 == *id))
            .expect("an unused name");
        document.features.push(Feature::new(
            &added,
            FeatureKind::Sketch(Sketch {
                plane: plane.clone(),
                entities: Vec::new(),
                constraints: Vec::new(),
                solved: Default::default(),
            }),
        ));
        Ok(())
    })?;
    Ok((change, format!("Added {added}")))
}

mod corners;
mod infer;
mod modify;
mod repeat;

#[cfg(test)]
mod tests;
