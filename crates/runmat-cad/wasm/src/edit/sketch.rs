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
//! the relations and dimensions the sketch model has, splines through points, ellipses and
//! elliptical arcs, and text. A tool the sketch model lacks (parabolas, conics, control-point splines,
//! text along a curve)
//! says so.

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
        projected: None,
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
        "sketch.polygon" => polygon(values, &mut name),
        "sketch.ellipse" => ellipse::draw(values, kind, &mut name),
        "sketch.text" => text::draw(values, &mut name),
        "sketch.spline" => {
            if !kind.is_empty() && kind != "through_points" {
                let what = match kind {
                    "control_points" => "control-point splines",
                    "equation" => "equation curves",
                    _ => "style splines",
                };
                return Err(model(format!(
                    "{what} aren't in the sketch model yet; draw a spline through points"
                )));
            }
            let points = spline_points(values)?;
            let closed = values
                .get("closed")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if points.len() < if closed { 3 } else { 2 } {
                return Err(model(if closed {
                    "a closed spline runs through at least three points"
                } else {
                    "a spline runs through at least two points"
                }));
            }
            Ok((
                vec![entity(&name("sp"), G::Spline { points, closed }, false)],
                Vec::new(),
            ))
        }
        "sketch.offset" => {
            let d = values
                .get("distance")
                .and_then(num)
                .ok_or_else(|| model("give the offset a distance"))?;
            let mut out = Vec::new();
            for pick in picks(values, "entities") {
                // A curve pick, bare (`l1`) or full (`sketch.s1.l1`); point picks are skipped.
                let rest = pick.rsplit('.').next().unwrap_or(&pick).to_owned();
                if role(&rest).is_some() {
                    continue;
                }
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

/// A spline's points in order: each a point value (`{ x, y }`, or `{ x, y, of }` when it snapped
/// to a sketch point) or a canvas pick of a sketch point (`sketch.<sketch>.l1.end`).
fn spline_points(values: &Map<String, Value>) -> Result<Vec<PointExpr>, EditFailure> {
    let items = match values.get("points") {
        Some(Value::Array(items)) => items.clone(),
        Some(v @ (Value::String(_) | Value::Object(_))) => vec![v.clone()],
        _ => Vec::new(),
    };
    items
        .iter()
        .map(|item| match item {
            Value::String(pick) => {
                // `sketch.<sketch>.l1.end` from the canvas, or `l1.end`.
                let rest = match pick.strip_prefix("sketch.") {
                    Some(r) => r.split_once('.').map_or(r, |(_, r)| r),
                    None => pick.as_str(),
                };
                match rest.rsplit_once('.') {
                    Some((entity, word)) if role(word).is_some() => Ok(of(entity, role(word).expect("checked"))),
                    _ => Err(model(format!(
                        "`{pick}` isn't a sketch point; a spline runs through points of the open sketch"
                    ))),
                }
            }
            Value::Object(_) => {
                let mut one = Map::new();
                one.insert("point".into(), item.clone());
                point(&one, "point")
            }
            _ => Err(model("a spline point is a picked sketch point or { x, y }")),
        })
        .collect()
}

/// A regular polygon: `sides` lines in a closed chain, each corner shared, all of equal length.
/// Its corners sit on a circle of `radius` about `center` (inscribed), or its sides touch that
/// circle (circumscribed); `rotation` (degrees) turns the first corner from the +x direction.
fn polygon(
    values: &Map<String, Value>,
    name: &mut dyn FnMut(&str) -> String,
) -> Result<(Vec<SketchEntity>, Vec<SketchConstraint>), EditFailure> {
    let n = values.get("sides").and_then(num).unwrap_or(6.0).round();
    if !(3.0..=64.0).contains(&n) {
        return Err(model("a polygon has 3 to 64 sides"));
    }
    let n = n as usize;
    let Some((cx, cy)) = xy(values, "center") else {
        return Err(model("pick the polygon's centre on the sketch"));
    };
    let r = values
        .get("radius")
        .and_then(num)
        .filter(|r| *r > 0.0)
        .ok_or_else(|| model("give the polygon a radius"))?;
    let step = std::f64::consts::TAU / n as f64;
    // Circumscribed: the sides touch the circle, so the corners sit further out.
    let corner_r = match values.get("fit").and_then(Value::as_str) {
        Some("circumscribed") => r / (step / 2.0).cos(),
        _ => r,
    };
    let turn = values
        .get("rotation")
        .and_then(num)
        .unwrap_or(0.0)
        .to_radians();
    let ids: Vec<String> = (0..n).map(|_| name("l")).collect();
    let corner = |k: usize| {
        let a = turn + step * k as f64;
        let round = |v: f64| (v * 1e6).round() / 1e6;
        PointExpr::Coords {
            x: mm(round(cx + corner_r * a.cos())),
            y: mm(round(cy + corner_r * a.sin())),
        }
    };
    let entities = (0..n)
        .map(|k| {
            let start = if k == 0 {
                corner(0)
            } else {
                of(&ids[k - 1], PointRole::End)
            };
            let end = if k == n - 1 {
                of(&ids[0], PointRole::Start)
            } else {
                corner(k + 1)
            };
            SketchEntity {
                id: SketchEntityId::new(&ids[k]),
                geometry: G::Line { start, end },
                construction: false,
                projected: None,
            }
        })
        .collect();
    let relations = (1..n)
        .map(|k| {
            constraint(K::Equal {
                a: SketchTarget::entity(&ids[0]),
                b: SketchTarget::entity(&ids[k]),
            })
        })
        .collect();
    Ok((entities, relations))
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
    let elliptic = |g: &G| matches!(g, G::Ellipse { .. });
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
        // An ellipse's semi-axis: the major unless the form names the minor.
        [t] if is(t, elliptic) => {
            let curve = t.clone();
            let value = mm(value);
            match values.get("axis").and_then(Value::as_str) {
                Some("minor") => K::MinorRadius { curve, value },
                _ => K::MajorRadius { curve, value },
            }
        }
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
    if command == "part.edit-sketch-plane" {
        return Some(edit_plane(path, text, values, picks_of));
    }
    if !command.starts_with("sketch.") {
        return None;
    }
    if command == "sketch.use-edges" {
        return Some(use_edges::use_edges(path, text, values));
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
        summary = apply(command, values, sketch, &id)?;
        Ok(())
    });
    Some(result.map(|change| (change, summary)))
}

/// One sketch tool on the open sketch `id`, answering the summary. Parts and assemblies both
/// run their sketch tools through this one function.
pub(crate) fn apply(
    command: &str,
    values: &Map<String, Value>,
    sketch: &mut Sketch,
    id: &str,
) -> Result<String, EditFailure> {
    let id = id.to_owned();
    let values = &picked_points(values, sketch, &id);
    Ok(match command {
        "sketch.constraint" => {
            let c = relation(values, &id)?;
            sketch.constraints.push(c);
            format!("Added a relation to {id}")
        }
        "sketch.dimension" => {
            let c = dimension(values, sketch, &id)?;
            sketch.constraints.push(c);
            format!("Dimensioned {id}")
        }
        "sketch.trim" => modify::trim(values, sketch, &id)?,
        "sketch.extend" => modify::extend(values, sketch, &id)?,
        "sketch.construction" => repeat::construction(values, sketch, &id)?,
        "sketch.mirror" => repeat::mirror(values, sketch, &id)?,
        "sketch.pattern" => repeat::pattern(values, sketch, &id)?,
        "sketch.transform" => repeat::transform(values, sketch, &id)?,
        "sketch.split" => repeat::split(values, sketch, &id)?,
        "sketch.fillet" => corners::fillet(values, sketch, &id)?,
        "sketch.chamfer" => corners::chamfer(values, sketch, &id)?,
        "sketch.auto-constrain" => infer::auto_constrain(values, sketch, &id)?,
        _ => {
            let (entities, relations) = draw(command, values, sketch)?;
            let summary = format!(
                "Added {} to {id}",
                entities
                    .iter()
                    .map(|e| e.id.0.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            sketch.entities.extend(entities);
            sketch.constraints.extend(relations);
            summary
        }
    })
}

/// The sketch entities among a selection, by sketch: canvas and Outline picks
/// `sketch.<sketch>.<entity>` (a point pick `sketch.<sketch>.<entity>.<role>` names its entity).
pub(crate) fn selected_entities(selection: &[String]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for pick in selection {
        let Some(rest) = pick.strip_prefix("sketch.") else {
            continue;
        };
        let mut parts = rest.split('.');
        if let (Some(sketch), Some(entity)) = (parts.next(), parts.next()) {
            // A point pick names its entity; Delete checks it is a point entity.
            let entity = match parts.next() {
                Some(role) => format!("{entity}.{role}"),
                None => entity.to_owned(),
            };
            let key = (sketch.to_owned(), entity);
            if !out.contains(&key) {
                out.push(key);
            }
        }
    }
    out
}

/// Deletes entities `ids` of `sketch`, as Delete does on picked curves and points: points other
/// curves took from them are written out where they sit, and the relations and dimensions that
/// name them go. Parts and assemblies both delete sketch entities through this one function.
pub(crate) fn delete_entities(sketch: &mut Sketch, ids: &[String]) -> Result<String, EditFailure> {
    for id in ids {
        // A picked point deletes a point entity (`p1.at`); a curve goes by picking the curve.
        let id = match id.split_once('.') {
            Some((entity, _)) => {
                let point = sketch
                    .entities
                    .iter()
                    .any(|e| e.id.0 == entity && matches!(e.geometry, G::Point { .. }));
                if !point {
                    return Err(model(format!(
                        "Delete takes whole curves: pick `{entity}` itself, not its point"
                    )));
                }
                SketchEntityId::new(entity)
            }
            None => SketchEntityId::new(id),
        };
        if !sketch.entities.iter().any(|e| e.id == id) {
            return Err(model(format!("the sketch has no `{id}`")));
        }
        modify::detach(sketch, &id);
        modify::drop_constraints_on(sketch, &id);
        sketch.entities.retain(|e| e.id != id);
    }
    Ok(format!("Deleted {}", ids.join(", ")))
}

/// A tool's values with the editor's selection as `values.selection` when the tool started from
/// one and its form has no field for it (Construction toggles the selected curves).
pub(crate) fn with_selection(
    values: &Map<String, Value>,
    selection: &[String],
) -> Map<String, Value> {
    let mut out = values.clone();
    if !selection.is_empty() && !out.contains_key("selection") {
        out.insert(
            "selection".into(),
            Value::Array(selection.iter().cloned().map(Value::String).collect()),
        );
    }
    out
}

/// The point fields a tool takes (`center`, `at`, `base`, `start`, `end`, `first`, `second`)
/// given as a pick of a sketch point (`sketch.<sketch>.l1.end`, from the Outline or the view)
/// rewritten as the point value the canvas sends, `{ x, y, of }`: the tool then sits on that
/// point, where it is now.
fn picked_points(values: &Map<String, Value>, sketch: &Sketch, id: &str) -> Map<String, Value> {
    let mut out = values.clone();
    for key in ["center", "at", "base", "start", "end", "first", "second"] {
        let pick = match values.get(key) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(a)) if a.len() == 1 => match a[0].as_str() {
                Some(s) => s.to_owned(),
                None => continue,
            },
            _ => continue,
        };
        let Ok(SketchTarget::Point { entity, role }) = target(&pick, id) else {
            continue;
        };
        let at = solved(sketch, &entity.0, role).or_else(|| {
            modify::position(
                sketch,
                &PointExpr::Of {
                    entity: entity.clone(),
                    role,
                },
                0,
            )
        });
        if let Some((x, y)) = at {
            out.insert(
                key.to_owned(),
                serde_json::json!({ "x": x, "y": y, "of": format!("{}.{}", entity.0, role.word()) }),
            );
        }
    }
    out
}

/// `part.edit-sketch-plane`: the selected (or open) sketch moves to the picked plane or planar
/// face. Its entities keep their sketch coordinates, so the drawing moves with the plane; a
/// face of a feature after the sketch is refused, since the sketch can't read what follows it.
fn edit_plane(
    path: &str,
    text: &str,
    values: &Map<String, Value>,
    picks_of: &crate::picks::Picks<'_>,
) -> Result<(SourceChange, String), EditFailure> {
    let id = values
        .get("sketch")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            picks(values, "selection").into_iter().find_map(|p| {
                let id = p
                    .strip_prefix("sketch:")
                    .or_else(|| p.strip_prefix("feature:"))?;
                picks_of
                    .document
                    .features
                    .iter()
                    .any(|f| f.id.0 == id && matches!(f.kind, FeatureKind::Sketch(_)))
                    .then(|| id.to_owned())
            })
        })
        .ok_or_else(|| model("select the sketch to move"))?;
    let pick = picks(values, "plane").into_iter().next().ok_or_else(|| {
        model(format!(
            "pick the plane or planar face to move {id} to. {}",
            crate::picks::PLANE_PICK_FORMS
        ))
    })?;
    let plane: PlaneRef = serde_json::from_value(picks_of.plane(&pick)?)
        .map_err(|e| model(format!("the sketch plane: {e}")))?;
    // A face the sketch's own feature or a later one made: the sketch can't sit on it.
    let at = picks_of.document.features.iter().position(|f| f.id.0 == id);
    if let (Some(at), PlaneRef::Face { face }) = (at, &plane) {
        let text = runmat_cad::syntax::refs::print_reference(face).unwrap_or_default();
        if let Some(later) = picks_of.document.features[at..]
            .iter()
            .find(|f| text.contains(&format!("feature.{}.", f.id.0)))
        {
            return Err(model(format!(
                "`{pick}` is made by `{}`, which comes after {id}; pick a plane or a face made before it",
                later.id
            )));
        }
    }
    let change = edit_part(text, path, |document| {
        let sketch = sketch_mut(document, &id)?;
        if sketch.plane == plane {
            return Err(model(format!("{id} is already on that plane")));
        }
        sketch.plane = plane.clone();
        Ok(())
    })?;
    Ok((change, format!("Moved {id} to {pick}")))
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
mod ellipse;
mod infer;
mod modify;
mod repeat;
mod text;
mod use_edges;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod ellipse_tests;

#[cfg(test)]
mod text_tests;

#[cfg(test)]
mod use_edges_tests;
