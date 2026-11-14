//! Component edges brought into an assembly sketch: `sketch.use-edges` (type Project) and
//! Offset of a component edge.
//!
//! Each picked edge is published from its component's part ([`Publisher`], the path mates use),
//! and the sketch gains an entity written `from = published(occurrence = <component>, name =
//! <published>)` with the edge's current projection onto the sketch plane. On every rebuild the
//! assembly re-reads the published edge, projects it again and holds the entity fixed
//! (`runmat_cad::assembly::projection`), so it follows the component when it moves and the edge
//! when the part changes.
//!
//! Offset of a component edge converts the edge as construction geometry, offsets that
//! entity with the part's Offset, and relates the copy to it: parallel at the distance for a
//! line, concentric at the offset radius for a circle. The copy follows the edge.

use super::anchors::{Anchor, Publisher};
use super::mate::resolve;
use super::{model, parse_pick, Compiled, Entity};
use crate::features::picks_of;
use runmat_cad::assembly::model::AssemblyDocument;
use runmat_cad::assembly::projection::{fit, in_plane};
use runmat_cad::ids::SketchEntityId;
use runmat_cad::model::datum::Frame;
use runmat_cad::model::sketch::{
    PointRole, Sketch, SketchConstraint, SketchConstraintKind, SketchEntity, SketchGeometry,
    SketchProjection, SketchTarget,
};
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

type Values = Map<String, Value>;

/// The sketch `values.sketch` names in `document`, and its id.
fn open_sketch<'d>(
    document: &'d mut AssemblyDocument,
    values: &Values,
) -> Result<(String, &'d mut Sketch), EditFailure> {
    let id = values
        .get("sketch")
        .and_then(Value::as_str)
        .ok_or_else(|| model("open a sketch first"))?
        .to_owned();
    match document
        .features
        .iter_mut()
        .find(|f| f.id.0 == id)
        .map(|f| &mut f.kind)
    {
        Some(runmat_cad::model::FeatureKind::Sketch(s)) => Ok((id, s)),
        Some(_) => Err(model(format!("`{id}` isn't a sketch"))),
        None => Err(model(format!("the assembly has no sketch `{id}`"))),
    }
}

/// The drawn frame of sketch `id` (assembly coordinates, metres).
fn sketch_frame(compiled: &Compiled, id: &str) -> Result<Frame, EditFailure> {
    let drawn = compiled.display["sketches"]
        .as_array()
        .and_then(|all| all.iter().find(|s| s["id"] == id))
        .ok_or_else(|| model(format!("`{id}` isn't drawn, so edges can't be projected onto it")))?;
    let v = |k: &str| {
        let a = &drawn["frame"][k];
        [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0))
    };
    Ok(Frame {
        origin: v("origin"),
        x: v("x"),
        y: v("y"),
        z: v("z"),
    })
}

/// An edge pick's drawn polyline in assembly coordinates (metres).
fn edge_points(compiled: &Compiled, uuid: &str, body: usize, edge: usize) -> Option<Vec<[f64; 3]>> {
    let (prototype, t) = compiled.placed(uuid)?;
    let e = compiled.body(prototype, body)?["edges"]
        .as_array()?
        .iter()
        .find(|e| e["index"].as_u64() == Some(edge as u64))?;
    let flat: Vec<f64> = e["points"].as_array()?.iter().filter_map(Value::as_f64).collect();
    Some(
        flat.chunks_exact(3)
            .map(|p| {
                if t.len() < 12 {
                    return [p[0], p[1], p[2]];
                }
                [
                    t[0] * p[0] + t[1] * p[1] + t[2] * p[2] + t[9],
                    t[3] * p[0] + t[4] * p[1] + t[5] * p[2] + t[10],
                    t[6] * p[0] + t[7] * p[1] + t[8] * p[2] + t[11],
                ]
            })
            .collect(),
    )
}

/// One picked component edge, published from its part: where it comes from and its current
/// projection onto `frame`.
fn convert(
    document: &AssemblyDocument,
    compiled: &Compiled,
    publisher: &mut Publisher,
    frame: &Frame,
    pick: &str,
) -> Result<(SketchProjection, SketchGeometry), EditFailure> {
    let Some(Entity::Edge { body, edge }) = parse_pick(pick).and_then(|p| p.entity) else {
        return Err(model(format!(
            "`{pick}` isn't a component edge; Project takes edges (pick each edge of a face's loop)"
        )));
    };
    let uuid = parse_pick(pick).map(|p| p.uuid).unwrap_or_default();
    let picked = resolve(document, compiled, pick, "the edge to convert")?;
    if !matches!(picked.anchor, Anchor::Entity { .. }) {
        return Err(model(format!(
            "the picked edge on `{}` has no stable name in its part to follow",
            picked.occurrence
        )));
    }
    let published = publisher
        .name_of(&picked.anchor)?
        .ok_or_else(|| model("the picked edge can't be published from its part"))?;
    let points = edge_points(compiled, &uuid, body, edge)
        .ok_or_else(|| model(format!("the picked edge on `{}` isn't drawn", picked.occurrence)))?;
    let local: Vec<[f64; 2]> = points.iter().map(|p| in_plane(frame, *p)).collect();
    let geometry = fit(&local).ok_or_else(|| {
        model(format!(
            "the picked edge on `{}` projects to a point on this sketch's plane",
            picked.occurrence
        ))
    })?;
    Ok((
        SketchProjection {
            occurrence: picked.occurrence.clone(),
            published: published.as_str().to_owned(),
        },
        geometry,
    ))
}

fn stem(geometry: &SketchGeometry) -> &'static str {
    match geometry {
        SketchGeometry::Line { .. } => "l",
        SketchGeometry::Circle { .. } => "c",
        SketchGeometry::Arc { .. } => "a",
        SketchGeometry::Spline { .. } => "sp",
        _ => "e",
    }
}

fn fresh(sketch: &Sketch, stem: &str) -> SketchEntityId {
    let id = (1..)
        .map(|n| format!("{stem}{n}"))
        .find(|id| !sketch.entities.iter().any(|e| e.id.0 == *id))
        .expect("an unused name");
    SketchEntityId::new(id)
}

/// Adds each picked edge to the open sketch as a converted entity; answers the new ids.
fn add_converted(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    publisher: &mut Publisher,
    values: &Values,
    picks: &[String],
    construction: bool,
) -> Result<(String, Vec<SketchEntityId>), EditFailure> {
    let id = open_sketch(document, values)?.0;
    let frame = sketch_frame(compiled, &id)?;
    let mut converted = Vec::new();
    for pick in picks {
        converted.push(convert(document, compiled, publisher, &frame, pick)?);
    }
    let sketch = open_sketch(document, values)?.1;
    let mut ids = Vec::new();
    for (projected, geometry) in converted {
        let entity = fresh(sketch, stem(&geometry));
        sketch.entities.push(SketchEntity {
            id: entity.clone(),
            geometry,
            construction,
            projected: Some(projected),
        });
        ids.push(entity);
    }
    Ok((id, ids))
}

/// `sketch.use-edges` on an assembly sketch.
pub(super) fn use_edges(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    publisher: &mut Publisher,
    values: &Values,
) -> Result<String, EditFailure> {
    match values.get("type").and_then(Value::as_str).unwrap_or("project") {
        "project" => {}
        other => {
            return Err(model(format!(
                "Use edges by {} isn't built yet; Project brings edges in",
                other.replace('_', " ")
            )))
        }
    }
    let picks = picks_of(values, "items");
    if picks.is_empty() {
        return Err(model("pick the component edges to bring into the sketch"));
    }
    let (sketch, ids) = add_converted(document, compiled, publisher, values, &picks, false)?;
    Ok(format!(
        "Projected {} into {sketch}",
        ids.iter().map(|e| e.0.as_str()).collect::<Vec<_>>().join(", ")
    ))
}

/// Whether `pick` is a component's entity (`occurrence:…/edge:…`) rather than a sketch curve.
pub(super) fn is_component_pick(pick: &str) -> bool {
    pick.starts_with("occurrence:")
}

/// `sketch.offset` with component edges among its picks: each edge converted as construction,
/// the shared Offset run on the converted entities, and each copy related to its edge so it
/// follows it.
pub(super) fn offset(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    publisher: &mut Publisher,
    values: &Values,
) -> Result<String, EditFailure> {
    let picks = picks_of(values, "entities");
    let (edges, curves): (Vec<String>, Vec<String>) =
        picks.into_iter().partition(|p| is_component_pick(p));
    let (sketch_id, converted) =
        add_converted(document, compiled, publisher, values, &edges, true)?;
    let mut values = values.clone();
    values.insert(
        "entities".into(),
        Value::from(
            curves
                .into_iter()
                .chain(converted.iter().map(|e| format!("sketch.{sketch_id}.{}", e.0)))
                .collect::<Vec<_>>(),
        ),
    );
    let distance = values.get("distance").and_then(Value::as_f64).unwrap_or(0.0);
    let sketch = open_sketch(document, &values)?.1;
    let before: Vec<SketchEntityId> = sketch.entities.iter().map(|e| e.id.clone()).collect();
    let summary = crate::edit::sketch::apply("sketch.offset", &values, sketch, &sketch_id)?;
    let made: Vec<SketchEntityId> = sketch
        .entities
        .iter()
        .map(|e| e.id.clone())
        .filter(|e| !before.contains(e))
        .collect();
    // The shared Offset makes one copy per picked curve, in pick order; the converted edges
    // were picked last.
    let copies = &made[made.len().saturating_sub(converted.len())..];
    for (source, copy) in converted.iter().zip(copies) {
        // The copy is profile geometry, though the edge it came from is construction.
        if let Some(e) = sketch.entities.iter_mut().find(|e| e.id == *copy) {
            e.construction = false;
        }
        relate(sketch, source, copy, distance);
    }
    Ok(summary)
}

fn entity(id: &SketchEntityId) -> SketchTarget {
    SketchTarget::Entity { entity: id.clone() }
}

/// The relations that keep an offset copy on its converted edge.
fn relate(sketch: &mut Sketch, source: &SketchEntityId, copy: &SketchEntityId, distance: f64) {
    let Some(geometry) = sketch.entities.iter().find(|e| e.id == *copy).map(|e| e.geometry.clone())
    else {
        return;
    };
    let mut push = |kind| sketch.constraints.push(SketchConstraint { id: None, kind });
    match geometry {
        SketchGeometry::Line { .. } => {
            push(SketchConstraintKind::Parallel {
                a: entity(source),
                b: entity(copy),
            });
            push(SketchConstraintKind::Distance {
                a: SketchTarget::Point {
                    entity: copy.clone(),
                    role: PointRole::Start,
                },
                b: entity(source),
                value: ValueExpr::mm(distance.abs()),
            });
        }
        SketchGeometry::Circle { radius, .. } => {
            push(SketchConstraintKind::Concentric {
                a: entity(source),
                b: entity(copy),
            });
            push(SketchConstraintKind::Radius {
                curve: entity(copy),
                value: radius,
            });
        }
        _ => {}
    }
}
