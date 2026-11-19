//! Offset of model edges, shared by part and assembly sketches.
//!
//! The caller converts each picked model edge (a part's own edge, or a component's edge in an
//! assembly) into the sketch's projection of it. This module adds each converted edge as
//! construction geometry, runs the sketch's Offset on those entities with the picked sketch
//! curves, and relates each copy to its edge so it follows the edge: parallel at the distance
//! for a line, concentric at the offset radius for a circle.

use super::super::{model, EditFailure, SourceChange};
use runmat_cad::ids::SketchEntityId;
use runmat_cad::model::sketch::{
    PointRole, Sketch, SketchConstraint, SketchConstraintKind, SketchEntity, SketchGeometry,
    SketchProjection, SketchTarget,
};
use runmat_cad::syntax::edit::edit_part;
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

/// One converted model edge: where it comes from and its current projection.
pub(crate) type Converted = (SketchProjection, SketchGeometry);

/// The id stem of a converted entity (`l`, `c`, `a`, `sp`).
pub(crate) fn stem(geometry: &SketchGeometry) -> &'static str {
    match geometry {
        SketchGeometry::Line { .. } => "l",
        SketchGeometry::Circle { .. } => "c",
        SketchGeometry::Arc { .. } => "a",
        SketchGeometry::Spline { .. } => "sp",
        _ => "e",
    }
}

/// The next unused entity id with `stem` in `sketch`.
pub(crate) fn fresh(sketch: &Sketch, stem: &str) -> SketchEntityId {
    let id = (1..)
        .map(|n| format!("{stem}{n}"))
        .find(|id| !sketch.entities.iter().any(|e| e.id.0 == *id))
        .expect("an unused name");
    SketchEntityId::new(id)
}

/// Adds each converted edge to `sketch`; answers the new ids in order.
pub(crate) fn add(
    sketch: &mut Sketch,
    converted: Vec<Converted>,
    construction: bool,
) -> Vec<SketchEntityId> {
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
    ids
}

/// `sketch.offset` of the picked sketch `curves` and the converted model `edges` in `sketch`
/// (`sketch_id`). Each edge is added as construction, the shared Offset runs on it, and the
/// copy is held to it.
pub(crate) fn offset(
    values: &Map<String, Value>,
    sketch: &mut Sketch,
    sketch_id: &str,
    curves: Vec<String>,
    edges: Vec<Converted>,
) -> Result<String, EditFailure> {
    if curves.is_empty() && edges.is_empty() {
        return Err(model("pick the curves to offset"));
    }
    let converted = add(sketch, edges, true);
    let mut values = values.clone();
    values.insert(
        "entities".into(),
        Value::from(
            curves
                .into_iter()
                .chain(
                    converted
                        .iter()
                        .map(|e| format!("sketch.{sketch_id}.{}", e.0)),
                )
                .collect::<Vec<_>>(),
        ),
    );
    let distance = values
        .get("distance")
        .and_then(super::num)
        .unwrap_or(0.0);
    let before: Vec<SketchEntityId> = sketch.entities.iter().map(|e| e.id.clone()).collect();
    let summary = super::apply("sketch.offset", &values, sketch, sketch_id)?;
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

/// `sketch.offset` in a part sketch when the picks include the part's own edges: each edge is
/// projected from the regenerated part, then offset with [`offset`]. `None` when no pick is
/// a model edge.
pub(super) fn in_part(
    path: &str,
    text: &str,
    sketch_id: &str,
    values: &Map<String, Value>,
) -> Option<Result<(SourceChange, String), EditFailure>> {
    let (edges, curves): (Vec<String>, Vec<String>) = super::picks(values, "entities")
        .into_iter()
        .partition(|p| super::use_edges::is_model_edge(p));
    if edges.is_empty() {
        return None;
    }
    let converted = match super::use_edges::project_edges(path, text, sketch_id, &edges) {
        Ok(converted) => converted,
        Err(e) => return Some(Err(e)),
    };
    let mut summary = String::new();
    let result = edit_part(text, path, |document| {
        let sketch = super::sketch_mut(document, sketch_id)?;
        summary = offset(values, sketch, sketch_id, curves.clone(), converted.clone())?;
        Ok(())
    });
    Some(result.map(|change| (change, summary)))
}

fn entity(id: &SketchEntityId) -> SketchTarget {
    SketchTarget::Entity { entity: id.clone() }
}

/// The relations that keep an offset copy on its converted edge.
fn relate(sketch: &mut Sketch, source: &SketchEntityId, copy: &SketchEntityId, distance: f64) {
    let Some(geometry) = sketch
        .entities
        .iter()
        .find(|e| e.id == *copy)
        .map(|e| e.geometry.clone())
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
