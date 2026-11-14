//! Sketches in an assembly: `assembly.sketch` starts one on an origin plane of the assembly or
//! on a planar face of a placed component, and the `sketch.*` tools draw in the open one through
//! the part's sketch tools ([`crate::edit::sketch::apply`]), the one implementation both
//! editors share.
//!
//! A face pick becomes a mate connector on its component, anchored to the face the component's
//! part publishes ([`super::mate::connector`]), and the sketch is written `on plane.<connector>`:
//! it follows the component when it moves and the face when the part changes.

use super::anchors::Publisher;
use super::frames::{add, scale, sub, unit, Frame, FrameKind, V3};
use super::{mate, model, parse_pick, Compiled};
use crate::features::picks_of;
use runmat_cad::assembly::model::AssemblyDocument;
use runmat_cad::model::datum::PlaneRef;
use runmat_cad::model::sketch::Sketch;
use runmat_cad::model::{Feature, FeatureKind};
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{Map, Value};

/// The plane forms `assembly.sketch` takes, for its refusal.
const PLANE_FORMS: &str = "Pick an origin plane of the assembly (`origin.xy`, `origin.yz`, `origin.zx`; the Outline's `origin:top`, `origin:front`, `origin:right`), one of its connectors (`connector:<name>`) or a planar face of a component.";

fn origin_plane(pick: &str) -> Option<PlaneRef> {
    let word = match pick {
        "origin.xy" | "origin:top" => "xy",
        "origin.zx" | "origin:front" => "zx",
        "origin.yz" | "origin:right" => "yz",
        _ => return None,
    };
    serde_json::from_value(
        serde_json::json!({ "on": "datum", "datum": { "datum": "origin_plane", "plane": word } }),
    )
    .ok()
}

/// `assembly.sketch`: an empty sketch on the picked plane, after the assembly's other sketches.
pub(super) fn new_sketch(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    publisher: &mut Publisher,
    values: &Map<String, Value>,
) -> Result<String, EditFailure> {
    let pick = picks_of(values, "plane")
        .into_iter()
        .next()
        .ok_or_else(|| {
            model(format!(
                "pick a plane or a planar face to sketch on. {PLANE_FORMS}"
            ))
        })?;
    let plane = if let Some(plane) = origin_plane(&pick) {
        plane
    } else if let Some(name) = pick.strip_prefix("connector:") {
        if document.connector(name).is_none() {
            return Err(model(format!("the assembly has no connector `{name}`")));
        }
        PlaneRef::datum(name)
    } else if parse_pick(&pick).is_some_and(|p| p.entity.is_some()) {
        let picked = mate::resolve(document, compiled, &pick, "the sketch plane")?;
        if picked
            .frame
            .as_ref()
            .is_some_and(|f| !matches!(f.kind, FrameKind::Plane | FrameKind::Frame))
        {
            return Err(model("pick a planar face to sketch on"));
        }
        PlaneRef::datum(&mate::connector(document, publisher, &picked, false)?)
    } else {
        return Err(model(format!(
            "`{pick}` isn't a plane or a face. {PLANE_FORMS}"
        )));
    };
    let id = (1..)
        .map(|n| format!("sketch{n}"))
        .find(|id| !document.features.iter().any(|f| f.id.0 == *id))
        .expect("an unused name");
    document.features.push(Feature::new(
        &id,
        FeatureKind::Sketch(Sketch {
            plane,
            entities: Vec::new(),
            constraints: Vec::new(),
            solved: Default::default(),
        }),
    ));
    Ok(format!("Added {id}"))
}

/// A `sketch.*` tool on the assembly's open sketch (`values.sketch`).
pub(super) fn tool(
    document: &mut AssemblyDocument,
    command: &str,
    values: &Map<String, Value>,
) -> Result<String, EditFailure> {
    let id = values
        .get("sketch")
        .and_then(Value::as_str)
        .ok_or_else(|| model("open a sketch first"))?
        .to_owned();
    let sketch = match document
        .features
        .iter_mut()
        .find(|f| f.id.0 == id)
        .map(|f| &mut f.kind)
    {
        Some(FeatureKind::Sketch(s)) => s,
        Some(_) => return Err(model(format!("`{id}` isn't a sketch"))),
        None => return Err(model(format!("the assembly has no sketch `{id}`"))),
    };
    crate::edit::sketch::apply(command, values, sketch, &id)
}

/// The sketches among the selection (`feature:<id>` rows of the Outline).
pub(super) fn selected(request: &super::EditRequest) -> impl Iterator<Item = &str> {
    request
        .selection
        .iter()
        .filter_map(|id| id.strip_prefix("feature:"))
}

/// `assembly.delete` and `assembly.suppress` on selected sketches.
pub(super) fn delete_or_suppress(
    document: &mut AssemblyDocument,
    request: &super::EditRequest,
) -> Result<String, EditFailure> {
    let ids: Vec<String> = selected(request).map(str::to_owned).collect();
    for id in &ids {
        let at = document
            .features
            .iter()
            .position(|f| f.id.0 == *id)
            .ok_or_else(|| model(format!("the assembly has no sketch `{id}`")))?;
        if request.command == "assembly.delete" {
            document.features.remove(at);
        } else {
            document.features[at].suppressed = true;
        }
    }
    let verb = if request.command == "assembly.delete" {
        "Deleted"
    } else {
        "Suppressed"
    };
    Ok(format!("{verb} {}", ids.join(", ")))
}

/// A sketch point or line picked for a mate (`sketch.<sketch>.<entity>[.<role>]`): the sketch,
/// the entity, the role and the picked frame in assembly coordinates, from the drawn sketch.
/// A point (or a point entity, or a circle's or arc's center) is a point; a line is an axis
/// along it from its start, as the resolver reads it
/// ([`runmat_cad::assembly::sketches::sketch_point_frame`]).
pub(super) struct LayoutPick {
    pub sketch: String,
    pub entity: String,
    pub role: Option<String>,
    pub frame: Frame,
}

pub(super) fn layout_pick(
    document: &AssemblyDocument,
    compiled: &Compiled,
    pick: &str,
) -> Option<Result<LayoutPick, EditFailure>> {
    let rest = pick.strip_prefix("sketch.")?;
    let mut parts = rest.split('.');
    let (Some(sketch), Some(entity)) = (parts.next(), parts.next()) else {
        return Some(Err(model(format!("`{pick}` isn't a sketch point or line"))));
    };
    let role = parts.next().map(str::to_owned);
    Some((|| {
        if !document
            .features
            .iter()
            .any(|f| f.id.0 == sketch && matches!(f.kind, FeatureKind::Sketch(_)))
        {
            return Err(model(format!("the assembly has no sketch `{sketch}`")));
        }
        let drawn = compiled.display()["sketches"]
            .as_array()
            .and_then(|a| a.iter().find(|s| s["id"] == sketch))
            .ok_or_else(|| model(format!("the sketch `{sketch}` isn't drawn, so it can't be mated")))?;
        let e = drawn["entities"]
            .as_array()
            .and_then(|a| a.iter().find(|e| e["id"] == entity))
            .ok_or_else(|| model(format!("the sketch `{sketch}` has no `{entity}`")))?;
        let v3 = |v: &Value| -> Option<V3> {
            Some([v[0].as_f64()?, v[1].as_f64()?, v[2].as_f64()?])
        };
        let f = &drawn["frame"];
        let (o, fx, fy, fz) = (
            v3(&f["origin"]),
            v3(&f["x"]),
            v3(&f["y"]),
            v3(&f["z"]),
        );
        let (Some(o), Some(fx), Some(fy), Some(fz)) = (o, fx, fy, fz) else {
            return Err(model(format!("the sketch `{sketch}` has no frame")));
        };
        let point = |r: &str| -> Result<V3, EditFailure> {
            let p = &e["points"][r];
            match (p[0].as_f64(), p[1].as_f64()) {
                (Some(u), Some(v)) => Ok(add(o, add(scale(fx, u), scale(fy, v)))),
                _ => Err(model(format!("`{entity}` has no point `{r}`"))),
            }
        };
        let at = |origin: V3| Frame {
            kind: FrameKind::Point,
            origin,
            z: fz,
            x: fx,
            radius: None,
        };
        let kind = e["kind"].as_str().unwrap_or_default();
        let frame = match (role.as_deref(), kind) {
            (Some(r), _) => at(point(r)?),
            (None, "point") => at(point("at")?),
            (None, "circle" | "arc" | "ellipse") => Frame {
                kind: FrameKind::Axis,
                radius: e["radius"].as_f64(),
                ..at(point("center")?)
            },
            (None, "line") => {
                let (a, b) = (point("start")?, point("end")?);
                let z = unit(sub(b, a));
                Frame {
                    kind: FrameKind::Axis,
                    origin: a,
                    z,
                    // The sketch's normal, square to a line in the sketch.
                    x: fz,
                    radius: None,
                }
            }
            _ => {
                return Err(model(format!(
                    "pick a point of `{entity}`, or a sketch point, line, circle or arc"
                )))
            }
        };
        Ok(LayoutPick {
            sketch: sketch.to_owned(),
            entity: entity.to_owned(),
            role,
            frame,
        })
    })())
}

/// Refuses a mate that would move the component sketch `sketch` sits on (`moving`), or that
/// reads a sketch on a component mates already move: the component would follow its own
/// sketch.
pub(super) fn refuse_cycle(
    document: &AssemblyDocument,
    sketch: &str,
    moving: &str,
) -> Result<(), EditFailure> {
    let Some(FeatureKind::Sketch(s)) = document
        .features
        .iter()
        .find(|f| f.id.0 == sketch)
        .map(|f| &f.kind)
    else {
        return Err(model(format!("the assembly has no sketch `{sketch}`")));
    };
    if let PlaneRef::Datum {
        datum: runmat_cad::model::datum::DatumRef::Feature { feature },
    } = &s.plane
    {
        let on = document
            .connector(&feature.0)
            .and_then(|c| c.occurrence.0.first())
            .map(|seg| seg.name.as_str());
        if on == Some(moving) {
            return Err(model(format!(
                "the sketch `{sketch}` sits on `{moving}`, which this mate moves: a component can't follow its own sketch. Sketch on an origin plane or a fixed component"
            )));
        }
    }
    if let Some(problem) = runmat_cad::assembly::sketches::moves_with_mates(document, &s.plane) {
        return Err(model(format!(
            "the sketch `{sketch}` {problem}, so it can't place mates: the component would follow its own sketch"
        )));
    }
    Ok(())
}
