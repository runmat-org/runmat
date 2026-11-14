//! Sketches in an assembly: `assembly.sketch` starts one on an origin plane of the assembly or
//! on a planar face of a placed component, and the `sketch.*` tools draw in the open one through
//! the part's sketch tools ([`crate::edit::sketch::apply`]), the one implementation both
//! editors share.
//!
//! A face pick becomes a mate connector on its component, anchored to the face the component's
//! part publishes ([`super::mate::connector`]), and the sketch is written `on plane.<connector>`:
//! it follows the component when it moves and the face when the part changes.

use super::anchors::Publisher;
use super::frames::FrameKind;
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
