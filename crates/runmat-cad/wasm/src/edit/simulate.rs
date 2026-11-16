//! Simulation preparation on a part, run from the canvas.
//!
//! - `simulate.region`: a named region (`region mount : face = { select = [ … ], expect = … }`)
//!   in the part's own source. The name travels with the geometry: a study that reads the part
//!   (`file = "./bracket.part"`) selects the region's faces with `select name "mount"`. No other
//!   file holds the name.
//! - `simulate.defeature` and `simulate.representation`: an analysis representation
//!   (`representation simple : defeatured = { max_fillet = 2 mm, … }`, `… : midsurface = { body =
//!   … }`, `… : beam = { … }`) in the part's source, built at once ([`crate::representation`]) so
//!   one that can't be made (a beam of a body that isn't prismatic) is refused and nothing is
//!   written. The design's features stay as they are. `full` writes nothing: the Desktop draws
//!   the design itself again.

use super::model;
use crate::features::picks_of;
use crate::picks::Picks;
use runmat_cad::ids::RegionName;
use runmat_cad::model::items::{Cardinality, Region};
use runmat_cad::refs::{Reference, TopoKind};
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use serde_json::{Map, Value};

pub(super) fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn text<'a>(values: &'a Map<String, Value>, key: &str) -> &'a str {
    values.get(key).and_then(Value::as_str).map(str::trim).unwrap_or("")
}

/// `simulate.region`: adds a region of the picked faces, edges, vertices or bodies. Every pick
/// must be of the chosen kind; the region's name must be new to the part.
pub fn region(
    path: &str,
    text_in: &str,
    values: &Map<String, Value>,
    selection: &[String],
    picks: &Picks,
) -> Result<(SourceChange, String), EditFailure> {
    let name = text(values, "name");
    if name.is_empty() {
        return Err(model("name the region"));
    }
    if !valid_name(name) {
        return Err(model(format!(
            "`{name}` isn't a region name: use letters, digits and `_`, starting with a letter"
        )));
    }
    let kind = match text(values, "kind") {
        "" | "face" => TopoKind::Face,
        "edge" => TopoKind::Edge,
        "body" => TopoKind::Body,
        "vertex" => TopoKind::Vertex,
        other => return Err(model(format!("a region holds no `{other}`"))),
    };
    let expect = match text(values, "expect") {
        "" | "at_least_one" => Cardinality::AtLeastOne,
        "exactly_one" => Cardinality::ExactlyOne,
        "any" => Cardinality::Any,
        other => return Err(model(format!("`{other}` isn't an expected count"))),
    };
    let mut ids = picks_of(values, "select");
    if ids.is_empty() {
        ids = selection.to_vec();
    }
    if ids.is_empty() {
        return Err(model(format!("pick the {kind}s the region holds")));
    }
    let mut select: Vec<Reference> = Vec::new();
    for id in &ids {
        let reference: Reference = serde_json::from_value(picks.reference(id)?)
            .map_err(|e| model(format!("`{id}`: {e}")))?;
        if let Some(found) = reference.kind().filter(|k| *k != kind) {
            return Err(model(format!(
                "`{id}` is a {found}; this region holds {kind}s"
            )));
        }
        if !select.contains(&reference) {
            select.push(reference);
        }
    }
    if expect == Cardinality::ExactlyOne && select.len() != 1 {
        return Err(model(format!(
            "the region expects exactly one {kind} and {} are picked",
            select.len()
        )));
    }
    let count = select.len();
    let change = edit_part(text_in, path, |document| {
        if document.regions.iter().any(|r| r.name.as_str() == name) {
            return Err(model(format!("the part already has a region `{name}`")));
        }
        document.regions.push(Region {
            name: RegionName::new(name),
            kind,
            select: select.clone(),
            expect,
            label: None,
            body: None,
        });
        Ok(())
    })?;
    let plural = if count == 1 { "" } else { "s" };
    Ok((change, format!("Region {name}: {count} {kind}{plural}")))
}

/// `simulate.defeature` (`kind` is `defeatured`) and `simulate.representation`.
pub fn representation(
    path: &str,
    text_in: &str,
    values: &Map<String, Value>,
    picks: &Picks,
    kind: &str,
) -> Result<(SourceChange, String), EditFailure> {
    use runmat_cad::assembly::model::Representation;
    use runmat_cad::model::PartRepresentation;
    let kind = match kind {
        "" | "defeatured" => Representation::Defeatured,
        "midsurface" => Representation::Midsurface,
        "beam" => Representation::Beam,
        other => return Err(model(format!("a part's representation is full, defeatured, midsurface or beam, not `{other}`"))),
    };
    let name = text(values, "name");
    if !valid_name(name) {
        return Err(model(if name.is_empty() {
            "name the representation".to_owned()
        } else {
            format!("`{name}` isn't a representation name: use letters, digits and `_`, starting with a letter")
        }));
    }
    let length = |key: &str| -> Result<Option<runmat_cad::value::ValueExpr>, EditFailure> {
        match values.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) if s.trim().is_empty() => Ok(None),
            Some(_) => {
                let v = crate::features::value(values, key, "mm")?;
                serde_json::from_value(v)
                    .map(Some)
                    .map_err(|e| model(format!("`{key}`: {e}")))
            }
        }
    };
    let body = match picks_of(values, "body").first() {
        Some(id) if kind != Representation::Defeatured => Some(
            serde_json::from_value::<Reference>(picks.reference(id)?)
                .map_err(|e| model(format!("`{id}`: {e}")))?,
        ),
        _ => None,
    };
    let entry = if kind == Representation::Defeatured {
        PartRepresentation {
            name: name.to_owned(),
            kind,
            body: None,
            max_fillet: length("max_fillet")?,
            max_hole: length("max_hole")?,
            max_chamfer: length("max_chamfer")?,
            keep_regions: values.get("keep_regions").and_then(Value::as_bool).unwrap_or(true),
        }
    } else {
        PartRepresentation {
            name: name.to_owned(),
            kind,
            body,
            max_fillet: None,
            max_hole: None,
            max_chamfer: None,
            keep_regions: true,
        }
    };
    let change = edit_part(text_in, path, |document| {
        if document.representations.iter().any(|r| r.name == name) {
            return Err(model(format!("the part already has a representation `{name}`")));
        }
        document.representations.push(entry.clone());
        Ok(())
    })?;
    // Built now: one that can't be made is refused, and the summary says what it is.
    let tessellation = runmat_cad::regen::kernel::TessellationOptions::default();
    let full = crate::runtime::regenerate_part_in(path, &change.source, None, &tessellation)
        .map_err(|d| model(d.first().map(|d| d.message.clone()).unwrap_or_default()))?;
    let built = crate::representation::build(path, &change.source, None, &full, name, &tessellation)
        .map_err(model)?;
    Ok((change, built.summary()))
}

#[cfg(test)]
mod tests {
    use crate::edit::edit_json;
    use crate::runtime::Files;
    use serde_json::{json, Value};

    const PUCK: &str = "runmat 0.7\nstd 1\n\npart puck {\n    sketch s on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = 7 mm)\n    }\n    feature puck = extrude(profile = sketch.s, distance = 5 mm)\n    output puck = { body = feature.puck.body }\n}\n";

    fn run(text: &str, values: Value) -> Value {
        let request = json!({ "command": "simulate.region", "values": values });
        serde_json::from_str(&edit_json("/p/puck.part", text, &Files::new(), &request.to_string()))
            .unwrap()
    }

    #[test]
    fn a_region_is_written_into_the_part_and_listed_with_its_faces() {
        let out = run(
            PUCK,
            json!({ "name": "mount", "kind": "face", "select": ["feature.puck.face.start"], "expect": "exactly_one" }),
        );
        assert_eq!(out["ok"], true, "{out}");
        assert_eq!(out["summary"], "Region mount: 1 face");
        let text = out["files"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("region mount : face = { select = [feature.puck.face.start], expect = exactly_one }"),
            "{text}"
        );
        // The Outline's Regions folder lists it, holding the bottom face.
        let compiled: Value = serde_json::from_str(&crate::runtime::compile_json(
            "/p/puck.part",
            text,
            &Files::new(),
        ))
        .unwrap();
        let regions = compiled["projection"]["data"]["index"]["regions"].as_array().unwrap();
        assert!(regions.iter().any(|r| r["name"] == "mount"), "{regions:?}");
        let faces = compiled["display"]["data"]["bodies"][0]["faces"].as_array().unwrap();
        let held: Vec<&Value> = faces.iter().filter(|f| f["regions"] == json!(["mount"])).collect();
        assert_eq!(held.len(), 1, "one face carries the region");
    }

    #[test]
    fn a_region_refuses_a_pick_of_another_kind_a_taken_name_and_a_wrong_count() {
        let out = run(PUCK, json!({ "name": "mount", "kind": "edge", "select": ["feature.puck.face.start"] }));
        assert_eq!(out["ok"], false);
        assert!(out["message"].as_str().unwrap().contains("is a face; this region holds edges"), "{out}");
        let out = run(
            PUCK,
            json!({ "name": "ends", "kind": "face", "select": ["feature.puck.face.start", "feature.puck.face.end"], "expect": "exactly_one" }),
        );
        assert!(out["message"].as_str().unwrap().contains("exactly one face and 2 are picked"), "{out}");
        let once = run(PUCK, json!({ "name": "mount", "select": ["feature.puck.face.start"] }));
        let text = once["files"][0]["text"].as_str().unwrap();
        let again = run(text, json!({ "name": "mount", "select": ["feature.puck.face.end"] }));
        assert!(again["message"].as_str().unwrap().contains("already has a region `mount`"), "{again}");
        let out = run(PUCK, json!({ "name": "2nd", "select": ["feature.puck.face.end"] }));
        assert!(out["message"].as_str().unwrap().contains("isn't a region name"), "{out}");
        let out = run(PUCK, json!({ "name": "body", "kind": "body", "select": ["feature:puck"] }));
        assert_eq!(out["ok"], true, "a body pick names the feature's body: {out}");
    }
}
