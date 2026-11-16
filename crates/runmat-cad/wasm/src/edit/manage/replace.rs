//! `manage.replace-reference` on a part: points the features that use a face, edge, vertex,
//! body, datum plane or sketch at another one, then rebuilds and reports what broke.
//!
//! Values: `reference` (what the features use now), `with` (what they should use) and `scope`:
//! `all` replaces every use, `one` only the uses in the feature `feature` names. Both picks are
//! lowered through the part grammar ([`Picks`]), so the written reference is the canonical
//! persistent name the solver matches, the same value its source text would give. A stored use
//! matches when it is that value, or a reference with a geometric fallback whose primary is that
//! value (the stale fallback is dropped with it). Uses inside an edge set
//! (`edges_between(face, …)`) or a piece count too: the set then follows the new face.
//!
//! The summary lists every feature that used the reference, which of them changed, and each
//! error the rebuild reports that the part did not have before.

use super::super::{model, EditRequest};
use crate::features::picks_of;
use crate::picks::{topo, Picks};
use runmat_cad::diagnostics::{Cause, Severity, Subject};
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use serde_json::Value;

/// What the old and new picks are, as the model stores them.
enum Target {
    /// A face, edge, vertex or body reference (`{"ref": …}`).
    Reference(Value, Value),
    /// A datum or origin plane (`{"on": "datum", …}`).
    Plane(Value, Value),
    /// A sketch, named by profiles and sketch-line axes (`sketch = "s1"`).
    Sketch(String, String),
}

fn only(request: &EditRequest, field: &str, what: &str) -> Result<String, EditFailure> {
    let mut found = picks_of(&request.values, field);
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(model(format!("pick the {what}"))),
        _ => Err(model(format!("pick one {what}, not {}", found.len()))),
    }
}

fn target(picks: &Picks, old: &str, new: &str) -> Result<Target, EditFailure> {
    if let Ok(sketch) = picks.sketch(old) {
        let with = picks.sketch(new).map_err(|_| {
            model(format!(
                "`{old}` is a sketch, so replace it with another sketch"
            ))
        })?;
        return Ok(Target::Sketch(sketch, with));
    }
    if let Ok(plane) = picks.plane(old) {
        if plane["on"] != "face" {
            let with = picks.plane(new)?;
            return Ok(Target::Plane(plane, with));
        }
    }
    let was = picks.reference(old)?;
    let with = picks.reference(new)?;
    let (a, b) = (topo(&was), topo(&with));
    if a != b {
        return Err(model(format!(
            "`{old}` is a {a} and `{new}` is a {b}; replace a {a} with another {a}"
        )));
    }
    Ok(Target::Reference(was, with))
}

/// Replaces each use of the target in `value` (a feature's parameters as JSON), recording the
/// parameter path of each use. Sketch names are not looked for inside topological references:
/// a generated face name's source tag names its sketch, and that tag is the name's identity.
fn replace(value: &mut Value, target: &Target, at: &str, uses: &mut Vec<String>) {
    match target {
        Target::Reference(was, with) | Target::Plane(was, with) => {
            let fallback_of = value["ref"] == "with_fallback" && value["primary"] == *was;
            if *value == *was || fallback_of {
                *value = with.clone();
                uses.push(at.to_owned());
                return;
            }
        }
        Target::Sketch(was, with) => {
            if value.get("ref").is_some() {
                return;
            }
            if let Some(map) = value.as_object_mut() {
                if map.get("sketch").and_then(Value::as_str) == Some(was.as_str()) {
                    map.insert("sketch".into(), Value::String(with.clone()));
                    // A profile's loops name the old sketch's curves: take every closed loop of
                    // the new sketch instead.
                    map.remove("loops");
                    uses.push(if at.is_empty() {
                        "sketch".into()
                    } else {
                        at.to_owned()
                    });
                }
            }
        }
    }
    match value {
        Value::Object(map) => {
            for (key, inner) in map.iter_mut() {
                let path = if at.is_empty() {
                    key.clone()
                } else {
                    format!("{at}.{key}")
                };
                replace(inner, target, &path, uses);
            }
        }
        Value::Array(items) => {
            for (i, inner) in items.iter_mut().enumerate() {
                replace(inner, target, &format!("{at}[{i}]"), uses);
            }
        }
        _ => {}
    }
}

/// Errors (not their cascades) of the part's rebuild, as `feature: message`.
fn errors(path: &str, text: &str) -> Vec<String> {
    let Ok(part) = crate::runtime::regenerate_part(path, text) else {
        return vec!["the part no longer parses".into()];
    };
    part.result
        .diagnostics
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error && !matches!(d.cause, Cause::Cascade { .. }))
        .map(|d| match &d.site.subject {
            Subject::Feature { feature } => format!("{}: {}", feature.0, d.message),
            _ => d.message.clone(),
        })
        .collect()
}

/// The pick as the summary names it (`feature.base.face.end`, `sketch s2`).
fn shown(pick: &str) -> String {
    for (prefix, word) in [("sketch:", "sketch "), ("feature:", ""), ("body:", "body ")] {
        if let Some(rest) = pick.strip_prefix(prefix) {
            return format!("{word}{rest}");
        }
    }
    pick.to_owned()
}

pub(in crate::edit) fn replace_reference(
    path: &str,
    text: &str,
    request: &EditRequest,
    picks: &Picks,
) -> Result<(SourceChange, String), EditFailure> {
    let old = only(request, "reference", "reference to replace")?;
    let new = only(request, "with", "reference to use instead")?;
    if old == new {
        return Err(model("pick a different reference to use instead"));
    }
    let target = target(picks, &old, &new)?;
    let one = request.values.get("scope").and_then(Value::as_str) == Some("one");
    let only_in = if one {
        let pick = only(request, "feature", "feature whose use to replace")?;
        Some(
            pick.strip_prefix("feature:")
                .or_else(|| pick.strip_prefix("sketch:"))
                .unwrap_or(&pick)
                .to_owned(),
        )
    } else {
        None
    };

    // Every use, feature by feature in history order, without changing anything yet.
    let mut users: Vec<(String, Vec<String>)> = Vec::new();
    for feature in &picks.document.features {
        let mut kind = serde_json::to_value(&feature.kind).map_err(|e| model(e.to_string()))?;
        let mut uses = Vec::new();
        replace(&mut kind, &target, "", &mut uses);
        if !uses.is_empty() {
            users.push((feature.id.0.clone(), uses));
        }
    }
    if users.is_empty() {
        return Err(model(format!("no feature uses {}", shown(&old))));
    }
    if let Some(id) = &only_in {
        if !users.iter().any(|(f, _)| f == id) {
            let named: Vec<&str> = users.iter().map(|(f, _)| f.as_str()).collect();
            return Err(model(format!(
                "{id} doesn't use {}; it is used by {}",
                shown(&old),
                named.join(", ")
            )));
        }
    }
    let changed: Vec<&(String, Vec<String>)> = users
        .iter()
        .filter(|(f, _)| only_in.as_ref().is_none_or(|id| id == f))
        .collect();

    let before = errors(path, text);
    let change = edit_part(text, path, |document| {
        for (id, _) in &changed {
            let feature = document
                .feature_mut(id)
                .ok_or_else(|| model(format!("the part has no feature `{id}`")))?;
            let mut kind = serde_json::to_value(&feature.kind).map_err(|e| model(e.to_string()))?;
            replace(&mut kind, &target, "", &mut Vec::new());
            feature.kind = serde_json::from_value(kind)
                .map_err(|e| model(format!("`{id}` with the new reference: {e}")))?;
        }
        Ok(())
    })?;

    let list = |items: &[&(String, Vec<String>)]| -> String {
        items
            .iter()
            .map(|(f, uses)| format!("{f} ({})", uses.join(", ")))
            .collect::<Vec<_>>()
            .join("; ")
    };
    let mut summary = format!(
        "Replaced {} with {} in {} of {} {}: {}",
        shown(&old),
        shown(&new),
        changed.len(),
        users.len(),
        if users.len() == 1 {
            "feature that used it"
        } else {
            "features that used it"
        },
        list(&changed)
    );
    let kept: Vec<&(String, Vec<String>)> = users
        .iter()
        .filter(|(f, _)| only_in.as_ref().is_some_and(|id| id != f))
        .collect();
    if !kept.is_empty() {
        summary.push_str(&format!(". Still on {}: {}", shown(&old), list(&kept)));
    }
    let broke: Vec<String> = errors(path, &change.source)
        .into_iter()
        .filter(|e| !before.contains(e))
        .collect();
    if broke.is_empty() {
        summary.push_str(". The part rebuilt without new errors");
    } else {
        summary.push_str(&format!(". After the rebuild: {}", broke.join("; ")));
    }
    Ok((change, summary))
}

#[cfg(test)]
mod tests {
    use crate::edit::edit_json;
    use crate::runtime::Files;
    use serde_json::{json, Value};

    const PATH: &str = "/p/plate.part";

    /// A plate, a sketch and a boss on its top face, a chamfer on a top edge, a second sketch
    /// on the origin plane, and a cube made after everything else.
    const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    sketch s2 on feature.body.face.end {\n        m1 = line(start = [5 mm, 5 mm], end = [10 mm, 5 mm])\n        m2 = line(start = [10 mm, 5 mm], end = [10 mm, 10 mm])\n        m3 = line(start = [10 mm, 10 mm], end = [5 mm, 10 mm])\n        m4 = line(start = [5 mm, 10 mm], end = [5 mm, 5 mm])\n    }\n    feature boss = extrude(profile = sketch.s2, distance = 2 mm, mode = add)\n    feature break = chamfer(edges = [edges_between(feature.body.face.end, feature.body.face.side(s1.l3))], distance = 1 mm)\n    sketch s3 on origin.xy {\n        c1 = line(start = [15 mm, 5 mm], end = [20 mm, 5 mm])\n        c2 = line(start = [20 mm, 5 mm], end = [20 mm, 10 mm])\n        c3 = line(start = [20 mm, 10 mm], end = [15 mm, 10 mm])\n        c4 = line(start = [15 mm, 10 mm], end = [15 mm, 5 mm])\n    }\n    sketch s4 on origin.xy {\n        k1 = line(start = [60 mm, 0 mm], end = [65 mm, 0 mm])\n        k2 = line(start = [65 mm, 0 mm], end = [65 mm, 5 mm])\n        k3 = line(start = [65 mm, 5 mm], end = [60 mm, 5 mm])\n        k4 = line(start = [60 mm, 5 mm], end = [60 mm, 0 mm])\n    }\n    feature cube = extrude(profile = sketch.s4, distance = 5 mm, mode = new)\n}\n";

    fn run(values: Value) -> Value {
        let request = json!({ "command": "manage.replace-reference", "values": values });
        serde_json::from_str(&edit_json(PATH, PLATE, &Files::new(), &request.to_string())).unwrap()
    }

    fn applied(values: Value) -> (String, String) {
        let out = run(values);
        assert_eq!(out["ok"], true, "{out}");
        (
            out["files"][0]["text"].as_str().unwrap().to_owned(),
            out["summary"].as_str().unwrap().to_owned(),
        )
    }

    #[test]
    fn every_use_of_a_face_follows_the_new_face_and_the_summary_lists_them() {
        let (text, summary) = applied(json!({
            "reference": "feature.body.face.end", "with": "feature.body.face.start", "scope": "all"
        }));
        assert!(
            text.contains("sketch s2 on feature.body.face.start {"),
            "{text}"
        );
        assert!(
            text.contains("edges_between(feature.body.face.start, feature.body.face.side(s1.l3))"),
            "{text}"
        );
        assert!(!text.contains("face.end"), "{text}");
        assert!(summary.starts_with("Replaced feature.body.face.end with feature.body.face.start in 2 of 2 features that used it: s2 (plane.face); break (edges[0].a)"), "{summary}");
        // The written names are canonical: the edited text reads back to the same model.
        let parsed = runmat_cad::syntax::parse_part(PATH, &text);
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    }

    #[test]
    fn one_feature_changes_and_the_others_stay_listed() {
        let (text, summary) = applied(json!({
            "reference": "feature.body.face.end", "with": "feature.body.face.start",
            "scope": "one", "feature": "feature:break"
        }));
        assert!(
            text.contains("sketch s2 on feature.body.face.end {"),
            "{text}"
        );
        assert!(
            text.contains("edges_between(feature.body.face.start,"),
            "{text}"
        );
        assert!(
            summary.contains("in 1 of 2 features that used it: break (edges[0].a)"),
            "{summary}"
        );
        assert!(
            summary.contains("Still on feature.body.face.end: s2 (plane.face)"),
            "{summary}"
        );
        let out = run(json!({
            "reference": "feature.body.face.end", "with": "feature.body.face.start",
            "scope": "one", "feature": "feature:cube"
        }));
        assert_eq!(out["ok"], false);
        assert_eq!(
            out["message"],
            "cube doesn't use feature.body.face.end; it is used by s2, break"
        );
    }

    #[test]
    fn a_profile_takes_another_sketch_and_the_rebuild_reports_its_order() {
        let (text, summary) = applied(json!({ "reference": "sketch:s2", "with": "sketch:s3" }));
        assert!(
            text.contains("feature boss = extrude(profile = sketch.s3,"),
            "{text}"
        );
        assert!(
            summary.contains(
                "Replaced sketch s2 with sketch s3 in 1 of 1 feature that used it: boss (profile)"
            ),
            "{summary}"
        );
        assert!(
            summary.ends_with("After the rebuild: boss: `s3` comes later in the history"),
            "{summary}"
        );
    }

    #[test]
    fn a_reference_made_later_is_written_and_the_rebuild_reports_it() {
        let (text, summary) = applied(json!({
            "reference": "feature.body.face.end", "with": "feature.cube.face.end", "scope": "one",
            "feature": "sketch:s2"
        }));
        assert!(
            text.contains("sketch s2 on feature.cube.face.end {"),
            "{text}"
        );
        assert!(summary.contains("After the rebuild: s2:"), "{summary}");
    }

    #[test]
    fn mismatched_or_unused_references_are_refused() {
        let out = run(json!({ "reference": "feature.body.face.end", "with": "sketch:s3" }));
        assert_eq!(out["ok"], false, "{out}");
        let out =
            run(json!({ "reference": "feature.cube.face.end", "with": "feature.body.face.end" }));
        assert_eq!(out["message"], "no feature uses feature.cube.face.end");
        let out = run(json!({ "reference": "sketch:s2", "with": "feature.body.face.end" }));
        assert_eq!(
            out["message"],
            "`sketch:s2` is a sketch, so replace it with another sketch"
        );
        let out = run(json!({ "reference": "feature.body.face.end" }));
        assert_eq!(out["message"], "pick the reference to use instead");
    }
}
