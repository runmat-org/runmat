//! The what's-wrong list's repairs (CAD-25): `repair.repair` with `{ diagnostic, repair,
//! candidate? }` regenerates the part, takes that diagnostic's typed repair and writes it to the
//! part's text as a model edit.
//!
//! - re-pick, disambiguate, rewrite and use-blend-face put the chosen reference at the
//!   diagnostic's parameter (`candidate`, default the first offered);
//! - use-relational writes the edges between the two faces;
//! - remove-reference drops the reference from its list (or the parameter);
//! - unsuppress, reorder and move-before move or re-enable the feature;
//! - suppress-feature suppresses the failing feature, so what follows it can rebuild.
//!
//! Repairs that edit another file (pins, publishes, releases) or an assembly are applied by the
//! assembly and file commands; here they say so.

use crate::edit::model;
use runmat_cad::diagnostics::{Diagnostic, Repair, Site, Subject};
use runmat_cad::refs::Reference;
use runmat_cad::syntax::edit::{edit_part, reorder_feature, suppress_feature, EditFailure, SourceChange};
use serde_json::{Map, Value};

type Values = Map<String, Value>;

fn index(values: &Values, id: &str) -> Option<usize> {
    match values.get(id) {
        Some(Value::Number(n)) => n.as_u64().map(|n| n as usize),
        Some(Value::String(s)) => s.trim().parse().ok(),
        _ => None,
    }
}

/// `edges[1].a` → ["edges", 1, "a"].
fn steps(param: &str) -> Vec<Result<usize, String>> {
    let mut out = Vec::new();
    for part in param.split('.') {
        let mut rest = part;
        if let Some(open) = rest.find('[') {
            if open > 0 {
                out.push(Err(rest[..open].to_owned()));
            }
            rest = &rest[open..];
            while let Some(close) = rest.find(']') {
                if let Ok(k) = rest[1..close].parse() {
                    out.push(Ok(k));
                }
                rest = &rest[close + 1..];
            }
        } else {
            out.push(Err(rest.to_owned()));
        }
    }
    out
}

fn slot<'a>(value: &'a mut Value, steps: &[Result<usize, String>]) -> Option<&'a mut Value> {
    let mut at = value;
    for step in steps {
        at = match step {
            Ok(k) => at.get_mut(*k)?,
            Err(key) => at.get_mut(key.as_str())?,
        };
    }
    // A boundary or fill edge entry holds its reference under `edge`.
    if at.get("edge").is_some() && at.get("ref").is_none() {
        at = at.get_mut("edge")?;
    }
    Some(at)
}

fn feature_of(site: &Site) -> Result<(String, String), EditFailure> {
    match (&site.subject, &site.param) {
        (Subject::Feature { feature }, Some(param)) => Ok((feature.0.clone(), param.clone())),
        _ => Err(model("this repair names no feature parameter; edit the reference in the feature's properties")),
    }
}

/// Writes `reference` at the feature parameter `site` names.
fn put(text: &str, path: &str, site: &Site, reference: &Reference) -> Result<SourceChange, EditFailure> {
    let (id, param) = feature_of(site)?;
    let new = serde_json::to_value(reference).map_err(|e| model(e.to_string()))?;
    edit_part(text, path, |document| {
        let feature = document.feature_mut(&id).ok_or_else(|| model(format!("the part has no feature `{id}`")))?;
        let mut kind = serde_json::to_value(&feature.kind).map_err(|e| model(e.to_string()))?;
        let at = slot(&mut kind, &steps(&param)).ok_or_else(|| model(format!("`{id}` has no parameter `{param}`")))?;
        *at = new;
        feature.kind = serde_json::from_value(kind).map_err(|e| model(format!("the repaired `{id}`: {e}")))?;
        Ok(())
    })
}

/// Drops the reference at `site`: from its list when it is one entry, else the parameter.
fn remove(text: &str, path: &str, site: &Site) -> Result<SourceChange, EditFailure> {
    let (id, param) = feature_of(site)?;
    edit_part(text, path, |document| {
        let feature = document.feature_mut(&id).ok_or_else(|| model(format!("the part has no feature `{id}`")))?;
        let mut kind = serde_json::to_value(&feature.kind).map_err(|e| model(e.to_string()))?;
        let mut path = steps(&param);
        let last = path.pop().ok_or_else(|| model("the repair names no parameter"))?;
        let parent = slot(&mut kind, &path).ok_or_else(|| model(format!("`{id}` has no parameter `{param}`")))?;
        match (parent, last) {
            (Value::Array(items), Ok(k)) if k < items.len() => {
                items.remove(k);
            }
            (Value::Object(fields), Err(key)) => {
                fields.remove(&key);
            }
            _ => return Err(model(format!("`{param}` of `{id}` can't be removed"))),
        }
        feature.kind = serde_json::from_value(kind).map_err(|e| model(format!("`{id}` without that reference: {e}")))?;
        Ok(())
    })
}

/// The feature that runs just before `before`, for a move ahead of it.
fn ahead_of(text: &str, path: &str, before: &str) -> Result<Option<String>, EditFailure> {
    let parsed = runmat_cad::syntax::parse_part(path, text);
    let document = parsed.document.ok_or(EditFailure::Invalid(parsed.diagnostics))?;
    let at = document.features.iter().position(|f| f.id.0 == before).ok_or_else(|| model(format!("the part has no feature `{before}`")))?;
    Ok(at.checked_sub(1).map(|k| document.features[k].id.0.clone()))
}

/// Applies `diagnostic`'s repair `at`; `candidate` picks among a repair's offered references.
pub fn apply(text: &str, path: &str, diagnostic: &Diagnostic, at: usize, candidate: usize) -> Result<(SourceChange, String), EditFailure> {
    let repair = diagnostic.repairs.get(at).ok_or_else(|| model(format!("the problem has no repair {at}")))?;
    let pick = |candidates: &[Reference]| -> Result<Reference, EditFailure> {
        candidates
            .get(candidate)
            .cloned()
            .ok_or_else(|| model("no candidate to use; pick the new reference in the feature's properties"))
    };
    Ok(match repair {
        Repair::RePick { site, candidates } | Repair::Disambiguate { site, candidates } => {
            (put(text, path, site, &pick(candidates)?)?, "Re-picked the reference".into())
        }
        Repair::RewriteReference { site, reference } => (put(text, path, site, reference)?, "Rewrote the reference".into()),
        Repair::UseRelational { site, between } => {
            let [a, b] = (**between).clone();
            (put(text, path, site, &Reference::edges_between(a, b))?, "Used the edges between the two faces".into())
        }
        Repair::UseBlendFace { site, face } => (
            put(text, path, site, &Reference::Name { kind: runmat_cad::refs::TopoKind::Face, name: face.clone() })?,
            "Used the blend face".into(),
        ),
        Repair::RemoveReference { site } => (remove(text, path, site)?, "Removed the reference".into()),
        Repair::Unsuppress { feature } => (suppress_feature(text, path, &feature.0, false)?, format!("Unsuppressed {feature}")),
        Repair::SuppressFeature { feature } => (suppress_feature(text, path, &feature.0, true)?, format!("Suppressed {feature}")),
        Repair::Reorder { feature, after } => (reorder_feature(text, path, &feature.0, Some(&after.0))?, format!("Moved {feature} after {after}")),
        Repair::MoveBefore { feature, before } => {
            let after = ahead_of(text, path, &before.0)?;
            (reorder_feature(text, path, &feature.0, after.as_deref())?, format!("Moved {feature} before {before}"))
        }
        other => {
            return Err(model(format!(
                "the {} repair edits another file or an assembly; run it from that file",
                serde_json::to_value(other).ok().and_then(|v| v["repair"].as_str().map(str::to_owned)).unwrap_or_default().replace('_', " ")
            )))
        }
    })
}

/// Gives each repair that offers candidates their text (`candidate_labels`), so a problem
/// list can offer one action per candidate: a re-pick's references as written in the source,
/// a retarget's component paths.
pub fn label_candidates(diagnostics: &mut Value) {
    for d in diagnostics.as_array_mut().into_iter().flatten() {
        for r in d["repairs"].as_array_mut().into_iter().flatten() {
            let labels: Option<Vec<String>> = match r["repair"].as_str() {
                Some("re_pick" | "disambiguate") => r["candidates"].as_array().map(|c| {
                    c.iter()
                        .filter_map(|c| serde_json::from_value::<Reference>(c.clone()).ok())
                        .map(|c| runmat_cad::syntax::refs::print_reference(&c).unwrap_or_else(|| "a candidate".into()))
                        .collect()
                }),
                Some("retarget_override") => r["candidates"].as_array().map(|c| {
                    c.iter()
                        .filter_map(|c| serde_json::from_value::<runmat_cad::assembly::RelPath>(c.clone()).ok())
                        .map(|c| c.to_string())
                        .collect()
                }),
                _ => None,
            };
            if let Some(labels) = labels.filter(|l| !l.is_empty()) {
                r["candidate_labels"] = labels.into();
            }
        }
    }
}

/// `repair.repair` on a part: regenerates it and applies the repair the values name.
pub fn repair_part(path: &str, text: &str, values: &Values) -> Result<(SourceChange, String), EditFailure> {
    let id = index(values, "diagnostic").ok_or_else(|| model("name the problem to repair"))?;
    let at = index(values, "repair").unwrap_or(0);
    let candidate = index(values, "candidate").unwrap_or(0);
    let part = crate::runtime::regenerate_part(path, text).map_err(|_| model("the part doesn't parse; fix the source first"))?;
    let diagnostic = part
        .result
        .diagnostics
        .diagnostics
        .iter()
        .find(|d| d.id as usize == id)
        .ok_or_else(|| model("that problem is gone; the part changed since it was reported"))?;
    apply(text, path, diagnostic, at, candidate)
}

#[cfg(test)]
mod tests {
    use super::{repair_part, steps};
    use runmat_cad::diagnostics::{Repair, Severity};
    use serde_json::json;

    const SKETCH: &str = "    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n";

    fn part(features: &str) -> String {
        format!("runmat 0.7\nstd 1\n\npart plate {{\n{SKETCH}{features}}}\n")
    }

    fn errors(text: &str) -> Vec<runmat_cad::diagnostics::Diagnostic> {
        let part = crate::runtime::regenerate_part("/p/plate.part", text).expect("parses");
        part.result.diagnostics.diagnostics.into_iter().filter(|d| d.severity == Severity::Error).collect()
    }

    /// Applies the first repair of the first root error whose repairs `want` accepts.
    fn fix(text: &str, want: impl Fn(&Repair) -> bool) -> String {
        let found = errors(text);
        let d = found.iter().find(|d| d.repairs.iter().any(&want)).unwrap_or_else(|| panic!("no such repair in {found:?}"));
        let at = d.repairs.iter().position(&want).unwrap();
        let values = json!({ "diagnostic": d.id, "repair": at }).as_object().unwrap().clone();
        repair_part("/p/plate.part", text, &values).expect("repairs").0.source
    }

    #[test]
    fn a_feature_that_runs_before_what_it_uses_moves_after_it() {
        let text = part("    feature round = fillet(edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end)], radius = 1 mm)\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n");
        let fixed = fix(&text, |r| matches!(r, Repair::Reorder { .. } | Repair::MoveBefore { .. }));
        assert!(fixed.find("feature body").unwrap() < fixed.find("feature round").unwrap(), "{fixed}");
        assert!(errors(&fixed).is_empty(), "{fixed}");
    }

    #[test]
    fn every_root_error_on_a_feature_offers_a_repair_and_suppress_clears_it() {
        // A shell with no wall fails before any kernel runs, with nothing more specific to offer.
        let text = part("    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature bad = shell(body = feature.body.body, open_faces = [feature.body.face.end], thickness = 0 mm)\n");
        let found = errors(&text);
        assert!(!found.is_empty(), "the zero-thickness shell fails");
        for d in found.iter().filter(|d| !matches!(d.cause, runmat_cad::diagnostics::Cause::Cascade { .. })) {
            assert!(!d.repairs.is_empty(), "{d:?}");
        }
        let fixed = fix(&text, |r| matches!(r, Repair::SuppressFeature { .. }));
        assert!(fixed.contains("suppressed"), "{fixed}");
        assert!(errors(&fixed).is_empty(), "{fixed}");
    }

    #[test]
    fn a_parameter_path_reads_as_keys_and_indices() {
        assert_eq!(steps("edges[1].a"), vec![Err("edges".to_owned()), Ok(1), Err("a".to_owned())]);
        assert_eq!(steps("profile"), vec![Err("profile".to_owned())]);
    }
}
