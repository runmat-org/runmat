//! Repairs that pin or publish across files (CAD-25), as source changes:
//!
//! - pin-to-revision and update-pin rewrite this file's references to the other file (under the
//!   diagnostic's subject: the feature, region, input or publish it names) to read that
//!   released revision;
//! - pin-occurrence pins an assembly's component to a released revision of its definition, in
//!   whichever assembly file the diagnostic names;
//! - add-publish and publish-anchor add the publish entry to the other file, so the name can be
//!   used from here.
//!
//! Releasing a file makes a project revision, which no source edit can do; that repair says
//! where to release.

use crate::edit::model;
use crate::runtime::Files;
use runmat_cad::diagnostics::{Repair, Site, Subject};
use runmat_cad::ids::{FilePath, PublishName, RevisionId};
use runmat_cad::model::{PartDocument, PublishEntry, PublishTarget};
use runmat_cad::refs::XrefContext;
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use runmat_design_syntax::edit::TextEdit;
use runmat_types::Span;
use serde_json::{json, Value};

/// A change to one file: its path, its text before, and the change.
pub type FileEdit = (String, String, SourceChange);

/// The key of `file` (a project path) among the files the runtime was given.
fn find<'a>(files: &'a Files, file: &FilePath) -> Option<(&'a String, &'a String)> {
    let want = file.0.trim_start_matches('/');
    files.iter().find(|(k, _)| {
        k.as_str() == file.0
            || k.trim_start_matches('/') == want
            || k.ends_with(&format!("/{want}"))
    })
}

/// The JSON pointer of the document item a diagnostic's subject names, or the whole document.
fn subject_pointer(document: &Value, subject: &Subject) -> String {
    let index_of = |list: &str, key: &str, name: &str| -> Option<String> {
        document[list]
            .as_array()?
            .iter()
            .position(|item| item[key].as_str() == Some(name))
            .map(|at| format!("/{list}/{at}"))
    };
    let found = match subject {
        Subject::Feature { feature } => index_of("features", "id", &feature.0),
        Subject::Region { region } => index_of("regions", "name", region.as_str()),
        Subject::Publish { publish } => index_of("publishes", "name", publish.as_str()),
        Subject::Part { part } => index_of("parts", "name", part.as_str()),
        Subject::Value { name } => {
            let bare = name
                .strip_prefix("input.")
                .or_else(|| name.strip_prefix("let."))
                .unwrap_or(name);
            index_of("inputs", "name", bare).or_else(|| index_of("lets", "name", bare))
        }
        _ => None,
    };
    found.unwrap_or_default()
}

/// Pins every reference to `file` under `at`: live ones (`from` None), or those pinned to
/// `from`. Returns how many it pinned.
fn pin_json(at: &mut Value, file: &str, from: Option<&RevisionId>, to: &RevisionId) -> usize {
    let mut n = 0;
    match at {
        Value::Object(fields) => {
            let names_file = fields.get("file").and_then(Value::as_str) == Some(file);
            if let (true, Some(context)) = (names_file, fields.get("context")) {
                let matches = match from {
                    None => context["context"] == "live",
                    Some(r) => {
                        context["context"] == "pinned"
                            && context["revision"].as_str() == Some(r.0.as_str())
                    }
                };
                if matches {
                    fields.insert(
                        "context".into(),
                        json!({ "context": "pinned", "revision": to.0 }),
                    );
                    n += 1;
                }
            }
            for value in fields.values_mut() {
                n += pin_json(value, file, from, to);
            }
        }
        Value::Array(items) => {
            for item in items {
                n += pin_json(item, file, from, to);
            }
        }
        _ => {}
    }
    n
}

/// This part's references to `file` under the site's subject, pinned to `to`.
pub fn pin_in_part(
    text: &str,
    path: &str,
    site: &Site,
    file: &FilePath,
    from: Option<&RevisionId>,
    to: &RevisionId,
) -> Result<SourceChange, EditFailure> {
    edit_part(text, path, |document| {
        let mut j = serde_json::to_value(&*document).map_err(|e| model(e.to_string()))?;
        let pointer = subject_pointer(&j, &site.subject);
        let at = j.pointer_mut(&pointer).ok_or_else(|| {
            model("the problem's subject is gone; the part changed since it was reported")
        })?;
        let pinned = pin_json(at, &file.0, from, to);
        if pinned == 0 {
            return Err(model(match from {
                None => format!("found no live reference to `{}` to pin", file.0),
                Some(r) => format!("found no reference to `{}` pinned to {r}", file.0),
            }));
        }
        *document =
            serde_json::from_value(j).map_err(|e| model(format!("the pinned part: {e}")))?;
        Ok(())
    })
}

/// The publish entry for `name` in a definition that doesn't publish it yet: the region, part
/// output, input or let of that name.
fn anchor_entry(document: &PartDocument, name: &PublishName) -> Option<PublishEntry> {
    let n = name.as_str();
    let target = if document.regions.iter().any(|r| r.name.as_str() == n) {
        PublishTarget::Region {
            region: runmat_cad::ids::RegionName::new(n),
        }
    } else if document.parts.iter().any(|p| p.name.as_str() == n) {
        PublishTarget::Part {
            part: runmat_cad::ids::PartName::new(n),
        }
    } else if document.inputs.iter().any(|i| i.name.as_str() == n) {
        PublishTarget::Value {
            input: runmat_cad::ids::InputName::new(n),
        }
    } else if document.lets.iter().any(|l| l.name.as_str() == n) {
        PublishTarget::Let {
            name: runmat_cad::ids::LetName::new(n),
        }
    } else {
        return None;
    };
    Some(PublishEntry {
        name: name.clone(),
        target,
        auto: false,
    })
}

/// Adds a publish entry to the part `file` (one of `files`).
fn add_publish(
    files: &Files,
    file: &FilePath,
    entry: impl FnOnce(&PartDocument) -> Result<PublishEntry, EditFailure>,
) -> Result<(FileEdit, String), EditFailure> {
    let (key, old) =
        find(files, file).ok_or_else(|| model(format!("`{}` isn't in the project", file.0)))?;
    if !key.ends_with(".part") {
        return Err(model(format!(
            "`{}` isn't a part; add the publish entry in its source",
            file.0
        )));
    }
    let mut added = String::new();
    let change = edit_part(old, key, |document| {
        let entry = entry(document)?;
        if document.publishes.iter().any(|p| p.name == entry.name) {
            return Err(model(format!(
                "`{}` already publishes `{}`",
                file.0,
                entry.name.as_str()
            )));
        }
        added = entry.name.as_str().to_owned();
        document.publishes.push(entry);
        Ok(())
    })?;
    Ok((
        (key.clone(), old.clone(), change),
        format!("Published {added} from {}", file.0),
    ))
}

/// Pins component `occurrence` of the assembly `key` (text `old`) to `revision`.
pub fn pin_occurrence(
    key: &str,
    old: &str,
    files: &Files,
    occurrence: &str,
    revision: &RevisionId,
) -> Result<SourceChange, EditFailure> {
    let scope = crate::runtime::module_scope(key, old, files);
    let parsed = runmat_cad::syntax::parse_assembly_in(key, old, &scope);
    let (Some(mut document), true) = (parsed.document, parsed.diagnostics.is_empty()) else {
        return Err(EditFailure::Invalid(parsed.diagnostics));
    };
    let o = document
        .occurrences
        .iter_mut()
        .find(|o| o.name == occurrence)
        .ok_or_else(|| model(format!("`{key}` has no component `{occurrence}`")))?;
    o.source.context = XrefContext::Pinned {
        revision: revision.clone(),
    };
    let printed =
        runmat_cad::syntax::print_assembly(&document, &parsed.uses).map_err(EditFailure::Print)?;
    Ok(SourceChange {
        edits: vec![TextEdit::replace(
            Span {
                start: 0,
                end: old.len(),
            },
            printed.clone(),
        )],
        source: printed,
        reprinted: true,
    })
}

/// Applies `repair` when it edits a file other than `path` or pins: `Ok(None)` for the repairs
/// this file's own appliers handle.
pub fn apply(
    path: &str,
    text: &str,
    files: &Files,
    repair: &Repair,
) -> Result<Option<(Vec<FileEdit>, String)>, EditFailure> {
    Ok(Some(match repair {
        Repair::PinToRevision { site, file, revision } if path.ends_with(".part") => {
            let change = pin_in_part(text, path, site, file, None, revision)?;
            (vec![(path.to_owned(), text.to_owned(), change)], format!("Pinned {} to revision {revision}", file.0))
        }
        Repair::UpdatePin { site, file, from, to } if path.ends_with(".part") => {
            let change = pin_in_part(text, path, site, file, Some(from), to)?;
            (vec![(path.to_owned(), text.to_owned(), change)], format!("Updated the pin on {} from {from} to {to}", file.0))
        }
        Repair::AddPublish { file, entry } => {
            let (edit, summary) = add_publish(files, file, |_| Ok(entry.clone()))?;
            (vec![edit], summary)
        }
        Repair::PublishAnchor { file, name } => {
            let (edit, summary) = add_publish(files, file, |document| {
                anchor_entry(document, name).ok_or_else(|| model(format!("`{}` has no region, part, input or let named `{}` to publish", file.0, name.as_str())))
            })?;
            (vec![edit], summary)
        }
        Repair::PinOccurrence { assembly, occurrence, revision } => {
            let (key, old) = if assembly.0 == path {
                (path.to_owned(), text.to_owned())
            } else {
                let (k, t) = find(files, assembly).ok_or_else(|| model(format!("`{}` isn't in the project", assembly.0)))?;
                (k.clone(), t.clone())
            };
            let change = pin_occurrence(&key, &old, files, occurrence, revision)?;
            (vec![(key, old, change)], format!("Pinned {occurrence} to revision {revision}"))
        }
        Repair::ReleaseFile { file } => {
            return Err(model(format!(
                "release `{}` to make a revision (a project action, from the file's history), then pin to it",
                file.0
            )))
        }
        _ => return Ok(None),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_cad::ids::FeatureId;

    fn site(subject: Subject) -> Site {
        Site::of(subject)
    }

    const CONSUMER: &str = "runmat 0.7\nstd 1\n\npart rib {\n    input width : length = part.skeleton.width\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])\n        l2 = line(start = [input.width, 0 mm], end = [input.width, 10 mm])\n        l3 = line(start = [input.width, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 5 mm)\n    output rib = { body = feature.body.body }\n}\n";

    #[test]
    fn a_live_reference_pins_to_the_release_and_a_pin_moves_to_the_newer_one() {
        let file = FilePath::new("skeleton.part");
        let input = site(Subject::Value {
            name: "input.width".into(),
        });
        let pinned = pin_in_part(
            CONSUMER,
            "/p/rib.part",
            &input,
            &file,
            None,
            &RevisionId::new("A"),
        )
        .expect("pins");
        let back = runmat_cad::syntax::parse_part("/p/rib.part", &pinned.source)
            .document
            .expect("parses");
        let pinned_json = serde_json::to_value(&back.inputs[0]).unwrap().to_string();
        assert!(
            pinned_json.contains(r#""context":"pinned","revision":"A""#),
            "{}\n{pinned_json}",
            pinned.source
        );
        let moved = pin_in_part(
            &pinned.source,
            "/p/rib.part",
            &input,
            &file,
            Some(&RevisionId::new("A")),
            &RevisionId::new("B"),
        )
        .expect("updates");
        let back = runmat_cad::syntax::parse_part("/p/rib.part", &moved.source)
            .document
            .unwrap();
        assert!(
            serde_json::to_value(&back.inputs[0])
                .unwrap()
                .to_string()
                .contains(r#""revision":"B""#),
            "{}",
            moved.source
        );
        // A pin from a revision nothing reads says so.
        let err = pin_in_part(
            &moved.source,
            "/p/rib.part",
            &input,
            &file,
            Some(&RevisionId::new("A")),
            &RevisionId::new("C"),
        )
        .unwrap_err();
        assert!(matches!(err, EditFailure::Model(m) if m.contains("pinned to A")));
        // The feature subject scopes the pin: the extrude reads nothing from the skeleton.
        let feature = site(Subject::Feature {
            feature: FeatureId::new("body"),
        });
        assert!(pin_in_part(
            CONSUMER,
            "/p/rib.part",
            &feature,
            &file,
            None,
            &RevisionId::new("A")
        )
        .is_err());
    }

    #[test]
    fn a_missing_publish_is_added_to_the_other_file() {
        let skeleton = "runmat 0.7\nstd 1\n\npart skeleton {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    region mount : face = { select = [feature.body.face.end], expect = exactly_one }\n    output skeleton = { body = feature.body.body }\n}\n";
        let files: Files = [(
            "/p/interfaces/skeleton.part".to_string(),
            skeleton.to_string(),
        )]
        .into();
        let repair = Repair::PublishAnchor {
            file: FilePath::new("interfaces/skeleton.part"),
            name: PublishName::new("mount"),
        };
        let (edits, summary) = apply("/p/rib.part", CONSUMER, &files, &repair)
            .unwrap()
            .expect("applies here");
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].0, "/p/interfaces/skeleton.part");
        assert!(
            edits[0]
                .2
                .source
                .contains("publish mount = region(region = mount)"),
            "{}",
            edits[0].2.source
        );
        assert_eq!(summary, "Published mount from interfaces/skeleton.part");
        let back =
            runmat_cad::syntax::parse_part("/p/interfaces/skeleton.part", &edits[0].2.source)
                .document
                .expect("parses");
        assert_eq!(back.publishes.len(), 1);
        // Publishing it twice says it's there.
        let files: Files = [(
            "/p/interfaces/skeleton.part".to_string(),
            edits[0].2.source.clone(),
        )]
        .into();
        assert!(apply("/p/rib.part", CONSUMER, &files, &repair).is_err());
        // Repairs of this file's own are left to its appliers.
        assert!(apply(
            "/p/rib.part",
            CONSUMER,
            &files,
            &Repair::SuppressFeature {
                feature: FeatureId::new("body")
            }
        )
        .unwrap()
        .is_none());
    }
}
