//! The argument registry against the model and the docs.
//!
//! - Every argument a corpus part writes, as the model serializes it, is a registry argument of
//!   its operation, so the registry cannot miss a field the files use.
//! - The docs' generated parameter tables match the registry. `RUNMAT_ARGUMENT_DOCS_WRITE=1`
//!   rewrites them.

use runmat_cad::syntax::arguments::{operation, rewrite_docs};
use std::path::{Path, PathBuf};

fn parts(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            parts(&path, out);
        } else if path.extension().is_some_and(|e| e == "part") {
            out.push(path);
        }
    }
}

#[test]
fn corpus_arguments_are_registered() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    parts(&root.join("tests/fixtures"), &mut files);
    let mut missing = Vec::new();
    let mut checked = 0;
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap();
        let parsed = runmat_cad::syntax::parse_part(&file.to_string_lossy(), &text);
        let Some(document) = parsed.document else { continue };
        for feature in &document.features {
            let value = serde_json::to_value(&feature.kind).unwrap();
            let serde_json::Value::Object(fields) = value else { continue };
            let Some(tag) = fields.get("feature").and_then(|v| v.as_str()) else { continue };
            if tag == "sketch" {
                continue;
            }
            let sub = fields.iter().find_map(|(k, v)| {
                let word = format!("{tag}.{}", v.as_str()?);
                operation(&word).map(|op| (k.clone(), op))
            });
            let (subtag_key, op) = match sub {
                Some((k, op)) => (Some(k), op),
                None => (None, operation(tag).unwrap_or_else(|| panic!("no operation `{tag}`"))),
            };
            if op.arguments.is_empty() {
                continue;
            }
            for key in fields.keys() {
                if key == "feature" || Some(key) == subtag_key.as_ref() {
                    continue;
                }
                checked += 1;
                let known = op
                    .arguments
                    .iter()
                    .any(|a| a.name == key || a.sites.contains(&key.as_str()));
                if !known {
                    missing.push(format!("{}: {}.{key}", file.display(), op.word));
                }
            }
        }
    }
    missing.sort();
    missing.dedup();
    assert!(checked > 50, "only {checked} arguments checked");
    assert!(missing.is_empty(), "unregistered arguments:\n{}", missing.join("\n"));
}

#[test]
fn docs_tables_are_generated() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design/mcad/part");
    for page in ["features.md"] {
        let path = root.join(page);
        let text = std::fs::read_to_string(&path).unwrap();
        let fresh = rewrite_docs(&text);
        if std::env::var_os("RUNMAT_ARGUMENT_DOCS_WRITE").is_some() {
            std::fs::write(&path, &fresh).unwrap();
        } else {
            assert_eq!(text, fresh, "{page} is stale: rerun with RUNMAT_ARGUMENT_DOCS_WRITE=1");
        }
    }
}

mod common;

#[test]
fn reference_facts_describe_what_each_reference_resolves_to() {
    use runmat_cad::index::{reference_facts, EntityFacts};
    let text = include_str!("fixtures/notes/l_bracket.part");
    let document = runmat_cad::syntax::parse_part("l_bracket.part", text)
        .document
        .expect("the L-bracket reads");
    let result = common::Env::new().regen(&document);
    let facts = reference_facts(&document, &result);
    // The hole's start face: on the test kernel it resolves like any feature face.
    let faces: Vec<&EntityFacts> = facts
        .iter()
        .filter(|(key, _)| key.contains("/face"))
        .flat_map(|(_, (_, facts))| facts)
        .collect();
    assert!(
        faces.iter().any(|f| matches!(f, EntityFacts::Face { surface, normal: Some(_), .. } if surface == "planar")),
        "no planar face among {:?}",
        facts.keys().collect::<Vec<_>>()
    );
}
