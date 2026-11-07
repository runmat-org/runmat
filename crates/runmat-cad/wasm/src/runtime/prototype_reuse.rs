//! Assembly prototypes a caller already holds.
//!
//! Every mesh prototype of an assembly display carries the `digest` of the part regeneration it
//! draws ([`super::part_cache::digest`]: the part's path, text, configuration and tessellation).
//! A caller that keeps part displays by digest passes the digests it holds (`known` in the
//! compile options), and those prototypes come back `reused`, without their `part` display: a
//! recompile after an edit sends only the parts that changed. A board's prototype carries the
//! digest of its model's key (the layout and every circuit file it reads,
//! [`part_cache::board_key`]), so an unchanged board isn't sent again either.

use super::part_cache;
use runmat_cad::assembly::model::DefinitionKind;
use runmat_cad::assembly::resolve::ResolvedAssembly;
use runmat_cad::assembly::source::ConfigKey;
use runmat_cad::display::{configuration_key, AssemblyDisplay, PartDisplay, PrototypeShape};
use runmat_cad::model::PartDocument;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// The `known` digests of compile `options` (`{ "known": ["…", …] }`).
pub fn known_from_options(options: &str) -> BTreeSet<String> {
    serde_json::from_str::<Value>(options)
        .ok()
        .and_then(|v| v.get("known").cloned())
        .and_then(|k| serde_json::from_value(k).ok())
        .unwrap_or_default()
}

/// The digest of each prototype `assembly_display_configured` makes from `resolved`, in its
/// order: a configured occurrence's from its document and configuration, a default one's from
/// `digests` (the part files regenerated, by file). `None` for boxes and empties.
pub fn prototype_digests(
    resolved: &ResolvedAssembly,
    documents: &BTreeMap<String, PartDocument>,
    digests: &BTreeMap<String, String>,
) -> Vec<Option<String>> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for node in &resolved.nodes {
        let drawn = matches!(node.definition.kind, DefinitionKind::Part | DefinitionKind::Circuit);
        if !drawn || node.represented_by_ancestor {
            continue;
        }
        let key = configuration_key(node);
        if !seen.insert(key.clone()) {
            continue;
        }
        let file = &key.0;
        let configured = node.definition.kind == DefinitionKind::Part && node.configuration != ConfigKey::default();
        let digest = match (configured, documents.get(file)) {
            (true, Some(document)) => Some(part_cache::digest(&part_cache::configured_key(file, document, &node.configuration))),
            _ => digests.get(file).cloned(),
        };
        out.push(digest);
    }
    out
}

/// Empties the part display of each mesh prototype whose digest is in `known`, so it isn't
/// written; returns which prototypes it emptied. Nothing is emptied when the digests don't line
/// up with the prototypes.
pub fn omit_known(display: &mut AssemblyDisplay, digests: &[Option<String>], known: &BTreeSet<String>) -> Vec<bool> {
    if known.is_empty() || digests.len() != display.prototypes.len() {
        return vec![false; display.prototypes.len()];
    }
    display
        .prototypes
        .iter_mut()
        .zip(digests)
        .map(|(p, d)| match (&mut p.shape, d) {
            (PrototypeShape::Mesh { part }, Some(d)) if known.contains(d) => {
                **part = PartDisplay::default();
                true
            }
            _ => false,
        })
        .collect()
}

/// Writes each mesh prototype's `digest` into the serialized `display`, and marks the emptied
/// ones `reused` (their `part` removed).
pub fn mark(display: &mut Value, digests: &[Option<String>], reused: &[bool]) {
    let Some(prototypes) = display.get_mut("prototypes").and_then(Value::as_array_mut) else { return };
    if prototypes.len() != digests.len() {
        return;
    }
    for (i, (p, d)) in prototypes.iter_mut().zip(digests).enumerate() {
        let (Some(d), Some(row)) = (d, p.as_object_mut()) else { continue };
        if row.get("shape").and_then(Value::as_str) != Some("mesh") {
            continue;
        }
        row.insert("digest".into(), Value::String(d.clone()));
        if reused.get(i).copied().unwrap_or(false) {
            row.remove("part");
            row.insert("reused".into(), Value::Bool(true));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::known_from_options;
    use crate::runtime::{compile_json_known, part_cache, Files};
    use serde_json::Value;
    use std::collections::BTreeSet;

    const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    input width : length = 20 mm\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])\n        l2 = line(start = l1.end, end = [input.width, 10 mm])\n        l3 = line(start = l2.end, end = [0 mm, 10 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature box = extrude(profile = sketch.s, distance = 5 mm)\n    output block = { body = feature.box.body }\n}\n";
    const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part block as plain\n    use part block as wide {\n        configuration = { select = inputs, values = { width = 40 mm } }\n        placement = { placement = offset, offset = { translation = [0 mm, 50 mm, 0 mm] } }\n    }\n}\n";

    fn compile(block: &str, known: &BTreeSet<String>) -> Value {
        let files: Files = [("/p/block.part".to_owned(), block.to_owned()), ("/p/rig.assembly".to_owned(), RIG.to_owned())].into();
        serde_json::from_str(&compile_json_known("/p/rig.assembly", RIG, &files, None, known)).unwrap()
    }

    fn prototypes(out: &Value) -> Vec<Value> {
        out["display"]["data"]["prototypes"].as_array().cloned().unwrap_or_default()
    }

    /// Known prototypes come back without their part; a changed part comes back whole, with a
    /// new digest.
    #[test]
    fn a_caller_gets_only_the_prototypes_it_does_not_hold() {
        part_cache::clear();
        let first = prototypes(&compile(BLOCK, &BTreeSet::new()));
        assert_eq!(first.len(), 2);
        let digests: BTreeSet<String> = first.iter().map(|p| p["digest"].as_str().expect("a digest").to_owned()).collect();
        assert_eq!(digests.len(), 2, "the default and the configured occurrence differ");
        assert!(first.iter().all(|p| p.get("part").is_some() && p.get("reused").is_none()));

        let again = prototypes(&compile(BLOCK, &digests));
        assert!(again.iter().all(|p| p.get("part").is_none() && p["reused"] == true), "{again:?}");
        let digests_again: BTreeSet<String> = again.iter().map(|p| p["digest"].as_str().unwrap().to_owned()).collect();
        assert_eq!(digests, digests_again);

        // The part's height changes: both prototypes are new and sent whole.
        let edited = prototypes(&compile(&BLOCK.replace("10 mm", "12 mm"), &digests));
        for p in &edited {
            assert!(p.get("part").is_some() && p.get("reused").is_none());
            assert!(!digests.contains(p["digest"].as_str().unwrap()));
        }
    }

    #[test]
    fn known_digests_read_from_the_options() {
        assert_eq!(known_from_options(r#"{"known":["a","b"],"configuration":"x"}"#), ["a".to_owned(), "b".to_owned()].into());
        assert!(known_from_options("").is_empty());
    }
}
