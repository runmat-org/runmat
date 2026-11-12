//! An assembly opened in stages, so the Desktop can show how far the open has got.
//!
//! Opening an assembly costs mostly part regeneration: every part file it places, and every
//! configuration an occurrence uses, that the part cache (`runtime::part_cache`) doesn't hold.
//! The Desktop therefore opens an assembly it hasn't compiled yet in three steps, the way a
//! drawing compiles in stages (`drawing::progress`):
//!
//! 1. **Probe**: the compile as a probe (`part_cache::probe`) with nothing allowed. The report's
//!    `entries` list every part, configured part and board the assembly reads, in order, each
//!    with its file and whether the cache held it; `misses` are the ones to regenerate. The
//!    probe's output draws the occurrences of every held part, so they paint at once.
//! 2. **Parts**: `{"assembly": {"stage": "parts"}}` as a probe that allows one digest
//!    regenerates exactly that entry and nothing else: no occurrence tree is built into an index
//!    and no display is written. The entry stays in the cache for the next call, so the Desktop
//!    learns each part as it finishes. A configured part's digest needs its part's document, so
//!    it shows in the probe after its part is held.
//! 3. **Assemble**: the full compile (no stage), which finds every part in the cache and builds
//!    the occurrences, mates and display.
//!
//! The Desktop's percentage weighs each entry by what it cost last time, and moves only when a
//! step finishes (desktop `cad-load-progress.ts`).

use super::{resolve_assembly, Diagnostic, Files};
use runmat_cad::assembly::model::DefinitionKind;
use serde_json::{json, Value};

/// Which part of an assembly a compile makes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stage {
    /// Everything.
    #[default]
    Full,
    /// The parts the compile may regenerate, and nothing else.
    Parts,
}

impl Stage {
    /// The stage `options` (the compile's options JSON) asks for.
    pub fn from_options(options: &str) -> Stage {
        let value: Value = serde_json::from_str(options).unwrap_or_default();
        if value["assembly"]["stage"] == "parts" {
            Stage::Parts
        } else {
            Stage::Full
        }
    }
}

/// The parts stage of assembly `path`: regenerates the part files and configured parts it reads
/// that the cache doesn't hold (during a probe, only the allowed ones) and keeps them in the
/// cache. Answers `{"stage": "parts", "components": n}`, `n` being the part and board
/// occurrences the assembly places as far as its held parts resolve (all of them once every
/// part is held), or the syntax errors that stop it.
pub fn compile_parts(path: &str, text: &str, files: &Files) -> String {
    match resolve_assembly(path, text, files) {
        Ok(resolution) => {
            let _ =
                crate::configured::configured_displays(&resolution.resolved, &resolution.documents);
            let components = resolution
                .resolved
                .nodes
                .iter()
                .filter(|n| {
                    matches!(
                        n.definition.kind,
                        DefinitionKind::Part | DefinitionKind::Circuit
                    )
                })
                .count();
            json!({ "stage": "parts", "components": components }).to_string()
        }
        Err(diagnostics) => json!({ "diagnostics": diagnostics_json(&diagnostics) }).to_string(),
    }
}

fn diagnostics_json(diagnostics: &[Diagnostic]) -> Value {
    serde_json::to_value(diagnostics).unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::part_cache::{self, probe, EntryKind};
    use crate::runtime::values_host::compile_json_with_values;

    fn block(name: &str, width_mm: u32) -> String {
        format!("runmat 0.7\nstd 1\n\npart {name} {{\n    input width : length = {width_mm} mm\n    sketch s on origin.xy {{\n        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])\n        l2 = line(start = l1.end, end = [input.width, 10 mm])\n        l3 = line(start = l2.end, end = [0 mm, 10 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }}\n    feature box = extrude(profile = sketch.s, distance = 5 mm)\n    output {name} = {{ body = feature.box.body }}\n}}\n")
    }

    const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part a as a\n    use part b as b {\n        placement = { placement = offset, offset = { translation = [0 mm, 50 mm, 0 mm] } }\n    }\n    use part b as wide {\n        configuration = { select = inputs, values = { width = 40 mm } }\n        placement = { placement = offset, offset = { translation = [0 mm, 100 mm, 0 mm] } }\n    }\n}\n";
    const PARTS: &str = r#"{"assembly":{"stage":"parts"}}"#;

    fn files() -> Files {
        [
            ("/p/a.part".to_owned(), block("a", 20)),
            ("/p/b.part".to_owned(), block("b", 30)),
            ("/p/rig.assembly".to_owned(), RIG.to_owned()),
        ]
        .into()
    }

    fn compile(options: &str, allow: &[&str]) -> (Value, part_cache::ProbeReport) {
        let files = files();
        let allow = allow.iter().map(|d| d.to_string()).collect();
        let (out, report) = probe(allow, || {
            compile_json_with_values("/p/rig.assembly", RIG, &files, options)
        });
        (serde_json::from_str(&out).unwrap(), report)
    }

    #[test]
    fn the_stage_comes_from_the_options() {
        assert_eq!(Stage::from_options(""), Stage::Full);
        assert_eq!(Stage::from_options(r#"{"configuration":"x"}"#), Stage::Full);
        assert_eq!(Stage::from_options(PARTS), Stage::Parts);
        assert_eq!(
            Stage::from_options(r#"{"drawing":{"stage":"sheet"}}"#),
            Stage::Full
        );
    }

    /// The probe lists every entry with its file and kind; each parts-stage call regenerates
    /// exactly the entry it allows and writes no display; the configured part shows once its
    /// part is held; the full compile then regenerates nothing.
    #[test]
    fn an_assembly_opens_one_part_per_call() {
        part_cache::clear();
        let (first, report) = compile("", &[]);
        assert!(first.get("display").is_some(), "the probe still draws");
        let files: Vec<(&str, EntryKind, bool)> = report
            .entries
            .iter()
            .map(|e| (e.path.as_str(), e.kind, e.held))
            .collect();
        assert_eq!(
            files,
            [
                ("/p/a.part", EntryKind::Part, false),
                ("/p/b.part", EntryKind::Part, false)
            ],
            "{:?}",
            report.entries
        );
        assert_eq!(report.misses.len(), 2);

        for entry in &report.entries {
            let before = part_cache::stats();
            let (out, _) = compile(PARTS, &[&entry.digest]);
            assert_eq!(out["stage"], "parts");
            assert!(out.get("display").is_none() && out.get("projection").is_none());
            let after = part_cache::stats();
            assert_eq!(after.misses - before.misses, 1, "exactly {}", entry.path);
            assert_eq!(after.entries, before.entries + 1);
        }

        // With a and b held, the probe meets b's configured part, still to regenerate.
        let (_, report) = compile("", &[]);
        let configured: Vec<_> = report
            .entries
            .iter()
            .filter(|e| e.kind == EntryKind::Configured)
            .collect();
        assert_eq!(configured.len(), 1, "{:?}", report.entries);
        assert_eq!(configured[0].path, "/p/b.part");
        assert!(!configured[0].held);
        assert!(report
            .entries
            .iter()
            .filter(|e| e.kind == EntryKind::Part)
            .all(|e| e.held));
        assert_eq!(report.misses, [configured[0].digest.clone()]);
        let (out, _) = compile(PARTS, &[&configured[0].digest]);
        assert_eq!(out, json!({ "stage": "parts", "components": 3 }));

        let before = part_cache::stats();
        let (full, report) = compile("", &[]);
        assert!(report.misses.is_empty(), "{:?}", report.misses);
        assert_eq!(
            part_cache::stats().misses,
            before.misses,
            "nothing regenerates"
        );
        assert_eq!(
            full["display"]["data"]["prototypes"]
                .as_array()
                .map(Vec::len),
            Some(3)
        );
    }

    #[test]
    fn a_parts_stage_with_errors_says_so() {
        let files = files();
        let out: Value = serde_json::from_str(&compile_json_with_values(
            "/p/rig.assembly",
            "assembly rig {",
            &files,
            PARTS,
        ))
        .unwrap();
        assert!(out["diagnostics"].as_array().is_some_and(|d| !d.is_empty()));
        assert!(out.get("stage").is_none());
    }
}
