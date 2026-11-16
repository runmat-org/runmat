//! Released revisions for an assembly's resolution, read from the project's release store
//! ([`runmat_cad::refs::release_store`]).
//!
//! For each file an assembly places, the store beside it says which revisions are released
//! (the latest is what Update pins → Latest released pins to). An occurrence pinned to one of
//! them (`@ "R2"`) regenerates that revision's text, held under `<file>@R2`
//! ([`runmat_cad::assembly::resolve::source_key`]), so it draws, weighs and exports as released
//! while the live file changes. A pin the store doesn't hold stays unloaded, and resolution
//! reports the revision missing.

use super::{
    assembly_part_entry, bbox, part_cache, resolve_path, Diagnostic, Files, ASSEMBLY_TESSELLATION,
};
use runmat_cad::assembly::model::{AssemblyDocument, DefinitionKind, DefinitionRef};
use runmat_cad::assembly::resolve::source_key;
use runmat_cad::assembly::source::{ConfiguredSummary, DefinitionSummary, MemorySource};
use runmat_cad::display::PartDisplay;
use runmat_cad::ids::RevisionId;
use runmat_cad::model::PartDocument;
use runmat_cad::refs::release_store::{store_path, ReleaseStore};
use runmat_cad::refs::XrefContext;
use runmat_cad::syntax::parse_assembly_in;
use std::collections::{BTreeMap, BTreeSet};

/// The release store of the design file at workspace path `file`, when the project has one.
pub fn store(file: &str, files: &Files) -> Option<ReleaseStore> {
    files
        .get(&store_path(file))
        .and_then(|text| ReleaseStore::parse(text).ok())
}

/// The revisions `doc`'s occurrences pin each definition to.
pub fn pinned(doc: &AssemblyDocument) -> BTreeMap<String, (DefinitionRef, BTreeSet<RevisionId>)> {
    let mut out: BTreeMap<String, (DefinitionRef, BTreeSet<RevisionId>)> = BTreeMap::new();
    for o in &doc.occurrences {
        if let XrefContext::Pinned { revision } = &o.source.context {
            out.entry(o.source.definition.file.0.clone())
                .or_insert_with(|| (o.source.definition.clone(), BTreeSet::new()))
                .1
                .insert(revision.clone());
        }
    }
    out
}

/// What a resolution collects that released revisions add to.
pub(super) struct Sink<'a> {
    pub source: &'a mut MemorySource,
    pub displays: &'a mut BTreeMap<String, PartDisplay>,
    pub documents: &'a mut BTreeMap<String, PartDocument>,
    pub digests: &'a mut BTreeMap<String, String>,
    pub problems: &'a mut Vec<Diagnostic>,
    pub queue: &'a mut Vec<(String, AssemblyDocument)>,
}

/// The pinned revisions loaded so far in one resolution, each once.
#[derive(Default)]
pub(super) struct PinnedLoads {
    done: BTreeSet<(String, RevisionId)>,
}

impl PinnedLoads {
    /// Reads the stores of the files `doc` (at workspace path `at`) places, and loads each
    /// pinned revision the store holds.
    pub fn load(
        &mut self,
        at: &str,
        doc: &AssemblyDocument,
        files: &Files,
        scope: &runmat_cad::modules::ModuleScope,
        sink: &mut Sink,
    ) {
        let pins = pinned(doc);
        let mut seen = BTreeSet::new();
        for o in &doc.occurrences {
            let def = &o.source.definition;
            let key = def.file.0.clone();
            if !seen.insert(key.clone()) {
                continue;
            }
            let file = resolve_path(at, &key, files);
            let Some(store) = store(&file, files) else {
                continue;
            };
            for revision in pins.get(&key).map(|(_, r)| r.clone()).unwrap_or_default() {
                if !self.done.insert((key.clone(), revision.clone())) {
                    continue;
                }
                let Some(text) = store.text(&revision) else {
                    continue;
                };
                load_revision(def, &key, &file, text, &revision, scope, sink);
            }
            sink.source
                .releases
                .insert(def.file.clone(), store.revisions());
        }
    }
}

/// Adds `def` at `revision` (its released `text`) to the source.
fn load_revision(
    def: &DefinitionRef,
    key: &str,
    file: &str,
    text: &str,
    revision: &RevisionId,
    scope: &runmat_cad::modules::ModuleScope,
    sink: &mut Sink,
) {
    let held = source_key(key, Some(revision));
    match def.kind {
        DefinitionKind::Assembly => match parse_assembly_in(file, text, scope).document {
            Some(sub) => {
                sink.queue.push((file.to_owned(), sub.clone()));
                sink.source.add_assembly_revision(sub, &revision.0);
            }
            None => sink.problems.push(failed(file, key, revision, "has errors")),
        },
        DefinitionKind::Part => {
            let cache_key = part_cache::assembly_key(&held, text, &ASSEMBLY_TESSELLATION);
            sink.digests
                .insert(held.clone(), part_cache::digest(&cache_key));
            match assembly_part_entry(key, text, cache_key) {
                Ok(part) => {
                    let (summary, configured) = part_summary(def, &part, &held, text);
                    sink.source
                        .add_part_revision(summary, configured, &revision.0);
                    sink.displays.insert(held.clone(), part.display.clone());
                    sink.documents.insert(held, part.document.clone());
                }
                Err(d) => {
                    let first = d
                        .iter()
                        .find(|e| e.severity == "error")
                        .map(|e| e.message.clone())
                        .unwrap_or_default();
                    sink.problems
                        .push(failed(file, key, revision, &format!("doesn't compile: {first}")));
                }
            }
        }
        // A released board or other design: its live file stands in until boards release.
        _ => {}
    }
}

fn failed(file: &str, key: &str, revision: &RevisionId, why: &str) -> Diagnostic {
    Diagnostic {
        path: file.to_owned(),
        start: 0,
        end: 0,
        severity: "error".into(),
        message: format!("{key} at released revision {revision} {why}"),
    }
}

/// The summaries resolution reads for the part `def`, regenerated from `text` and held under
/// `held` (its file, or `<file>@<revision>`).
pub(super) fn part_summary(
    def: &DefinitionRef,
    part: &part_cache::AssemblyPart,
    held: &str,
    text: &str,
) -> (DefinitionSummary, ConfiguredSummary) {
    let mut summary = DefinitionSummary::new(def.clone());
    summary.configurations = part
        .document
        .configurations
        .iter()
        .map(|c| c.name.clone())
        .collect();
    summary.configuration_part_numbers = part
        .document
        .all_configurations()
        .into_iter()
        .filter_map(|c| c.part_number.map(|pn| (c.name, pn)))
        .collect();
    summary.inputs = part
        .document
        .inputs
        .iter()
        .map(|i| (i.name.clone(), i.kind.clone()))
        .collect();
    if let Some(output) = part.document.parts.first() {
        summary.material = output.material.clone();
        if summary.part_number.is_none() {
            summary.part_number = output.part_number.clone();
        }
    }
    if part.document.parts.len() > 1 {
        summary.outputs = part
            .document
            .parts
            .iter()
            .map(|o| runmat_cad::assembly::source::OutputSummary {
                name: o.name.as_str().to_owned(),
                part_number: o.part_number.clone(),
                material: o.material.clone(),
            })
            .collect();
    }
    summary.digest = format!("text:{}", text.len());
    let configured = ConfiguredSummary {
        bbox: bbox(&part.display),
        artifact: format!("text:{held}"),
        publishes: part.publishes.clone(),
        ..Default::default()
    };
    (summary, configured)
}
