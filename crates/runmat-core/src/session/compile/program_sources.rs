use super::{
    source_context_function_names_in_statements, CompanionSourceDiscovery, SourceContextData,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use runmat_types::ProgramSourceId;

#[path = "program_sources/identity.rs"]
pub(super) mod identity;
use identity::executable_source_identity;

/// A retained definition may refer to an older generation at the same source
/// location. The content revision is explicit; the original text remains in
/// the key so identity does not rely on a digest collision assumption.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct SourceContentRevision(runmat_execution::Digest);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct SourceKey {
    owner_identity: String,
    relative_path: String,
    revision: SourceContentRevision,
    text: String,
}

impl SourceKey {
    fn new(owner_identity: String, relative_path: String, text: String) -> Self {
        let revision = SourceContentRevision(runmat_execution::Digest::sha256(text.as_bytes()));
        Self {
            owner_identity,
            relative_path,
            revision,
            text,
        }
    }
}

/// One source table for the immutable compiler product. These ordinals never
/// borrow authority from the session SourcePool or a runtime source lease.
pub(super) struct ProgramSourceTable {
    pub source_map: crate::ExecutableSourceMap,
    pub statement_sources: Vec<ProgramSourceId>,
    pub function_sources: HashMap<String, ProgramSourceId>,
    pub retained_runtime_sources: HashMap<runmat_hir::SourceId, ProgramSourceId>,
}

pub(super) struct SourceCompileProjection {
    pub runtime_to_program: HashMap<runmat_hir::SourceId, ProgramSourceId>,
    pub program_to_runtime: HashMap<ProgramSourceId, runmat_hir::SourceId>,
}

impl SourceCompileProjection {
    pub fn program_ref_for_runtime_ref(
        &self,
        source: runmat_vm::BytecodeSourceRef,
    ) -> Option<runmat_vm::BytecodeSourceRef> {
        let runtime = source.runtime()?;
        self.runtime_to_program
            .get(&runtime)
            .copied()
            .map(runmat_vm::BytecodeSourceRef::Program)
    }

    pub fn runtime_ref_for_program_ref(
        &self,
        source: runmat_vm::BytecodeSourceRef,
    ) -> Option<runmat_vm::BytecodeSourceRef> {
        let ordinal = source.program()?;
        self.program_to_runtime
            .get(&ordinal)
            .copied()
            .map(runmat_vm::BytecodeSourceRef::Runtime)
    }
}

struct CompanionSource {
    context: Arc<SourceContextData>,
    owner_identity: String,
    relative_path: String,
}

impl ProgramSourceTable {
    pub fn intern_runtime_projection(
        &self,
        pool: &mut crate::source_pool::SourcePool,
    ) -> Result<SourceCompileProjection, String> {
        let mut program_to_runtime = HashMap::new();
        let mut runtime_to_program = HashMap::new();
        for entry in self.source_map.entries() {
            let program = entry.source_id;
            let runtime = pool.intern_with_fullpath(
                &entry.display_name,
                entry.full_path.as_deref(),
                &entry.text,
            );
            if let Some(previous) = runtime_to_program.insert(runtime, program) {
                if previous != program {
                    return Err("distinct program sources share one session source identity".into());
                }
            }
            program_to_runtime.insert(program, runtime);
        }
        for (runtime, program) in &self.retained_runtime_sources {
            let program = *program;
            if let Some(previous) = runtime_to_program.insert(*runtime, program) {
                if previous != program {
                    return Err("retained source identity disagrees with the source table".into());
                }
            }
        }
        Ok(SourceCompileProjection {
            runtime_to_program,
            program_to_runtime,
        })
    }

    pub fn build(
        root: &crate::ExecutableSource,
        root_full_path: Option<&str>,
        discovery: &CompanionSourceDiscovery,
        primary_statement_count: usize,
        handoff: Option<&runmat_package::FrozenProjectHandoff>,
        retained: &[(runmat_hir::SourceId, SourceContextData)],
    ) -> Result<Self, String> {
        let mut companions = BTreeMap::<SourceKey, CompanionSource>::new();
        let mut discovered_locations = BTreeMap::<(String, String), String>::new();
        for discovered in &discovery.statements {
            let Some(context) = &discovered.source else {
                return Err("companion statement is missing source provenance".into());
            };
            let (owner_identity, relative_path) = executable_source_identity(
                root,
                handoff,
                root_full_path,
                &context.display_name,
                context.fullpath_name.as_deref(),
            )?;
            if owner_identity == root.owner_identity && relative_path == root.relative_path {
                return Err("companion source resolves to the primary source".into());
            }
            let location = (owner_identity.clone(), relative_path.clone());
            if let Some(previous_text) = discovered_locations.get(&location) {
                if previous_text != &context.text {
                    return Err("one companion source has conflicting discovered text".into());
                }
            } else {
                discovered_locations.insert(location, context.text.clone());
            }
            let key = SourceKey::new(
                owner_identity.clone(),
                relative_path.clone(),
                context.text.clone(),
            );
            if !companions.contains_key(&key) {
                companions.insert(
                    key,
                    CompanionSource {
                        context: Arc::clone(context),
                        owner_identity,
                        relative_path,
                    },
                );
            }
        }

        let primary_key = SourceKey::new(
            root.owner_identity.clone(),
            root.relative_path.clone(),
            root.text.clone(),
        );
        let mut retained_keys = HashMap::new();
        for (runtime_id, context) in retained {
            let (owner_identity, relative_path) = executable_source_identity(
                root,
                handoff,
                root_full_path,
                &context.display_name,
                context.fullpath_name.as_deref(),
            )?;
            let key = SourceKey::new(
                owner_identity.clone(),
                relative_path.clone(),
                context.text.clone(),
            );
            retained_keys.insert(*runtime_id, key.clone());
            if key != primary_key && !companions.contains_key(&key) {
                companions.insert(
                    key,
                    CompanionSource {
                        context: Arc::new(context.clone()),
                        owner_identity,
                        relative_path,
                    },
                );
            }
        }

        let mut entries = Vec::with_capacity(companions.len() + 1);
        entries.push(crate::SourceMapEntry {
            source_id: ProgramSourceId(0),
            owner_identity: root.owner_identity.clone(),
            relative_path: root.relative_path.clone(),
            content_revision: runmat_execution::Digest::sha256(root.text.as_bytes()),
            display_name: root.relative_path.clone(),
            full_path: root_full_path.map(ToString::to_string),
            text: root.text.clone(),
        });
        let mut ids = BTreeMap::new();
        ids.insert(primary_key, ProgramSourceId(0));
        for (key, source) in companions {
            let ordinal = u32::try_from(entries.len())
                .map_err(|_| "portable program has too many source units".to_string())?;
            ids.insert(key, ProgramSourceId(ordinal));
            entries.push(crate::SourceMapEntry {
                source_id: ProgramSourceId(ordinal),
                owner_identity: source.owner_identity,
                relative_path: source.relative_path,
                content_revision: runmat_execution::Digest::sha256(source.context.text.as_bytes()),
                display_name: source.context.display_name.clone(),
                full_path: source.context.fullpath_name.clone(),
                text: source.context.text.clone(),
            });
        }

        let mut statement_sources = vec![ProgramSourceId(0); primary_statement_count];
        let mut function_sources = HashMap::new();
        for discovered in &discovery.statements {
            let context = discovered
                .source
                .as_ref()
                .expect("companion source was validated above");
            let key = executable_source_identity(
                root,
                handoff,
                root_full_path,
                &context.display_name,
                context.fullpath_name.as_deref(),
            )?;
            let source_id = ids[&SourceKey::new(key.0, key.1, context.text.clone())];
            statement_sources.push(source_id);
            for name in source_context_function_names_in_statements(std::slice::from_ref(
                &discovered.statement,
            )) {
                function_sources.insert(name, source_id);
            }
        }
        let retained_runtime_sources = retained_keys
            .into_iter()
            .map(|(runtime_id, key)| (runtime_id, ids[&key]))
            .collect();
        Ok(Self {
            source_map: crate::ExecutableSourceMap::new(entries),
            statement_sources,
            function_sources,
            retained_runtime_sources,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::compile::DiscoveredSourceStatement;

    fn discovered(text: &str, name: &str, display: &str) -> DiscoveredSourceStatement {
        let statement = runmat_parser::parse(text)
            .expect("source parses")
            .body
            .into_iter()
            .next()
            .expect("source declares one function");
        DiscoveredSourceStatement {
            statement,
            is_private: false,
            source: Some(Arc::new(SourceContextData {
                display_name: display.into(),
                fullpath_name: Some(format!("/project/{name}")),
                text: text.into(),
            })),
        }
    }

    #[test]
    fn companion_order_does_not_change_source_ordinals_or_map() {
        let root = crate::ExecutableSource::new("package", "main.m", "value = a() + z();");
        let a = discovered("function y = a(); y = 1; end", "a.m", "a.m");
        let z = discovered("function y = z(); y = 2; end", "z.m", "z.m");
        let one = CompanionSourceDiscovery {
            statements: vec![z, a],
            ..Default::default()
        };
        let first = ProgramSourceTable::build(&root, None, &one, 1, None, &[]).unwrap();
        assert_eq!(first.function_sources["a"], ProgramSourceId(1));
        assert_eq!(first.function_sources["z"], ProgramSourceId(2));
        assert_eq!(
            first.statement_sources,
            vec![ProgramSourceId(0), ProgramSourceId(2), ProgramSourceId(1)]
        );
        let two = CompanionSourceDiscovery {
            statements: vec![
                discovered("function y = a(); y = 1; end", "a.m", "a.m"),
                discovered("function y = z(); y = 2; end", "z.m", "z.m"),
            ],
            ..Default::default()
        };
        let second = ProgramSourceTable::build(&root, None, &two, 1, None, &[]).unwrap();
        assert_eq!(first.source_map, second.source_map);
        assert_eq!(second.statement_sources[0], ProgramSourceId(0));
    }

    #[test]
    fn one_canonical_source_cannot_publish_conflicting_text() {
        let root = crate::ExecutableSource::new("package", "main.m", "x = 1;");
        let discovery = CompanionSourceDiscovery {
            statements: vec![
                discovered("function y = a(); y = 1; end", "a.m", "a.m"),
                discovered("function y = a(); y = 2; end", "a.m", "a.m"),
            ],
            ..Default::default()
        };
        assert!(
            ProgramSourceTable::build(&root, None, &discovery, 1, None, &[])
                .err()
                .expect("conflicting source must fail")
                .contains("conflicting discovered text")
        );
    }

    #[test]
    fn retained_definition_sources_use_stable_ordinals_not_runtime_ids() {
        let root = crate::ExecutableSource::new("interactive", "main.m", "value = prior();");
        let prior = SourceContextData {
            display_name: "prior.m".into(),
            fullpath_name: Some("/project/prior.m".into()),
            text: "function y = prior(); y = 4; end".into(),
        };
        let empty = CompanionSourceDiscovery::default();
        let first = ProgramSourceTable::build(
            &root,
            None,
            &empty,
            1,
            None,
            &[(runmat_hir::SourceId(99), prior.clone())],
        )
        .unwrap();
        let second = ProgramSourceTable::build(
            &root,
            None,
            &empty,
            1,
            None,
            &[(runmat_hir::SourceId(2), prior)],
        )
        .unwrap();
        assert_eq!(first.source_map, second.source_map);
        assert_eq!(
            first.retained_runtime_sources[&runmat_hir::SourceId(99)],
            ProgramSourceId(1)
        );
        assert_eq!(
            second.retained_runtime_sources[&runmat_hir::SourceId(2)],
            ProgramSourceId(1)
        );
    }

    #[test]
    fn retained_generations_at_one_location_have_explicit_content_revisions() {
        let root = crate::ExecutableSource::new("interactive", "main.m", "value = prior();");
        let source = |text: &str| SourceContextData {
            display_name: "prior.m".into(),
            fullpath_name: Some("/project/prior.m".into()),
            text: text.into(),
        };
        let table = ProgramSourceTable::build(
            &root,
            None,
            &CompanionSourceDiscovery::default(),
            1,
            None,
            &[
                (
                    runmat_hir::SourceId(5),
                    source("function y = prior(); y = 1; end"),
                ),
                (
                    runmat_hir::SourceId(6),
                    source("function y = prior(); y = 2; end"),
                ),
            ],
        )
        .unwrap();
        assert_eq!(table.source_map.entries().len(), 3);
        assert_ne!(
            table.retained_runtime_sources[&runmat_hir::SourceId(5)],
            table.retained_runtime_sources[&runmat_hir::SourceId(6)]
        );
    }

    #[test]
    fn same_retained_sources_ignore_runtime_issuance_order_but_not_content() {
        let root = crate::ExecutableSource::new("interactive", "main.m", "y = prior();");
        let source = |name: &str, text: &str| SourceContextData {
            display_name: name.into(),
            fullpath_name: Some(format!("/project/{name}")),
            text: text.into(),
        };
        let a = source("a.m", "function y = a(); y = 1; end");
        let b = source("b.m", "function y = b(); y = 2; end");
        let build = |retained: &[(runmat_hir::SourceId, SourceContextData)]| {
            ProgramSourceTable::build(
                &root,
                None,
                &CompanionSourceDiscovery::default(),
                1,
                None,
                retained,
            )
            .unwrap()
            .source_map
        };
        let first = build(&[
            (runmat_hir::SourceId(99), a.clone()),
            (runmat_hir::SourceId(100), b.clone()),
        ]);
        let reordered = build(&[
            (runmat_hir::SourceId(2), b.clone()),
            (runmat_hir::SourceId(1), a),
        ]);
        assert_eq!(first, reordered);
        let changed = build(&[
            (runmat_hir::SourceId(2), b),
            (
                runmat_hir::SourceId(1),
                source("a.m", "function y = a(); y = 3; end"),
            ),
        ]);
        assert_ne!(first, changed);
    }
}
