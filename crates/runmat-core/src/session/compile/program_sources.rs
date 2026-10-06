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
pub(in crate::session) struct ProgramSourceTable {
    pub(in crate::session) source_map: crate::ExecutableSourceMap,
    pub(in crate::session) statement_sources: Vec<ProgramSourceId>,
    pub(in crate::session) function_sources: HashMap<String, ProgramSourceId>,
    pub retained_runtime_sources:
        HashMap<runmat_runtime::source_context::RuntimeSourceRef, ProgramSourceId>,
}

pub(super) struct SourceCompileProjection {
    pub runtime_to_program:
        HashMap<runmat_runtime::source_context::RuntimeSourceRef, ProgramSourceId>,
    pub program_to_runtime:
        HashMap<ProgramSourceId, runmat_runtime::source_context::RuntimeSourceRef>,
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
    pub(in crate::session) fn validate_compiled_sources(
        &self,
        bytecode: &runmat_vm::Bytecode,
    ) -> Result<(), String> {
        let contains = |source: Option<runmat_vm::BytecodeSourceRef>| {
            source
                .and_then(runmat_vm::BytecodeSourceRef::program)
                .is_some_and(|ordinal| {
                    usize::try_from(ordinal.0)
                        .ok()
                        .and_then(|index| self.source_map.entries().get(index))
                        .is_some_and(|entry| entry.source_id == ordinal)
                })
        };
        if !contains(bytecode.source_id) {
            return Err("compiled entrypoint has no admitted program source".into());
        }
        for function in bytecode.function_registry.functions.values() {
            if !contains(function.source_id) {
                return Err(format!(
                    "compiled function {:?} has no admitted program source",
                    function.function
                ));
            }
        }
        Ok(())
    }

    pub(in crate::session) fn dynamic_root(
        path: &str,
        text: &str,
        handoff: Option<&runmat_package::FrozenProjectHandoff>,
        origin: &runmat_runtime::builtins::common::path_search::FileCandidateOrigin,
    ) -> Result<crate::ExecutableSource, String> {
        let (owner, relative) = match origin {
            runmat_runtime::builtins::common::path_search::FileCandidateOrigin::SearchRoot {
                ordinal,
                relative,
                ..
            } => (
                format!("dynamic-search-root:{ordinal}"),
                relative.to_string_lossy().replace('\\', "/"),
            ),
            runmat_runtime::builtins::common::path_search::FileCandidateOrigin::Direct {
                spelling,
            } => (
                "dynamic-explicit-path".to_string(),
                spelling.to_string_lossy().replace('\\', "/"),
            ),
        };
        let fallback = crate::ExecutableSource::new(owner, relative, text);
        if handoff.is_none() {
            return Ok(fallback);
        }
        let (owner, relative) =
            executable_source_identity(&fallback, handoff, Some(path), path, Some(path))?;
        Ok(crate::ExecutableSource::new(owner, relative, text))
    }

    pub fn intern_runtime_projection(
        &self,
        pool: &mut crate::source_pool::SourcePool,
        runtime_context: &runmat_runtime::context::RuntimeContext,
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
            let issued = pool
                .issued_source_ref(runtime, runtime_context)
                .map_err(|error| error.to_string())?;
            if let Some(previous) = runtime_to_program.insert(issued, program) {
                if previous != program {
                    return Err("distinct program sources share one session source identity".into());
                }
            }
            program_to_runtime.insert(program, issued);
        }
        for (issued, program) in &self.retained_runtime_sources {
            if issued.resolve_in(runtime_context).is_none() {
                return Err("retained source belongs to a missing or foreign session owner".into());
            }
            if let Some(previous) = runtime_to_program.insert(*issued, *program) {
                if previous != *program {
                    return Err("retained source identity disagrees with the source table".into());
                }
            }
        }
        Ok(SourceCompileProjection {
            runtime_to_program,
            program_to_runtime,
        })
    }

    pub(in crate::session) fn build(
        root: &crate::ExecutableSource,
        root_full_path: Option<&str>,
        discovery: &CompanionSourceDiscovery,
        primary_statement_count: usize,
        handoff: Option<&runmat_package::FrozenProjectHandoff>,
        retained: &[(
            runmat_runtime::source_context::RuntimeSourceRef,
            SourceContextData,
        )],
    ) -> Result<Self, String> {
        Self::build_with_origin(
            root,
            root_full_path,
            discovery,
            primary_statement_count,
            handoff,
            None,
            retained,
        )
    }

    pub(in crate::session) fn build_dynamic(
        root: &crate::ExecutableSource,
        root_full_path: &str,
        discovery: &CompanionSourceDiscovery,
        primary_statement_count: usize,
        handoff: Option<&runmat_package::FrozenProjectHandoff>,
        origin: &runmat_runtime::builtins::common::path_search::FileCandidateOrigin,
    ) -> Result<Self, String> {
        Self::build_with_origin(
            root,
            Some(root_full_path),
            discovery,
            primary_statement_count,
            handoff,
            Some(origin),
            &[],
        )
    }

    fn build_with_origin(
        root: &crate::ExecutableSource,
        root_full_path: Option<&str>,
        discovery: &CompanionSourceDiscovery,
        primary_statement_count: usize,
        handoff: Option<&runmat_package::FrozenProjectHandoff>,
        origin: Option<&runmat_runtime::builtins::common::path_search::FileCandidateOrigin>,
        retained: &[(
            runmat_runtime::source_context::RuntimeSourceRef,
            SourceContextData,
        )],
    ) -> Result<Self, String> {
        let source_identity = |context: &SourceContextData| {
            if handoff.is_none() {
                if let Some(runmat_runtime::builtins::common::path_search::FileCandidateOrigin::SearchRoot {
                    ordinal,
                    access_root,
                    ..
                }) = origin
                {
                    let full = context.fullpath_name.as_deref().ok_or_else(|| {
                        "loose companion lacks captured full access path".to_string()
                    })?;
                    let relative = std::path::Path::new(full)
                        .strip_prefix(access_root)
                        .map_err(|_| "loose companion is outside its selected search root".to_string())?;
                    return Ok((
                        format!("dynamic-search-root:{ordinal}"),
                        relative.to_string_lossy().replace('\\', "/"),
                    ));
                }
            }
            executable_source_identity(
                root,
                handoff,
                root_full_path,
                &context.display_name,
                context.fullpath_name.as_deref(),
            )
        };
        let mut companions = BTreeMap::<SourceKey, CompanionSource>::new();
        let mut discovered_locations = BTreeMap::<(String, String), String>::new();
        for discovered in &discovery.statements {
            let Some(context) = &discovered.source else {
                return Err("companion statement is missing source provenance".into());
            };
            let (owner_identity, relative_path) = source_identity(context)?;
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
            let (owner_identity, relative_path) = source_identity(context)?;
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
            let display_name = if handoff.is_none()
                && origin.is_some_and(|origin| {
                    matches!(
                        origin,
                        runmat_runtime::builtins::common::path_search::FileCandidateOrigin::SearchRoot { .. }
                    )
                })
            {
                source.relative_path.clone()
            } else {
                source.context.display_name.clone()
            };
            entries.push(crate::SourceMapEntry {
                source_id: ProgramSourceId(ordinal),
                owner_identity: source.owner_identity,
                relative_path: source.relative_path,
                content_revision: runmat_execution::Digest::sha256(source.context.text.as_bytes()),
                display_name,
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
            let key = source_identity(context)?;
            let source_id = *ids
                .get(&SourceKey::new(key.0, key.1, context.text.clone()))
                .ok_or_else(|| {
                    "companion source is absent from the admitted source table".to_string()
                })?;
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
mod authority_tests {
    use super::*;

    #[test]
    fn reverse_projection_uses_issued_authority_not_pool_integer() {
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let _unrelated = runmat_runtime::source_context::lease_source_context_in_context(
            &runtime,
            "unrelated.m",
            None,
            "unrelated = 1;",
        )
        .expect("advance the selected source issuer");
        let text = "function y = main(); y = 2; end";
        let table = ProgramSourceTable {
            source_map: crate::ExecutableSourceMap::new(vec![crate::SourceMapEntry {
                source_id: ProgramSourceId(0),
                owner_identity: "fixture".into(),
                relative_path: "main.m".into(),
                content_revision: runmat_execution::Digest::sha256(text.as_bytes()),
                display_name: "main.m".into(),
                full_path: None,
                text: text.into(),
            }]),
            statement_sources: Vec::new(),
            function_sources: HashMap::new(),
            retained_runtime_sources: HashMap::new(),
        };
        let mut pool = crate::source_pool::SourcePool::default();
        let projection = table
            .intern_runtime_projection(&mut pool, &runtime)
            .expect("issued projection");
        let issued = projection.program_to_runtime[&ProgramSourceId(0)];
        assert_eq!(pool.entries().next().map(|(id, _)| id.0), Some(0));
        assert_eq!(issued.source_id().0, 1);
        assert_eq!(
            projection.program_ref_for_runtime_ref(runmat_vm::BytecodeSourceRef::Runtime(issued)),
            Some(runmat_vm::BytecodeSourceRef::Program(ProgramSourceId(0)))
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::compile::DiscoveredSourceStatement;

    #[test]
    fn loose_search_root_identity_is_relocatable_and_distinguishes_equal_leaf_sources() {
        use runmat_runtime::builtins::common::path_search::FileCandidateOrigin;
        let text = "function y = same(); y = 1; end";
        let origin = |ordinal| FileCandidateOrigin::SearchRoot {
            ordinal,
            access_root: std::path::PathBuf::from("/checkout-one/pkg"),
            relative: std::path::PathBuf::from("+pkg/same.m"),
        };
        let first = ProgramSourceTable::dynamic_root(
            "/checkout-one/pkg/+pkg/same.m",
            text,
            None,
            &origin(0),
        )
        .unwrap();
        let relocated = ProgramSourceTable::dynamic_root(
            "/checkout-two/pkg/+pkg/same.m",
            text,
            None,
            &origin(0),
        )
        .unwrap();
        let next_root = ProgramSourceTable::dynamic_root(
            "/checkout-one/other/+pkg/same.m",
            text,
            None,
            &origin(1),
        )
        .unwrap();
        assert_eq!(first, relocated);
        assert_ne!(first, next_root);
        let build = |root: &crate::ExecutableSource, full_path: &str| {
            let map = ProgramSourceTable::build(
                root,
                Some(full_path),
                &CompanionSourceDiscovery::default(),
                1,
                None,
                &[],
            )
            .unwrap()
            .source_map
            .portable_v2()
            .unwrap();
            serde_json::to_vec(&map).unwrap()
        };
        assert_eq!(
            build(&first, "/checkout-one/pkg/+pkg/same.m"),
            build(&relocated, "/checkout-two/pkg/+pkg/same.m")
        );
        assert_ne!(
            build(&first, "/checkout-one/pkg/+pkg/same.m"),
            build(&next_root, "/checkout-one/other/+pkg/same.m")
        );
    }

    #[test]
    fn loose_companion_uses_the_same_captured_search_root_not_cwd_display_text() {
        use runmat_runtime::builtins::common::path_search::FileCandidateOrigin;
        let build = |prefix: &str| {
            let access_root = std::path::PathBuf::from(prefix);
            let primary_access = format!("{prefix}/+pkg/main.m");
            let companion_access = format!("{prefix}/+pkg/helper.m");
            let origin = FileCandidateOrigin::SearchRoot {
                ordinal: 2,
                access_root,
                relative: std::path::PathBuf::from("+pkg/main.m"),
            };
            let primary = ProgramSourceTable::dynamic_root(
                &primary_access,
                "value = helper();",
                None,
                &origin,
            )
            .unwrap();
            let mut discovery = CompanionSourceDiscovery::default();
            discovery.statements.push(DiscoveredSourceStatement {
                statement: runmat_parser::parse("function y = helper(); y = 5; end")
                    .unwrap()
                    .body
                    .remove(0),
                is_private: false,
                source: Some(Arc::new(SourceContextData {
                    display_name: companion_access.clone(),
                    fullpath_name: Some(companion_access),
                    text: "function y = helper(); y = 5; end".into(),
                })),
            });
            let table = ProgramSourceTable::build_dynamic(
                &primary,
                &primary_access,
                &discovery,
                1,
                None,
                &origin,
            )
            .unwrap();
            assert_eq!(table.source_map.entries()[1].relative_path, "+pkg/helper.m");
            assert_eq!(
                table.source_map.entries()[1].owner_identity,
                "dynamic-search-root:2"
            );
            serde_json::to_vec(&table.source_map.portable_v2().unwrap()).unwrap()
        };
        assert_eq!(build("/checkout-one"), build("/checkout-two"));
    }

    fn runtime() -> runmat_runtime::context::RuntimeContext {
        runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
    }

    fn lease(
        runtime: &runmat_runtime::context::RuntimeContext,
        source: &SourceContextData,
    ) -> runmat_runtime::source_context::RuntimeSourceLease {
        runmat_runtime::source_context::lease_source_context_in_context(
            runtime,
            source.display_name.as_str(),
            source.fullpath_name.as_deref().map(Arc::<str>::from),
            source.text.as_str(),
        )
        .expect("canonical retained source")
    }

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
        let runtime = runtime();
        let first_lease = lease(&runtime, &prior);
        let second_lease = lease(&runtime, &prior);
        let empty = CompanionSourceDiscovery::default();
        let first = ProgramSourceTable::build(
            &root,
            None,
            &empty,
            1,
            None,
            &[(first_lease.source_ref(), prior.clone())],
        )
        .unwrap();
        let second = ProgramSourceTable::build(
            &root,
            None,
            &empty,
            1,
            None,
            &[(second_lease.source_ref(), prior)],
        )
        .unwrap();
        assert_eq!(first.source_map, second.source_map);
        assert_eq!(
            first.retained_runtime_sources[&first_lease.source_ref()],
            ProgramSourceId(1)
        );
        assert_eq!(
            second.retained_runtime_sources[&second_lease.source_ref()],
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
        let runtime = runtime();
        let first_source = source("function y = prior(); y = 1; end");
        let second_source = source("function y = prior(); y = 2; end");
        let first_lease = lease(&runtime, &first_source);
        let second_lease = lease(&runtime, &second_source);
        let table = ProgramSourceTable::build(
            &root,
            None,
            &CompanionSourceDiscovery::default(),
            1,
            None,
            &[
                (first_lease.source_ref(), first_source),
                (second_lease.source_ref(), second_source),
            ],
        )
        .unwrap();
        assert_eq!(table.source_map.entries().len(), 3);
        assert_ne!(
            table.retained_runtime_sources[&first_lease.source_ref()],
            table.retained_runtime_sources[&second_lease.source_ref()]
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
        let changed_a = source("a.m", "function y = a(); y = 3; end");
        let runtime = runtime();
        let first_a = lease(&runtime, &a);
        let first_b = lease(&runtime, &b);
        let second_b = lease(&runtime, &b);
        let second_a = lease(&runtime, &a);
        let changed = lease(&runtime, &changed_a);
        let build = |retained: &[(
            runmat_runtime::source_context::RuntimeSourceRef,
            SourceContextData,
        )]| {
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
            (first_a.source_ref(), a.clone()),
            (first_b.source_ref(), b.clone()),
        ]);
        let reordered = build(&[
            (second_b.source_ref(), b.clone()),
            (second_a.source_ref(), a),
        ]);
        assert_eq!(first, reordered);
        let changed = build(&[
            (second_b.source_ref(), b),
            (changed.source_ref(), changed_a),
        ]);
        assert_ne!(first, changed);
    }
}
