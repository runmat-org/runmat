#[cfg(test)]
use super::source_context_function_names_in_statements;
use super::{CompanionSourceDiscovery, SourceContextData};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use runmat_types::ProgramSourceId;

#[path = "program_sources/identity.rs"]
pub(super) mod identity;
use super::test_source_authority::CapturedTestSourceAuthority;
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
    primary_source: ProgramSourceId,
    pub(in crate::session) source_map: crate::ExecutableSourceMap,
    pub(in crate::session) statement_sources: Vec<ProgramSourceId>,
    #[cfg(test)]
    pub(in crate::session) function_sources: HashMap<String, ProgramSourceId>,
    pub retained_runtime_sources:
        HashMap<runmat_runtime::source_context::RuntimeSourceRef, ProgramSourceId>,
}

pub(crate) struct SourceCompileProjection {
    pub runtime_to_program:
        HashMap<runmat_runtime::source_context::RuntimeSourceRef, ProgramSourceId>,
    pub program_to_runtime:
        HashMap<ProgramSourceId, runmat_runtime::source_context::RuntimeSourceRef>,
}

impl SourceCompileProjection {
    pub(crate) fn program_ref_for_runtime_ref(
        &self,
        source: runmat_vm::BytecodeSourceRef,
    ) -> Option<runmat_vm::BytecodeSourceRef> {
        let runtime = source.runtime()?;
        self.runtime_to_program
            .get(&runtime)
            .copied()
            .map(runmat_vm::BytecodeSourceRef::Program)
    }

    pub(crate) fn runtime_ref_for_program_ref(
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

enum SelectedSourceAuthority<'a, 'snapshot> {
    Ordinary,
    CapturedTest(&'a CapturedTestSourceAuthority<'snapshot>),
    Dynamic(&'a runmat_runtime::builtins::common::path_search::FileCandidateOrigin),
}

impl ProgramSourceTable {
    pub(in crate::session) fn primary_source(&self) -> ProgramSourceId {
        self.primary_source
    }

    pub(in crate::session) fn validate_compiled_sources(
        &self,
        bytecode: &runmat_vm::Bytecode,
        hir: &runmat_hir::HirAssembly,
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
        let primary = self.primary_source;
        if hir.primary_source != Some(primary)
            || !hir.modules.iter().any(|module| module.source_id == primary)
        {
            return Err("HIR primary source differs from the admitted source table".into());
        }
        for class in &hir.classes {
            let module_source = hir
                .modules
                .get(class.module.0)
                .filter(|module| module.id == class.module)
                .map(|module| module.source_id)
                .ok_or("compiled class has no HIR source module")?;
            if class.declaration.source != module_source
                || !contains(Some(runmat_vm::BytecodeSourceRef::Program(module_source)))
            {
                return Err("class declaration has no admitted original program source".into());
            }
        }
        match (bytecode.active_function, bytecode.source_id) {
            (None, Some(source)) if hir.entrypoints.is_empty() => {
                if source.program() != Some(primary) || !contains(Some(source)) {
                    return Err("compiled function file has no admitted primary source".into());
                }
            }
            (Some(active), Some(source)) => {
                let function = hir
                    .functions
                    .iter()
                    .find(|function| function.id == active)
                    .ok_or("compiled entrypoint is absent from HIR")?;
                let module_source = hir
                    .modules
                    .get(function.module.0)
                    .filter(|module| module.id == function.module)
                    .map(|module| module.source_id)
                    .ok_or("compiled entrypoint has no HIR source module")?;
                if !hir.entrypoints.iter().any(|entry| entry.target == active)
                    || source.program() != Some(module_source)
                    || !contains(Some(source))
                {
                    return Err("compiled entrypoint has no admitted program source".into());
                }
            }
            _ => return Err("compiled entrypoint has no admitted program source".into()),
        }
        for function in &hir.functions {
            if Some(function.id) == bytecode.active_function {
                continue;
            }
            let module_source = hir
                .modules
                .get(function.module.0)
                .filter(|module| module.id == function.module)
                .map(|module| module.source_id)
                .ok_or("compiled function has no HIR source module")?;
            let compiled = bytecode
                .function_registry
                .functions
                .get(&function.id)
                .ok_or("HIR function is absent from compiled registry")?;
            let compiled_source = compiled
                .source_id
                .and_then(runmat_vm::BytecodeSourceRef::program);
            let bound_source = bytecode
                .bound_functions
                .get(&function.id)
                .and_then(|body| body.source_id)
                .and_then(runmat_vm::BytecodeSourceRef::program);
            let admitted = contains(compiled.source_id);
            if compiled_source != Some(module_source)
                || bound_source != Some(module_source)
                || !admitted
            {
                return Err(format!(
                    "compiled function {:?} differs from its admitted HIR source module: HIR={module_source:?}, registry={compiled_source:?}, bound={bound_source:?}, registry_admitted={admitted}",
                    function.id,
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

    pub(super) fn intern_runtime_projection(
        &self,
        pool: &mut crate::source_pool::SourcePool,
        runtime_context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<SourceCompileProjection, String> {
        let admitted = runmat_runtime::source_context::ValidatedPortableSourceMap::new(Arc::new(
            self.source_map.portable_v2()?,
        ))
        .map_err(|error| error.to_string())?;
        let mut program_to_runtime = HashMap::new();
        let mut runtime_to_program = HashMap::new();
        for entry in self.source_map.entries() {
            let program = entry.source_id;
            let selected = admitted.record(program).ok_or_else(|| {
                "portable source ordinal is absent from its admitted map".to_string()
            })?;
            let runtime = pool
                .intern_portable_entry(entry, selected)
                .map_err(|error| error.to_string())?;
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
            runmat_runtime::source_context::PortableSourceIdentity,
        )],
    ) -> Result<Self, String> {
        Self::build_with_origin(
            root,
            root_full_path,
            discovery,
            primary_statement_count,
            handoff,
            SelectedSourceAuthority::Ordinary,
            retained,
        )
    }

    pub(in crate::session) fn build_with_test(
        root: &crate::ExecutableSource,
        root_full_path: Option<&str>,
        discovery: &CompanionSourceDiscovery,
        primary_statement_count: usize,
        handoff: Option<&runmat_package::FrozenProjectHandoff>,
        test_authority: &CapturedTestSourceAuthority<'_>,
        retained: &[(
            runmat_runtime::source_context::RuntimeSourceRef,
            SourceContextData,
            runmat_runtime::source_context::PortableSourceIdentity,
        )],
    ) -> Result<Self, String> {
        Self::build_with_origin(
            root,
            root_full_path,
            discovery,
            primary_statement_count,
            handoff,
            SelectedSourceAuthority::CapturedTest(test_authority),
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
            SelectedSourceAuthority::Dynamic(origin),
            &[],
        )
    }

    fn build_with_origin(
        root: &crate::ExecutableSource,
        root_full_path: Option<&str>,
        discovery: &CompanionSourceDiscovery,
        primary_statement_count: usize,
        handoff: Option<&runmat_package::FrozenProjectHandoff>,
        authority: SelectedSourceAuthority<'_, '_>,
        retained: &[(
            runmat_runtime::source_context::RuntimeSourceRef,
            SourceContextData,
            runmat_runtime::source_context::PortableSourceIdentity,
        )],
    ) -> Result<Self, String> {
        let (test_authority, origin) = match authority {
            SelectedSourceAuthority::Ordinary => (None, None),
            SelectedSourceAuthority::CapturedTest(test) => (Some(test), None),
            SelectedSourceAuthority::Dynamic(origin) => (None, Some(origin)),
        };
        let source_identity = |context: &SourceContextData| {
            if let (Some(authority), Some(full)) =
                (test_authority, context.fullpath_name.as_deref())
            {
                if let Some(identity) = authority.source_identity(full, &context.text)? {
                    return Ok(identity);
                }
            }
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
                    #[cfg(windows)]
                    let access_root = if std::path::Path::new(full).starts_with(access_root) {
                        access_root.as_path()
                    } else {
                        discovery.canonical_search_root.as_deref().unwrap_or(access_root)
                    };
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
            companions.entry(key).or_insert_with(|| CompanionSource {
                context: Arc::clone(context),
                owner_identity,
                relative_path,
            });
        }

        let primary_key = SourceKey::new(
            root.owner_identity.clone(),
            root.relative_path.clone(),
            root.text.clone(),
        );
        let mut retained_keys = HashMap::new();
        for (runtime_id, context, identity) in retained {
            if identity.content_revision
                != runmat_execution::Digest::sha256(context.text.as_bytes())
            {
                return Err(
                    "retained source text disagrees with its issued portable identity".into(),
                );
            }
            let owner_identity = identity.owner_identity.clone();
            let relative_path = identity.relative_path.clone();
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

        let primary_source = ProgramSourceId(0);
        let mut entries = Vec::with_capacity(companions.len() + 1);
        entries.push(crate::SourceMapEntry {
            source_id: primary_source,
            owner_identity: root.owner_identity.clone(),
            relative_path: root.relative_path.clone(),
            content_revision: runmat_execution::Digest::sha256(root.text.as_bytes()),
            display_name: root.relative_path.clone(),
            full_path: root_full_path.map(ToString::to_string),
            text: root.text.clone(),
        });
        let mut ids = BTreeMap::new();
        ids.insert(primary_key, primary_source);
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

        let mut statement_sources = vec![primary_source; primary_statement_count];
        #[cfg(test)]
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
            #[cfg(test)]
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
            primary_source,
            source_map: crate::ExecutableSourceMap::new(entries),
            statement_sources,
            #[cfg(test)]
            function_sources,
            retained_runtime_sources,
        })
    }
}

#[cfg(test)]
#[path = "program_sources/authority_tests.rs"]
mod authority_tests;

#[cfg(test)]
#[path = "program_sources/tests.rs"]
mod tests;
