//! Read-only observation of the selected runtime name environment.
//!
//! Executable selection stays with the semantic resolver and dynamic loader.
//! This view asks those owners for their selected facts and uses the shared
//! path candidate iterator; it never loads a module or compiles source.

use std::collections::HashSet;

use runmat_builtins::builtin_id_by_name;

use crate::builtins::common::path_search::{
    class_file_candidates_with_origin, class_folder_candidates_with_origin,
    directory_candidates_with_origin, find_all_files_with_extensions_with_origin,
    find_class_file_with_origin, DirectoryCandidate, FileCandidate, CLASS_M_FILE_EXTENSIONS,
    GENERAL_FILE_EXTENSIONS,
};
use crate::user_functions::{
    current_scoped_name_target_resolver, current_semantic_name_candidates,
    resolve_semantic_function_by_name, source_function_for, SemanticNameCandidate,
};

use super::{NativeNameFact, NativeResolutionScope, SourceFunctionInfo};

#[derive(Debug)]
pub enum RuntimeNameResolutionError {
    Workspace(Box<crate::RuntimeError>),
    WasmRegistryNotReady,
    SearchPathChanged,
    ClassRegistryChanged,
    Search(crate::builtins::common::path_search::PathSearchError),
    Control(Box<crate::RuntimeError>),
}

impl std::fmt::Display for RuntimeNameResolutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Workspace(error) => formatter.write_str(error.message()),
            Self::WasmRegistryNotReady => {
                formatter.write_str("name resolution: WASM builtin registry is not ready")
            }
            Self::SearchPathChanged => {
                formatter.write_str("name resolution: search path changed during observation")
            }
            Self::ClassRegistryChanged => {
                formatter.write_str("name resolution: class registry changed during observation")
            }
            Self::Search(error) => std::fmt::Display::fmt(error, formatter),
            Self::Control(error) => formatter.write_str(error.message()),
        }
    }
}

impl From<crate::builtins::common::path_search::PathSearchError> for RuntimeNameResolutionError {
    fn from(error: crate::builtins::common::path_search::PathSearchError) -> Self {
        Self::Search(error)
    }
}

impl From<crate::RuntimeError> for RuntimeNameResolutionError {
    fn from(error: crate::RuntimeError) -> Self {
        Self::Control(Box::new(error))
    }
}

/// The direct dynamic-name route probes native modules before public semantic
/// names; the active caller's exact private selection precedes native modules.
/// Named `feval` resolves semantic functions and builtins first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeNameRoute {
    DynamicCall,
    NamedFeval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeNameFilter {
    Any,
    Workspace,
    Builtin,
    File,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeNameQuery {
    pub route: RuntimeNameRoute,
    pub filter: RuntimeNameFilter,
    pub all: bool,
}

#[derive(Debug, Clone)]
pub enum RuntimeNameCandidate {
    WorkspaceVariable,
    RegisteredConstructor(runmat_types::ClassIdentity),
    RegisteredStaticMethod {
        resolution: crate::dispatcher::RegisteredStaticMethodResolution,
        source: Option<SourceFunctionInfo>,
    },
    Native(NativeNameFact),
    Semantic {
        function: usize,
        source: Option<SourceFunctionInfo>,
        definition_owner: Option<std::sync::Arc<runmat_value::FunctionDefinitionOwner>>,
    },
    Builtin(runmat_types::BuiltinId),
    Source(FileCandidate),
    ClassFolder(DirectoryCandidate),
    ClassFile(FileCandidate),
    File(FileCandidate),
    Directory(DirectoryCandidate),
}

#[derive(Debug, Clone)]
pub struct RuntimeNameResolution {
    pub name: String,
    pub query: RuntimeNameQuery,
    pub captured_path_revision: Option<u64>,
    pub captured_class_revision: crate::class_registry::ClassRegistryRevision,
    /// The first candidate is selected; later candidates are shadowed.
    pub candidates: Vec<RuntimeNameCandidate>,
}

impl RuntimeNameResolution {
    pub fn selected(&self) -> Option<&RuntimeNameCandidate> {
        self.candidates.first()
    }

    pub fn shadowed(&self) -> &[RuntimeNameCandidate] {
        self.candidates.get(1..).unwrap_or(&[])
    }
}

/// Observe without invoking any candidate. A native observer is supplied by
/// the installed session; standalone/browser sessions simply have no native
/// selection. Path candidates carry their captured search-root origin.
pub async fn observe_runtime_name(
    name: &str,
    query: RuntimeNameQuery,
) -> Result<RuntimeNameResolution, RuntimeNameResolutionError> {
    // Dispatch bootstraps WASM builtin registration before invoking `which`.
    // A direct observer caller must wait for that same inventory; installing it
    // here would turn observation into a state-changing operation.
    #[cfg(target_arch = "wasm32")]
    if !runmat_builtins::wasm_registry::is_registered() {
        return Err(RuntimeNameResolutionError::WasmRegistryNotReady);
    }
    let context = crate::context::legacy::active();
    let search_path = context
        .as_ref()
        .and_then(|context| context.search_path())
        .cloned();
    let captured_path_revision = search_path.as_ref().map(|path| path.generation());
    let captured_class_revision = crate::class_registry::class_registry_generation();
    let finish = |candidates| {
        if search_path.as_ref().map(|path| path.generation()) != captured_path_revision {
            return Err(RuntimeNameResolutionError::SearchPathChanged);
        }
        if crate::class_registry::class_registry_generation() != captured_class_revision {
            return Err(RuntimeNameResolutionError::ClassRegistryChanged);
        }
        Ok(RuntimeNameResolution {
            name: name.to_owned(),
            query,
            captured_path_revision,
            captured_class_revision,
            candidates,
        })
    };
    let mut candidates = Vec::new();
    if query.route == RuntimeNameRoute::DynamicCall
        && matches!(
            query.filter,
            RuntimeNameFilter::Any | RuntimeNameFilter::Workspace
        )
    {
        if crate::workspace::try_lookup(name)
            .map_err(|error| RuntimeNameResolutionError::Workspace(Box::new(error)))?
            .is_some()
        {
            candidates.push(RuntimeNameCandidate::WorkspaceVariable);
        }
        if (!query.all && !candidates.is_empty()) || query.filter == RuntimeNameFilter::Workspace {
            return finish(candidates);
        }
    }
    if query.filter == RuntimeNameFilter::Workspace {
        return finish(candidates);
    }

    if query.route == RuntimeNameRoute::NamedFeval {
        append_semantic(name, query.filter, &mut candidates);
        if !query.all && !candidates.is_empty() {
            return finish(candidates);
        }
        append_builtin(name, query.route, query.filter, &mut candidates);
        if !query.all && !candidates.is_empty() {
            return finish(candidates);
        }
    }
    // A registered DynamicName constructor is diverted to dispatcher before
    // BeforeSemantic MEX/source lookup. Named-feval has already offered its
    // semantic/builtin bindings; direct dynamic fallback offers exact builtin
    // binding here before the constructor.
    if query.filter == RuntimeNameFilter::Any {
        if let Some(class) = crate::class_registry::registered_constructor_identity(name) {
            if query.route == RuntimeNameRoute::DynamicCall {
                append_builtin(name, query.route, query.filter, &mut candidates);
                if !query.all && !candidates.is_empty() {
                    return finish(candidates);
                }
            }
            candidates.push(RuntimeNameCandidate::RegisteredConstructor(class));
            if !query.all {
                return finish(candidates);
            }
        }
    }
    if query.route == RuntimeNameRoute::DynamicCall
        && matches!(
            query.filter,
            RuntimeNameFilter::Any | RuntimeNameFilter::File
        )
    {
        append_scoped_private_semantic(name, query.filter, &mut candidates);
        if !query.all && !candidates.is_empty() {
            return finish(candidates);
        }
    }
    if matches!(
        query.filter,
        RuntimeNameFilter::Any | RuntimeNameFilter::File
    ) {
        let native = match context
            .as_ref()
            .map(|context| (context, context.native_resolution_scope()))
        {
            Some((runtime, NativeResolutionScope::Selected(route))) => {
                route.observe(runtime.clone(), name.to_owned()).await?
            }
            #[cfg(not(target_arch = "wasm32"))]
            Some((
                runtime,
                NativeResolutionScope::SelectedOrdinary {
                    route,
                    source_owner,
                    ..
                },
            )) if source_owner.is_bound_in(runtime) => {
                route.observe(runtime.clone(), name.to_owned()).await?
            }
            #[cfg(not(target_arch = "wasm32"))]
            Some((_, NativeResolutionScope::SelectedOrdinary { .. })) => None,
            Some((_, NativeResolutionScope::SelectedClosed)) | None => None,
            Some((runtime, NativeResolutionScope::Session)) => {
                let observer = runtime.state().call.borrow().native_name_observer.clone();
                if let Some(observer) = observer {
                    super::await_native_name_observation(observer(runtime.clone(), name.to_owned()))
                        .await?
                } else {
                    None
                }
            }
        };
        if let Some(native) = native {
            candidates.push(RuntimeNameCandidate::Native(native));
        }
        if !query.all && !candidates.is_empty() {
            return finish(candidates);
        }
    }
    if query.route == RuntimeNameRoute::DynamicCall {
        append_semantic(name, query.filter, &mut candidates);
        if !query.all && !candidates.is_empty() {
            return finish(candidates);
        }
    }
    if matches!(
        query.filter,
        RuntimeNameFilter::Any | RuntimeNameFilter::File
    ) {
        let sources = if query.all {
            find_all_files_with_extensions_with_origin(name, &[".m"], "which").await?
        } else {
            crate::builtins::common::path_search::find_file_with_extensions_with_origin(
                name,
                &[".m"],
                "which",
            )
            .await?
            .into_iter()
            .collect()
        };
        candidates.extend(sources.into_iter().map(RuntimeNameCandidate::Source));
        if !query.all && !candidates.is_empty() {
            return finish(candidates);
        }
    }
    if query.route == RuntimeNameRoute::DynamicCall {
        append_builtin(name, query.route, query.filter, &mut candidates);
        if !query.all && !candidates.is_empty() {
            return finish(candidates);
        }
    }
    if query.filter == RuntimeNameFilter::Builtin {
        return finish(candidates);
    }

    if matches!(
        query.filter,
        RuntimeNameFilter::Any | RuntimeNameFilter::File
    ) {
        if let Some(method) = crate::dispatcher::observe_registered_static_method(name) {
            let source = match &method.target {
                crate::dispatcher::RegisteredStaticMethodTarget::Bound(binding) => {
                    source_function_for(binding.function)
                }
                crate::dispatcher::RegisteredStaticMethodTarget::Semantic(function) => {
                    source_function_for(*function)
                }
                crate::dispatcher::RegisteredStaticMethodTarget::Builtin(_) => None,
            };
            let has_file = source
                .as_ref()
                .and_then(|info| info.source_file.as_ref())
                .is_some();
            if query.filter == RuntimeNameFilter::Any || has_file {
                candidates.push(RuntimeNameCandidate::RegisteredStaticMethod {
                    resolution: method,
                    source,
                });
                if !query.all {
                    return finish(candidates);
                }
            }
        }
    }

    // Presentation-only file and folder matches follow executable candidates.
    // These are still gathered here, not independently resolved by `which`.
    let mut seen_paths = HashSet::new();
    for candidate in &candidates {
        match candidate {
            RuntimeNameCandidate::Source(file)
            | RuntimeNameCandidate::ClassFile(file)
            | RuntimeNameCandidate::File(file) => {
                seen_paths.insert(file.path().to_path_buf());
            }
            RuntimeNameCandidate::Native(native) => {
                seen_paths.insert(native.path.clone());
            }
            _ => {}
        }
    }
    for folder in class_folder_candidates_with_origin(name, "which")? {
        if crate::builtins::common::path_search::path_is_directory(folder.path()).await?
            && seen_paths.insert(folder.path().to_path_buf())
        {
            candidates.push(RuntimeNameCandidate::ClassFolder(folder));
            if !query.all {
                return finish(candidates);
            }
        }
    }
    let class_files = if query.all {
        class_file_candidates_with_origin(name, CLASS_M_FILE_EXTENSIONS, "classdef", "which")
            .await?
    } else {
        find_class_file_with_origin(name, CLASS_M_FILE_EXTENSIONS, "classdef", "which")
            .await?
            .into_iter()
            .collect()
    };
    for file in class_files {
        if seen_paths.insert(file.path().to_path_buf()) {
            candidates.push(RuntimeNameCandidate::ClassFile(file));
            if !query.all {
                return finish(candidates);
            }
        }
    }
    let files = if query.all {
        find_all_files_with_extensions_with_origin(name, GENERAL_FILE_EXTENSIONS, "which").await?
    } else {
        crate::builtins::common::path_search::find_file_with_extensions_with_origin(
            name,
            GENERAL_FILE_EXTENSIONS,
            "which",
        )
        .await?
        .into_iter()
        .collect()
    };
    for file in files {
        if seen_paths.insert(file.path().to_path_buf()) {
            candidates.push(RuntimeNameCandidate::File(file));
            if !query.all {
                return finish(candidates);
            }
        }
    }
    for directory in directory_candidates_with_origin(name, "which")? {
        if crate::builtins::common::path_search::path_is_directory(directory.path()).await?
            && seen_paths.insert(directory.path().to_path_buf())
        {
            candidates.push(RuntimeNameCandidate::Directory(directory));
            if !query.all {
                return finish(candidates);
            }
        }
    }
    finish(candidates)
}

fn append_semantic(
    name: &str,
    filter: RuntimeNameFilter,
    candidates: &mut Vec<RuntimeNameCandidate>,
) {
    if matches!(filter, RuntimeNameFilter::Any | RuntimeNameFilter::File)
        && crate::user_functions::current_semantic_function_invoker().is_some()
    {
        let selected = resolve_semantic_function_by_name(name);
        let mut ordered = current_semantic_name_candidates()
            .map(|observer| observer(name))
            .unwrap_or_default();
        if let Some(function) = selected {
            if ordered
                .first()
                .is_none_or(|candidate| candidate.function != function)
            {
                ordered.insert(
                    0,
                    SemanticNameCandidate {
                        function,
                        source: source_function_for(function),
                        definition_owner: None,
                    },
                );
            }
        }
        for offered in ordered {
            if candidates
                .iter()
                .any(|candidate| same_semantic_definition(candidate, &offered))
            {
                continue;
            }
            let source = offered
                .source
                .or_else(|| source_function_for(offered.function));
            if filter == RuntimeNameFilter::File
                && source
                    .as_ref()
                    .and_then(|info| info.source_file.as_ref())
                    .is_none()
            {
                continue;
            }
            candidates.push(RuntimeNameCandidate::Semantic {
                function: offered.function,
                source,
                definition_owner: offered.definition_owner,
            });
        }
    }
}

fn append_scoped_private_semantic(
    name: &str,
    filter: RuntimeNameFilter,
    candidates: &mut Vec<RuntimeNameCandidate>,
) {
    if crate::user_functions::current_semantic_function_invoker().is_none() {
        return;
    }
    let Some(function) = resolve_semantic_function_by_name(name) else {
        return;
    };
    let Some(target) = current_scoped_name_target_resolver().and_then(|resolver| resolver(name))
    else {
        return;
    };
    if target.function().0 != function {
        return;
    }
    let offered = current_semantic_name_candidates()
        .map(|observer| observer(name))
        .and_then(|ordered| {
            ordered
                .into_iter()
                .find(|candidate| candidate.function == function)
        });
    let source = offered
        .as_ref()
        .and_then(|candidate| candidate.source.clone())
        .or_else(|| source_function_for(function));
    if filter == RuntimeNameFilter::File
        && source
            .as_ref()
            .and_then(|info| info.source_file.as_ref())
            .is_none()
    {
        return;
    }
    candidates.push(RuntimeNameCandidate::Semantic {
        function,
        source,
        definition_owner: offered.and_then(|candidate| candidate.definition_owner),
    });
}

fn same_semantic_definition(
    candidate: &RuntimeNameCandidate,
    offered: &SemanticNameCandidate,
) -> bool {
    let RuntimeNameCandidate::Semantic {
        function,
        source,
        definition_owner,
    } = candidate
    else {
        return false;
    };
    match (definition_owner, &offered.definition_owner) {
        (Some(left), Some(right)) => std::sync::Arc::ptr_eq(left, right),
        (None, None) => match (
            source.as_ref().and_then(|source| source.binding_origin),
            offered
                .source
                .as_ref()
                .and_then(|source| source.binding_origin),
        ) {
            (Some(left), Some(right)) => left == right,
            _ => *function == offered.function,
        },
        _ => false,
    }
}

fn append_builtin(
    name: &str,
    route: RuntimeNameRoute,
    filter: RuntimeNameFilter,
    candidates: &mut Vec<RuntimeNameCandidate>,
) {
    if matches!(filter, RuntimeNameFilter::Any | RuntimeNameFilter::Builtin) {
        let identity = match route {
            RuntimeNameRoute::DynamicCall => {
                crate::dispatcher::observed_dispatch_builtin_name(name)
            }
            RuntimeNameRoute::NamedFeval => builtin_id_by_name(name),
        };
        if let Some(identity) = identity {
            if !candidates.iter().any(|candidate| {
                matches!(candidate, RuntimeNameCandidate::Builtin(existing) if existing == &identity)
            }) {
                candidates.push(RuntimeNameCandidate::Builtin(identity));
            }
        }
    }
}

#[cfg(test)]
#[path = "name_resolution/tests.rs"]
mod tests;
