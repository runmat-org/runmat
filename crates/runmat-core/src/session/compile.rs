use super::*;
use crate::fusion::FusionPlannerMetadata;
pub(super) use runmat_static_analysis::project::composition::qualify_companion_classdefs;
use runmat_static_analysis::project::composition::{
    function_leaf_name, function_owner_scope_from_qualified_name, is_class_source_body,
    is_function_source_body, project_symbol_aliases, qualify_companion_functions,
    qualify_private_companion_functions,
};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[path = "compile/program_sources.rs"]
pub(super) mod program_sources;
#[path = "compile/test_source_authority.rs"]
mod test_source_authority;
use program_sources::identity::{executable_primary_access_path, executable_source_identity};
use test_source_authority::CapturedTestSourceAuthority;

fn fatal_frontend_diagnostic(
    diagnostics: &[runmat_hir::HirDiagnostic],
) -> Option<&runmat_hir::HirDiagnostic> {
    diagnostics
        .iter()
        .find(|diagnostic| diagnostic.blocks_execution())
}

fn bind_frontend_diagnostic(
    diagnostic: &runmat_hir::HirDiagnostic,
    source_table: &program_sources::ProgramSourceTable,
) -> RunError {
    let mut error = runmat_vm::CompileError::new(diagnostic.message.clone())
        .with_identifier(diagnostic.code.clone())
        .with_span(diagnostic.primary.span);
    if let Some(source) = diagnostic.primary.source {
        error = error.with_source(source);
    }
    bind_compile_error(error, source_table)
}

fn bind_compile_error(
    error: runmat_vm::CompileError,
    source_table: &program_sources::ProgramSourceTable,
) -> RunError {
    let Some(source_id) = error.source else {
        return RunError::Compile(error);
    };
    let Some(source) = source_table
        .source_map
        .entries()
        .iter()
        .find(|entry| entry.source_id == source_id)
    else {
        return RunError::Runtime(
            build_runtime_error("frontend diagnostic refers to an unadmitted program source")
                .with_identifier("RunMat:ExecutableSourceIdentity")
                .build(),
        );
    };
    if source_id == source_table.primary_source() {
        RunError::Compile(error)
    } else {
        RunError::SourceCompile(crate::SourceCompileError {
            error,
            source: source.clone(),
        })
    }
}

mod facts;
use facts::{function_output_arities, mir_local_fact_count_for_entrypoint};

fn source_lookup_cwd(source_name: &str) -> Result<PathBuf, RuntimeError> {
    let source_path = PathBuf::from(source_name);
    if crate::diagnostic_path::host_path_is_absolute(&source_path) {
        return Ok(source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")));
    }
    runmat_filesystem::current_dir().map_err(|source| {
        build_runtime_error(format!(
            "failed to resolve source working directory: {source}"
        ))
        .with_source(source)
        .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
        .build()
    })
}

fn resolved_source_path(source_name: &str, cwd: &Path) -> PathBuf {
    crate::diagnostic_path::resolve_against_base(source_name, cwd)
}

fn path_to_source_name(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn package_class_name_from_path(source_path: &Path, root_dir: &Path) -> Option<String> {
    let relative = source_path.strip_prefix(root_dir).ok()?;
    let class_name = source_path.file_stem()?.to_str()?;
    let mut package_segments = Vec::new();
    if let Some(parent) = relative.parent() {
        for component in parent.components() {
            let segment = component.as_os_str().to_str()?;
            if let Some(pkg) = segment.strip_prefix('+') {
                if pkg.is_empty() {
                    return None;
                }
                package_segments.push(pkg.to_string());
            } else if let Some(class) = segment.strip_prefix('@') {
                if class.is_empty() {
                    return None;
                }
                package_segments.push(class.to_string());
            } else {
                return None;
            }
        }
    }
    if package_segments.is_empty() {
        return None;
    }
    package_segments.push(class_name.to_string());
    Some(package_segments.join("."))
}

fn source_index_qualified_function_name(
    source: &runmat_package::FrozenSourceDescriptor,
) -> Option<&str> {
    source.function_qualified_name()
}

fn source_index_qualified_class_name(
    source: &runmat_package::FrozenSourceDescriptor,
) -> Option<&str> {
    source.class_definition_qualified_name()
}

fn is_private_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("private"))
}

fn private_parent_dir_for_source(path: &Path) -> Option<PathBuf> {
    let private_dir = path.parent()?;
    if !is_private_dir(private_dir) {
        return None;
    }
    private_dir.parent().map(Path::to_path_buf)
}

fn source_paths_equivalent(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        if let (Ok(left), Ok(right)) = (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
            if left == right {
                return true;
            }
        }
    }

    #[cfg(windows)]
    {
        windows_source_path_key(left) == windows_source_path_key(right)
    }
    #[cfg(not(windows))]
    {
        false
    }
}

#[cfg(windows)]
fn windows_source_path_key(path: &Path) -> String {
    let mut text = path.to_string_lossy().replace('/', "\\");
    if let Some(stripped) = text.strip_prefix(r"\\?\UNC\") {
        text = format!(r"\\{stripped}");
    } else if let Some(stripped) = text.strip_prefix(r"\\?\") {
        text = stripped.to_string();
    }
    while text.ends_with('\\') && text.len() > 3 {
        text.pop();
    }
    text.to_ascii_lowercase()
}

fn private_source_visible_to(primary_source_path: &Path, source_path: &Path) -> bool {
    let Some(private_parent) = private_parent_dir_for_source(source_path) else {
        return true;
    };
    primary_source_path
        .parent()
        .is_some_and(|caller_dir| source_paths_equivalent(caller_dir, &private_parent))
}

#[derive(Default)]
pub(super) struct CompanionSourceDiscovery {
    #[cfg(windows)]
    pub canonical_search_root: Option<PathBuf>,
    pub statements: Vec<DiscoveredSourceStatement>,
    pub source_catalog: Option<runmat_package::DiscoveredSourceSymbols>,
    pub project_symbol_aliases: HashMap<String, String>,
    pub private_function_names: HashSet<String>,
    pub private_function_owners: HashMap<String, String>,
    pub private_function_aliases: HashMap<String, HashMap<String, String>>,
    pub observed_sources: Vec<ObservedSourceRevision>,
}

pub(super) struct ObservedSourceRevision {
    pub access_path: PathBuf,
    pub content_revision: Option<runmat_execution::Digest>,
    pub uses_pcode_reader: bool,
}

pub(super) struct DiscoveredSourceStatement {
    pub statement: runmat_parser::Stmt,
    pub is_private: bool,
    pub source: Option<Arc<SourceContextData>>,
}

pub(super) struct ProjectCompilationContext {
    cwd: PathBuf,
    primary_source_path: PathBuf,
    frozen: Option<runmat_package::FrozenProject>,
    loose_index: Option<runmat_config::project::ProjectSourceIndex>,
    pub source_catalog: Option<runmat_package::DiscoveredSourceSymbols>,
}

impl ProjectCompilationContext {
    /// Revision of the selected source inventory before companion contents are
    /// read. It invalidates dynamic products when path precedence, package or
    /// class ownership, or the visible private-file set changes.
    pub(super) fn dynamic_inventory_revision(&self) -> runmat_execution::Digest {
        if let Some(revision) = self
            .source_catalog
            .as_ref()
            .and_then(|catalog| catalog.project_revision())
        {
            return runmat_execution::Digest::sha256(revision.cache_namespace());
        }
        let mut files = self
            .loose_index
            .as_ref()
            .map(|index| {
                index
                    .files
                    .iter()
                    .map(|file| {
                        (
                            file.relative_path.to_string_lossy().replace('\\', "/"),
                            file.qualified_name.clone(),
                            file.package_path.clone(),
                            file.class_name.clone(),
                            file.class_qualified_name.clone(),
                            file.is_private,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        files.sort();
        runmat_execution::Digest::sha256(
            serde_json::to_vec(&files).expect("source inventory rows are serializable"),
        )
    }
}

#[derive(Clone)]
pub(super) struct SourceContextData {
    pub display_name: String,
    pub fullpath_name: Option<String>,
    pub text: String,
}

fn function_names_in_statements(stmts: &[runmat_parser::Stmt]) -> impl Iterator<Item = &str> {
    stmts.iter().filter_map(|stmt| {
        if let runmat_parser::Stmt::Function { name, .. } = stmt {
            Some(name.as_str())
        } else {
            None
        }
    })
}

fn class_name_in_statements(stmts: &[runmat_parser::Stmt]) -> Option<&str> {
    stmts.iter().find_map(|stmt| {
        if let runmat_parser::Stmt::ClassDef { name, .. } = stmt {
            Some(name.as_str())
        } else {
            None
        }
    })
}

pub(super) fn source_context_function_names_in_statements(
    stmts: &[runmat_parser::Stmt],
) -> Vec<String> {
    let mut names = Vec::new();
    for stmt in stmts {
        match stmt {
            runmat_parser::Stmt::Function { name, .. } => names.push(name.clone()),
            runmat_parser::Stmt::ClassDef {
                name: class_name,
                members,
                ..
            } => {
                for member in members {
                    if let runmat_parser::ClassMember::Methods { body, .. } = member {
                        for stmt in body {
                            if let runmat_parser::Stmt::Function { name, .. } = stmt {
                                let display_name = if name.contains('.') {
                                    name.clone()
                                } else {
                                    format!("{class_name}.{name}")
                                };
                                names.push(display_name);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    names
}

impl CompanionSourceDiscovery {
    fn extend_body(
        &mut self,
        body: Vec<runmat_parser::Stmt>,
        private_owner_scope: Option<&str>,
        private_aliases: HashMap<String, String>,
        source_context: Option<SourceContextData>,
    ) {
        let source_context = source_context.map(Arc::new);
        let is_private = private_owner_scope.is_some();
        if is_private {
            let owner_scope = private_owner_scope.unwrap_or_default();
            for function_name in function_names_in_statements(&body) {
                self.private_function_names
                    .insert(function_name.to_string());
                self.private_function_owners
                    .insert(function_name.to_string(), owner_scope.to_string());
            }
            if !private_aliases.is_empty() {
                self.private_function_aliases
                    .entry(owner_scope.to_string())
                    .or_default()
                    .extend(private_aliases);
            }
        }
        for stmt in body {
            self.statements.push(DiscoveredSourceStatement {
                statement: stmt,
                is_private,
                source: source_context.clone(),
            });
        }
    }

    pub(super) fn apply_function_precedence(&mut self, primary_function_names: &HashSet<String>) {
        let discovered_private_function_names: HashSet<String> = self
            .statements
            .iter()
            .filter_map(|discovered| {
                if !discovered.is_private {
                    return None;
                }
                if let runmat_parser::Stmt::Function { name, .. } = &discovered.statement {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        let old_statements = std::mem::take(&mut self.statements);
        self.private_function_names.clear();
        self.private_function_owners.clear();
        self.private_function_aliases.clear();

        for discovered in old_statements {
            let keep = match &discovered.statement {
                runmat_parser::Stmt::Function { name, .. } => {
                    !primary_function_names.contains(name)
                        && (discovered.is_private
                            || !discovered_private_function_names.contains(name))
                }
                _ => true,
            };
            if !keep {
                continue;
            }
            if discovered.is_private {
                if let runmat_parser::Stmt::Function { name, .. } = &discovered.statement {
                    self.private_function_names.insert(name.clone());
                    let owner_scope = function_owner_scope_from_qualified_name(name);
                    let owner_scope = if let Some((owner, _)) = name.split_once(".__private__.") {
                        owner.to_string()
                    } else {
                        owner_scope
                    };
                    self.private_function_owners
                        .insert(name.clone(), owner_scope.clone());
                    self.private_function_aliases
                        .entry(owner_scope)
                        .or_default()
                        .insert(function_leaf_name(name).to_string(), name.clone());
                }
            }
            self.statements.push(discovered);
        }
    }
}

fn dependency_composed_name(
    catalog: Option<&runmat_package::DiscoveredSourceSymbols>,
    alias: Option<&runmat_package::PackageAlias>,
    source_id: &runmat_package::StableSourceId,
    original: String,
) -> String {
    let Some(alias) = alias else {
        return original;
    };
    let has_unique_bare_name = catalog.is_some_and(|catalog| {
        catalog.definitions.iter().any(|definition| {
            definition.name == original
                && definition.dependency_alias.as_deref() == Some(alias.as_str())
                && definition.source_id.as_ref() == Some(source_id)
        })
    });
    if has_unique_bare_name {
        original
    } else {
        format!("{alias}.{original}")
    }
}

pub(super) async fn discover_project_compilation_context_async(
    source_name: &str,
) -> Result<Option<ProjectCompilationContext>, RuntimeError> {
    let cwd = source_lookup_cwd(source_name)?;
    let primary_source_path = resolved_source_path(source_name, &cwd);
    let primary_is_local_file =
        runmat_runtime::builtins::common::path_search::path_is_file(&primary_source_path)
            .await
            .map_err(|error| error.into_runtime_error("source resolution"))?;
    if !primary_is_local_file {
        return Ok(None);
    }
    let Some(parent) = primary_source_path.parent() else {
        return Ok(None);
    };
    let frozen = runmat_package::discover_frozen_project_from_async(
        &primary_source_path,
        Default::default(),
    )
    .await
    .map_err(|error| {
        build_runtime_error(error.to_string())
            .with_source(error)
            .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
            .build()
    })?;
    let frozen = if let Some(frozen) = frozen {
        let handoff = runmat_package::FrozenProjectHandoff::new(frozen);
        handoff.validate().map_err(|error| {
            build_runtime_error(error.to_string())
                .with_source(error)
                .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                .build()
        })?;
        Some(handoff.into_project())
    } else {
        None
    };
    let (loose_index, source_catalog) = if let Some(frozen) = frozen.as_ref() {
        (
            None,
            Some(runmat_package::source_symbols_from_frozen(
                frozen,
                &primary_source_path,
            )),
        )
    } else {
        let index = runmat_config::project::build_loose_source_index_async(parent)
            .await
            .map_err(|error| {
                build_runtime_error(error.to_string())
                    .with_source(error)
                    .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                    .build()
            })?;
        let catalog =
            runmat_package::source_symbols_from_index(&index, parent, &primary_source_path, None);
        (Some(index), Some(catalog))
    };
    Ok(Some(ProjectCompilationContext {
        cwd,
        primary_source_path,
        frozen,
        loose_index,
        source_catalog,
    }))
}

pub(super) fn project_compilation_context_from_handoff(
    source_name: &str,
    handoff: &runmat_package::FrozenProjectHandoff,
) -> Result<ProjectCompilationContext, RuntimeError> {
    handoff.validate().map_err(|error| {
        build_runtime_error(error.to_string())
            .with_source(error)
            .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
            .build()
    })?;
    let cwd = source_lookup_cwd(source_name)?;
    let primary_source_path = resolved_source_path(source_name, &cwd);
    let frozen = handoff.project.clone();
    let source_catalog = Some(runmat_package::source_symbols_from_frozen(
        &frozen,
        &primary_source_path,
    ));
    Ok(ProjectCompilationContext {
        cwd,
        primary_source_path,
        frozen: Some(frozen),
        loose_index: None,
        source_catalog,
    })
}

/// Exact project source bytes captured with a frozen handoff, keyed by access path.
pub(crate) type CapturedSourceBytes = std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>;

async fn compose_frozen_project_async(
    context: &ProjectCompilationContext,
    frozen: &runmat_package::FrozenProject,
    compat_mode: runmat_parser::CompatMode,
    test_authority: Option<&CapturedTestSourceAuthority<'_>>,
) -> Result<CompanionSourceDiscovery, RuntimeError> {
    compose_frozen_source_view_async(context, frozen, compat_mode, test_authority, None).await
}

/// Compose graph-declared companions. Test authorities and captured
/// snapshots supply exact bytes; only an unconstrained view reads providers.
async fn compose_frozen_source_view_async(
    context: &ProjectCompilationContext,
    frozen: &runmat_package::FrozenProject,
    compat_mode: runmat_parser::CompatMode,
    test_authority: Option<&CapturedTestSourceAuthority<'_>>,
    captured: Option<&CapturedSourceBytes>,
) -> Result<CompanionSourceDiscovery, RuntimeError> {
    let options = ParserOptions::new(compat_mode);
    let mut out = CompanionSourceDiscovery::default();
    out.source_catalog = context.source_catalog.clone();
    out.project_symbol_aliases = project_symbol_aliases(out.source_catalog.as_ref());
    for visible in frozen.visible_sources(&context.primary_source_path) {
        let source = visible.source;
        let access_path = visible.access_path;
        if source_paths_equivalent(access_path, &context.primary_source_path) {
            continue;
        }
        let test_captured =
            test_authority.and_then(|authority| authority.captured_text_for(access_path));
        let (file_path, contents, uses_pcode_reader) = if let Some(text) = test_captured {
            (access_path.to_path_buf(), Cow::Borrowed(text), false)
        } else if let Some(captured) = captured {
            let file_path = access_path.to_path_buf();
            let bytes = captured.get(&file_path).ok_or_else(|| {
                format!("captured source is unavailable: {}", file_path.display())
            })?;
            if runmat_package::ContentDigest::sha256(bytes) != source.id.content_digest {
                return Err(format!(
                    "captured source differs from its frozen identity: {}",
                    file_path.display()
                )
                .into());
            }
            let text = std::str::from_utf8(bytes).map_err(|_| {
                format!(
                    "captured MATLAB source is not UTF-8: {}",
                    file_path.display()
                )
            })?;
            (file_path, Cow::Borrowed(text), false)
        } else {
            let file_path =
                runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(access_path)
                    .await
                    .map_err(|error| error.into_runtime_error("source resolution"))?;
            let contents =
                runmat_runtime::builtins::io::repl_fs::pcode::read_source_text_async(&file_path)
                    .await
                    .map_err(|error| {
                        build_runtime_error(format!(
                            "failed to read graph-declared source {}: {error}",
                            file_path.display()
                        ))
                        .with_source(error)
                        .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                        .build()
                    })?;
            (file_path, Cow::Owned(contents), true)
        };
        out.observed_sources.push(ObservedSourceRevision {
            access_path: file_path.clone(),
            content_revision: Some(runmat_execution::Digest::sha256(contents.as_bytes())),
            uses_pcode_reader,
        });
        if !contents.contains("classdef") && !contents.contains("function") {
            continue;
        }
        let program = parse_with_options(&contents, options).map_err(|error| {
            format!(
                "failed to parse graph-declared source {}: {error}",
                file_path.display()
            )
        })?;
        let is_class_source = is_class_source_body(&program.body);
        let is_function_source = is_function_source_body(&program.body);
        if !is_class_source && !is_function_source {
            continue;
        }
        let mut body = program.body;
        let private_owner_scope = source
            .is_private
            .then(|| function_owner_scope_from_qualified_name(&source.qualified_name));
        let primary_visible_private = source.is_private && visible.directly_visible;
        let private_aliases = if let Some(owner_scope) = private_owner_scope.as_deref() {
            qualify_private_companion_functions(&mut body, owner_scope, primary_visible_private)
        } else {
            HashMap::new()
        };
        if is_class_source {
            let qualified = source_index_qualified_class_name(source)
                .map(ToOwned::to_owned)
                .or_else(|| package_class_name_from_path(&file_path, &context.cwd))
                .or_else(|| {
                    visible
                        .dependency_alias
                        .and_then(|_| class_name_in_statements(&body).map(ToOwned::to_owned))
                });
            if let Some(qualified) = qualified {
                let qualified = dependency_composed_name(
                    context.source_catalog.as_ref(),
                    visible.dependency_alias,
                    &source.id,
                    qualified,
                );
                qualify_companion_classdefs(&mut body, &qualified);
            }
        } else if private_owner_scope.is_none() {
            let qualified = source_index_qualified_function_name(source)
                .map(ToOwned::to_owned)
                .or_else(|| package_class_name_from_path(&file_path, &context.cwd))
                .or_else(|| {
                    visible.dependency_alias.and_then(|_| {
                        function_names_in_statements(&body)
                            .next()
                            .map(ToOwned::to_owned)
                    })
                });
            if let Some(qualified) = qualified {
                let qualified = dependency_composed_name(
                    context.source_catalog.as_ref(),
                    visible.dependency_alias,
                    &source.id,
                    qualified,
                );
                qualify_companion_functions(&mut body, &qualified);
            }
        }
        let source_context_path = access_path;
        out.extend_body(
            body,
            private_owner_scope.as_deref(),
            private_aliases,
            Some(SourceContextData {
                display_name: crate::diagnostic_path::display_path_from_base(
                    source_context_path,
                    &context.cwd,
                ),
                fullpath_name: Some(path_to_source_name(source_context_path)),
                text: contents.into_owned(),
            }),
        );
    }
    Ok(out)
}

async fn compose_loose_project_async(
    context: &ProjectCompilationContext,
    source_index: &runmat_config::project::ProjectSourceIndex,
    compat_mode: runmat_parser::CompatMode,
) -> Result<CompanionSourceDiscovery, RuntimeError> {
    let mut out = CompanionSourceDiscovery {
        source_catalog: context.source_catalog.clone(),
        project_symbol_aliases: project_symbol_aliases(context.source_catalog.as_ref()),
        ..Default::default()
    };
    let options = ParserOptions::new(compat_mode);
    let Some(parent) = context.primary_source_path.parent() else {
        return Ok(out);
    };
    for source in &source_index.files {
        let path = parent.join(&source.source_root).join(&source.relative_path);
        if source_paths_equivalent(&path, &context.primary_source_path) {
            continue;
        }
        let contents = runmat_filesystem::read_to_string_async(&path).await;
        out.observed_sources.push(ObservedSourceRevision {
            access_path: path.clone(),
            content_revision: contents
                .as_ref()
                .ok()
                .map(|text| runmat_execution::Digest::sha256(text.as_bytes())),
            uses_pcode_reader: false,
        });
        let contents = match contents {
            Ok(contents) => contents,
            Err(source)
                if matches!(
                    source.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                ) =>
            {
                continue
            }
            Err(source) => {
                return Err(build_runtime_error(format!(
                    "failed to read companion source '{}': {source}",
                    path.display()
                ))
                .with_source(source)
                .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                .build());
            }
        };
        if !contents.contains("classdef") && !contents.contains("function") {
            continue;
        }
        let Ok(program) = parse_with_options(&contents, options) else {
            continue;
        };
        let is_class_source = is_class_source_body(&program.body);
        let is_function_source = is_function_source_body(&program.body);
        if !is_class_source && !is_function_source {
            continue;
        }
        let mut body = program.body;
        let private_owner_scope = source
            .is_private
            .then(|| function_owner_scope_from_qualified_name(&source.qualified_name));
        let primary_visible_private = private_owner_scope.is_some()
            && private_source_visible_to(&context.primary_source_path, &path);
        let private_aliases = if let Some(owner_scope) = private_owner_scope.as_deref() {
            qualify_private_companion_functions(&mut body, owner_scope, primary_visible_private)
        } else {
            HashMap::new()
        };
        if is_class_source {
            if let Some(qualified) = source.class_definition_qualified_name() {
                qualify_companion_classdefs(&mut body, qualified);
            }
        } else if private_owner_scope.is_none() {
            if let Some(qualified) = source.function_qualified_name() {
                qualify_companion_functions(&mut body, qualified);
            }
        }
        out.extend_body(
            body,
            private_owner_scope.as_deref(),
            private_aliases,
            Some(SourceContextData {
                display_name: crate::diagnostic_path::display_path_from_base(&path, &context.cwd),
                fullpath_name: Some(path_to_source_name(&path)),
                text: contents,
            }),
        );
    }
    Ok(out)
}

pub(super) async fn compose_project_compilation_context_async(
    context: &ProjectCompilationContext,
    compat_mode: runmat_parser::CompatMode,
) -> Result<CompanionSourceDiscovery, RuntimeError> {
    if let Some(frozen) = context.frozen.as_ref() {
        compose_frozen_project_async(context, frozen, compat_mode, None).await
    } else if let Some(loose_index) = context.loose_index.as_ref() {
        compose_loose_project_async(context, loose_index, compat_mode).await
    } else {
        Ok(CompanionSourceDiscovery::default())
    }
}

pub(super) async fn discover_companion_source_statements_async(
    source_name: &str,
    compat_mode: runmat_parser::CompatMode,
) -> Result<CompanionSourceDiscovery, RuntimeError> {
    let Some(context) = discover_project_compilation_context_async(source_name).await? else {
        return Ok(CompanionSourceDiscovery::default());
    };
    compose_project_compilation_context_async(&context, compat_mode).await
}

/// A diagnostic name never grants filesystem access. Inline input can use an
/// installed project's manifest as its lookup anchor; only a resolved primary
/// path selects a source file or starts implicit project discovery.
pub(super) async fn companion_source_statements_for_input_async(
    primary_access_path: Option<&str>,
    compat_mode: runmat_parser::CompatMode,
    handoff: Option<&runmat_package::FrozenProjectHandoff>,
) -> Result<CompanionSourceDiscovery, RuntimeError> {
    if let Some(handoff) = handoff {
        let anchor = handoff.project.manifest_path.to_string_lossy();
        companion_source_statements_from_handoff_async(
            primary_access_path.unwrap_or(anchor.as_ref()),
            compat_mode,
            handoff,
            None,
        )
        .await
    } else if let Some(path) = primary_access_path {
        discover_companion_source_statements_async(path, compat_mode).await
    } else {
        Ok(CompanionSourceDiscovery::default())
    }
}

pub(super) async fn companion_source_statements_from_handoff_async(
    source_name: &str,
    compat_mode: runmat_parser::CompatMode,
    handoff: &runmat_package::FrozenProjectHandoff,
    test_authority: Option<&CapturedTestSourceAuthority<'_>>,
) -> Result<CompanionSourceDiscovery, RuntimeError> {
    let context = project_compilation_context_from_handoff(source_name, handoff)?;
    compose_frozen_project_async(&context, &handoff.project, compat_mode, test_authority).await
}

impl RunMatSession {
    /// Compile one immutable source/revision product without executing it or
    /// publishing its functions into the interactive session.
    pub async fn compile_executable_unit<'a>(
        &mut self,
        source: impl Into<Cow<'a, crate::ExecutableSource>>,
        program_revision: Option<runmat_execution::ProgramRevision>,
    ) -> std::result::Result<crate::ExecutableUnit, RunError> {
        self.compile_executable_unit_inner(source.into(), program_revision, None, None)
            .await
    }

    /// Compile one selected test source from its frozen run, while retaining
    /// the installed project's independent dependency/source authority.
    pub async fn compile_test_executable_unit(
        &mut self,
        source: crate::ExecutableSource,
        descriptor: &runmat_test::descriptor::SourceDescriptor,
        snapshot: &runmat_test::discovery::FrozenTestRunSnapshot,
        span_only: bool,
    ) -> std::result::Result<crate::ExecutableUnit, RunError> {
        let authority = CapturedTestSourceAuthority::new(
            snapshot,
            descriptor,
            &source,
            span_only,
            self.project_handoff.as_ref(),
        )
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error)
                    .with_identifier("RunMat:ExecutableSourceIdentity")
                    .build(),
            )
        })?;
        self.compile_executable_unit_inner(
            Cow::Owned(source),
            Some(snapshot.program_revision.clone()),
            Some(&authority),
            None,
        )
        .await
    }

    /// Compile against an explicit immutable source inventory. Missing captures
    /// are errors, never permission to discover live bytes from a provider.
    pub async fn compile_executable_unit_from_snapshot(
        &mut self,
        source: crate::ExecutableSource,
        program_revision: runmat_execution::ProgramRevision,
        handoff: &runmat_package::FrozenProjectHandoff,
        captured: &CapturedSourceBytes,
    ) -> std::result::Result<crate::ExecutableUnit, RunError> {
        handoff.validate().map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error.to_string())
                    .with_identifier("RunMat:CapturedSourceIdentity")
                    .build(),
            )
        })?;
        let captured_bytes = captured.get(Path::new(source.lookup_path()));
        let descriptor = handoff
            .project
            .all_sources()
            .find(|(_, path)| path.as_path() == Path::new(source.lookup_path()))
            .map(|(descriptor, _)| descriptor);
        let project = handoff.project.revision();
        if !matches!((captured_bytes, descriptor), (Some(bytes), Some(descriptor))
            if bytes.as_slice() == source.text.as_bytes() && runmat_package::ContentDigest::sha256(bytes) == descriptor.id.content_digest)
            || *program_revision.graph_digest()
                != runmat_execution::Digest::from_bytes(*project.graph_digest.bytes())
            || *program_revision.source_digest()
                != runmat_execution::Digest::from_bytes(*project.source_revision.bytes())
        {
            return Err(RunError::Runtime(
                build_runtime_error(
                    "captured executable source does not match its admitted project revision",
                )
                .with_identifier("RunMat:CapturedSourceIdentity")
                .build(),
            ));
        }
        self.compile_executable_unit_inner(
            Cow::Owned(source),
            Some(program_revision),
            None,
            Some((handoff, captured)),
        )
        .await
    }

    async fn compile_executable_unit_inner(
        &mut self,
        source: Cow<'_, crate::ExecutableSource>,
        program_revision: Option<runmat_execution::ProgramRevision>,
        test_authority: Option<&CapturedTestSourceAuthority<'_>>,
        captured_snapshot: Option<(&runmat_package::FrozenProjectHandoff, &CapturedSourceBytes)>,
    ) -> std::result::Result<crate::ExecutableUnit, RunError> {
        self.ensure_user_path_hydrated().await?;
        let snapshot = captured_snapshot.map(|(handoff, _)| handoff);
        if let (Some(requested), Some(installed)) =
            (program_revision.as_ref(), self.project_revision())
        {
            if *requested.graph_digest()
                != runmat_execution::Digest::from_bytes(*installed.graph_digest.bytes())
            {
                return Err(RunError::Runtime(
                    build_runtime_error(
                        "executable source revision does not match the installed project",
                    )
                    .with_identifier("RunMat:ExecutableRevisionMismatch")
                    .build(),
                ));
            }
        }

        let primary_access_path = if let Some(authority) = test_authority {
            Ok(authority.primary_access_path().map(str::to_owned))
        } else {
            executable_primary_access_path(&source, snapshot.or(self.project_handoff.as_ref()))
        }
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error)
                    .with_identifier("RunMat:ExecutableSourceIdentity")
                    .build(),
            )
        })?;
        let result = {
            let mut scope = super::source_scope::SourceCompositionScope::enter(self);
            let session = scope.session();
            session.active_source_name = primary_access_path
                .clone()
                .unwrap_or_else(|| source.relative_path.clone());
            session.active_source_fullpath_name = primary_access_path;

            let companion_result = if let Some((handoff, captured)) = captured_snapshot {
                let anchor = handoff.project.manifest_path.to_string_lossy();
                match project_compilation_context_from_handoff(
                    session
                        .active_source_fullpath_name
                        .as_deref()
                        .unwrap_or(anchor.as_ref()),
                    handoff,
                ) {
                    Ok(context) => {
                        compose_frozen_source_view_async(
                            &context,
                            &handoff.project,
                            session.compat_mode,
                            None,
                            Some(captured),
                        )
                        .await
                    }
                    Err(error) => Err(error),
                }
            } else if let Some(handoff) = session.project_handoff.as_ref() {
                let anchor = handoff.project.manifest_path.to_string_lossy();
                companion_source_statements_from_handoff_async(
                    session
                        .active_source_fullpath_name
                        .as_deref()
                        .unwrap_or(anchor.as_ref()),
                    session.compat_mode,
                    handoff,
                    test_authority,
                )
                .await
            } else if test_authority.is_some() {
                Ok(CompanionSourceDiscovery::default())
            } else {
                companion_source_statements_for_input_async(
                    session.active_source_fullpath_name.as_deref(),
                    session.compat_mode,
                    None,
                )
                .await
            };
            let companion_result = companion_result.and_then(|mut companion| {
                if let Some(authority) = test_authority {
                    authority.add_companions(&mut companion, session.compat_mode)?;
                }
                Ok(companion)
            });
            match companion_result {
                Ok(companion) => {
                    session.pending_companion_source_discovery = Some(companion);
                    session.compile_input_with_portable_source(
                        &source.text,
                        Some(&source),
                        test_authority,
                    )
                }
                Err(error) => Err(RunError::Runtime(
                    build_runtime_error(format!("project composition failed: {error}"))
                        .with_source(error)
                        .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                        .with_identifier("RunMat:ProjectComposition")
                        .build(),
                )),
            }
        };

        let mut prepared = result?;
        let source_map = prepared
            .portable_source_map
            .take()
            .expect("portable compilation must retain its source table");
        let revision = crate::ExecutableRevision::derive_with_selection(
            &source,
            program_revision,
            crate::program_environment(self.compat_mode),
            test_authority.and_then(CapturedTestSourceAuthority::executable_selection),
        );
        prepared
            .attach_function_origins(&revision.program_revision, &source_map)
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error.to_string())
                        .with_identifier("RunMat:ExecutableFunctionIdentity")
                        .with_source(error)
                        .build(),
                )
            })?;
        let PreparedExecution {
            mir,
            analysis,
            bytecode,
            published_registry_for_reuse,
            private_companion_function_names,
            mut current_function_ids,
            ..
        } = prepared;
        let mut draft = crate::executable::ExecutableUnitDraft::prepare(
            source.into_owned(),
            revision,
            source_map,
            mir,
            analysis,
            bytecode,
        )
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error)
                    .with_identifier("RunMat:ExecutableProduct")
                    .build(),
            )
        })?;
        let mut excluded_reused_ids = HashSet::new();
        if let Some(remap) = super::semantic_definition::unchanged_product_remap(
            draft.bytecode(),
            &current_function_ids,
            &published_registry_for_reuse,
            &private_companion_function_names,
        ) {
            excluded_reused_ids.extend(remap.values().copied());
            draft
                .rebind_current_definitions(&published_registry_for_reuse, &remap)
                .map_err(|error| {
                    RunError::Runtime(
                        build_runtime_error(error)
                            .with_identifier("RunMat:ExecutableFunctionIdentity")
                            .build(),
                    )
                })?;
            current_function_ids = current_function_ids
                .into_iter()
                .map(|id| remap.get(&id).copied().unwrap_or(id))
                .collect();
        }
        let unit = self
            .program_service
            .bind_original_dependencies(draft.seal())
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableProduct")
                        .build(),
                )
            })?;
        // The prepared product, not parser membership, decides which bodies
        // this immutable program actually owns after unchanged-body reuse.
        let current = unit.compiled_current_body_ids().map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error)
                    .with_identifier("RunMat:ExecutableProduct")
                    .build(),
            )
        })?;
        if !current.is_disjoint(&excluded_reused_ids) || !current.is_subset(&current_function_ids) {
            return Err(RunError::Runtime(
                build_runtime_error("compiled body selection disagrees with reused definitions")
                    .with_identifier("RunMat:ExecutableProduct")
                    .build(),
            ));
        }
        let selected = super::source_evaluation::selected_static_imports(
            &unit,
            &self.function_registry,
            &current,
            &excluded_reused_ids,
        )
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error)
                    .with_identifier("RunMat:ExecutableStaticImport")
                    .build(),
            )
        })?;
        unit.with_portable_selection(current, selected.imports, selected.targets)
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableProduct")
                        .build(),
                )
            })
    }

    #[cfg(test)]
    pub(super) fn executable_source_map(
        &self,
        source: &crate::ExecutableSource,
    ) -> Result<crate::ExecutableSourceMap, String> {
        let entries = self
            .source_pool
            .entries()
            .map(|(source_id, entry)| {
                let display_name = entry.name.to_string();
                let full_path = entry.fullpath_name.as_ref().map(ToString::to_string);
                let (owner_identity, relative_path) = executable_source_identity(
                    source,
                    self.project_handoff.as_ref(),
                    self.current_source_fullpath_name(),
                    &display_name,
                    full_path.as_deref(),
                )?;
                Ok(crate::SourceMapEntry {
                    source_id: runmat_types::ProgramSourceId(
                        u32::try_from(source_id.0)
                            .map_err(|_| "source ordinal exceeds portable identity")?,
                    ),
                    owner_identity,
                    relative_path,
                    content_revision: runmat_execution::Digest::sha256(entry.text.as_bytes()),
                    display_name,
                    full_path,
                    text: entry.text.to_string(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(crate::ExecutableSourceMap::new(entries))
    }

    #[cfg(test)]
    pub(crate) fn published_function_registry_for_testing(&self) -> runmat_vm::FunctionRegistry {
        self.function_registry.clone()
    }

    #[cfg(test)]
    pub(crate) fn compile_input_for_source_name(
        &mut self,
        source_name: &str,
        input: &str,
    ) -> std::result::Result<PreparedExecution, RunError> {
        self.compile_input_with_source_access(source_name, None, input)
    }

    #[cfg(test)]
    pub(crate) fn compile_input_for_source_path(
        &mut self,
        source_path: &str,
        input: &str,
    ) -> std::result::Result<PreparedExecution, RunError> {
        self.compile_input_with_source_access(source_path, Some(source_path), input)
    }

    #[cfg(test)]
    fn compile_input_with_source_access(
        &mut self,
        source_name: &str,
        access_path: Option<&str>,
        input: &str,
    ) -> std::result::Result<PreparedExecution, RunError> {
        let mut scope = super::source_scope::SourceCompositionScope::enter(self);
        let session = scope.session();
        session.active_source_name = source_name.to_string();
        session.active_source_fullpath_name = access_path.map(str::to_owned);
        let handoff = session.project_handoff.clone();
        let companion = futures::executor::block_on(async {
            companion_source_statements_for_input_async(
                access_path,
                session.compat_mode,
                handoff.as_ref(),
            )
            .await
        })
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(format!("project composition failed: {error}"))
                    .with_source(error)
                    .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                    .with_identifier("RunMat:ProjectComposition")
                    .build(),
            )
        })?;
        session.pending_companion_source_discovery = Some(companion);
        session.compile_input(input)
    }

    pub(crate) fn compile_input(
        &mut self,
        input: &str,
    ) -> std::result::Result<PreparedExecution, RunError> {
        self.compile_input_with_portable_source(input, None, None)
    }

    pub(super) fn canonical_interactive_source(
        &self,
        input: &str,
    ) -> std::result::Result<crate::ExecutableSource, RunError> {
        let display_name = self.current_source_name();
        if let Some(captured) = self.active_primary_source.as_ref() {
            let source = crate::ExecutableSource::with_captured_primary(input, captured.clone())
                .map_err(|error| {
                    RunError::Runtime(
                        build_runtime_error(error)
                            .with_identifier("RunMat:ExecutableSourceIdentity")
                            .build(),
                    )
                })?;
            captured
                .validate(
                    self.project_handoff.as_ref().ok_or_else(|| {
                        RunError::Runtime(
                            build_runtime_error("captured primary has no installed project")
                                .with_identifier("RunMat:ExecutableSourceIdentity")
                                .build(),
                        )
                    })?,
                    &source,
                )
                .map_err(|error| {
                    RunError::Runtime(
                        build_runtime_error(error)
                            .with_identifier("RunMat:ExecutableSourceIdentity")
                            .build(),
                    )
                })?;
            return Ok(source);
        }
        let fallback = crate::ExecutableSource::new("interactive", display_name, input);
        let (owner_identity, relative_path) = executable_source_identity(
            &fallback,
            self.project_handoff.as_ref(),
            self.current_source_fullpath_name(),
            display_name,
            self.current_source_fullpath_name(),
        )
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error)
                    .with_identifier("RunMat:ExecutableSourceIdentity")
                    .build(),
            )
        })?;
        Ok(crate::ExecutableSource::new(
            owner_identity,
            relative_path,
            input,
        ))
    }

    #[tracing::instrument(name = "runtime.compile", skip_all, fields(input_len = input.len()))]
    fn compile_input_with_portable_source(
        &mut self,
        input: &str,
        portable_source: Option<&crate::ExecutableSource>,
        test_authority: Option<&CapturedTestSourceAuthority<'_>>,
    ) -> std::result::Result<PreparedExecution, RunError> {
        self.sync_workspace_binding_metadata()
            .map_err(RunError::Runtime)?;
        let source_name = self.current_source_name().to_string();
        let source_fullpath_name = self.current_source_fullpath_name().map(ToString::to_string);
        // Selected executable sources have already crossed their own authority
        // boundary; resolve an interactive source only for interactive input.
        let interactive_source;
        let source_for_table = match portable_source {
            Some(source) => source,
            None => {
                interactive_source = self.canonical_interactive_source(input)?;
                &interactive_source
            }
        };
        self.source_pool
            .intern_with_fullpath(&source_name, source_fullpath_name.as_deref(), input);
        let (
            ast,
            private_companion_function_names,
            private_companion_function_owners,
            private_companion_function_aliases,
            source_table,
            source_catalog,
            project_symbol_aliases,
        ) = {
            let _span = info_span!("runtime.parse").entered();
            let mut ast = parse_with_options(input, ParserOptions::new(self.compat_mode))?;
            if let Some(authority) = test_authority {
                authority
                    .select_program(&mut ast, input.len())
                    .map_err(|error| {
                        RunError::Runtime(
                            build_runtime_error(error)
                                .with_identifier("RunMat:ExecutableSourceIdentity")
                                .build(),
                        )
                    })?;
            }
            let primary_function_names = function_names_in_statements(&ast.body)
                .map(ToString::to_string)
                .collect::<HashSet<_>>();
            let mut companion = self
                .pending_companion_source_discovery
                .take()
                .unwrap_or_default();
            companion.apply_function_precedence(&primary_function_names);
            let mut retained_sources = Vec::new();
            {
                let retained_ids = self
                    .function_registry
                    .functions
                    .values()
                    .filter_map(|function| {
                        function
                            .source_id
                            .and_then(runmat_vm::BytecodeSourceRef::runtime)
                    })
                    .collect::<HashSet<_>>();
                for runtime_id in retained_ids {
                    let source = runtime_id
                        .resolve_in(&self.runtime_context)
                        .and_then(|id| {
                            runmat_runtime::source_context::source_info_in_context(
                                &self.runtime_context,
                                id,
                            )
                        })
                        .ok_or_else(|| {
                            RunError::Runtime(
                            build_runtime_error(
                                "retained function refers to a missing or foreign session source",
                            )
                            .with_identifier("RunMat:ExecutableSourceIdentity")
                            .build(),
                        )
                        })?;
                    let identity =
                        runmat_runtime::source_context::portable_source_identity_in_context(
                            &self.runtime_context,
                            runtime_id,
                        )
                        .ok_or_else(|| {
                            RunError::Runtime(
                                build_runtime_error(
                                    "retained function lacks its original portable source owner",
                                )
                                .with_identifier("RunMat:ExecutableSourceIdentity")
                                .build(),
                            )
                        })?;
                    retained_sources.push((
                        runtime_id,
                        SourceContextData {
                            display_name: source.name.to_string(),
                            fullpath_name: source.fullpath_name.as_deref().map(ToString::to_string),
                            text: source.text.to_string(),
                        },
                        identity,
                    ));
                }
            }
            let source_table = if let Some(authority) = test_authority {
                program_sources::ProgramSourceTable::build_with_test(
                    source_for_table,
                    source_fullpath_name.as_deref(),
                    &companion,
                    ast.body.len(),
                    self.project_handoff.as_ref(),
                    authority,
                    &retained_sources,
                )
            } else {
                program_sources::ProgramSourceTable::build(
                    source_for_table,
                    source_fullpath_name.as_deref(),
                    &companion,
                    ast.body.len(),
                    self.project_handoff.as_ref(),
                    &retained_sources,
                )
            }
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableSourceIdentity")
                        .build(),
                )
            })?;
            let private_companion_function_names =
                std::mem::take(&mut companion.private_function_names);
            let private_companion_function_owners =
                std::mem::take(&mut companion.private_function_owners);
            let private_companion_function_aliases =
                std::mem::take(&mut companion.private_function_aliases);
            let source_catalog = companion.source_catalog.take();
            let project_symbol_aliases = std::mem::take(&mut companion.project_symbol_aliases);
            if !companion.statements.is_empty() {
                ast.body.extend(
                    companion
                        .statements
                        .drain(..)
                        .map(|discovered| discovered.statement),
                );
            }
            (
                ast,
                private_companion_function_names,
                private_companion_function_owners,
                private_companion_function_aliases,
                source_table,
                source_catalog,
                project_symbol_aliases,
            )
        };
        let (lowering, mut mir, mut bytecode) = {
            let _span = info_span!("runtime.lower").entered();
            // WASM registrations are explicit rather than inventory-backed. Install
            // them before analysis so the retained catalog revision is identical to
            // the program environment subsequently bound into the executable.
            runmat_runtime::builtins::wasm_registry::register_all();
            let function_names = self.function_registry.names.clone();
            let function_output_arities = function_output_arities(&self.function_registry);
            let function_modifiers = self.program_service.function_modifiers();
            let workspace_bindings = self.lowering_workspace_bindings();
            let known_project_symbols = source_catalog
                .as_ref()
                .map(|catalog| &catalog.symbols)
                .cloned()
                .unwrap_or_default();
            let primary_source = source_table.primary_source();
            let frontend = runmat_static_analysis::frontend::analyze_program_with_catalog(
                &ast,
                &LoweringContext::new(&workspace_bindings)
                    .with_bound_functions(&function_names)
                    .with_function_output_arities(&function_output_arities)
                    .with_function_modifiers(&function_modifiers)
                    .with_known_project_symbols(&known_project_symbols)
                    .with_project_symbol_aliases(&project_symbol_aliases)
                    .with_top_level_sources(&source_table.statement_sources)
                    .with_primary_source(primary_source)
                    .with_private_functions(
                        &private_companion_function_owners,
                        &private_companion_function_aliases,
                    )
                    .with_runmat_extensions_enabled(self.compat_mode.allows_runmat_extensions())
                    .with_top_level_await_enabled(self.top_level_await_enabled),
                source_catalog.as_ref(),
            );
            if let Some(error) = frontend.lowering_failure {
                return Err(error.into());
            }
            if let Some(error) = frontend.compile_failure {
                return Err(bind_compile_error(error, &source_table));
            }
            if let Some(diagnostic) = fatal_frontend_diagnostic(&frontend.diagnostics) {
                return Err(bind_frontend_diagnostic(diagnostic, &source_table));
            }
            (
                frontend
                    .lowering
                    .expect("canonical frontend returned no lowering without an error"),
                frontend
                    .mir
                    .expect("canonical frontend returned no MIR without an error"),
                frontend
                    .bytecode
                    .expect("canonical frontend returned no bytecode without an error"),
            )
        };
        let sources_span = info_span!("runtime.compile.bind_sources").entered();
        crate::ExecutableUnit::retain_vm_layout(&mut bytecode, &lowering.assembly, &mir).map_err(
            |error| {
                RunError::Runtime(
                    build_runtime_error(format!(
                        "failed to retain executable VM layout: {error:?}"
                    ))
                    .with_identifier("RunMat:ExecutableLayout")
                    .build(),
                )
            },
        )?;
        source_table
            .validate_compiled_sources(&bytecode, &lowering.assembly)
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableSourceIdentity")
                        .build(),
                )
            })?;
        let private_owners =
            super::private_source_owner::CapturedPrivateSourceOwners::new(&source_table.source_map);
        for function in bytecode.function_registry.functions.values_mut() {
            function.private_source_owner =
                function.source_id.and_then(|id| private_owners.get(id));
        }
        bytecode.bound_functions = bytecode.function_registry.functions.clone();
        let source_projection = source_table
            .intern_runtime_projection(&mut self.source_pool, &self.runtime_context)
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableSourceIdentity")
                        .build(),
                )
            })?;
        // Full source/catalog/companion assembly is complete. The following
        // session publication only renames compiler coordinates; retain this
        // analysis's original revision-level function/point identities.
        drop(sources_span);
        let analysis = info_span!("runtime.compile.mir_analysis")
            .in_scope(|| runmat_mir::analysis::analyze_assembly(&mir));
        let _registry_span = info_span!("runtime.compile.publish_registry").entered();
        let (
            function_registry_after_success,
            next_semantic_function_id_after_success,
            function_id_remap,
            published_registry_for_reuse,
        ) = self.prepare_session_semantic_function_registry(
            &mut bytecode,
            &private_companion_function_names,
            Some(&source_projection),
        )?;
        if !function_id_remap.is_empty() {
            runmat_mir::remap_function_ids(&mut mir, &function_id_remap).map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableFunctionIdentity")
                        .build(),
                )
            })?;
            if let Some(layout) = &mut bytecode.layout {
                runmat_vm::remap_layout_function_ids(layout, &function_id_remap).map_err(
                    |error| {
                        RunError::Runtime(
                            build_runtime_error(format!(
                                "failed to publish executable VM layout identities: {error:?}"
                            ))
                            .with_identifier("RunMat:ExecutableFunctionIdentity")
                            .build(),
                        )
                    },
                )?;
            }
        }
        let analysis = analysis
            .rebind_eval_local_functions(&function_id_remap)
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableFunctionIdentity")
                        .build(),
                )
            })?;
        Ok(PreparedExecution {
            ast,
            lowering,
            mir,
            analysis,
            bytecode,
            portable_source_map: Some(source_table.source_map),
            source_projection,
            function_registry_after_success,
            published_registry_for_reuse,
            private_companion_function_names,
            current_function_ids: function_id_remap.values().copied().collect(),
            next_semantic_function_id_after_success,
        })
    }

    pub(crate) fn populate_callstack(
        &self,
        error: &mut RuntimeError,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) {
        runmat_runtime::execution::preserve_source_call_stack_in_context(error, runtime);
    }

    fn prepare_session_semantic_function_registry(
        &self,
        bytecode: &mut runmat_vm::Bytecode,
        private_companion_function_names: &HashSet<String>,
        source_projection: Option<&program_sources::SourceCompileProjection>,
    ) -> std::result::Result<
        (
            runmat_vm::FunctionRegistry,
            usize,
            HashMap<runmat_hir::FunctionId, runmat_hir::FunctionId>,
            runmat_vm::FunctionRegistry,
        ),
        RunError,
    > {
        let mut session_registry = self.function_registry.clone();
        let mut execution_registry = session_registry.clone();
        if let Some(projection) = source_projection {
            execution_registry
                .project_current_source_ids(|runtime| {
                    projection.program_ref_for_runtime_ref(runtime)
                })
                .map_err(|missing| {
                    RunError::Runtime(
                        build_runtime_error(format!(
                            "retained source {missing:?} is absent from executable provenance"
                        ))
                        .with_identifier("RunMat:ExecutableSourceIdentity")
                        .build(),
                    )
                })?;
        }
        let published_registry_for_reuse = execution_registry.clone();
        // Published semantic functions share the retained executable layout's
        // function-id namespace. Reserve every unit-local layout id before
        // assigning session-stable ids so a closure/local function can never
        // overwrite the entrypoint (or another unremapped unit function).
        let first_id_after_unit = bytecode
            .layout
            .as_ref()
            .and_then(|layout| layout.functions.keys().map(|function| function.0).max())
            .map(|function| {
                function
                    .checked_add(1)
                    .expect("unit-local function identity space exhausted")
            })
            .unwrap_or(0);
        let mut next_semantic_function_id = self.next_semantic_function_id.max(first_id_after_unit);
        let current_registry = bytecode.function_registry();
        if current_registry.functions.is_empty() {
            bytecode.function_registry = execution_registry;
            bytecode.bound_functions = bytecode.function_registry.functions.clone();
            let bindings = semantic_function_bindings(bytecode, &HashSet::new())?;
            runmat_vm::remap_session_entrypoint(bytecode, &HashMap::new(), &bindings).map_err(
                |error| {
                    RunError::Runtime(
                        build_runtime_error(format!("session entrypoint binding: {error}"))
                            .with_identifier("RunMat:SelectedFusionStageProjection")
                            .build(),
                    )
                },
            )?;
            bytecode.function_registry.finalize_selected_image_views();
            session_registry.finalize_selected_image_views();
            return Ok((
                session_registry,
                next_semantic_function_id,
                HashMap::new(),
                published_registry_for_reuse,
            ));
        }

        // Provisional identities remain fresh until the *decorated* product
        // proves same-stage equality immediately before immutable sealing.
        let mut remap = HashMap::new();
        let mut ids: Vec<_> = current_registry.functions.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        for old_id in ids {
            let new_id = runmat_hir::FunctionId(next_semantic_function_id);
            next_semantic_function_id += 1;
            remap.insert(old_id, new_id);
        }

        let name_remaps: Vec<(String, runmat_hir::FunctionId)> = current_registry
            .names
            .iter()
            .map(|(name, function)| (name.clone(), *function))
            .collect();

        let mut replaced_sources = Vec::new();
        for function in current_registry.functions.values() {
            if private_companion_function_names.contains(&function.display_name) {
                continue;
            }
            if let Some(existing_id) = session_registry.resolve_name(&function.display_name) {
                if let Some(source_id) = session_registry.get(existing_id).and_then(|existing| {
                    existing
                        .source_id
                        .and_then(runmat_vm::BytecodeSourceRef::runtime)
                }) {
                    if !replaced_sources.contains(&source_id) {
                        replaced_sources.push(source_id);
                    }
                }
            }
        }
        for source_id in replaced_sources {
            // Retire the complete replaced source in both views. The execution
            // view has projected source references, but shares these exact
            // function identities with the durable session view.
            for function in session_registry.functions_for_source(source_id) {
                execution_registry.remove(*function);
            }
            session_registry.remove_source(source_id);
        }

        for (old_id, function) in current_registry.into_functions() {
            let Some(new_id) = remap.get(&old_id).copied() else {
                continue;
            };
            let mut function = function;
            let source = function.source_id.or(bytecode.source_id);
            runmat_vm::remap_session_function(&mut function, new_id, source, &remap).map_err(
                |error| {
                    RunError::Runtime(
                        build_runtime_error(format!("session function remap: {error}"))
                            .with_identifier("RunMat:SelectedFusionStageProjection")
                            .build(),
                    )
                },
            )?;
            let persist_function =
                !private_companion_function_names.contains(&function.display_name);
            execution_registry.insert_replacing_name(function.clone());
            if persist_function {
                if let Some(projection) = source_projection {
                    if let Some(program) = function.source_id {
                        let runtime = projection.runtime_ref_for_program_ref(program).ok_or_else(|| {
                            RunError::Runtime(
                                build_runtime_error("published function source is absent from executable provenance")
                                    .with_identifier("RunMat:ExecutableSourceIdentity")
                                    .build(),
                            )
                        })?;
                        runmat_vm::rebind_session_source(&mut function, Some(runtime)).map_err(
                            |error| {
                                RunError::Runtime(
                                    build_runtime_error(format!(
                                        "session function source rebind: {error}"
                                    ))
                                    .with_identifier("RunMat:SelectedFusionStageProjection")
                                    .build(),
                                )
                            },
                        )?;
                    }
                }
                session_registry.insert_replacing_name(function);
            }
        }
        for (name, old_id) in name_remaps {
            let Some(new_id) = remap.get(&old_id).copied() else {
                continue;
            };
            execution_registry.names.insert(name.clone(), new_id);
            if !private_companion_function_names.contains(&name) {
                session_registry.names.insert(name, new_id);
            }
        }

        bytecode.function_registry = execution_registry;
        bytecode.bound_functions = bytecode.function_registry.functions.clone();
        let bindings = semantic_function_bindings(bytecode, &remap.values().copied().collect())?;
        runmat_vm::remap_session_entrypoint(bytecode, &remap, &bindings).map_err(|error| {
            RunError::Runtime(
                build_runtime_error(format!("session entrypoint remap: {error}"))
                    .with_identifier("RunMat:SelectedFusionStageProjection")
                    .build(),
            )
        })?;
        // Remapping and session publication invalidate the compiler's
        // generation, so seal both published registries before any call.
        bytecode.function_registry.finalize_selected_image_views();
        session_registry.finalize_selected_image_views();
        Ok((
            session_registry,
            next_semantic_function_id,
            remap,
            published_registry_for_reuse,
        ))
    }

    pub(crate) fn normalize_error_namespace(&self, error: &mut RuntimeError) {
        let Some(identifier) = error.identifier.clone() else {
            return;
        };
        let suffix = identifier
            .split_once(':')
            .map(|(_, suffix)| suffix)
            .unwrap_or(identifier.as_str());
        error.identifier = Some(format!("{}:{suffix}", self.error_namespace));
    }

    /// Compile the input and produce a fusion plan snapshot without executing.
    pub fn compile_fusion_plan(
        &mut self,
        input: &str,
    ) -> std::result::Result<Option<FusionPlanSnapshot>, RunError> {
        let prepared = self.compile_input(input)?;
        let runtime_groups = prepared.bytecode.runtime_fusion_groups();
        let (runtime_graph, runtime_graph_source) = prepared
            .bytecode
            .runtime_accel_graph_for_fusion_with_source(&runtime_groups);
        Ok(build_fusion_snapshot(
            &runtime_groups,
            &prepared
                .bytecode
                .fusion_metadata
                .mir_fusion_candidate_groups,
            &prepared.bytecode.fusion_metadata.instruction_windows,
            Some(FusionPlannerMetadata {
                source: "semantic-mir-analysis".to_string(),
                accel_graph_state: if runtime_graph.is_some() {
                    "present".to_string()
                } else {
                    "missing".to_string()
                },
                accel_graph_source: runtime_graph_source.as_str().to_string(),
                mir_local_fact_count: mir_local_fact_count_for_entrypoint(
                    &prepared.analysis,
                    &prepared.lowering.assembly,
                ),
                mir_diagnostic_count: prepared.analysis.diagnostics.len(),
                mir_fusion_signal_count: prepared.bytecode.fusion_metadata.mir_fusion_signal_count,
                mir_fusion_candidate_group_count: prepared
                    .bytecode
                    .fusion_metadata
                    .mir_fusion_candidate_group_count,
                mir_semantic_instruction_window_count: prepared
                    .bytecode
                    .fusion_metadata
                    .instruction_window_count,
            }),
        ))
    }

    pub(crate) fn prepare_variable_array_for_execution(
        &self,
        bytecode: &runmat_vm::Bytecode,
        updated_var_mapping: &HashMap<String, usize>,
        workspace: &runmat_runtime::workspace::BaseWorkspaceProjectionService,
        runtime: &runmat_runtime::context::RuntimeContext,
        debug_trace: bool,
    ) -> std::result::Result<runmat_vm::InterpreterVars, RuntimeError> {
        let max_var_id = updated_var_mapping.values().copied().max();
        let required_len = max_var_id
            .map(|slot| slot.checked_add(1))
            .unwrap_or(Some(0))
            .ok_or_else(|| {
                build_runtime_error("workspace slot extent overflow")
                    .with_identifier("RunMat:WorkspaceBindingSlots")
                    .build()
            })?
            .max(bytecode.var_count);
        let generation = runmat_runtime::context::RuntimeWorkspaceService::capture_workspace_generation_with_context(
            workspace, runtime,
        )?.ok_or_else(|| {
            build_runtime_error("selected workspace has no canonical read generation")
                .with_identifier("RunMat:Interpreter:WorkspaceGeneration").build()
        })?;
        if debug_trace {
            debug!(
                bytecode_var_count = bytecode.var_count,
                required_len, max_var_id, "[repl] prepare canonical workspace slot leases"
            );
        }
        runmat_vm::InterpreterVars::from_workspace_generation(
            runtime,
            &generation,
            updated_var_mapping
                .iter()
                .map(|(name, slot)| (name.as_str(), *slot)),
            required_len,
        )
    }
}

fn semantic_function_bindings(
    bytecode: &runmat_vm::Bytecode,
    current: &HashSet<runmat_hir::FunctionId>,
) -> Result<Vec<runmat_vm::SessionCallableBinding>, RunError> {
    let registry = &bytecode.function_registry;
    let mut bindings = Vec::new();
    for (pc, instr) in bytecode.instructions.iter().enumerate() {
        if let runmat_vm::Instr::CreateFunctionHandle(name)
        | runmat_vm::Instr::CreateExternalFunctionHandle(name) = instr
        {
            if let Some(function) = registry.resolve_name(name) {
                if matches!(instr, runmat_vm::Instr::CreateFunctionHandle(_)) {
                    bindings.push(runmat_vm::SessionCallableBinding::named(
                        pc,
                        function,
                        name.clone(),
                        session_reference_kind(function, current),
                    ));
                }
                continue;
            }
            if runmat_builtins::builtin_name_is_known(name) {
                continue;
            }
            let constructor =
                registry
                    .resolve_session_constructor_handle(name)
                    .map_err(|message| {
                        RunError::Runtime(
                            build_runtime_error(message)
                                .with_identifier("RunMat:SelectedConstructorHandle")
                                .build(),
                        )
                    })?;
            if let Some(function) = constructor {
                bindings.push(runmat_vm::SessionCallableBinding::constructor(
                    pc,
                    function,
                    name.clone(),
                    session_reference_kind(function, current),
                ));
            }
        }
    }
    Ok(bindings)
}

fn session_reference_kind(
    function: runmat_hir::FunctionId,
    current: &HashSet<runmat_hir::FunctionId>,
) -> runmat_vm::FunctionReferenceKind {
    if current.contains(&function) {
        runmat_vm::FunctionReferenceKind::Local
    } else {
        runmat_vm::FunctionReferenceKind::External
    }
}
#[cfg(test)]
mod source_discovery_tests {
    use super::*;
    use std::fs;

    #[test]
    fn successive_primary_functions_and_nested_companions_keep_source_ownership() {
        let _provider_guard = runmat_filesystem::provider_override_lock();
        let temp = tempfile::TempDir::new().unwrap();
        fs::write(
            temp.path().join("helper.m"),
            "function y = helper(x)\ny = inner(x);\nfunction z = inner(v)\nz = v + 1;\nend\nend\n",
        )
        .unwrap();
        let mut session = RunMatSession::with_options(false, false).unwrap();
        for name in ["alpha", "beta"] {
            let path = temp.path().join(format!("{name}.m"));
            let text =
                format!("function y = {name}(x)\narguments\nx double\nend\ny = helper(x);\nend\n");
            fs::write(&path, &text).unwrap();
            let unit = futures::executor::block_on(session.compile_executable_unit(
                crate::ExecutableSource::frozen(
                    "source-attribution",
                    path.to_str().unwrap(),
                    path.to_str().unwrap(),
                    text,
                ),
                None,
            ))
            .unwrap();
            unit.primary_function_interface(name).unwrap();
            assert!(
                unit.primary_function_interface("helper").is_err(),
                "companion must not qualify as a primary-file function"
            );
            let primary = unit
                .source_map()
                .entries()
                .iter()
                .find(|entry| entry.relative_path == path.to_str().unwrap())
                .unwrap();
            let helper = unit
                .mir()
                .functions
                .values()
                .find(|metadata| metadata.name.0 == "helper")
                .unwrap();
            assert_ne!(helper.source, primary.source_id);
            let helper_source = unit
                .source_map()
                .entries()
                .iter()
                .find(|entry| entry.source_id == helper.source)
                .unwrap();
            assert!(helper_source.relative_path.ends_with("helper.m"));
            assert!(unit
                .mir()
                .functions
                .values()
                .any(|metadata| metadata.parent.is_some()));
            for (id, metadata) in &unit.mir().functions {
                if let Some(parent) = metadata.parent {
                    assert_eq!(metadata.source, unit.mir().functions[&parent].source);
                }
                if metadata.name.0 == name {
                    assert_eq!(metadata.source, primary.source_id);
                }
                if let Some(function) = unit.bytecode().function_registry.get(*id) {
                    assert_eq!(function.source_id.unwrap().program(), Some(metadata.source));
                }
            }
            for region in &unit.analysis().regions {
                let facts = unit.function_analysis(region.contract.id.function).unwrap();
                let Some(runmat_types::CallableIdentity::BoundFunction(function)) =
                    facts.callable.identity.as_ref()
                else {
                    panic!("region owner self binding");
                };
                assert_eq!(
                    region.contract.source,
                    unit.mir().functions[function].source
                );
            }
        }
    }

    fn discovered_function_names(discovery: &CompanionSourceDiscovery) -> HashSet<String> {
        discovery
            .statements
            .iter()
            .filter_map(|discovered| match &discovered.statement {
                runmat_parser::Stmt::Function { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn precedence_keeps_the_winning_statement_and_its_source_together() {
        let mut discovery = CompanionSourceDiscovery::default();
        let first = "function y = helper(); y = 1; end";
        let private = "function y = helper(); y = 2; end";
        discovery.extend_body(
            runmat_parser::parse(first).unwrap().body,
            None,
            HashMap::new(),
            Some(SourceContextData {
                display_name: "ordinary/helper.m".into(),
                fullpath_name: Some("/project/ordinary/helper.m".into()),
                text: first.into(),
            }),
        );
        discovery.extend_body(
            runmat_parser::parse(private).unwrap().body,
            Some("owner"),
            HashMap::new(),
            Some(SourceContextData {
                display_name: "private/helper.m".into(),
                fullpath_name: Some("/project/private/helper.m".into()),
                text: private.into(),
            }),
        );
        discovery.apply_function_precedence(&HashSet::new());
        assert_eq!(discovery.statements.len(), 1);
        assert!(discovery.statements[0].is_private);
        assert_eq!(
            discovery.statements[0]
                .source
                .as_ref()
                .unwrap()
                .display_name,
            "private/helper.m"
        );
    }

    #[test]
    fn loose_runtime_companions_use_the_shared_matlab_source_index() {
        let _provider_guard = runmat_filesystem::provider_override_lock();
        let temp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(temp.path().join("+pkg")).unwrap();
        fs::create_dir_all(temp.path().join("private")).unwrap();
        fs::create_dir_all(temp.path().join("ordinary-nested")).unwrap();
        let main = temp.path().join("main.m");
        fs::write(&main, "value = helper(1);").unwrap();
        fs::write(
            temp.path().join("helper.m"),
            "function y = helper(x)\ny = x;\nend",
        )
        .unwrap();
        fs::write(
            temp.path().join("+pkg/tool.m"),
            "function y = tool(x)\ny = x;\nend",
        )
        .unwrap();
        fs::write(
            temp.path().join("private/secret.m"),
            "function y = secret(x)\ny = x;\nend",
        )
        .unwrap();
        fs::write(
            temp.path().join("ordinary-nested/hidden.m"),
            "function y = hidden(x)\ny = x;\nend",
        )
        .unwrap();

        let discovery = futures::executor::block_on(discover_companion_source_statements_async(
            main.to_str().unwrap(),
            runmat_parser::CompatMode::RunMat,
        ))
        .expect("discover loose companion sources");
        let names = discovered_function_names(&discovery);
        assert!(names.contains("helper"));
        assert!(names.contains("pkg.tool"));
        assert!(names.contains("secret"));
        assert!(!names.iter().any(|name| name.ends_with("hidden")));
    }

    #[test]
    fn inline_project_lookup_uses_manifest_authority_and_root_private_visibility() {
        let _provider_guard = runmat_filesystem::provider_override_lock();
        let temp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(temp.path().join("private")).unwrap();
        fs::create_dir_all(temp.path().join("nested/private")).unwrap();
        fs::write(
            temp.path().join("runmat.toml"),
            "[package]\nname = \"inline-project\"\n[sources]\nroots = [\".\"]\n",
        )
        .unwrap();
        let main = temp.path().join("main.m");
        fs::write(&main, "value = helper();").unwrap();
        fs::write(
            temp.path().join("helper.m"),
            "function y = helper(); y = 1; end",
        )
        .unwrap();
        fs::write(
            temp.path().join("private/root_secret.m"),
            "function y = root_secret(); y = 2; end",
        )
        .unwrap();
        fs::write(
            temp.path().join("nested/private/nested_secret.m"),
            "function y = nested_secret(); y = 3; end",
        )
        .unwrap();
        let project = futures::executor::block_on(
            runmat_package::discover_frozen_project_from_async(&main, Default::default()),
        )
        .unwrap()
        .unwrap();
        let handoff = runmat_package::FrozenProjectHandoff::new(project);
        let discovery = futures::executor::block_on(companion_source_statements_for_input_async(
            None,
            runmat_parser::CompatMode::RunMat,
            Some(&handoff),
        ))
        .unwrap();
        let names = discovered_function_names(&discovery);
        assert!(names.contains("helper"));
        assert!(names.contains("root_secret"));
        assert!(!names.contains("nested_secret"));
        let symbols = &discovery.source_catalog.as_ref().unwrap().symbols;
        assert!(symbols.contains("root_secret"));
        assert!(!symbols.contains("nested_secret"));
        let inline = futures::executor::block_on(companion_source_statements_for_input_async(
            None,
            runmat_parser::CompatMode::RunMat,
            None,
        ))
        .unwrap();
        assert!(inline.statements.is_empty());
        assert!(inline.source_catalog.is_none());
    }

    #[test]
    fn frozen_runtime_companions_enforce_alias_transitive_and_private_visibility() {
        let _provider_guard = runmat_filesystem::provider_override_lock();
        let temp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(temp.path().join("src")).unwrap();
        fs::write(
            temp.path().join("runmat.toml"),
            r#"
[package]
name = "root"

[sources]
roots = ["src"]

[dependencies]
middle = { path = "deps/middle" }
"#,
        )
        .unwrap();
        let main = temp.path().join("src/main.m");
        fs::write(&main, "value = middle.api();").unwrap();
        fs::create_dir_all(temp.path().join("deps/middle/src/private")).unwrap();
        fs::write(
            temp.path().join("deps/middle/runmat.toml"),
            r#"
[package]
name = "middle"

[sources]
roots = ["src"]

[dependencies]
leaf = { path = "deps/leaf" }
"#,
        )
        .unwrap();
        fs::write(
            temp.path().join("deps/middle/src/api.m"),
            "function y = api(); y = 1; end",
        )
        .unwrap();
        fs::write(
            temp.path().join("deps/middle/src/private/secret.m"),
            "function y = secret(); y = 2; end",
        )
        .unwrap();
        fs::create_dir_all(temp.path().join("deps/middle/deps/leaf/src")).unwrap();
        fs::write(
            temp.path().join("deps/middle/deps/leaf/runmat.toml"),
            r#"
[package]
name = "leaf"

[sources]
roots = ["src"]
"#,
        )
        .unwrap();
        fs::write(
            temp.path().join("deps/middle/deps/leaf/src/transitive.m"),
            "function y = transitive(); y = 3; end",
        )
        .unwrap();

        let discovery = futures::executor::block_on(discover_companion_source_statements_async(
            main.to_str().unwrap(),
            runmat_parser::CompatMode::RunMat,
        ))
        .expect("discover frozen companion sources");
        let names = discovered_function_names(&discovery);
        assert!(names.contains("api"));
        assert!(!names.iter().any(|name| name.ends_with("secret")));
        assert!(!names.iter().any(|name| name.ends_with("transitive")));
        assert_eq!(
            discovery.project_symbol_aliases.get("middle.api"),
            Some(&"api".to_string())
        );
    }
}

#[cfg(test)]
#[path = "compile/name_binding_tests.rs"]
mod name_binding_tests;
#[cfg(test)]
#[path = "compile/portable_source_tests.rs"]
mod portable_source_tests;
#[cfg(test)]
#[path = "compile/stage_remap_tests.rs"]
mod stage_remap_tests;
