use super::*;
use crate::fusion::FusionPlannerMetadata;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[path = "compile/program_sources.rs"]
pub(super) mod program_sources;
use program_sources::identity::{executable_primary_access_path, executable_source_identity};

fn entrypoint_target_function(
    assembly: &runmat_hir::HirAssembly,
) -> Option<runmat_hir::FunctionId> {
    assembly
        .entrypoints
        .first()
        .map(|entrypoint| entrypoint.target)
}

fn mir_local_fact_count_for_entrypoint(
    analysis: &runmat_mir::analysis::AnalysisStore,
    assembly: &runmat_hir::HirAssembly,
) -> usize {
    let function = entrypoint_target_function(assembly)
        .and_then(|function| u32::try_from(function.0).ok())
        .map(runmat_types::ProgramFunctionId);
    analysis.local_value_count(function)
}

fn function_output_arities(
    registry: &runmat_vm::FunctionRegistry,
) -> HashMap<runmat_hir::FunctionId, runmat_hir::FunctionOutputArity> {
    registry
        .functions
        .iter()
        .map(|(id, function)| {
            (
                *id,
                runmat_hir::FunctionOutputArity::new(
                    function.output_slots.len(),
                    function.varargout_slot.is_some(),
                ),
            )
        })
        .collect()
}

fn source_lookup_cwd(source_name: &str) -> Option<PathBuf> {
    let source_path = PathBuf::from(source_name);
    if source_path.is_absolute() {
        return source_path
            .parent()
            .map(Path::to_path_buf)
            .or_else(|| Some(PathBuf::from(".")));
    }
    Some(runmat_filesystem::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn resolved_source_path(source_name: &str, cwd: &Path) -> PathBuf {
    crate::diagnostic_path::resolve_against_base(source_name, cwd)
}

fn path_to_source_name(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn is_class_source_body(stmts: &[runmat_parser::Stmt]) -> bool {
    let has_classdef = stmts
        .iter()
        .any(|stmt| matches!(stmt, runmat_parser::Stmt::ClassDef { .. }));
    if !has_classdef {
        return false;
    }
    stmts.iter().all(|stmt| {
        matches!(
            stmt,
            runmat_parser::Stmt::ClassDef { .. } | runmat_parser::Stmt::Function { .. }
        )
    })
}

fn is_function_source_body(stmts: &[runmat_parser::Stmt]) -> bool {
    !stmts.is_empty()
        && stmts
            .iter()
            .all(|stmt| matches!(stmt, runmat_parser::Stmt::Function { .. }))
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

fn qualify_companion_classdefs(stmts: &mut [runmat_parser::Stmt], qualified_name: &str) {
    for stmt in stmts {
        if let runmat_parser::Stmt::ClassDef { name, .. } = stmt {
            if !name.contains('.') {
                *name = qualified_name.to_string();
            }
        }
    }
}

fn qualify_companion_functions(stmts: &mut [runmat_parser::Stmt], qualified_name: &str) {
    for stmt in stmts {
        if let runmat_parser::Stmt::Function { name, .. } = stmt {
            if !name.contains('.') {
                *name = qualified_name.to_string();
            }
        }
    }
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

fn function_owner_scope_from_qualified_name(qualified_name: &str) -> String {
    qualified_name
        .rsplit_once('.')
        .map(|(owner, _)| owner.to_string())
        .unwrap_or_default()
}

fn function_leaf_name(name: &str) -> &str {
    name.rsplit_once('.').map(|(_, leaf)| leaf).unwrap_or(name)
}

fn synthetic_private_function_name(owner_scope: &str, leaf_name: &str) -> String {
    if owner_scope.is_empty() {
        format!("__private__.{leaf_name}")
    } else {
        format!("{owner_scope}.__private__.{leaf_name}")
    }
}

fn qualify_private_companion_functions(
    stmts: &mut [runmat_parser::Stmt],
    owner_scope: &str,
    primary_visible: bool,
) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    for stmt in stmts {
        if let runmat_parser::Stmt::Function { name, .. } = stmt {
            let leaf = function_leaf_name(name).to_string();
            let display_name = if primary_visible {
                leaf.clone()
            } else {
                synthetic_private_function_name(owner_scope, &leaf)
            };
            *name = display_name.clone();
            aliases.insert(leaf, display_name);
        }
    }
    aliases
}

#[derive(Default)]
pub(super) struct CompanionSourceDiscovery {
    pub statements: Vec<DiscoveredSourceStatement>,
    pub source_catalog: Option<runmat_package::DiscoveredSourceSymbols>,
    pub project_symbol_aliases: HashMap<String, String>,
    pub private_function_names: HashSet<String>,
    pub private_function_owners: HashMap<String, String>,
    pub private_function_aliases: HashMap<String, HashMap<String, String>>,
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

fn project_symbol_aliases(
    catalog: Option<&runmat_package::DiscoveredSourceSymbols>,
) -> HashMap<String, String> {
    let Some(catalog) = catalog else {
        return HashMap::new();
    };
    catalog
        .definitions
        .iter()
        .filter_map(|definition| {
            let alias = definition.dependency_alias.as_deref()?;
            let original = definition.name.strip_prefix(&format!("{alias}."))?;
            let has_unique_bare_name = catalog.definitions.iter().any(|candidate| {
                candidate.name == original
                    && candidate.dependency_alias.as_deref() == Some(alias)
                    && candidate.source_id == definition.source_id
            });
            Some((
                definition.name.clone(),
                if has_unique_bare_name {
                    original.to_string()
                } else {
                    definition.name.clone()
                },
            ))
        })
        .collect()
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
) -> Result<Option<ProjectCompilationContext>, String> {
    let Some(cwd) = source_lookup_cwd(source_name) else {
        return Ok(None);
    };
    let primary_source_path = resolved_source_path(source_name, &cwd);
    let primary_is_local_file = runmat_filesystem::metadata_async(&primary_source_path)
        .await
        .map(|metadata| metadata.is_file())
        .unwrap_or(false);
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
    .map_err(|error| error.to_string())?;
    let frozen = if let Some(frozen) = frozen {
        let handoff = runmat_package::FrozenProjectHandoff::new(frozen);
        handoff.validate().map_err(|error| error.to_string())?;
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
            .map_err(|error| error.to_string())?;
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
) -> Result<ProjectCompilationContext, String> {
    handoff.validate().map_err(|error| error.to_string())?;
    let cwd = source_lookup_cwd(source_name)
        .ok_or_else(|| format!("cannot resolve project requester `{source_name}`"))?;
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

async fn compose_frozen_project_async(
    context: &ProjectCompilationContext,
    frozen: &runmat_package::FrozenProject,
    compat_mode: runmat_parser::CompatMode,
) -> Result<CompanionSourceDiscovery, String> {
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
        let file_path = access_path.to_path_buf();
        let file_path =
            runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(&file_path)
                .await;
        let contents =
            runmat_runtime::builtins::io::repl_fs::pcode::read_source_text_async(&file_path)
                .await
                .map_err(|error| {
                    format!(
                        "failed to read graph-declared source {}: {error}",
                        file_path.display()
                    )
                })?;
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
                text: contents,
            }),
        );
    }
    Ok(out)
}

async fn compose_loose_project_async(
    context: &ProjectCompilationContext,
    source_index: &runmat_config::project::ProjectSourceIndex,
    compat_mode: runmat_parser::CompatMode,
) -> Result<CompanionSourceDiscovery, String> {
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
        let Ok(contents) = runmat_filesystem::read_to_string_async(&path).await else {
            continue;
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
) -> Result<CompanionSourceDiscovery, String> {
    if let Some(frozen) = context.frozen.as_ref() {
        compose_frozen_project_async(context, frozen, compat_mode).await
    } else if let Some(loose_index) = context.loose_index.as_ref() {
        compose_loose_project_async(context, loose_index, compat_mode).await
    } else {
        Ok(CompanionSourceDiscovery::default())
    }
}

pub(super) async fn discover_companion_source_statements_async(
    source_name: &str,
    compat_mode: runmat_parser::CompatMode,
) -> Result<CompanionSourceDiscovery, String> {
    let Some(context) = discover_project_compilation_context_async(source_name).await? else {
        return Ok(CompanionSourceDiscovery::default());
    };
    compose_project_compilation_context_async(&context, compat_mode).await
}

pub(super) async fn companion_source_statements_from_handoff_async(
    source_name: &str,
    compat_mode: runmat_parser::CompatMode,
    handoff: &runmat_package::FrozenProjectHandoff,
) -> Result<CompanionSourceDiscovery, String> {
    let context = project_compilation_context_from_handoff(source_name, handoff)?;
    compose_project_compilation_context_async(&context, compat_mode).await
}

impl RunMatSession {
    /// Compile one immutable source/revision product without executing it or
    /// publishing its functions into the interactive session.
    pub async fn compile_executable_unit(
        &mut self,
        source: crate::ExecutableSource,
        program_revision: Option<runmat_execution::ProgramRevision>,
    ) -> std::result::Result<crate::ExecutableUnit, RunError> {
        self.ensure_user_path_hydrated().await?;
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

        let primary_access_path =
            executable_primary_access_path(&source, self.project_handoff.as_ref()).map_err(
                |error| {
                    RunError::Runtime(
                        build_runtime_error(error)
                            .with_identifier("RunMat:ExecutableSourceIdentity")
                            .build(),
                    )
                },
            )?;
        let previous_source_name = self.active_source_name.clone();
        let previous_source_fullpath_name = self.active_source_fullpath_name.clone();
        let previous_pending = self.pending_companion_source_discovery.take();
        self.active_source_name = source.relative_path.clone();
        self.active_source_fullpath_name = primary_access_path;

        let companion_result = if let Some(handoff) = self.project_handoff.as_ref() {
            companion_source_statements_from_handoff_async(
                &source.relative_path,
                self.compat_mode,
                handoff,
            )
            .await
        } else {
            discover_companion_source_statements_async(&source.relative_path, self.compat_mode)
                .await
        };
        let result = match companion_result {
            Ok(companion) => {
                self.pending_companion_source_discovery = Some(companion);
                self.compile_input_with_portable_source(&source.text, Some(&source))
            }
            Err(error) => Err(RunError::Runtime(
                build_runtime_error(format!("project composition failed: {error}"))
                    .with_identifier("RunMat:ProjectComposition")
                    .build(),
            )),
        };

        self.active_source_name = previous_source_name;
        self.active_source_fullpath_name = previous_source_fullpath_name;
        self.pending_companion_source_discovery = previous_pending;

        let mut prepared = result?;
        let source_map = prepared
            .portable_source_map
            .take()
            .expect("portable compilation must retain its source table");
        let revision = crate::ExecutableRevision::derive(
            &source,
            program_revision,
            crate::program_environment(self.compat_mode),
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
        crate::ExecutableUnit::new(
            source,
            revision,
            source_map,
            prepared.mir,
            prepared.analysis,
            prepared.bytecode,
        )
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
    pub(crate) fn compile_input_for_source_name(
        &mut self,
        source_name: &str,
        input: &str,
    ) -> std::result::Result<PreparedExecution, RunError> {
        let previous_source_name = self.active_source_name.clone();
        let previous_source_fullpath_name = self.active_source_fullpath_name.clone();
        self.active_source_name = source_name.to_string();
        self.active_source_fullpath_name = None;
        let handoff = self.project_handoff.clone();
        let companion = futures::executor::block_on(async {
            if let Some(handoff) = handoff.as_ref() {
                companion_source_statements_from_handoff_async(
                    source_name,
                    self.compat_mode,
                    handoff,
                )
                .await
            } else {
                discover_companion_source_statements_async(source_name, self.compat_mode).await
            }
        })
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(format!("project composition failed: {error}"))
                    .with_identifier("RunMat:ProjectComposition")
                    .build(),
            )
        })?;
        self.pending_companion_source_discovery = Some(companion);
        let result = self.compile_input(input);
        self.active_source_name = previous_source_name;
        self.active_source_fullpath_name = previous_source_fullpath_name;
        result
    }

    pub(crate) fn compile_input(
        &mut self,
        input: &str,
    ) -> std::result::Result<PreparedExecution, RunError> {
        self.compile_input_with_portable_source(input, None)
    }

    pub(super) fn canonical_interactive_source(
        &self,
        input: &str,
    ) -> std::result::Result<crate::ExecutableSource, RunError> {
        let display_name = self.current_source_name();
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

    fn compile_input_with_portable_source(
        &mut self,
        input: &str,
        portable_source: Option<&crate::ExecutableSource>,
    ) -> std::result::Result<PreparedExecution, RunError> {
        let source_name = self.current_source_name().to_string();
        let source_fullpath_name = self.current_source_fullpath_name().map(ToString::to_string);
        let interactive_source = self.canonical_interactive_source(input)?;
        let source_for_table = portable_source.unwrap_or(&interactive_source);
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
                    retained_sources.push((
                        runtime_id,
                        SourceContextData {
                            display_name: source.name.to_string(),
                            fullpath_name: source.fullpath_name.as_deref().map(ToString::to_string),
                            text: source.text.to_string(),
                        },
                    ));
                }
            }
            let source_table = program_sources::ProgramSourceTable::build(
                source_for_table,
                source_fullpath_name.as_deref(),
                &companion,
                ast.body.len(),
                self.project_handoff.as_ref(),
                &retained_sources,
            )
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
        let (lowering, mut mir, mut analysis, mut bytecode) = {
            let _span = info_span!("runtime.lower").entered();
            // WASM registrations are explicit rather than inventory-backed. Install
            // them before analysis so the retained catalog revision is identical to
            // the program environment subsequently bound into the executable.
            runmat_runtime::builtins::wasm_registry::register_all();
            let function_names = self.function_registry.names.clone();
            let function_output_arities = function_output_arities(&self.function_registry);
            let workspace_bindings = self.lowering_workspace_bindings();
            let known_project_symbols = source_catalog
                .as_ref()
                .map(|catalog| &catalog.symbols)
                .cloned()
                .unwrap_or_default();
            let frontend = runmat_static_analysis::frontend::analyze_program_with_catalog(
                &ast,
                &LoweringContext::new(&workspace_bindings)
                    .with_bound_functions(&function_names)
                    .with_function_output_arities(&function_output_arities)
                    .with_known_project_symbols(&known_project_symbols)
                    .with_project_symbol_aliases(&project_symbol_aliases)
                    .with_top_level_sources(&source_table.statement_sources)
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
                return Err(error.into());
            }
            (
                frontend
                    .lowering
                    .expect("canonical frontend returned no lowering without an error"),
                frontend
                    .mir
                    .expect("canonical frontend returned no MIR without an error"),
                frontend
                    .facts
                    .expect("canonical frontend returned no facts without an error"),
                frontend
                    .bytecode
                    .expect("canonical frontend returned no bytecode without an error"),
            )
        };
        if bytecode.layout.is_none() {
            bytecode.layout = Some(runmat_vm::derive_layout(&lowering.assembly, &mir).map_err(
                |error| {
                    RunError::Runtime(
                        build_runtime_error(format!(
                            "failed to retain executable VM layout: {error:?}"
                        ))
                        .with_identifier("RunMat:ExecutableLayout")
                        .build(),
                    )
                },
            )?);
        }
        source_table
            .validate_compiled_sources(&bytecode)
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableSourceIdentity")
                        .build(),
                )
            })?;
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
        let (
            function_registry_after_success,
            next_semantic_function_id_after_success,
            function_id_remap,
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
            analysis = runmat_mir::analysis::analyze_assembly(&mir);
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
        Ok(PreparedExecution {
            ast,
            lowering,
            mir,
            analysis,
            bytecode,
            portable_source_map: Some(source_table.source_map),
            function_registry_after_success,
            current_function_ids: function_id_remap.values().copied().collect(),
            next_semantic_function_id_after_success,
        })
    }

    pub(crate) fn populate_callstack(
        &self,
        error: &mut RuntimeError,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) {
        if !error.context.call_stack.is_empty() || error.context.call_frames.is_empty() {
            return;
        }
        let mut rendered = Vec::new();
        if error.context.call_frames_elided > 0 {
            rendered.push(format!(
                "... {} frames elided ...",
                error.context.call_frames_elided
            ));
        }
        for frame in error.context.call_frames.iter().rev() {
            let mut line = frame.function.clone();
            if let (Some(source_id), Some((start, _end))) = (frame.source_id, frame.span) {
                if let Some(source) = runmat_runtime::source_context::source_info_in_context(
                    runtime,
                    SourceId(source_id),
                ) {
                    let (line_num, col) = line_col_from_offset(&source.text, start);
                    line = format!("{} @ {}:{}:{}", frame.function, source.name, line_num, col);
                }
            }
            rendered.push(line);
        }
        error.context.call_stack = rendered;
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
            bind_semantic_function_references(bytecode);
            return Ok((session_registry, next_semantic_function_id, HashMap::new()));
        }

        let reused = super::semantic_definition::unchanged_unit_remap(
            bytecode,
            &current_registry,
            &execution_registry,
            private_companion_function_names,
        );
        let reuse_definitions = reused.is_some();
        let mut remap = reused.unwrap_or_default();
        if !reuse_definitions {
            let mut ids: Vec<_> = current_registry.functions.keys().copied().collect();
            ids.sort_by_key(|id| id.0);
            for old_id in ids {
                let new_id = runmat_hir::FunctionId(next_semantic_function_id);
                next_semantic_function_id += 1;
                remap.insert(old_id, new_id);
            }
        }

        for instr in &mut bytecode.instructions {
            super::semantic_definition::remap_semantic_function_instr(instr, &remap);
        }
        let name_remaps: Vec<(String, runmat_hir::FunctionId)> = current_registry
            .names
            .iter()
            .map(|(name, function)| (name.clone(), *function))
            .collect();

        let mut replaced_sources = Vec::new();
        for function in current_registry
            .functions
            .values()
            .filter(|_| !reuse_definitions)
        {
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
            session_registry.remove_source(source_id);
        }

        for (old_id, function) in current_registry.into_functions() {
            let Some(new_id) = remap.get(&old_id).copied() else {
                continue;
            };
            let mut function = function;
            let program_function = runmat_types::ProgramFunctionId(
                u32::try_from(new_id.0).expect("session function identity exceeds portable schema"),
            );
            function.rebind_identity(new_id, program_function);
            function.source_id = function.source_id.or(bytecode.source_id);
            for instr in &mut function.instructions {
                super::semantic_definition::remap_semantic_function_instr(instr, &remap);
            }
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
                        function.source_id = Some(runtime);
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
        bind_semantic_function_references(bytecode);
        Ok((session_registry, next_semantic_function_id, remap))
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
        &mut self,
        bytecode: &runmat_vm::Bytecode,
        updated_var_mapping: &HashMap<String, usize>,
        debug_trace: bool,
    ) {
        // Create a new variable array of the correct size
        let max_var_id = updated_var_mapping.values().copied().max().unwrap_or(0);
        let required_len = std::cmp::max(bytecode.var_count, max_var_id + 1);
        let mut new_variable_array = vec![Value::Num(0.0); required_len];
        if debug_trace {
            debug!(
                bytecode_var_count = bytecode.var_count,
                required_len, max_var_id, "[repl] prepare variable array"
            );
        }

        // Populate with existing values based on the variable mapping
        for (var_name, &new_var_id) in updated_var_mapping {
            if new_var_id < new_variable_array.len() {
                if let Some(value) = self.workspace_values.get(var_name) {
                    if debug_trace {
                        debug!(
                            var_name,
                            var_id = new_var_id,
                            ?value,
                            "[repl] prepare set var"
                        );
                    }
                    new_variable_array[new_var_id] = value.clone();
                }
            } else if debug_trace {
                debug!(
                    var_name,
                    var_id = new_var_id,
                    len = new_variable_array.len(),
                    "[repl] prepare skipping var"
                );
            }
        }

        // Update our variable array and mapping
        self.variable_array = new_variable_array;
    }
}

fn bind_semantic_function_references(bytecode: &mut runmat_vm::Bytecode) {
    let registry = bytecode.function_registry.clone();
    bind_semantic_callback_literals(bytecode, &registry);
    for instr in &mut bytecode.instructions {
        if let runmat_vm::Instr::CreateFunctionHandle(name) = instr {
            if let Some(function) = registry.resolve_name(name) {
                let role = registry
                    .get(function)
                    .map_or(runmat_types::FunctionSourceRole::Unknown, |definition| {
                        definition.source_role
                    });
                *instr = runmat_vm::Instr::CreateBoundFunctionHandle(function, name.clone(), role);
            }
        }
    }
}

fn bind_semantic_callback_literals(
    bytecode: &mut runmat_vm::Bytecode,
    registry: &runmat_vm::FunctionRegistry,
) {
    let mut stack: Vec<usize> = Vec::new();
    let mut replacements = Vec::new();

    for (pc, instr) in bytecode.instructions.iter().enumerate() {
        match instr {
            runmat_vm::Instr::CallBuiltinMulti(name, argc, _)
                if matches!(name.as_str(), "cellfun" | "arrayfun") && *argc > 0 =>
            {
                if stack.len() >= *argc {
                    let producer = stack[stack.len() - *argc];
                    if let Some((function, display_name)) =
                        callback_literal(bytecode.instructions.get(producer), registry)
                    {
                        replacements.push((producer, function, display_name));
                    }
                }
            }
            runmat_vm::Instr::CallFevalMulti(argc, _) => {
                let pops = *argc + 1;
                if stack.len() >= pops {
                    let producer = stack[stack.len() - pops];
                    if let Some((function, display_name)) =
                        callback_literal(bytecode.instructions.get(producer), registry)
                    {
                        replacements.push((producer, function, display_name));
                    }
                }
            }
            runmat_vm::Instr::CallFevalMultiUsingOutputSlot(argc, _) => {
                let pops = *argc + 1;
                if stack.len() >= pops {
                    let producer = stack[stack.len() - pops];
                    if let Some((function, display_name)) =
                        callback_literal(bytecode.instructions.get(producer), registry)
                    {
                        replacements.push((producer, function, display_name));
                    }
                }
            }
            runmat_vm::Instr::CallFevalExpandMultiOutput(_, _) => {
                if let Some(effect) = instr.stack_effect() {
                    if stack.len() >= effect.pops {
                        let producer = stack[stack.len() - effect.pops];
                        if let Some((function, display_name)) =
                            callback_literal(bytecode.instructions.get(producer), registry)
                        {
                            replacements.push((producer, function, display_name));
                        }
                    }
                }
            }
            runmat_vm::Instr::CallFevalExpandMultiOutputUsingOutputSlot(_, _) => {
                if let Some(effect) = instr.stack_effect() {
                    if stack.len() >= effect.pops {
                        let producer = stack[stack.len() - effect.pops];
                        if let Some((function, display_name)) =
                            callback_literal(bytecode.instructions.get(producer), registry)
                        {
                            replacements.push((producer, function, display_name));
                        }
                    }
                }
            }
            _ => {}
        }

        let Some(effect) = instr.stack_effect() else {
            stack.clear();
            continue;
        };
        if effect.pops > stack.len() {
            stack.clear();
        } else {
            for _ in 0..effect.pops {
                stack.pop();
            }
        }
        for _ in 0..effect.pushes {
            stack.push(pc);
        }
    }

    for (producer, function, display_name) in replacements {
        let role = registry
            .get(function)
            .map_or(runmat_types::FunctionSourceRole::Unknown, |definition| {
                definition.source_role
            });
        bytecode.instructions[producer] =
            runmat_vm::Instr::CreateBoundFunctionHandle(function, display_name, role);
    }
}

fn callback_literal(
    instr: Option<&runmat_vm::Instr>,
    registry: &runmat_vm::FunctionRegistry,
) -> Option<(runmat_hir::FunctionId, String)> {
    let text = match instr? {
        runmat_vm::Instr::LoadString(text) | runmat_vm::Instr::LoadCharRow(text) => text,
        _ => return None,
    };
    let name = text.trim().strip_prefix('@').unwrap_or(text.trim()).trim();
    if name.is_empty() {
        return None;
    }
    registry
        .resolve_name(name)
        .map(|function| (function, name.to_string()))
}

#[cfg(test)]
mod source_discovery_tests {
    use super::*;
    use std::fs;

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
#[path = "compile/portable_source_tests.rs"]
mod portable_source_tests;
