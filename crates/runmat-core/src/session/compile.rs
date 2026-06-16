use super::*;
use crate::fusion::FusionPlannerMetadata;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

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
    let Some(entrypoint_target) = entrypoint_target_function(assembly) else {
        return analysis.mir_locals.len();
    };
    analysis
        .mir_locals
        .keys()
        .filter(|key| key.function == entrypoint_target)
        .count()
}

fn discover_known_project_symbols(source_name: &str) -> HashSet<String> {
    use runmat_config::project::discover_known_project_symbols_from_source_name;

    let source_path = PathBuf::from(source_name);
    let cwd = if source_path.is_absolute() {
        source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    } else {
        runmat_filesystem::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    };
    discover_known_project_symbols_from_source_name(Some(source_name), &cwd)
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
    let source_path = PathBuf::from(source_name);
    if source_path.is_absolute() {
        source_path
    } else {
        cwd.join(source_path)
    }
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
    source: &runmat_config::project::ProjectSourceFile,
) -> Option<&str> {
    if source.is_private {
        return None;
    }
    (source.package_path.is_some() || source.class_name.is_some())
        .then_some(source.qualified_name.as_str())
        .filter(|name| name.contains('.'))
}

fn source_index_qualified_class_name(
    source: &runmat_config::project::ProjectSourceFile,
) -> Option<&str> {
    source.package_path.as_ref().and_then(|_| {
        source
            .qualified_name
            .contains('.')
            .then_some(source.qualified_name.as_str())
    })
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

fn owner_scope_from_path_skipping_private(source_path: &Path, root_dir: &Path) -> Option<String> {
    let relative = source_path.strip_prefix(root_dir).ok()?;
    let parent = relative.parent()?;
    let mut segments = Vec::new();
    for component in parent.components() {
        let segment = component.as_os_str().to_str()?;
        if segment.eq_ignore_ascii_case("private") {
            continue;
        }
        if let Some(pkg) = segment.strip_prefix('+') {
            if pkg.is_empty() {
                return None;
            }
            segments.push(pkg.to_string());
        } else if let Some(class) = segment.strip_prefix('@') {
            if class.is_empty() {
                return None;
            }
            segments.push(class.to_string());
        } else {
            segments.push(segment.to_string());
        }
    }
    Some(segments.join("."))
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
    pub statements: Vec<runmat_parser::Stmt>,
    pub private_function_names: HashSet<String>,
    pub private_function_owners: HashMap<String, String>,
    pub private_function_aliases: HashMap<String, HashMap<String, String>>,
    pub function_source_contexts: HashMap<String, (String, String)>,
    private_statement_flags: Vec<bool>,
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

fn source_context_function_names_in_statements(stmts: &[runmat_parser::Stmt]) -> Vec<String> {
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
        source_context: Option<(String, String)>,
    ) {
        if let Some((source_name, source_text)) = source_context {
            for function_name in source_context_function_names_in_statements(&body) {
                self.function_source_contexts
                    .insert(function_name, (source_name.clone(), source_text.clone()));
            }
        }
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
            self.statements.push(stmt);
            self.private_statement_flags.push(is_private);
        }
    }

    fn apply_function_precedence(&mut self, primary_function_names: &HashSet<String>) {
        let discovered_private_function_names: HashSet<String> = self
            .statements
            .iter()
            .zip(self.private_statement_flags.iter())
            .filter_map(|(stmt, is_private)| {
                if !*is_private {
                    return None;
                }
                if let runmat_parser::Stmt::Function { name, .. } = stmt {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        let old_statements = std::mem::take(&mut self.statements);
        let old_private_flags = std::mem::take(&mut self.private_statement_flags);
        self.private_function_names.clear();
        self.private_function_owners.clear();
        self.private_function_aliases.clear();

        for (stmt, is_private) in old_statements.into_iter().zip(old_private_flags) {
            let keep = match &stmt {
                runmat_parser::Stmt::Function { name, .. } => {
                    !primary_function_names.contains(name)
                        && (is_private || !discovered_private_function_names.contains(name))
                }
                _ => true,
            };
            if !keep {
                if let runmat_parser::Stmt::Function { name, .. } = &stmt {
                    self.function_source_contexts.remove(name);
                }
                continue;
            }
            if is_private {
                if let runmat_parser::Stmt::Function { name, .. } = &stmt {
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
            self.statements.push(stmt);
            self.private_statement_flags.push(is_private);
        }
    }
}

async fn discover_companion_from_composition_graph_async(
    source_name: &str,
    cwd: &Path,
    primary_source_path: &Path,
    compat_mode: runmat_parser::CompatMode,
) -> CompanionSourceDiscovery {
    use runmat_config::project::{
        build_project_composition_graph_async, discover_project_symbols_from_source_name_async,
    };
    let options = ParserOptions::new(compat_mode);
    let mut out = CompanionSourceDiscovery::default();

    if let Ok(Some(discovered_symbols)) =
        discover_project_symbols_from_source_name_async(source_name, cwd).await
    {
        if let Ok(composition) =
            build_project_composition_graph_async(&discovered_symbols.manifest_path).await
        {
            for package in composition.packages.values() {
                for source in &package.source_index.files {
                    let file_path = package
                        .project_root
                        .join(&source.source_root)
                        .join(&source.relative_path);
                    if source_paths_equivalent(&file_path, primary_source_path) {
                        continue;
                    }
                    let Ok(contents) = runmat_filesystem::read_to_string_async(&file_path).await
                    else {
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
                    let primary_visible_private = source.is_private
                        && private_source_visible_to(primary_source_path, &file_path);
                    let private_aliases = if let Some(owner_scope) = private_owner_scope.as_deref()
                    {
                        qualify_private_companion_functions(
                            &mut body,
                            owner_scope,
                            primary_visible_private,
                        )
                    } else {
                        HashMap::new()
                    };
                    if is_class_source {
                        if let Some(qualified) = source_index_qualified_class_name(source) {
                            qualify_companion_classdefs(&mut body, qualified);
                        } else if let Some(qualified) =
                            package_class_name_from_path(&file_path, cwd)
                        {
                            qualify_companion_classdefs(&mut body, &qualified);
                        }
                    } else if private_owner_scope.is_none() {
                        if let Some(qualified) = source_index_qualified_function_name(source) {
                            qualify_companion_functions(&mut body, qualified);
                        } else if let Some(qualified) =
                            package_class_name_from_path(&file_path, cwd)
                        {
                            qualify_companion_functions(&mut body, &qualified);
                        }
                    }
                    out.extend_body(
                        body,
                        private_owner_scope.as_deref(),
                        private_aliases,
                        Some((file_path.to_string_lossy().to_string(), contents)),
                    );
                }
            }
        }
    }

    out
}

pub(super) async fn discover_companion_source_statements_async(
    source_name: &str,
    compat_mode: runmat_parser::CompatMode,
) -> CompanionSourceDiscovery {
    let Some(cwd) = source_lookup_cwd(source_name) else {
        return CompanionSourceDiscovery::default();
    };
    let primary_source_path = resolved_source_path(source_name, &cwd);
    let Some(parent) = primary_source_path.parent() else {
        return CompanionSourceDiscovery::default();
    };
    let mut out = discover_companion_from_composition_graph_async(
        source_name,
        &cwd,
        &primary_source_path,
        compat_mode,
    )
    .await;
    if !out.statements.is_empty() {
        return out;
    }
    let options = ParserOptions::new(compat_mode);
    let mut stack = vec![parent.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = runmat_filesystem::read_dir_async(&dir).await else {
            continue;
        };
        for entry in entries {
            let path = entry.path().to_path_buf();
            if source_paths_equivalent(&path, &primary_source_path) {
                continue;
            }
            if entry.is_dir() {
                let is_package_dir = entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with('+'));
                let is_class_dir = entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with('@'));
                if is_package_dir || is_class_dir || is_private_dir(&path) {
                    stack.push(path);
                }
                continue;
            }
            if !path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("m"))
            {
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
            let private_owner_scope = private_parent_dir_for_source(&path)
                .and_then(|_| owner_scope_from_path_skipping_private(&path, parent));
            let primary_visible_private = private_owner_scope.is_some()
                && private_source_visible_to(&primary_source_path, &path);
            let private_aliases = if let Some(owner_scope) = private_owner_scope.as_deref() {
                qualify_private_companion_functions(&mut body, owner_scope, primary_visible_private)
            } else {
                HashMap::new()
            };
            if is_class_source {
                if let Some(qualified) = package_class_name_from_path(&path, parent) {
                    qualify_companion_classdefs(&mut body, &qualified);
                }
            } else if private_owner_scope.is_none() {
                if let Some(qualified) = package_class_name_from_path(&path, parent) {
                    qualify_companion_functions(&mut body, &qualified);
                }
            }
            out.extend_body(
                body,
                private_owner_scope.as_deref(),
                private_aliases,
                Some((path.to_string_lossy().to_string(), contents)),
            );
        }
    }
    out
}

impl RunMatSession {
    #[cfg(test)]
    pub(crate) fn compile_input_for_source_name(
        &mut self,
        source_name: &str,
        input: &str,
    ) -> std::result::Result<PreparedExecution, RunError> {
        let previous_source_name = self.active_source_name.clone();
        self.active_source_name = source_name.to_string();
        let result = self.compile_input(input);
        self.active_source_name = previous_source_name;
        result
    }

    pub(crate) fn compile_input(
        &mut self,
        input: &str,
    ) -> std::result::Result<PreparedExecution, RunError> {
        let source_name = self.current_source_name().to_string();
        let source_id = self.source_pool.intern(&source_name, input);
        let (
            ast,
            private_companion_function_names,
            private_companion_function_owners,
            private_companion_function_aliases,
            companion_function_source_ids,
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
            let private_companion_function_names =
                std::mem::take(&mut companion.private_function_names);
            let private_companion_function_owners =
                std::mem::take(&mut companion.private_function_owners);
            let private_companion_function_aliases =
                std::mem::take(&mut companion.private_function_aliases);
            let companion_function_source_ids =
                std::mem::take(&mut companion.function_source_contexts)
                    .into_iter()
                    .map(|(function_name, (source_name, source_text))| {
                        (
                            function_name,
                            self.source_pool.intern(&source_name, &source_text),
                        )
                    })
                    .collect::<HashMap<_, _>>();
            if !companion.statements.is_empty() {
                ast.body.append(&mut companion.statements);
            }
            (
                ast,
                private_companion_function_names,
                private_companion_function_owners,
                private_companion_function_aliases,
                companion_function_source_ids,
            )
        };
        let lowering = {
            let _span = info_span!("runtime.lower").entered();
            let function_names = self.function_registry.names.clone();
            let function_output_arities = function_output_arities(&self.function_registry);
            let workspace_bindings = self.lowering_workspace_bindings();
            let known_project_symbols = discover_known_project_symbols(&source_name);
            runmat_hir::lower(
                &ast,
                &LoweringContext::new(&workspace_bindings)
                    .with_bound_functions(&function_names)
                    .with_function_output_arities(&function_output_arities)
                    .with_known_project_symbols(&known_project_symbols)
                    .with_private_functions(
                        &private_companion_function_owners,
                        &private_companion_function_aliases,
                    )
                    .with_runmat_extensions_enabled(self.compat_mode.allows_runmat_extensions())
                    .with_top_level_await_enabled(self.top_level_await_enabled),
            )?
        };
        let mir = {
            let _span = info_span!("runtime.compile.mir").entered();
            runmat_mir::lowering::lower_assembly(&lowering.assembly)?
        };
        let analysis = {
            let _span = info_span!("runtime.analyze").entered();
            runmat_mir::analysis::analyze_assembly(&mir)
        };
        let mut bytecode = {
            let _span = info_span!("runtime.compile.bytecode").entered();
            self.compile_semantic_bytecode_from_mir(&lowering.assembly, &mir)?
        };
        bytecode.source_id = Some(source_id);
        for function in bytecode.function_registry.functions.values_mut() {
            function.source_id = companion_function_source_ids
                .get(&function.display_name)
                .copied()
                .or(Some(source_id));
        }
        bytecode.bound_functions = bytecode.function_registry.functions.clone();
        let (function_registry_after_success, next_semantic_function_id_after_success) = self
            .prepare_session_semantic_function_registry(
                &mut bytecode,
                &private_companion_function_names,
            );
        Ok(PreparedExecution {
            ast,
            lowering,
            analysis,
            bytecode,
            function_registry_after_success,
            next_semantic_function_id_after_success,
        })
    }

    pub(crate) fn populate_callstack(&self, error: &mut RuntimeError) {
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
                if let Some(source) = self.source_pool.get(SourceId(source_id)) {
                    let (line_num, col) = line_col_from_offset(&source.text, start);
                    line = format!("{} @ {}:{}:{}", frame.function, source.name, line_num, col);
                }
            }
            rendered.push(line);
        }
        error.context.call_stack = rendered;
    }

    fn compile_semantic_bytecode_from_mir(
        &self,
        assembly: &runmat_hir::HirAssembly,
        mir: &runmat_mir::MirAssembly,
    ) -> std::result::Result<runmat_vm::Bytecode, RunError> {
        let Some(entrypoint) = assembly.entrypoints.first() else {
            let bound_functions = runmat_vm::compile_semantic_function_registry(assembly, mir)?;
            let function_registry = runmat_vm::FunctionRegistry::new(bound_functions.clone());
            let mut bytecode = runmat_vm::Bytecode::empty();
            bytecode.bound_functions = bound_functions;
            bytecode.function_registry = function_registry;
            return Ok(bytecode);
        };
        Ok(runmat_vm::compile(assembly, mir, entrypoint.id)?)
    }

    fn prepare_session_semantic_function_registry(
        &self,
        bytecode: &mut runmat_vm::Bytecode,
        private_companion_function_names: &HashSet<String>,
    ) -> (runmat_vm::FunctionRegistry, usize) {
        let mut session_registry = self.function_registry.clone();
        let mut execution_registry = session_registry.clone();
        let mut next_semantic_function_id = self.next_semantic_function_id;
        let current_registry = bytecode.function_registry();
        if current_registry.functions.is_empty() {
            bytecode.function_registry = session_registry.clone();
            bytecode.bound_functions = bytecode.function_registry.functions.clone();
            bind_semantic_function_references(bytecode);
            return (session_registry, next_semantic_function_id);
        }

        let mut remap = HashMap::new();
        let mut ids: Vec<_> = current_registry.functions.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        for old_id in ids {
            let new_id = runmat_hir::FunctionId(next_semantic_function_id);
            next_semantic_function_id += 1;
            remap.insert(old_id, new_id);
        }

        for instr in &mut bytecode.instructions {
            remap_semantic_function_instr(instr, &remap);
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
                if let Some(source_id) = session_registry
                    .get(existing_id)
                    .and_then(|existing| existing.source_id)
                {
                    if !replaced_sources.contains(&source_id) {
                        replaced_sources.push(source_id);
                    }
                }
            }
        }
        for source_id in replaced_sources {
            session_registry.remove_source(source_id);
        }

        for (old_id, function) in current_registry.functions {
            let Some(new_id) = remap.get(&old_id).copied() else {
                continue;
            };
            let mut function = function;
            function.function = new_id;
            function.source_id = function.source_id.or(bytecode.source_id);
            for instr in &mut function.instructions {
                remap_semantic_function_instr(instr, &remap);
            }
            let persist_function =
                !private_companion_function_names.contains(&function.display_name);
            execution_registry.insert_replacing_name(function.clone());
            if persist_function {
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
        (session_registry, next_semantic_function_id)
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

fn remap_semantic_function_instr(
    instr: &mut runmat_vm::Instr,
    remap: &HashMap<runmat_hir::FunctionId, runmat_hir::FunctionId>,
) {
    match instr {
        runmat_vm::Instr::CreateSemanticClosure(function, _, _)
        | runmat_vm::Instr::CreateBoundFunctionHandle(function, _)
        | runmat_vm::Instr::CreateSemanticFuture(function, _, _)
        | runmat_vm::Instr::CreateSemanticFutureExpandMultiOutput(function, _, _)
        | runmat_vm::Instr::CallSemanticFunctionMulti(function, _, _)
        | runmat_vm::Instr::CallSemanticFunctionMultiUsingOutputSlot(function, _, _)
        | runmat_vm::Instr::CallSemanticFunctionExpandMultiOutput(function, _, _) => {
            if let Some(new_id) = remap.get(function).copied() {
                *function = new_id;
            }
        }
        runmat_vm::Instr::CallSemanticNestedFunctionMulti { function, .. }
        | runmat_vm::Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { function, .. }
        | runmat_vm::Instr::CallSemanticNestedFunctionExpandMultiOutput { function, .. } => {
            if let Some(new_id) = remap.get(function).copied() {
                *function = new_id;
            }
        }
        runmat_vm::Instr::IndexSliceExpr {
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            end_numeric_exprs,
            ..
        }
        | runmat_vm::Instr::StoreSliceExpr {
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            end_numeric_exprs,
            ..
        } => {
            remap_optional_end_exprs(range_start_exprs, remap);
            remap_optional_end_exprs(range_step_exprs, remap);
            for expr in range_end_exprs {
                remap_semantic_function_end_expr(expr, remap);
            }
            for (_, expr) in end_numeric_exprs {
                remap_semantic_function_end_expr(expr, remap);
            }
        }
        _ => {}
    }
}

fn bind_semantic_function_references(bytecode: &mut runmat_vm::Bytecode) {
    let registry = bytecode.function_registry.clone();
    bind_semantic_callback_literals(bytecode, &registry);
    for instr in &mut bytecode.instructions {
        match instr {
            runmat_vm::Instr::CreateFunctionHandle(name) => {
                if let Some(function) = registry.resolve_name(name) {
                    *instr = runmat_vm::Instr::CreateBoundFunctionHandle(function, name.clone());
                }
            }
            runmat_vm::Instr::IndexSliceExpr {
                range_start_exprs,
                range_step_exprs,
                range_end_exprs,
                end_numeric_exprs,
                ..
            }
            | runmat_vm::Instr::StoreSliceExpr {
                range_start_exprs,
                range_step_exprs,
                range_end_exprs,
                end_numeric_exprs,
                ..
            } => {
                bind_optional_end_exprs(range_start_exprs, &registry);
                bind_optional_end_exprs(range_step_exprs, &registry);
                for expr in range_end_exprs {
                    bind_semantic_function_end_expr(expr, &registry);
                }
                for (_, expr) in end_numeric_exprs {
                    bind_semantic_function_end_expr(expr, &registry);
                }
            }
            _ => {}
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
        bytecode.instructions[producer] =
            runmat_vm::Instr::CreateBoundFunctionHandle(function, display_name);
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

fn bind_optional_end_exprs(
    exprs: &mut [Option<runmat_vm::EndExpr>],
    registry: &runmat_vm::FunctionRegistry,
) {
    for expr in exprs.iter_mut().flatten() {
        bind_semantic_function_end_expr(expr, registry);
    }
}

fn bind_semantic_function_end_expr(
    expr: &mut runmat_vm::EndExpr,
    registry: &runmat_vm::FunctionRegistry,
) {
    match expr {
        runmat_vm::EndExpr::ResolvedCall { identity, args, .. } => {
            if let runmat_hir::CallableIdentity::DynamicName(name) = identity {
                let dynamic_name = name.0.clone();
                if let Some(function) = registry.resolve_name(&dynamic_name) {
                    *identity = runmat_hir::CallableIdentity::BoundFunction(function);
                }
            }
            for arg in args {
                bind_semantic_function_end_expr(arg, registry);
            }
        }
        runmat_vm::EndExpr::Add(lhs, rhs)
        | runmat_vm::EndExpr::Sub(lhs, rhs)
        | runmat_vm::EndExpr::Mul(lhs, rhs)
        | runmat_vm::EndExpr::Div(lhs, rhs)
        | runmat_vm::EndExpr::LeftDiv(lhs, rhs)
        | runmat_vm::EndExpr::Pow(lhs, rhs) => {
            bind_semantic_function_end_expr(lhs, registry);
            bind_semantic_function_end_expr(rhs, registry);
        }
        runmat_vm::EndExpr::Neg(inner)
        | runmat_vm::EndExpr::Pos(inner)
        | runmat_vm::EndExpr::Floor(inner)
        | runmat_vm::EndExpr::Ceil(inner)
        | runmat_vm::EndExpr::Round(inner)
        | runmat_vm::EndExpr::Fix(inner) => bind_semantic_function_end_expr(inner, registry),
        runmat_vm::EndExpr::End | runmat_vm::EndExpr::Const(_) | runmat_vm::EndExpr::Var(_) => {}
    }
}

fn remap_optional_end_exprs(
    exprs: &mut [Option<runmat_vm::EndExpr>],
    remap: &HashMap<runmat_hir::FunctionId, runmat_hir::FunctionId>,
) {
    for expr in exprs.iter_mut().flatten() {
        remap_semantic_function_end_expr(expr, remap);
    }
}

fn remap_semantic_function_end_expr(
    expr: &mut runmat_vm::EndExpr,
    remap: &HashMap<runmat_hir::FunctionId, runmat_hir::FunctionId>,
) {
    match expr {
        runmat_vm::EndExpr::ResolvedCall { identity, args, .. } => {
            match identity {
                runmat_hir::CallableIdentity::BoundFunction(function)
                | runmat_hir::CallableIdentity::AnonymousFunction(function) => {
                    if let Some(new_id) = remap.get(function).copied() {
                        *function = new_id;
                    }
                }
                _ => {}
            }
            for arg in args {
                remap_semantic_function_end_expr(arg, remap);
            }
        }
        runmat_vm::EndExpr::Add(lhs, rhs)
        | runmat_vm::EndExpr::Sub(lhs, rhs)
        | runmat_vm::EndExpr::Mul(lhs, rhs)
        | runmat_vm::EndExpr::Div(lhs, rhs)
        | runmat_vm::EndExpr::LeftDiv(lhs, rhs)
        | runmat_vm::EndExpr::Pow(lhs, rhs) => {
            remap_semantic_function_end_expr(lhs, remap);
            remap_semantic_function_end_expr(rhs, remap);
        }
        runmat_vm::EndExpr::Neg(inner)
        | runmat_vm::EndExpr::Pos(inner)
        | runmat_vm::EndExpr::Floor(inner)
        | runmat_vm::EndExpr::Ceil(inner)
        | runmat_vm::EndExpr::Round(inner)
        | runmat_vm::EndExpr::Fix(inner) => remap_semantic_function_end_expr(inner, remap),
        runmat_vm::EndExpr::End | runmat_vm::EndExpr::Const(_) | runmat_vm::EndExpr::Var(_) => {}
    }
}
