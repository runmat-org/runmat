use super::*;
use crate::fusion::FusionPlannerMetadata;
use std::collections::HashSet;
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
    use runmat_config::discover_known_project_symbols_from_source_name;

    let source_path = PathBuf::from(source_name);
    let cwd = if source_path.is_absolute() {
        source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("/"))
    } else {
        match std::env::current_dir() {
            Ok(cwd) => cwd,
            Err(_) => return HashSet::new(),
        }
    };
    discover_known_project_symbols_from_source_name(Some(source_name), &cwd)
}

#[cfg(not(target_arch = "wasm32"))]
fn source_discovery_start_dir(
    source_name: &str,
    cwd: &std::path::Path,
) -> Option<std::path::PathBuf> {
    let source_path = PathBuf::from(source_name);
    let local_candidate = if source_path.is_absolute() {
        source_path.clone()
    } else {
        cwd.join(&source_path)
    };
    if source_name.contains(':') && !local_candidate.exists() {
        return None;
    }
    if (source_path.is_absolute() || source_path.components().count() > 1)
        && !local_candidate.exists()
    {
        return None;
    }
    if source_path.is_file() {
        return source_path.parent().map(|parent| parent.to_path_buf());
    }
    if source_path.is_absolute() {
        return source_path.parent().map(|parent| parent.to_path_buf());
    }
    if source_path.components().count() > 1 {
        return local_candidate.parent().map(|parent| parent.to_path_buf());
    }
    Some(cwd.to_path_buf())
}

#[cfg(not(target_arch = "wasm32"))]
fn discover_project_symbol_sources(source_name: &str) -> HashMap<String, std::path::PathBuf> {
    use runmat_config::{build_project_composition_graph, discover_project_manifest_from};

    let source_path = PathBuf::from(source_name);
    let cwd = if source_path.is_absolute() {
        source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("/"))
    } else {
        let Ok(cwd) = std::env::current_dir() else {
            return HashMap::new();
        };
        cwd
    };
    let Some(start_dir) = source_discovery_start_dir(source_name, &cwd) else {
        return HashMap::new();
    };
    let Some(manifest_path) = discover_project_manifest_from(&start_dir) else {
        return HashMap::new();
    };
    let Ok(composition) = build_project_composition_graph(&manifest_path) else {
        return HashMap::new();
    };
    let Some(root) = composition.packages.get(&composition.root_package) else {
        return HashMap::new();
    };
    let root_dependencies = &root.dependencies;
    let mut symbols = HashMap::new();
    for package in composition.packages.values() {
        for source in &package.source_index.files {
            if source.is_private {
                continue;
            }
            let full_path = package
                .project_root
                .join(&source.source_root)
                .join(&source.relative_path);
            if let Some(leaf) = source.qualified_name.rsplit('.').next() {
                symbols
                    .entry(leaf.to_string())
                    .or_insert_with(|| full_path.clone());
            }
            symbols.insert(source.qualified_name.clone(), full_path.clone());
            symbols.insert(
                format!("{}.{}", package.package_name, source.qualified_name),
                full_path.clone(),
            );
            for (alias, dependency_package) in root_dependencies {
                if dependency_package == &package.package_name {
                    symbols.insert(
                        format!("{alias}.{}", source.qualified_name),
                        full_path.clone(),
                    );
                }
            }
        }
    }
    symbols
}

#[cfg(not(target_arch = "wasm32"))]
fn insert_registry_function_without_replacing_name(
    registry: &mut runmat_vm::FunctionRegistry,
    function: runmat_vm::FunctionBytecode,
) {
    let function_id = function.function;
    if let Some(source_id) = function.source_id {
        let source_functions = registry.source_functions.entry(source_id).or_default();
        if !source_functions.contains(&function_id) {
            source_functions.push(function_id);
        }
    }
    registry
        .names
        .entry(function.display_name.clone())
        .or_insert(function_id);
    registry.functions.insert(function_id, function);
}

impl RunMatSession {
    pub(crate) fn compile_input(
        &mut self,
        input: &str,
    ) -> std::result::Result<PreparedExecution, RunError> {
        let source_name = self.current_source_name().to_string();
        let source_id = self.source_pool.intern(&source_name, input);
        let ast = {
            let _span = info_span!("runtime.parse").entered();
            parse_with_options(input, ParserOptions::new(self.compat_mode))?
        };
        let lowering = {
            let _span = info_span!("runtime.lower").entered();
            let function_names = self.function_registry.names.clone();
            let workspace_bindings = self.lowering_workspace_bindings();
            let known_project_symbols = discover_known_project_symbols(&source_name);
            runmat_hir::lower(
                &ast,
                &LoweringContext::new(&workspace_bindings)
                    .with_bound_functions(&function_names)
                    .with_known_project_symbols(&known_project_symbols)
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
        self.preload_project_symbol_functions_for_compile(&source_name, &lowering, &mut bytecode)?;
        let (function_registry_after_success, next_semantic_function_id_after_success) =
            self.prepare_session_semantic_function_registry(&mut bytecode);
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

    #[cfg(not(target_arch = "wasm32"))]
    fn preload_project_symbol_functions_for_compile(
        &self,
        source_name: &str,
        lowering: &LoweringResult,
        bytecode: &mut runmat_vm::Bytecode,
    ) -> std::result::Result<(), RunError> {
        let mut symbol_sources = discover_project_symbol_sources(source_name);
        let mut required_symbols = self.required_external_symbols(lowering);
        if let Some(stem) = std::path::Path::new(source_name)
            .file_stem()
            .and_then(|stem| stem.to_str())
        {
            required_symbols.remove(stem);
        }
        let source_dir = std::path::PathBuf::from(source_name)
            .canonicalize()
            .ok()
            .and_then(|path| path.parent().map(|parent| parent.to_path_buf()));
        for symbol in required_symbols {
            if symbol_sources.contains_key(&symbol) {
                continue;
            }
            if let Some(local_dir) = source_dir.as_ref() {
                let (packages, base_name) =
                    runmat_runtime::builtins::common::path_search::split_package_components(
                        &symbol,
                    );
                let mut local_candidate = local_dir.clone();
                for package in packages {
                    local_candidate.push(format!("+{package}"));
                }
                local_candidate.push(format!("{base_name}.m"));
                if local_candidate.is_file() {
                    symbol_sources.insert(symbol.clone(), local_candidate);
                    continue;
                }
            }
            let candidates = runmat_runtime::builtins::common::path_search::file_candidates(
                &symbol,
                &[".m"],
                "project symbol preload",
            )
            .unwrap_or_default();
            if let Some(path) = candidates.into_iter().find(|candidate| candidate.is_file()) {
                symbol_sources.insert(symbol, path);
            }
        }
        if symbol_sources.is_empty() {
            return Ok(());
        }

        let mut symbols_by_path: HashMap<std::path::PathBuf, Vec<String>> = HashMap::new();
        for (symbol, path) in symbol_sources {
            symbols_by_path.entry(path).or_default().push(symbol);
        }

        let current_source = std::path::PathBuf::from(source_name).canonicalize().ok();
        let known_project_symbols = discover_known_project_symbols(source_name);
        let mut next_function_id = bytecode
            .function_registry
            .functions
            .keys()
            .map(|id| id.0)
            .max()
            .unwrap_or(0)
            + 1;

        for (path, symbols) in symbols_by_path {
            let canonical_path = path.canonicalize().ok();
            if current_source.is_some() && current_source == canonical_path {
                continue;
            }

            let source_text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(_) => continue,
            };
            let ast = match parse_with_options(&source_text, ParserOptions::new(self.compat_mode)) {
                Ok(ast) => ast,
                Err(_) => continue,
            };
            let mut bound_functions = self.function_registry.names.clone();
            for (name, function_id) in &bytecode.function_registry.names {
                bound_functions.entry(name.clone()).or_insert(*function_id);
            }
            let lowering = match runmat_hir::lower(
                &ast,
                &LoweringContext::new(&HashMap::new())
                    .with_bound_functions(&bound_functions)
                    .with_known_project_symbols(&known_project_symbols)
                    .with_runmat_extensions_enabled(self.compat_mode.allows_runmat_extensions())
                    .with_top_level_await_enabled(self.top_level_await_enabled),
            ) {
                Ok(lowering) => lowering,
                Err(_) => continue,
            };
            let mir = match runmat_mir::lowering::lower_assembly(&lowering.assembly) {
                Ok(mir) => mir,
                Err(_) => continue,
            };
            let mut compiled_functions =
                match runmat_vm::compile_semantic_function_registry(&lowering.assembly, &mir) {
                    Ok(compiled) => compiled,
                    Err(_) => continue,
                };
            if compiled_functions.is_empty() {
                continue;
            }

            let mut ids: Vec<_> = compiled_functions.keys().copied().collect();
            ids.sort_by_key(|id| id.0);
            let mut remap: HashMap<runmat_hir::FunctionId, runmat_hir::FunctionId> = HashMap::new();
            let mut remapped_functions = Vec::new();
            for old_id in ids {
                let Some(mut function) = compiled_functions.remove(&old_id) else {
                    continue;
                };
                let new_id = runmat_hir::FunctionId(next_function_id);
                next_function_id += 1;
                remap.insert(old_id, new_id);
                function.function = new_id;
                remapped_functions.push(function);
            }
            for function in &mut remapped_functions {
                for instr in &mut function.instructions {
                    remap_semantic_function_instr(instr, &remap);
                }
            }

            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string);
            let fallback_primary = remapped_functions.first().map(|function| function.function);
            let preferred_primary = stem.as_ref().and_then(|file_stem| {
                remapped_functions
                    .iter()
                    .find(|function| function.display_name == *file_stem)
                    .map(|function| function.function)
            });
            let Some(primary_function) = preferred_primary.or(fallback_primary) else {
                continue;
            };

            for function in remapped_functions {
                insert_registry_function_without_replacing_name(
                    &mut bytecode.function_registry,
                    function,
                );
            }
            for symbol in symbols {
                bytecode
                    .function_registry
                    .names
                    .entry(symbol)
                    .or_insert(primary_function);
            }
        }
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    fn preload_project_symbol_functions_for_compile(
        &self,
        _source_name: &str,
        _lowering: &LoweringResult,
        _bytecode: &mut runmat_vm::Bytecode,
    ) -> std::result::Result<(), RunError> {
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn required_external_symbols(&self, lowering: &LoweringResult) -> HashSet<String> {
        let mut required = HashSet::new();
        let mut wildcard_import_prefixes = Vec::new();
        let mut specific_imports_by_leaf: HashMap<String, String> = HashMap::new();
        if let Some(module) = lowering.assembly.modules.first() {
            for import in &module.imports {
                let Some(path_display) = import.path.display_name() else {
                    continue;
                };
                if import.wildcard {
                    wildcard_import_prefixes.push(path_display);
                    continue;
                }
                if let Some(leaf) = import.path.0.last().map(|segment| segment.0.clone()) {
                    specific_imports_by_leaf.insert(leaf, path_display);
                }
            }
        }
        for call in &lowering.hir_index.calls {
            match (&call.kind, &call.callee) {
                (runmat_hir::CallKind::PackageFunction(path), _) => {
                    if let Some(name) = path.module.display_name() {
                        required.insert(name);
                    }
                }
                (
                    runmat_hir::CallKind::Dynamic,
                    runmat_hir::HirCallableRef::Unresolved(qualified),
                ) => {
                    if let Some(name) = qualified.display_name() {
                        let is_builtin =
                            runmat_builtins::builtin_function_by_name(name.as_str()).is_some();
                        if !is_builtin {
                            required.insert(name);
                            if qualified.0.len() == 1 {
                                let leaf = qualified.0[0].0.as_str();
                                if let Some(specific_target) = specific_imports_by_leaf.get(leaf) {
                                    required.insert(specific_target.clone());
                                }
                                for import_prefix in &wildcard_import_prefixes {
                                    required.insert(format!("{import_prefix}.{leaf}"));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        required
    }

    fn prepare_session_semantic_function_registry(
        &self,
        bytecode: &mut runmat_vm::Bytecode,
    ) -> (runmat_vm::FunctionRegistry, usize) {
        let mut session_registry = self.function_registry.clone();
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
            function.source_id = bytecode.source_id;
            for instr in &mut function.instructions {
                remap_semantic_function_instr(instr, &remap);
            }
            session_registry.insert_replacing_name(function);
        }
        for (name, old_id) in name_remaps {
            let Some(new_id) = remap.get(&old_id).copied() else {
                continue;
            };
            session_registry.names.insert(name, new_id);
        }

        bytecode.function_registry = session_registry.clone();
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
        | runmat_vm::Instr::CallSemanticFunctionMulti(function, _, _)
        | runmat_vm::Instr::CallSemanticFunctionExpandMultiOutput(function, _, _) => {
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
