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
    use runmat_config::discover_project_symbols_from;

    let cwd = match std::env::current_dir() {
        Ok(cwd) => cwd,
        Err(_) => return HashSet::new(),
    };
    let source_path = PathBuf::from(source_name);
    let start = if source_path.is_file() {
        source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.clone())
    } else if source_path.is_absolute() {
        source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.clone())
    } else if source_path.components().count() > 1 {
        let joined = cwd.join(&source_path);
        joined
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.clone())
    } else {
        cwd.clone()
    };
    let Ok(discovered) = discover_project_symbols_from(&start) else {
        return HashSet::new();
    };
    discovered
        .map(|discovered| discovered.symbols)
        .unwrap_or_default()
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
            let semantic_function_names = self.semantic_function_registry.names.clone();
            let workspace_bindings = self.lowering_workspace_bindings();
            let known_project_symbols = discover_known_project_symbols(&source_name);
            runmat_hir::lower(
                &ast,
                &LoweringContext::new(&workspace_bindings)
                    .with_semantic_functions(&semantic_function_names)
                    .with_known_project_symbols(&known_project_symbols)
                    .with_runmat_extensions_enabled(self.compat_mode.allows_runmat_extensions())
                    .with_top_level_await_enabled(self.top_level_await_enabled),
            )?
        };
        let (mut bytecode, mir) = {
            let _span = info_span!("runtime.compile.bytecode").entered();
            self.compile_semantic_bytecode(&lowering)?
        };
        bytecode.source_id = Some(source_id);
        let (semantic_function_registry_after_success, next_semantic_function_id_after_success) =
            self.prepare_session_semantic_function_registry(&mut bytecode);
        Ok(PreparedExecution {
            ast,
            lowering,
            mir,
            bytecode,
            semantic_function_registry_after_success,
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

    fn compile_semantic_bytecode(
        &self,
        lowering: &LoweringResult,
    ) -> std::result::Result<(runmat_vm::Bytecode, runmat_mir::MirAssembly), RunError> {
        let mir = runmat_mir::lowering::lower_assembly(&lowering.assembly)?;
        let Some(entrypoint) = lowering.assembly.entrypoints.first() else {
            let semantic_functions =
                runmat_vm::compile_semantic_function_registry(&lowering.assembly, &mir)?;
            let semantic_function_registry =
                runmat_vm::SemanticFunctionRegistry::new(semantic_functions.clone());
            let mut bytecode = runmat_vm::Bytecode::empty();
            bytecode.semantic_functions = semantic_functions;
            bytecode.semantic_function_registry = semantic_function_registry;
            return Ok((bytecode, mir));
        };
        Ok((
            runmat_vm::compile(&lowering.assembly, &mir, entrypoint.id)?,
            mir,
        ))
    }

    fn prepare_session_semantic_function_registry(
        &self,
        bytecode: &mut runmat_vm::Bytecode,
    ) -> (runmat_vm::SemanticFunctionRegistry, usize) {
        let mut session_registry = self.semantic_function_registry.clone();
        let mut next_semantic_function_id = self.next_semantic_function_id;
        let current_registry = bytecode.semantic_registry();
        if current_registry.functions.is_empty() {
            bytecode.semantic_function_registry = session_registry.clone();
            bytecode.semantic_functions = bytecode.semantic_function_registry.functions.clone();
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

        bytecode.semantic_function_registry = session_registry.clone();
        bytecode.semantic_functions = bytecode.semantic_function_registry.functions.clone();
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
        let analysis = runmat_mir::analysis::analyze_assembly(&prepared.mir);
        Ok(build_fusion_snapshot(
            prepared.bytecode.accel_graph.as_ref(),
            &prepared.bytecode.fusion_groups,
            &prepared
                .bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_groups,
            Some(FusionPlannerMetadata {
                source: "semantic-mir-analysis+bytecode-accel-graph".to_string(),
                mir_local_fact_count: mir_local_fact_count_for_entrypoint(
                    &analysis,
                    &prepared.lowering.assembly,
                ),
                mir_diagnostic_count: analysis.diagnostics.len(),
                mir_fusion_signal_count: prepared
                    .bytecode
                    .semantic_fusion_metadata
                    .mir_fusion_signal_count,
                mir_fusion_candidate_group_count: prepared
                    .bytecode
                    .semantic_fusion_metadata
                    .mir_fusion_candidate_group_count,
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
        | runmat_vm::Instr::CreateSemanticFunctionHandle(function, _)
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
    let registry = bytecode.semantic_function_registry.clone();
    bind_semantic_callback_literals(bytecode, &registry);
    for instr in &mut bytecode.instructions {
        match instr {
            runmat_vm::Instr::CreateFunctionHandle(name) => {
                if let Some(function) = registry.resolve_name(name) {
                    *instr = runmat_vm::Instr::CreateSemanticFunctionHandle(function, name.clone());
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
    registry: &runmat_vm::SemanticFunctionRegistry,
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
                        semantic_callback_literal(bytecode.instructions.get(producer), registry)
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
                        semantic_callback_literal(bytecode.instructions.get(producer), registry)
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
                            semantic_callback_literal(bytecode.instructions.get(producer), registry)
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
            runmat_vm::Instr::CreateSemanticFunctionHandle(function, display_name);
    }
}

fn semantic_callback_literal(
    instr: Option<&runmat_vm::Instr>,
    registry: &runmat_vm::SemanticFunctionRegistry,
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
    registry: &runmat_vm::SemanticFunctionRegistry,
) {
    for expr in exprs.iter_mut().flatten() {
        bind_semantic_function_end_expr(expr, registry);
    }
}

fn bind_semantic_function_end_expr(
    expr: &mut runmat_vm::EndExpr,
    registry: &runmat_vm::SemanticFunctionRegistry,
) {
    match expr {
        runmat_vm::EndExpr::ResolvedCall { identity, args, .. } => {
            if let runmat_hir::CallableIdentity::DynamicName(name) = identity {
                let dynamic_name = name.0.clone();
                if let Some(function) = registry.resolve_name(&dynamic_name) {
                    *identity = runmat_hir::CallableIdentity::SemanticFunction(function);
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
                runmat_hir::CallableIdentity::SemanticFunction(function)
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
