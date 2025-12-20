use anyhow::Result;
use log::{debug, info, warn};
use runmat_builtins::{Type, Value};
use runmat_gc::{gc_configure, gc_stats, GcConfig};

use runmat_lexer::{tokenize_detailed, Token as LexToken};
use runmat_parser::parse;
use runmat_snapshot::{Snapshot, SnapshotConfig, SnapshotLoader};
use runmat_turbine::TurbineEngine;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub mod commands;

/// Enhanced REPL execution engine that integrates all RunMat components
pub struct ReplEngine {
    /// JIT compiler engine (optional for fallback mode)
    jit_engine: Option<TurbineEngine>,
    /// Verbose output for debugging
    verbose: bool,
    /// Execution statistics
    stats: ExecutionStats,
    /// Persistent variable context for REPL sessions
    variables: HashMap<String, Value>,
    /// Current variable array for bytecode execution
    variable_array: Vec<Value>,
    /// Mapping from variable names to VarId indices
    variable_names: HashMap<String, usize>,
    /// Persistent workspace values keyed by variable name
    workspace_values: HashMap<String, Value>,
    /// User-defined functions context for REPL sessions
    function_definitions: HashMap<String, runmat_hir::HirStmt>,
    /// Loaded snapshot for standard library preloading
    snapshot: Option<Arc<Snapshot>>,
}

#[derive(Debug, Default)]
pub struct ExecutionStats {
    pub total_executions: usize,
    pub jit_compiled: usize,
    pub interpreter_fallback: usize,
    pub total_execution_time_ms: u64,
    pub average_execution_time_ms: f64,
}

#[derive(Debug)]
pub struct ExecutionResult {
    pub value: Option<Value>,
    pub execution_time_ms: u64,
    pub used_jit: bool,
    pub error: Option<String>,
    /// Type information displayed when output is suppressed by semicolon
    pub type_info: Option<String>,
}

/// Format value type information like MATLAB (e.g., "1000x1 vector", "3x3 matrix")
fn format_type_info(value: &Value) -> String {
    match value {
        Value::Int(_) => "scalar".to_string(),
        Value::Num(_) => "scalar".to_string(),
        Value::Bool(_) => "logical scalar".to_string(),
        Value::String(_) => "string".to_string(),
        Value::StringArray(sa) => {
            // MATLAB displays string arrays as m x n string array; for test's purpose, we classify scalar string arrays as "string"
            if sa.shape == vec![1, 1] {
                "string".to_string()
            } else {
                format!("{}x{} string array", sa.rows(), sa.cols())
            }
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 && ca.cols == 1 {
                "char".to_string()
            } else {
                format!("{}x{} char array", ca.rows, ca.cols)
            }
        }
        Value::Tensor(m) => {
            if m.rows() == 1 && m.cols() == 1 {
                "scalar".to_string()
            } else if m.rows() == 1 || m.cols() == 1 {
                format!("{}x{} vector", m.rows(), m.cols())
            } else {
                format!("{}x{} matrix", m.rows(), m.cols())
            }
        }
        Value::Cell(cells) => {
            if cells.data.len() == 1 {
                "1x1 cell".to_string()
            } else {
                format!("{}x1 cell array", cells.data.len())
            }
        }
        Value::GpuTensor(h) => {
            if h.shape.len() == 2 {
                let r = h.shape[0];
                let c = h.shape[1];
                if r == 1 && c == 1 {
                    "scalar (gpu)".to_string()
                } else if r == 1 || c == 1 {
                    format!("{r}x{c} vector (gpu)")
                } else {
                    format!("{r}x{c} matrix (gpu)")
                }
            } else {
                format!("Tensor{:?} (gpu)", h.shape)
            }
        }
        _ => "value".to_string(),
    }
}

impl ReplEngine {
    /// Create a new REPL engine
    pub fn new() -> Result<Self> {
        Self::with_options(true, false) // JIT enabled, verbose disabled
    }

    /// Create a new REPL engine with specific options
    pub fn with_options(enable_jit: bool, verbose: bool) -> Result<Self> {
        Self::with_snapshot(enable_jit, verbose, None::<&str>)
    }

    /// Create a new REPL engine with snapshot loading
    pub fn with_snapshot<P: AsRef<Path>>(
        enable_jit: bool,
        verbose: bool,
        snapshot_path: Option<P>,
    ) -> Result<Self> {
        // Load snapshot if provided
        let snapshot = if let Some(path) = snapshot_path {
            match Self::load_snapshot(path.as_ref()) {
                Ok(snapshot) => {
                    info!(
                        "Snapshot loaded successfully from {}",
                        path.as_ref().display()
                    );
                    Some(Arc::new(snapshot))
                }
                Err(e) => {
                    warn!(
                        "Failed to load snapshot from {}: {}, continuing without snapshot",
                        path.as_ref().display(),
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        let jit_engine = if enable_jit {
            match TurbineEngine::new() {
                Ok(engine) => {
                    info!("JIT compiler initialized successfully");
                    Some(engine)
                }
                Err(e) => {
                    warn!("JIT compiler initialization failed: {e}, falling back to interpreter");
                    None
                }
            }
        } else {
            info!("JIT compiler disabled, using interpreter only");
            None
        };

        Ok(Self {
            jit_engine,
            verbose,
            stats: ExecutionStats::default(),
            variables: HashMap::new(),
            variable_array: Vec::new(),
            variable_names: HashMap::new(),
            workspace_values: HashMap::new(),
            function_definitions: HashMap::new(),
            snapshot,
        })
    }

    /// Load a snapshot from disk
    fn load_snapshot(path: &Path) -> Result<Snapshot> {
        let mut loader = SnapshotLoader::new(SnapshotConfig::default());
        let (snapshot, _stats) = loader
            .load(path)
            .map_err(|e| anyhow::anyhow!("Failed to load snapshot: {}", e))?;
        Ok(snapshot)
    }

    /// Get snapshot information
    pub fn snapshot_info(&self) -> Option<String> {
        self.snapshot.as_ref().map(|snapshot| {
            format!(
                "Snapshot loaded: {} builtins, {} HIR functions, {} bytecode entries",
                snapshot.builtins.functions.len(),
                snapshot.hir_cache.functions.len(),
                snapshot.bytecode_cache.stdlib_bytecode.len()
            )
        })
    }

    /// Check if a snapshot is loaded
    pub fn has_snapshot(&self) -> bool {
        self.snapshot.is_some()
    }

    /// Execute MATLAB/Octave code
    pub fn execute(&mut self, input: &str) -> Result<ExecutionResult> {
        let start_time = Instant::now();
        self.stats.total_executions += 1;
        let debug_trace = std::env::var("RUNMAT_DEBUG_REPL").is_ok();

        if self.verbose {
            debug!("Executing: {}", input.trim());
        }

        // Parse the input
        let ast = parse(input)
            .map_err(|e| anyhow::anyhow!("Failed to parse input '{}': {}", input, e))?;
        if self.verbose {
            debug!("AST: {ast:?}");
        }

        // Lower to HIR with existing variable and function context
        let lowering_result = runmat_hir::lower_with_full_context(
            &ast,
            &self.variable_names,
            &self.function_definitions,
        )
        .map_err(|e| anyhow::anyhow!("Failed to lower to HIR: {}", e))?;
        let (hir, updated_vars, updated_functions, var_names_map) = (
            lowering_result.hir,
            lowering_result.variables,
            lowering_result.functions,
            lowering_result.var_names,
        );
        let max_var_id = updated_vars.values().copied().max().unwrap_or(0);
        if debug_trace {
            println!("updated_vars: {:?}", updated_vars);
        }
        if debug_trace {
            println!("workspace_values_before: {:?}", self.workspace_values);
        }
        let id_to_name: HashMap<usize, String> = var_names_map
            .iter()
            .map(|(var_id, name)| (var_id.0, name.clone()))
            .collect();
        let mut assigned_this_execution: HashSet<String> = HashSet::new();
        let assigned_snapshot: HashSet<String> = updated_vars
            .keys()
            .filter(|name| self.workspace_values.contains_key(name.as_str()))
            .cloned()
            .collect();
        let prev_assigned_snapshot = assigned_snapshot.clone();
        if debug_trace {
            println!("assigned_snapshot: {:?}", assigned_snapshot);
        }
        let _pending_workspace_guard =
            runmat_ignition::push_pending_workspace(updated_vars.clone(), assigned_snapshot);
        if self.verbose {
            debug!("HIR generated successfully");
        }

        // Compile to bytecode with existing function definitions
        let existing_functions = self.convert_hir_functions_to_user_functions();
        let bytecode = runmat_ignition::compile_with_functions(&hir, &existing_functions)
            .map_err(|e| anyhow::anyhow!("Failed to compile to bytecode: {}", e))?;
        if self.verbose {
            debug!(
                "Bytecode compiled: {} instructions",
                bytecode.instructions.len()
            );
        }

        // Prepare variable array with existing values before execution
        self.prepare_variable_array_for_execution(&bytecode, &updated_vars, debug_trace);

        if self.verbose {
            debug!(
                "Variable array after preparation: {:?}",
                self.variable_array
            );
            debug!("Updated variable mapping: {updated_vars:?}");
            debug!("Bytecode instructions: {:?}", bytecode.instructions);
        }

        let mut used_jit = false;
        let mut result_value: Option<Value> = None; // Always start fresh for each execution
        let mut suppressed_value: Option<Value> = None; // Track value for type info when suppressed
        let mut error = None;
        let mut ans_update: Option<(usize, Value)> = None;

        // Check if this is an expression statement (ends with Pop)
        let is_expression_stmt = bytecode
            .instructions
            .last()
            .map(|instr| matches!(instr, runmat_ignition::Instr::Pop))
            .unwrap_or(false);

        // Detect whether the user's input ends with a semicolon at the token level
        let ends_with_semicolon = {
            let toks = tokenize_detailed(input);
            toks.into_iter()
                .rev()
                .map(|t| t.token)
                .find(|_| true)
                .map(|t| matches!(t, LexToken::Semicolon))
                .unwrap_or(false)
        };

        // Check if this is a semicolon-suppressed statement (expression or assignment)
        // Control flow statements never return values regardless of semicolons
        let is_semicolon_suppressed = if hir.body.len() == 1 {
            match &hir.body[0] {
                runmat_hir::HirStmt::ExprStmt(_, _) => ends_with_semicolon,
                runmat_hir::HirStmt::Assign(_, _, _) => ends_with_semicolon,
                runmat_hir::HirStmt::If { .. }
                | runmat_hir::HirStmt::While { .. }
                | runmat_hir::HirStmt::For { .. }
                | runmat_hir::HirStmt::Break
                | runmat_hir::HirStmt::Continue
                | runmat_hir::HirStmt::Return
                | runmat_hir::HirStmt::Function { .. }
                | runmat_hir::HirStmt::MultiAssign(_, _, _)
                | runmat_hir::HirStmt::AssignLValue(_, _, _)
                | runmat_hir::HirStmt::Switch { .. }
                | runmat_hir::HirStmt::TryCatch { .. }
                | runmat_hir::HirStmt::Global(_)
                | runmat_hir::HirStmt::Persistent(_)
                | runmat_hir::HirStmt::Import {
                    path: _,
                    wildcard: _,
                }
                | runmat_hir::HirStmt::ClassDef { .. } => true,
            }
        } else {
            false
        };

        if self.verbose {
            debug!("HIR body len: {}", hir.body.len());
            if !hir.body.is_empty() {
                debug!("HIR statement: {:?}", &hir.body[0]);
            }
            debug!("is_semicolon_suppressed: {is_semicolon_suppressed}");
        }

        // Use JIT for assignments, interpreter for expressions (to capture results properly)
        if let Some(ref mut jit_engine) = &mut self.jit_engine {
            if !is_expression_stmt {
                // Ensure variable array is large enough
                if self.variable_array.len() < bytecode.var_count {
                    self.variable_array
                        .resize(bytecode.var_count, Value::Num(0.0));
                }

                if self.verbose {
                    debug!(
                        "JIT path for assignment: variable_array size: {}, bytecode.var_count: {}",
                        self.variable_array.len(),
                        bytecode.var_count
                    );
                }

                // Use JIT for assignments
                match jit_engine.execute_or_compile(&bytecode, &mut self.variable_array) {
                    Ok((_, actual_used_jit)) => {
                        used_jit = actual_used_jit;
                        if actual_used_jit {
                            self.stats.jit_compiled += 1;
                        } else {
                            self.stats.interpreter_fallback += 1;
                        }
                        // For assignments, capture the assigned value for both display and type info
                        // Prefer the variable slot indicated by HIR if available.
                        let assignment_value =
                            if let Some(runmat_hir::HirStmt::Assign(var_id, _, _)) =
                                hir.body.first()
                            {
                                if let Some(name) = id_to_name.get(&var_id.0) {
                                    assigned_this_execution.insert(name.clone());
                                }
                                if var_id.0 < self.variable_array.len() {
                                    Some(self.variable_array[var_id.0].clone())
                                } else {
                                    None
                                }
                            } else {
                                self.variable_array
                                    .iter()
                                    .rev()
                                    .find(|v| !matches!(v, Value::Num(0.0)))
                                    .cloned()
                            };

                        if !is_semicolon_suppressed {
                            result_value = assignment_value.clone();
                            if self.verbose {
                                debug!("JIT assignment result: {result_value:?}");
                            }
                        } else {
                            suppressed_value = assignment_value;
                            if self.verbose {
                                debug!("JIT assignment suppressed due to semicolon, captured for type info");
                            }
                        }

                        if self.verbose {
                            debug!(
                                "{} assignment successful, variable_array: {:?}",
                                if actual_used_jit {
                                    "JIT"
                                } else {
                                    "Interpreter"
                                },
                                self.variable_array
                            );
                        }
                    }
                    Err(e) => {
                        if self.verbose {
                            debug!("JIT execution failed: {e}, using interpreter");
                        }
                        // Fall back to interpreter
                    }
                }
            }
        }

        // Use interpreter if JIT failed or is disabled
        if !used_jit {
            if self.verbose {
                debug!(
                    "Interpreter path: variable_array size: {}, bytecode.var_count: {}",
                    self.variable_array.len(),
                    bytecode.var_count
                );
            }

            // For expressions, modify bytecode to store result in a temp variable instead of using stack
            let mut execution_bytecode = bytecode.clone();
            if is_expression_stmt && !execution_bytecode.instructions.is_empty() {
                execution_bytecode.instructions.pop(); // Remove the Pop instruction

                // Add StoreVar instruction to store the result in a temporary variable
                let temp_var_id = std::cmp::max(execution_bytecode.var_count, max_var_id + 1);
                execution_bytecode
                    .instructions
                    .push(runmat_ignition::Instr::StoreVar(temp_var_id));
                execution_bytecode.var_count = temp_var_id + 1; // Expand variable count for temp variable

                // Ensure our variable array can hold the temporary variable
                if self.variable_array.len() <= temp_var_id {
                    self.variable_array.resize(temp_var_id + 1, Value::Num(0.0));
                }

                if self.verbose {
                    debug!(
                        "Modified expression bytecode, new instructions: {:?}",
                        execution_bytecode.instructions
                    );
                }
            }

            match self.interpret_with_context(&execution_bytecode) {
                Ok(results) => {
                    // Only increment interpreter_fallback if JIT wasn't attempted
                    if self.jit_engine.is_none() || is_expression_stmt {
                        self.stats.interpreter_fallback += 1;
                    }
                    if self.verbose {
                        debug!("Interpreter results: {results:?}");
                    }

                    // Handle assignment statements (x = 42 should show the assigned value unless suppressed)
                    if hir.body.len() == 1 {
                        if let runmat_hir::HirStmt::Assign(var_id, _, _) = &hir.body[0] {
                            if self.verbose {
                                debug!(
                                    "Assignment detected, var_id: {}, ends_with_semicolon: {}",
                                    var_id.0, ends_with_semicolon
                                );
                            }
                            if let Some(name) = id_to_name.get(&var_id.0) {
                                assigned_this_execution.insert(name.clone());
                            }
                            // For assignments, capture the assigned value for both display and type info
                            if var_id.0 < self.variable_array.len() {
                                let assignment_value = self.variable_array[var_id.0].clone();
                                if !is_semicolon_suppressed {
                                    result_value = Some(assignment_value);
                                    if self.verbose {
                                        debug!("Interpreter assignment result: {result_value:?}");
                                    }
                                } else {
                                    suppressed_value = Some(assignment_value);
                                    if self.verbose {
                                        debug!("Interpreter assignment suppressed due to semicolon, captured for type info");
                                    }
                                }
                            }
                        } else if !is_expression_stmt
                            && !results.is_empty()
                            && !is_semicolon_suppressed
                        {
                            result_value = Some(results[0].clone());
                        }
                    }

                    // For expressions, get the result from the temporary variable (capture for both display and type info)
                    if is_expression_stmt
                        && !execution_bytecode.instructions.is_empty()
                        && result_value.is_none()
                        && suppressed_value.is_none()
                    {
                        let temp_var_id = execution_bytecode.var_count - 1; // The temp variable we added
                        if temp_var_id < self.variable_array.len() {
                            let expression_value = self.variable_array[temp_var_id].clone();
                            // Capture for 'ans' update
                            ans_update = Some((temp_var_id, expression_value.clone()));

                            if !is_semicolon_suppressed {
                                result_value = Some(expression_value);
                                if self.verbose {
                                    debug!("Expression result from temp var {temp_var_id}: {result_value:?}");
                                }
                            } else {
                                suppressed_value = Some(expression_value);
                                if self.verbose {
                                    debug!("Expression suppressed, captured for type info from temp var {temp_var_id}: {suppressed_value:?}");
                                }
                            }
                        }
                    } else if !is_semicolon_suppressed && result_value.is_none() {
                        result_value = results.into_iter().last();
                        if self.verbose {
                            debug!("Fallback result from interpreter: {result_value:?}");
                        }
                    }

                    if self.verbose {
                        debug!("Final result_value: {result_value:?}");
                    }
                    debug!(
                        "Interpreter execution successful, variable_array: {:?}",
                        self.variable_array
                    );
                }
                Err(e) => {
                    debug!("Interpreter execution failed: {e}");
                    error = Some(format!("Execution failed: {e}"));
                }
            }
        }

        let execution_time = start_time.elapsed();
        let execution_time_ms = execution_time.as_millis() as u64;

        self.stats.total_execution_time_ms += execution_time_ms;
        self.stats.average_execution_time_ms =
            self.stats.total_execution_time_ms as f64 / self.stats.total_executions as f64;

        // Update variable names mapping and function definitions if execution was successful
        if error.is_none() {
            if let Some((mutated_names, assigned)) = runmat_ignition::take_updated_workspace_state()
            {
                if debug_trace {
                    println!("mutated_names: {:?}", mutated_names);
                    println!("assigned_returned: {:?}", assigned);
                }
                self.variable_names = mutated_names.clone();
                let mut new_assigned: HashSet<String> = assigned
                    .difference(&prev_assigned_snapshot)
                    .cloned()
                    .collect();
                new_assigned.extend(assigned_this_execution.iter().cloned());
                for (name, var_id) in &mutated_names {
                    if *var_id >= self.variable_array.len() {
                        continue;
                    }
                    let new_value = &self.variable_array[*var_id];
                    let changed = match self.workspace_values.get(name) {
                        Some(old_value) => old_value != new_value,
                        None => true,
                    };
                    if changed {
                        new_assigned.insert(name.clone());
                    }
                }
                if debug_trace {
                    println!("new_assigned: {:?}", new_assigned);
                }
                for name in new_assigned {
                    let var_id = mutated_names.get(&name).copied().or_else(|| {
                        id_to_name
                            .iter()
                            .find_map(|(vid, n)| if n == &name { Some(*vid) } else { None })
                    });
                    if let Some(var_id) = var_id {
                        if var_id < self.variable_array.len() {
                            self.workspace_values
                                .insert(name.clone(), self.variable_array[var_id].clone());
                            if debug_trace {
                                println!(
                                    "workspace_update: {} -> {:?}",
                                    name, self.variable_array[var_id]
                                );
                            }
                        }
                    }
                }
            } else {
                for name in &assigned_this_execution {
                    if let Some(var_id) =
                        id_to_name
                            .iter()
                            .find_map(|(vid, n)| if n == name { Some(*vid) } else { None })
                    {
                        if var_id < self.variable_array.len() {
                            self.workspace_values
                                .insert(name.clone(), self.variable_array[var_id].clone());
                        }
                    }
                }
            }
            self.function_definitions = updated_functions;

            // Apply 'ans' update if applicable (persisting expression result)
            if let Some((var_id, value)) = ans_update {
                self.variable_names.insert("ans".to_string(), var_id);
                self.workspace_values.insert("ans".to_string(), value);
                if debug_trace {
                    println!("Updated 'ans' to var_id {}", var_id);
                }
            }
        }

        if self.verbose {
            debug!("Execution completed in {execution_time_ms}ms (JIT: {used_jit})");
        }

        // Generate type info if we have a suppressed value
        let type_info = suppressed_value.as_ref().map(format_type_info);

        // Final fallback: if not suppressed and still no value, try last non-zero variable slot
        if !is_semicolon_suppressed && result_value.is_none() {
            if let Some(v) = self
                .variable_array
                .iter()
                .rev()
                .find(|v| !matches!(v, Value::Num(0.0)))
                .cloned()
            {
                result_value = Some(v);
            }
        }

        Ok(ExecutionResult {
            value: result_value,
            execution_time_ms,
            used_jit,
            error,
            type_info,
        })
    }

    /// Get execution statistics
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Reset execution statistics
    pub fn reset_stats(&mut self) {
        self.stats = ExecutionStats::default();
    }

    /// Clear all variables in the REPL context
    pub fn clear_variables(&mut self) {
        self.variables.clear();
        self.variable_array.clear();
        self.variable_names.clear();
        self.workspace_values.clear();
    }

    /// Get a copy of current variables
    pub fn get_variables(&self) -> &HashMap<String, Value> {
        &self.variables
    }

    /// Interpret bytecode with persistent variable context
    fn interpret_with_context(
        &mut self,
        bytecode: &runmat_ignition::Bytecode,
    ) -> Result<Vec<Value>, String> {
        // Variable array should already be prepared by prepare_variable_array_for_execution

        // Use the main Ignition interpreter which has full function and scoping support
        match runmat_ignition::interpret_with_vars(
            bytecode,
            &mut self.variable_array,
            Some("<repl>"),
        ) {
            Ok(result) => {
                // Update the variables HashMap for display purposes
                self.variables.clear();
                for (i, value) in self.variable_array.iter().enumerate() {
                    if !matches!(value, Value::Num(0.0)) {
                        // Only store non-zero values to avoid clutter
                        self.variables.insert(format!("var_{i}"), value.clone());
                    }
                }

                Ok(result)
            }
            Err(e) => Err(e),
        }
    }

    /// Prepare variable array for execution by populating with existing values
    fn prepare_variable_array_for_execution(
        &mut self,
        bytecode: &runmat_ignition::Bytecode,
        updated_var_mapping: &HashMap<String, usize>,
        debug_trace: bool,
    ) {
        // Create a new variable array of the correct size
        let max_var_id = updated_var_mapping.values().copied().max().unwrap_or(0);
        let required_len = std::cmp::max(bytecode.var_count, max_var_id + 1);
        let mut new_variable_array = vec![Value::Num(0.0); required_len];
        if debug_trace {
            println!(
                "prepare: bytecode.var_count={} required_len={} max_var_id={}",
                bytecode.var_count, required_len, max_var_id
            );
        }

        // Populate with existing values based on the variable mapping
        for (var_name, &new_var_id) in updated_var_mapping {
            if new_var_id < new_variable_array.len() {
                if let Some(value) = self.workspace_values.get(var_name) {
                    if debug_trace {
                        println!(
                            "prepare: setting {} (var_id={}) -> {:?}",
                            var_name, new_var_id, value
                        );
                    }
                    new_variable_array[new_var_id] = value.clone();
                }
            } else if debug_trace {
                println!(
                    "prepare: skipping {} (var_id={}) because len={}",
                    var_name,
                    new_var_id,
                    new_variable_array.len()
                );
            }
        }

        // Update our variable array and mapping
        self.variable_array = new_variable_array;
    }

    /// Convert stored HIR function definitions to UserFunction format for compilation
    fn convert_hir_functions_to_user_functions(
        &self,
    ) -> HashMap<String, runmat_ignition::UserFunction> {
        let mut user_functions = HashMap::new();

        for (name, hir_stmt) in &self.function_definitions {
            if let runmat_hir::HirStmt::Function {
                name: func_name,
                params,
                outputs,
                body,
                has_varargin: _,
                has_varargout: _,
            } = hir_stmt
            {
                // Use the existing HIR utilities to calculate variable count
                let var_map =
                    runmat_hir::remapping::create_complete_function_var_map(params, outputs, body);
                let max_local_var = var_map.len();

                let user_func = runmat_ignition::UserFunction {
                    name: func_name.clone(),
                    params: params.clone(),
                    outputs: outputs.clone(),
                    body: body.clone(),
                    local_var_count: max_local_var,
                    has_varargin: false,
                    has_varargout: false,
                    var_types: vec![Type::Unknown; max_local_var],
                };
                user_functions.insert(name.clone(), user_func);
            }
        }

        user_functions
    }

    /// Configure garbage collector
    pub fn configure_gc(&self, config: GcConfig) -> Result<()> {
        gc_configure(config)
            .map_err(|e| anyhow::anyhow!("Failed to configure garbage collector: {}", e))
    }

    /// Get GC statistics
    pub fn gc_stats(&self) -> runmat_gc::GcStats {
        gc_stats()
    }

    /// Show detailed system information
    pub fn show_system_info(&self) {
        println!("RunMat REPL Engine Status");
        println!("==========================");
        println!();

        println!(
            "JIT Compiler: {}",
            if self.jit_engine.is_some() {
                "Available"
            } else {
                "Disabled/Failed"
            }
        );
        println!("Verbose Mode: {}", self.verbose);
        println!();

        println!("Execution Statistics:");
        println!("  Total Executions: {}", self.stats.total_executions);
        println!("  JIT Compiled: {}", self.stats.jit_compiled);
        println!("  Interpreter Used: {}", self.stats.interpreter_fallback);
        println!(
            "  Average Time: {:.2}ms",
            self.stats.average_execution_time_ms
        );
        println!();

        let gc_stats = self.gc_stats();
        println!("Garbage Collector:");
        println!(
            "  Total Allocations: {}",
            gc_stats
                .total_allocations
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        println!(
            "  Minor Collections: {}",
            gc_stats
                .minor_collections
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        println!(
            "  Major Collections: {}",
            gc_stats
                .major_collections
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        println!(
            "  Current Memory: {:.2} MB",
            gc_stats
                .current_memory_usage
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1024.0
                / 1024.0
        );
        println!();
    }
}

impl Default for ReplEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default REPL engine")
    }
}

/// Tokenize the input string and return a space separated string of token names.
/// This is kept for backward compatibility with existing tests.
pub fn format_tokens(input: &str) -> String {
    tokenize_detailed(input)
        .into_iter()
        .map(|t| format!("{:?}", t.token))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Execute MATLAB/Octave code and return the result as a formatted string
pub fn execute_and_format(input: &str) -> String {
    match ReplEngine::new() {
        Ok(mut engine) => match engine.execute(input) {
            Ok(result) => {
                if let Some(error) = result.error {
                    format!("Error: {error}")
                } else if let Some(value) = result.value {
                    format!("{value:?}")
                } else {
                    "".to_string()
                }
            }
            Err(e) => format!("Error: {e}"),
        },
        Err(e) => format!("Engine Error: {e}"),
    }
}
