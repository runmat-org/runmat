use anyhow::Result;
use log::{debug, info, warn};
use rustmat_builtins::Value;
use rustmat_gc::{gc_configure, gc_stats, GcConfig};
use rustmat_hir::lower_with_context;
use rustmat_ignition::compile;
use rustmat_lexer::tokenize;
use rustmat_parser::parse;
use rustmat_snapshot::{Snapshot, SnapshotConfig, SnapshotLoader};
use rustmat_turbine::TurbineEngine;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Enhanced REPL execution engine that integrates all RustMat components
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

        if self.verbose {
            debug!("Executing: {}", input.trim());
        }

        // Parse the input
        let ast = parse(input).map_err(|e| anyhow::anyhow!("Failed to parse input: {}", e))?;
        if self.verbose {
            debug!("AST: {ast:?}");
        }

        // Lower to HIR with existing variable context
        let (hir, updated_vars) = lower_with_context(&ast, &self.variable_names)
            .map_err(|e| anyhow::anyhow!("Failed to lower to HIR: {}", e))?;
        if self.verbose {
            debug!("HIR generated successfully");
        }

        // Compile to bytecode
        let bytecode =
            compile(&hir).map_err(|e| anyhow::anyhow!("Failed to compile to bytecode: {}", e))?;
        if self.verbose {
            debug!(
                "Bytecode compiled: {} instructions",
                bytecode.instructions.len()
            );
        }

        let mut used_jit = false;
        let mut result_value = None;
        let mut error = None;

        // Check if this is an expression statement (ends with Pop)
        let is_expression_stmt = bytecode
            .instructions
            .last()
            .map(|instr| matches!(instr, rustmat_ignition::Instr::Pop))
            .unwrap_or(false);

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
                        // For assignments, the result is typically the assigned value
                        result_value = self
                            .variable_array
                            .iter()
                            .rev()
                            .find(|v| !matches!(v, Value::Num(0.0)))
                            .cloned();

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
            match self.interpret_with_context(&bytecode) {
                Ok(results) => {
                    // Only increment interpreter_fallback if JIT wasn't attempted
                    if self.jit_engine.is_none() || is_expression_stmt {
                        self.stats.interpreter_fallback += 1;
                    }
                    result_value = results.into_iter().last();
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

        // Update variable names mapping if execution was successful
        if error.is_none() {
            self.variable_names = updated_vars;
        }

        if self.verbose {
            debug!("Execution completed in {execution_time_ms}ms (JIT: {used_jit})");
        }

        Ok(ExecutionResult {
            value: result_value,
            execution_time_ms,
            used_jit,
            error,
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
    }

    /// Get a copy of current variables
    pub fn get_variables(&self) -> &HashMap<String, Value> {
        &self.variables
    }

    /// Interpret bytecode with persistent variable context
    fn interpret_with_context(
        &mut self,
        bytecode: &rustmat_ignition::Bytecode,
    ) -> Result<Vec<Value>, String> {
        // Ensure our variable array is large enough
        if self.variable_array.len() < bytecode.var_count {
            self.variable_array
                .resize(bytecode.var_count, Value::Num(0.0));
        }

        // Take ownership of the variable array temporarily to avoid borrowing issues
        let mut vars = std::mem::take(&mut self.variable_array);

        // Use the existing interpreter with stack result capture
        let (result, stack_result) = Self::interpret_bytecode_static(bytecode, &mut vars)?;

        // Restore the variable array
        self.variable_array = vars;

        // Update the variables HashMap for display purposes
        self.variables.clear();
        for (i, value) in self.variable_array.iter().enumerate() {
            if !matches!(value, Value::Num(0.0)) {
                // Only store non-zero values to avoid clutter
                self.variables.insert(format!("var_{i}"), value.clone());
            }
        }

        // Prefer stack result over variable array result for expressions
        let final_result = if let Some(stack_val) = stack_result {
            vec![stack_val]
        } else {
            result
        };

        Ok(final_result)
    }

    /// Internal static interpreter that accepts a mutable variable array
    fn interpret_bytecode_static(
        bytecode: &rustmat_ignition::Bytecode,
        vars: &mut Vec<Value>,
    ) -> Result<(Vec<Value>, Option<Value>), String> {
        use rustmat_ignition::Instr;
        use std::convert::TryInto;

        let mut stack: Vec<Value> = Vec::new();
        let mut pc: usize = 0;

        // Check if the last instruction is Pop (indicating an expression statement that should return a value in REPL)
        let skip_final_pop = if let Some(last_instr) = bytecode.instructions.last() {
            matches!(last_instr, Instr::Pop)
        } else {
            false
        };

        while pc < bytecode.instructions.len() {
            match &bytecode.instructions[pc] {
                Instr::LoadConst(c) => stack.push(Value::Num(*c)),
                Instr::LoadVar(i) => {
                    if *i < vars.len() {
                        stack.push(vars[*i].clone())
                    } else {
                        stack.push(Value::Num(0.0))
                    }
                }
                Instr::StoreVar(i) => {
                    let val = stack.pop().ok_or("stack underflow")?;
                    if *i >= vars.len() {
                        vars.resize(i + 1, Value::Num(0.0));
                    }
                    vars[*i] = val;
                }
                Instr::Add => Self::binary_op(&mut stack, |a, b| a + b)?,
                Instr::Sub => Self::binary_op(&mut stack, |a, b| a - b)?,
                Instr::Mul => Self::binary_op(&mut stack, |a, b| a * b)?,
                Instr::Div => Self::binary_op(&mut stack, |a, b| a / b)?,
                Instr::Pow => Self::binary_op(&mut stack, |a, b| a.powf(b))?,
                Instr::Neg => Self::unary_op(&mut stack, |a| -a)?,
                Instr::CallBuiltin(name, argc) => {
                    let mut args = Vec::new();
                    for _ in 0..*argc {
                        args.push(stack.pop().ok_or("stack underflow")?);
                    }
                    args.reverse(); // Fix argument order

                    let result = rustmat_runtime::call_builtin(name, &args)
                        .map_err(|e| format!("Builtin call failed: {e}"))?;
                    stack.push(result);
                }
                Instr::CreateMatrix(rows, cols) => {
                    let total_elements = rows * cols;
                    let mut data = Vec::with_capacity(total_elements);

                    for _ in 0..total_elements {
                        let val: f64 = (&stack.pop().ok_or("stack underflow")?)
                            .try_into()
                            .map_err(|_| "Matrix element must be numeric")?;
                        data.push(val);
                    }

                    data.reverse(); // Fix element order

                    let matrix = rustmat_builtins::Matrix::new(data, *rows, *cols)
                        .map_err(|e| format!("Matrix creation error: {e}"))?;
                    stack.push(Value::Matrix(matrix));
                }
                Instr::LessEqual => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if a <= b { 1.0 } else { 0.0 }));
                }
                Instr::Less => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if a < b { 1.0 } else { 0.0 }));
                }
                Instr::Greater => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if a > b { 1.0 } else { 0.0 }));
                }
                Instr::GreaterEqual => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if a >= b { 1.0 } else { 0.0 }));
                }
                Instr::Equal => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if (a - b).abs() < f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }));
                }
                Instr::NotEqual => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if (a - b).abs() >= f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }));
                }
                Instr::JumpIfFalse(target) => {
                    let val: f64 = (&stack.pop().ok_or("stack underflow")?)
                        .try_into()
                        .map_err(|_| "Condition must be numeric")?;
                    if val == 0.0 {
                        pc = *target;
                        continue; // Skip the pc += 1 at the end
                    }
                }
                Instr::Jump(target) => {
                    pc = *target;
                    continue; // Skip the pc += 1 at the end
                }
                Instr::Pop => {
                    // Skip the final pop if it's the last instruction and we want to capture the expression result
                    if !(skip_final_pop && pc == bytecode.instructions.len() - 1) {
                        stack.pop();
                    }
                }
                Instr::Return => break,
            }
            pc += 1;
        }

        // Return both variable array and stack top (if any) for expression results
        let stack_result = stack.pop();
        Ok((vars.clone(), stack_result))
    }

    /// Helper for binary operations
    fn binary_op<F>(stack: &mut Vec<Value>, f: F) -> Result<(), String>
    where
        F: Fn(f64, f64) -> f64,
    {
        use std::convert::TryInto;
        let b: f64 = (&stack.pop().ok_or("stack underflow")?)
            .try_into()
            .map_err(|_| "Right operand must be numeric")?;
        let a: f64 = (&stack.pop().ok_or("stack underflow")?)
            .try_into()
            .map_err(|_| "Left operand must be numeric")?;
        stack.push(Value::Num(f(a, b)));
        Ok(())
    }

    /// Helper for unary operations
    fn unary_op<F>(stack: &mut Vec<Value>, f: F) -> Result<(), String>
    where
        F: Fn(f64) -> f64,
    {
        use std::convert::TryInto;
        let a: f64 = (&stack.pop().ok_or("stack underflow")?)
            .try_into()
            .map_err(|_| "Operand must be numeric")?;
        stack.push(Value::Num(f(a)));
        Ok(())
    }

    /// Configure garbage collector
    pub fn configure_gc(&self, config: GcConfig) -> Result<()> {
        gc_configure(config)
            .map_err(|e| anyhow::anyhow!("Failed to configure garbage collector: {}", e))
    }

    /// Get GC statistics
    pub fn gc_stats(&self) -> rustmat_gc::GcStats {
        gc_stats()
    }

    /// Show detailed system information
    pub fn show_system_info(&self) {
        println!("RustMat REPL Engine Status");
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
    let tokens = tokenize(input);
    tokens
        .into_iter()
        .map(|t| format!("{t:?}"))
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
