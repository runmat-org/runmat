use rustmat_lexer::tokenize;
use rustmat_parser::parse;
use rustmat_hir::lower;
use rustmat_ignition::compile;
use rustmat_turbine::TurbineEngine;
use rustmat_gc::{gc_configure, gc_stats, GcConfig};
use rustmat_builtins::Value;
use anyhow::Result;
use log::{debug, info, warn};
use std::time::Instant;
use std::collections::HashMap;

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
        Self::with_options(true, false)
    }

    /// Create a new REPL engine with specific options
    pub fn with_options(enable_jit: bool, verbose: bool) -> Result<Self> {
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
        })
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

        // Lower to HIR
        let hir = lower(&ast).map_err(|e| anyhow::anyhow!("Failed to lower to HIR: {}", e))?;
        if self.verbose {
            debug!("HIR generated successfully");
        }

        // Compile to bytecode
        let bytecode = compile(&hir).map_err(|e| anyhow::anyhow!("Failed to compile to bytecode: {}", e))?;
        if self.verbose {
            debug!("Bytecode compiled: {} instructions", bytecode.instructions.len());
        }

        let mut used_jit = false;
        let mut result_value = None;
        let mut error = None;

        // Try JIT compilation/execution first if available
        if let Some(ref mut jit_engine) = self.jit_engine {
            // Create a mutable variable array for JIT execution
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            
            match jit_engine.execute_or_compile(&bytecode, &mut vars) {
                Ok(_) => {
                    used_jit = true;
                    self.stats.jit_compiled += 1;
                    // Find the last non-zero result in vars as the result value
                    result_value = vars.into_iter().rev().find(|v| !matches!(v, Value::Num(0.0)));
                    if self.verbose {
                        debug!("JIT execution successful");
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

        // Use interpreter if JIT failed or is disabled
        if !used_jit {
            match self.interpret_with_context(&bytecode) {
                Ok(results) => {
                    self.stats.interpreter_fallback += 1;
                    result_value = results.into_iter().last();
                    if self.verbose {
                        debug!("Interpreter execution successful");
                    }
                }
                Err(e) => {
                    error = Some(format!("Execution failed: {e}"));
                }
            }
        }

        let execution_time = start_time.elapsed();
        let execution_time_ms = execution_time.as_millis() as u64;
        
        self.stats.total_execution_time_ms += execution_time_ms;
        self.stats.average_execution_time_ms = 
            self.stats.total_execution_time_ms as f64 / self.stats.total_executions as f64;

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
    fn interpret_with_context(&mut self, bytecode: &rustmat_ignition::Bytecode) -> Result<Vec<Value>, String> {
        // Ensure our variable array is large enough
        if self.variable_array.len() < bytecode.var_count {
            self.variable_array.resize(bytecode.var_count, Value::Num(0.0));
        }

        // Take ownership of the variable array temporarily to avoid borrowing issues
        let mut vars = std::mem::take(&mut self.variable_array);
        
        // Use the existing interpreter
        let result = Self::interpret_bytecode_static(bytecode, &mut vars)?;
        
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

        Ok(result)
    }

    /// Internal static interpreter that accepts a mutable variable array
    fn interpret_bytecode_static(bytecode: &rustmat_ignition::Bytecode, vars: &mut Vec<Value>) -> Result<Vec<Value>, String> {
        use rustmat_ignition::Instr;
        use std::convert::TryInto;

        let mut stack: Vec<Value> = Vec::new();
        let mut pc: usize = 0;

        while pc < bytecode.instructions.len() {
            match &bytecode.instructions[pc] {
                Instr::LoadConst(c) => stack.push(Value::Num(*c)),
                Instr::LoadVar(i) => {
                    if *i < vars.len() {
                        stack.push(vars[*i].clone())
                    } else {
                        stack.push(Value::Num(0.0))
                    }
                },
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
                        let val: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                            .map_err(|_| "Matrix element must be numeric")?;
                        data.push(val);
                    }
                    
                    data.reverse(); // Fix element order
                    
                    let matrix = rustmat_builtins::Matrix::new(data, *rows, *cols)
                        .map_err(|e| format!("Matrix creation error: {e}"))?;
                    stack.push(Value::Matrix(matrix));
                }
                Instr::LessEqual => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if a <= b { 1.0 } else { 0.0 }));
                }
                Instr::Less => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if a < b { 1.0 } else { 0.0 }));
                }
                Instr::Greater => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if a > b { 1.0 } else { 0.0 }));
                }
                Instr::GreaterEqual => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if a >= b { 1.0 } else { 0.0 }));
                }
                Instr::Equal => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if (a - b).abs() < f64::EPSILON { 1.0 } else { 0.0 }));
                }
                Instr::NotEqual => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Right operand must be numeric")?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
                        .map_err(|_| "Left operand must be numeric")?;
                    stack.push(Value::Num(if (a - b).abs() >= f64::EPSILON { 1.0 } else { 0.0 }));
                }
                Instr::JumpIfFalse(target) => {
                    let val: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
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
                Instr::Pop => { stack.pop(); }
                Instr::Return => break,
            }
            pc += 1;
        }
        
        Ok(vars.clone())
    }

    /// Helper for binary operations
    fn binary_op<F>(stack: &mut Vec<Value>, f: F) -> Result<(), String>
    where
        F: Fn(f64, f64) -> f64,
    {
        use std::convert::TryInto;
        let b: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
            .map_err(|_| "Right operand must be numeric")?;
        let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
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
        let a: f64 = (&stack.pop().ok_or("stack underflow")?).try_into()
            .map_err(|_| "Operand must be numeric")?;
        stack.push(Value::Num(f(a)));
        Ok(())
    }

    /// Configure garbage collector
    pub fn configure_gc(&self, config: GcConfig) -> Result<()> {
        gc_configure(config).map_err(|e| anyhow::anyhow!("Failed to configure garbage collector: {}", e))
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
        
        println!("JIT Compiler: {}", 
            if self.jit_engine.is_some() { "Available" } else { "Disabled/Failed" });
        println!("Verbose Mode: {}", self.verbose);
        println!();

        println!("Execution Statistics:");
        println!("  Total Executions: {}", self.stats.total_executions);
        println!("  JIT Compiled: {}", self.stats.jit_compiled);
        println!("  Interpreter Used: {}", self.stats.interpreter_fallback);
        println!("  Average Time: {:.2}ms", self.stats.average_execution_time_ms);
        println!();

        let gc_stats = self.gc_stats();
        println!("Garbage Collector:");
        println!("  Total Allocations: {}", gc_stats.total_allocations.load(std::sync::atomic::Ordering::Relaxed));
        println!("  Minor Collections: {}", gc_stats.minor_collections.load(std::sync::atomic::Ordering::Relaxed));
        println!("  Major Collections: {}", gc_stats.major_collections.load(std::sync::atomic::Ordering::Relaxed));
        println!("  Current Memory: {:.2} MB", 
            gc_stats.current_memory_usage.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1024.0 / 1024.0);
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
        Ok(mut engine) => {
            match engine.execute(input) {
                Ok(result) => {
                    if let Some(error) = result.error {
                        format!("Error: {error}")
                    } else if let Some(value) = result.value {
                        format!("{value:?}")
                    } else {
                        "".to_string()
                    }
                }
                Err(e) => format!("Error: {e}")
            }
        }
        Err(e) => format!("Engine Error: {e}")
    }
}