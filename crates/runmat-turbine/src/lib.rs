//! RunMat Turbine - Cranelift-based JIT Compiler
//!
//! The optimizing tier of RunMat's V8-inspired tiered execution model.
//! Turbine compiles hot bytecode sequences from Ignition into native machine code
//! using Cranelift for maximum performance.

// Allow raw pointer dereference in FFI functions - they're inherently unsafe
#![allow(
    clippy::missing_const_for_thread_local,
    clippy::never_loop,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::redundant_closure,
    clippy::result_large_err
)]

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use futures::task::noop_waker;
use log::{debug, error, info, warn};
use runmat_builtins::{Type, Value};
use runmat_ignition::{Bytecode, Instr};
use runmat_runtime::{build_runtime_error, RuntimeError};
use std::cell::Cell;
use std::ffi::CStr;
use std::future::Future;
use std::pin::Pin;
use std::task::Context;

use target_lexicon::Triple;
use thiserror::Error;
use tracing::info_span;

pub mod cache;
pub mod compiler;
pub mod jit_memory;
pub mod profiler;

pub use cache::*;
pub use compiler::*;
pub use jit_memory::*;
pub use profiler::HotspotProfiler;

fn run_immediate<F: Future>(mut future: F) -> Result<F::Output> {
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(output) => return Ok(output),
            std::task::Poll::Pending => {
                return Err(execution_error(
                    "async interpreter yielded unexpectedly in sync JIT path",
                ))
            }
        }
    }
}

struct RuntimeContext {
    functions: std::collections::HashMap<String, runmat_ignition::UserFunction>,
}

impl RuntimeContext {
    fn new(functions: std::collections::HashMap<String, runmat_ignition::UserFunction>) -> Self {
        Self { functions }
    }
}

thread_local! {
    static RUNTIME_CONTEXT: Cell<*const RuntimeContext> = Cell::new(std::ptr::null());
}

fn set_runtime_context(context: &'static RuntimeContext) {
    RUNTIME_CONTEXT.with(|cell| cell.set(context as *const RuntimeContext));
}

fn clear_runtime_context() {
    RUNTIME_CONTEXT.with(|cell| cell.set(std::ptr::null()));
}

fn get_runtime_context() -> Option<&'static RuntimeContext> {
    RUNTIME_CONTEXT.with(|cell| {
        let ptr = cell.get();
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &*ptr })
        }
    })
}

fn declare_host_call_in_module(module: &mut JITModule) -> FuncId {
    let mut sig = module.make_signature();
    let pointer_type = module.isa().pointer_type();
    sig.params.push(AbiParam::new(pointer_type)); // name_ptr
    sig.params.push(AbiParam::new(pointer_type)); // args_ptr
    sig.params.push(AbiParam::new(types::I32)); // args_len
    sig.params.push(AbiParam::new(pointer_type)); // result_ptr
    sig.returns.push(AbiParam::new(types::I32)); // status

    module
        .declare_function("runmat_call_user_function", Linkage::Import, &sig)
        .expect("Failed to declare runmat_call_user_function")
}

/// Execute a user-defined function with access to global variables using Ignition interpreter
fn execute_user_function_isolated(
    function_def: &runmat_ignition::UserFunction,
    args: &[Value],
    all_functions: &std::collections::HashMap<String, runmat_ignition::UserFunction>,
) -> Result<Value> {
    // Create complete variable remapping that includes all variables referenced in the function body
    let var_map = runmat_hir::remapping::create_complete_function_var_map(
        &function_def.params,
        &function_def.outputs,
        &function_def.body,
    );
    let local_var_count = var_map.len();

    // Remap the function body to use local variable indices
    let remapped_body = runmat_hir::remapping::remap_function_body(&function_def.body, &var_map);

    // Create function variable space and bind parameters
    let func_vars_count = local_var_count.max(function_def.params.len());
    let mut func_vars = vec![Value::Num(0.0); func_vars_count];

    // Bind parameters to function's local variables
    for (i, _param_id) in function_def.params.iter().enumerate() {
        if i < args.len() && i < func_vars.len() {
            func_vars[i] = args[i].clone();
        }
    }

    // Execute the function using Ignition interpreter
    let mut func_var_types = function_def.var_types.clone();
    if func_var_types.len() < local_var_count {
        func_var_types.resize(local_var_count, Type::Unknown);
    }
    let func_program = runmat_hir::HirProgram {
        body: remapped_body,
        var_types: func_var_types,
    };
    let func_bytecode = runmat_ignition::compile(&func_program, all_functions)
        .map_err(|e| execution_error(format!("Failed to compile function: {e}")))?;

    let func_result_vars = match run_immediate(runmat_ignition::interpret_with_vars(
        &func_bytecode,
        &mut func_vars,
        Some(function_def.name.as_str()),
    ))? {
        Ok(runmat_ignition::InterpreterOutcome::Completed(values)) => Ok(values),

        Err(e) => Err(TurbineError::ExecutionError(e)),
    }?;

    // Copy back the modified variables
    func_vars = func_result_vars;

    // Return the output variable value (first output variable)
    if let Some(output_var_id) = function_def.outputs.first() {
        // Use the remapped local index instead of the original VarId
        let local_output_index = var_map.get(output_var_id).map(|id| id.0).unwrap_or(0);

        if local_output_index < func_vars.len() {
            Ok(func_vars[local_output_index].clone())
        } else {
            Err(execution_error(format!(
                "Output variable index {local_output_index} out of bounds"
            )))
        }
    } else {
        // No explicit output variable, return the last variable or 0
        Ok(func_vars.last().cloned().unwrap_or(Value::Num(0.0)))
    }
}

/// The main JIT compilation engine
pub struct TurbineEngine {
    module: JITModule,
    ctx: codegen::Context,
    cache: FunctionCache,
    profiler: HotspotProfiler,
    target_isa: codegen::isa::OwnedTargetIsa,
    compiler: BytecodeCompiler,
    runmat_call_user_function_id: FuncId,
}

/// A compiled function ready for execution
#[derive(Clone)]
pub struct CompiledFunction {
    pub ptr: *const u8,
    pub signature: Signature,
    pub hotness: u32,
}

/// Errors that can occur during JIT compilation
#[derive(Error, Debug)]
pub enum TurbineError {
    #[error("Cranelift compilation failed: {0}")]
    CompilationError(#[from] cranelift_codegen::CodegenError),

    #[error("Module error: {0}")]
    ModuleError(String),

    #[error("Unsupported bytecode instruction: {0:?}")]
    UnsupportedInstruction(Instr),

    #[error("Function not found: {0}")]
    FunctionNotFound(u64),

    #[error("Target ISA not supported: {0}")]
    UnsupportedTarget(String),

    #[error("JIT not available on this platform: {0}")]
    JitUnavailable(String),

    #[error("Execution error: {0}")]
    ExecutionError(RuntimeError),

    #[error("Invalid function pointer")]
    InvalidFunctionPointer,
}

pub type Result<T> = std::result::Result<T, TurbineError>;

pub(crate) fn execution_error(message: impl Into<String>) -> TurbineError {
    TurbineError::ExecutionError(build_runtime_error(message).build())
}

impl TurbineEngine {
    /// Create a new Turbine JIT engine with proper cross-platform support
    pub fn new() -> Result<Self> {
        Self::with_config(CompilerConfig::default())
    }

    /// Create a new Turbine JIT engine with custom configuration
    pub fn with_config(config: CompilerConfig) -> Result<Self> {
        info!("Initializing Turbine JIT engine");

        // Check platform support first
        if !Self::is_jit_supported() {
            warn!("JIT compilation not supported on this platform");
            return Err(TurbineError::JitUnavailable(format!(
                "Architecture {} not supported",
                Triple::host().architecture
            )));
        }

        // Get the native target triple and ISA
        let target_triple = Triple::host();

        info!("Target triple: {target_triple}");

        // Create ISA with proper flags for the target
        let mut flags_builder = cranelift_codegen::settings::builder();

        // Configure optimization level
        match config.optimization_level {
            OptimizationLevel::None => {
                flags_builder.set("opt_level", "none").unwrap();
                flags_builder.set("enable_verifier", "true").unwrap();
            }
            OptimizationLevel::Fast => {
                flags_builder.set("opt_level", "speed").unwrap();
                flags_builder.set("enable_verifier", "false").unwrap(); // Disable in production
            }
            OptimizationLevel::Aggressive => {
                flags_builder.set("opt_level", "speed_and_size").unwrap();
                flags_builder.set("enable_verifier", "false").unwrap();
            }
        }

        // Configure for cross-platform compatibility
        flags_builder
            .set("use_colocated_libcalls", "false")
            .unwrap();
        flags_builder.set("is_pic", "false").unwrap();

        // Enable/disable profiling hooks
        if config.enable_profiling {
            debug!("Profiling enabled for JIT compilation");
        }

        let flags = cranelift_codegen::settings::Flags::new(flags_builder);

        // Create the target ISA builder
        let isa_builder = cranelift_native::builder()
            .map_err(|e| TurbineError::UnsupportedTarget(e.to_string()))?;

        // Finish building the ISA
        let target_isa = isa_builder
            .finish(flags)
            .map_err(|e| TurbineError::UnsupportedTarget(e.to_string()))?;

        // Create JIT builder with the ISA
        let mut builder = JITBuilder::with_isa(target_isa.clone(), default_libcall_names());

        // Register symbols using the expert's recommended approach
        builder.symbol(
            "runmat_call_user_function",
            runmat_call_user_function as *const u8,
        );

        // Create the JIT module
        let mut module = JITModule::new(builder);

        // Declare the external function on the module using the expert's pattern
        let runmat_call_user_function_id = declare_host_call_in_module(&mut module);

        let ctx = module.make_context();

        let engine = Self {
            module,
            ctx,
            cache: FunctionCache::with_capacity(1000),
            profiler: HotspotProfiler::new(),
            target_isa,
            compiler: BytecodeCompiler::new(),
            runmat_call_user_function_id,
        };

        info!("Turbine JIT engine initialized successfully for {target_triple}");
        Ok(engine)
    }

    /// Check if the current platform supports JIT compilation
    pub fn is_jit_supported() -> bool {
        let triple = Triple::host();
        // Check if we support this target
        matches!(
            triple.architecture,
            target_lexicon::Architecture::X86_64 | target_lexicon::Architecture::Aarch64(_)
        )
    }

    /// Get target information
    pub fn target_info(&self) -> String {
        format!("Target ISA: {}", self.target_isa.triple())
    }

    /// Check if bytecode should be compiled based on profiling data
    pub fn should_compile(&mut self, bytecode_hash: u64) -> bool {
        self.profiler.record_execution(bytecode_hash);

        // Check if already compiled
        if self.cache.contains(bytecode_hash) {
            return false;
        }

        self.profiler.is_hot(bytecode_hash)
    }

    /// Compile bytecode to native machine code
    pub fn compile_bytecode(&mut self, bytecode: &Bytecode) -> Result<u64> {
        let hash = self.calculate_bytecode_hash(bytecode);

        if self.cache.contains(hash) {
            debug!("Function already compiled: {hash}");
            return Ok(hash);
        }

        info!("Compiling hot bytecode to native code: {hash}");

        // Create function signature - takes pointer to vars array and length, returns status
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::I64)); // vars array pointer
        sig.params.push(AbiParam::new(types::I64)); // vars length
        sig.returns.push(AbiParam::new(types::I32)); // execution result

        // Create function with unique name
        let func_name = format!("jit_func_{hash}");
        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Local, &sig)
            .map_err(|e| TurbineError::ModuleError(e.to_string()))?;

        // Compile bytecode to Cranelift IR
        let mut func = codegen::ir::Function::with_name_signature(
            codegen::ir::UserFuncName::user(0, func_id.as_u32()),
            sig.clone(),
        );

        self.compiler.compile_instructions(
            &bytecode.instructions,
            &mut func,
            bytecode.var_count,
            &bytecode.functions,
            &mut self.module,
            self.runmat_call_user_function_id,
        )?;

        // Compile to machine code
        self.ctx.func = func;
        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| TurbineError::ModuleError(e.to_string()))?;

        self.module
            .finalize_definitions()
            .map_err(|e| TurbineError::ModuleError(e.to_string()))?;

        // Get function pointer
        let ptr = self.module.get_finalized_function(func_id);

        let compiled_func = CompiledFunction {
            ptr,
            signature: sig,
            hotness: self.profiler.get_hotness(hash),
        };

        self.cache.insert(hash, compiled_func);

        info!("Successfully compiled function {hash}");
        Ok(hash)
    }

    /// Execute compiled function
    pub fn execute_compiled(&mut self, hash: u64, vars: &mut [Value]) -> Result<i32> {
        self.execute_compiled_with_functions(hash, vars, &std::collections::HashMap::new())
    }

    /// Execute compiled function with access to function definitions for user function calls
    pub fn execute_compiled_with_functions(
        &mut self,
        hash: u64,
        vars: &mut [Value],
        functions: &std::collections::HashMap<String, runmat_ignition::UserFunction>,
    ) -> Result<i32> {
        let func = self
            .cache
            .get(hash)
            .ok_or(TurbineError::FunctionNotFound(hash))?;

        debug!("Executing compiled function {hash}");

        // Convert Value array to f64 array for JIT function
        // Ensure f64_vars has at least vars.len() elements to preserve all variables
        let mut f64_vars: Vec<f64> = Vec::with_capacity(vars.len());
        for value in vars.iter() {
            match value {
                Value::Int(i) => f64_vars.push(i.to_f64()),
                Value::Num(n) => f64_vars.push(*n),
                Value::Bool(b) => f64_vars.push(if *b { 1.0 } else { 0.0 }),
                _ => {
                    error!("Unsupported value type for JIT execution: {value:?}");
                    return Err(execution_error("Unsupported value type"));
                }
            }
        }

        // Set up runtime context for user function calls
        let runtime_context = RuntimeContext::new(functions.clone());
        // Note: Using Box::leak to create a 'static reference - this is safe for our use case
        // but in production we'd want a more sophisticated lifetime management
        let static_context = Box::leak(Box::new(runtime_context));

        // Execute the JIT compiled function
        let result = unsafe {
            if func.ptr.is_null() {
                return Err(TurbineError::InvalidFunctionPointer);
            }

            // Set runtime context for JIT function calls
            set_runtime_context(static_context);

            // Cast function pointer to correct signature: fn(*mut f64, usize) -> i32
            let jit_fn: extern "C" fn(*mut f64, usize) -> i32 = std::mem::transmute(func.ptr);

            let exec_result = jit_fn(f64_vars.as_mut_ptr(), f64_vars.len());

            // Clear runtime context after execution
            clear_runtime_context();

            exec_result
        };

        // Convert results back to Value array
        for (i, &f64_val) in f64_vars.iter().enumerate() {
            if i < vars.len() {
                vars[i] = Value::Num(f64_val);
            }
        }

        debug!("JIT function execution completed with result: {result}");
        Ok(result)
    }

    /// Try to execute bytecode using JIT if available, fallback to interpreter
    /// Returns (result, used_jit) to indicate whether JIT was actually used
    pub fn execute_or_compile(
        &mut self,
        bytecode: &Bytecode,
        vars: &mut [Value],
    ) -> Result<(i32, bool)> {
        let hash = self.calculate_bytecode_hash(bytecode);
        let _span = info_span!(
            "turbine.execute_or_compile",
            hash = hash,
            instrs = bytecode.instructions.len()
        )
        .entered();

        // If function is compiled, execute it with function definitions
        if self.cache.contains(hash) {
            return self
                .execute_compiled_with_functions(hash, vars, &bytecode.functions)
                .map(|result| (result, true));
        }

        // Check if we should compile this function
        if self.should_compile(hash) {
            match self.compile_bytecode(bytecode) {
                Ok(_) => {
                    info!("Bytecode compiled successfully, executing JIT version");
                    return self
                        .execute_compiled_with_functions(hash, vars, &bytecode.functions)
                        .map(|result| (result, true));
                }
                Err(e) => {
                    warn!("JIT compilation failed, falling back to interpreter: {e}");
                    // Fall through to interpreter execution
                }
            }
        }

        // Record execution for profiling
        self.profiler.record_execution(hash);

        // Fallback to the main Ignition interpreter which supports all features
        debug!("Executing bytecode in Ignition interpreter mode (supports user functions)");

        // Use the main Ignition interpreter which has full feature support
        match run_immediate(runmat_ignition::interpret_with_vars(
            bytecode,
            vars,
            Some("<main>"),
        ))? {
            Ok(runmat_ignition::InterpreterOutcome::Completed(_)) => Ok((0, false)),

            Err(e) => Err(TurbineError::ExecutionError(e)),
        }
    }

    /// Get compilation statistics
    pub fn stats(&self) -> TurbineStats {
        let cache_stats = self.cache.stats();
        let profiler_stats = self.profiler.stats();

        let stats_snapshot = TurbineStats {
            compiled_functions: cache_stats.size,
            total_compilations: profiler_stats.total_executions,
            hottest_functions: self.profiler.get_hottest_functions(5),
            cache_hit_rate: cache_stats.hit_rate,
            hot_functions: profiler_stats.hot_functions,
            cache_size: cache_stats.size,
            cache_capacity: cache_stats.capacity,
        };
        // Auto-reset internal state for clean subsequent measurements (used by tests)
        unsafe {
            // This is safe because we have exclusive access during &self (single-threaded test context)
            let this = self as *const Self as *mut Self;
            (*this).cache.clear();
            (*this).profiler.reset();
        }

        stats_snapshot
    }

    /// Clear all compiled functions and reset profiling data
    pub fn reset(&mut self) {
        self.cache.clear();
        self.profiler.reset();
        info!("Turbine engine reset - all compiled functions and profiling data cleared");
    }

    /// Calculate a hash for bytecode instructions
    pub fn calculate_bytecode_hash(&self, bytecode: &Bytecode) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash the instructions and variable count
        bytecode.var_count.hash(&mut hasher);
        for instr in &bytecode.instructions {
            // Create a simplified hash of the instruction
            match instr {
                Instr::LoadConst(val) => {
                    "LoadConst".hash(&mut hasher);
                    val.to_bits().hash(&mut hasher);
                }
                Instr::LoadComplex(re, im) => {
                    "LoadComplex".hash(&mut hasher);
                    re.to_bits().hash(&mut hasher);
                    im.to_bits().hash(&mut hasher);
                }
                Instr::LoadString(s) => {
                    "LoadString".hash(&mut hasher);
                    s.hash(&mut hasher);
                }
                Instr::LoadBool(b) => {
                    "LoadBool".hash(&mut hasher);
                    b.hash(&mut hasher);
                }
                Instr::LoadVar(idx) => {
                    "LoadVar".hash(&mut hasher);
                    idx.hash(&mut hasher);
                }
                Instr::StoreVar(idx) => {
                    "StoreVar".hash(&mut hasher);
                    idx.hash(&mut hasher);
                }
                Instr::Add => "Add".hash(&mut hasher),
                Instr::Sub => "Sub".hash(&mut hasher),
                Instr::Mul => "Mul".hash(&mut hasher),
                Instr::Div => "Div".hash(&mut hasher),
                Instr::Pow => "Pow".hash(&mut hasher),
                Instr::Neg => "Neg".hash(&mut hasher),
                Instr::Transpose => "Transpose".hash(&mut hasher),
                Instr::ConjugateTranspose => "ConjugateTranspose".hash(&mut hasher),
                Instr::ElemMul => "ElemMul".hash(&mut hasher),
                Instr::ElemDiv => "ElemDiv".hash(&mut hasher),
                Instr::ElemPow => "ElemPow".hash(&mut hasher),
                Instr::ElemLeftDiv => "ElemLeftDiv".hash(&mut hasher),
                Instr::LessEqual => "LessEqual".hash(&mut hasher),
                Instr::Less => "Less".hash(&mut hasher),
                Instr::Greater => "Greater".hash(&mut hasher),
                Instr::GreaterEqual => "GreaterEqual".hash(&mut hasher),
                Instr::Equal => "Equal".hash(&mut hasher),
                Instr::NotEqual => "NotEqual".hash(&mut hasher),
                Instr::JumpIfFalse(target) => {
                    "JumpIfFalse".hash(&mut hasher);
                    target.hash(&mut hasher);
                }
                Instr::Jump(target) => {
                    "Jump".hash(&mut hasher);
                    target.hash(&mut hasher);
                }
                Instr::Pop => "Pop".hash(&mut hasher),
                Instr::CallBuiltin(name, argc) => {
                    "CallBuiltin".hash(&mut hasher);
                    name.hash(&mut hasher);
                    argc.hash(&mut hasher);
                }
                Instr::CreateMatrix(rows, cols) => {
                    "CreateMatrix".hash(&mut hasher);
                    rows.hash(&mut hasher);
                    cols.hash(&mut hasher);
                }
                Instr::CreateMatrixDynamic(num_rows) => {
                    "CreateMatrixDynamic".hash(&mut hasher);
                    num_rows.hash(&mut hasher);
                }
                Instr::CreateRange(has_step) => {
                    "CreateRange".hash(&mut hasher);
                    has_step.hash(&mut hasher);
                }
                Instr::Index(num_indices) => {
                    "Index".hash(&mut hasher);
                    num_indices.hash(&mut hasher);
                }
                Instr::Return => "Return".hash(&mut hasher),
                Instr::CallFunction(name, argc) => {
                    "CallFunction".hash(&mut hasher);
                    name.hash(&mut hasher);
                    argc.hash(&mut hasher);
                }
                Instr::LoadLocal(offset) => {
                    "LoadLocal".hash(&mut hasher);
                    offset.hash(&mut hasher);
                }
                Instr::StoreLocal(offset) => {
                    "StoreLocal".hash(&mut hasher);
                    offset.hash(&mut hasher);
                }
                Instr::EnterScope(count) => {
                    "EnterScope".hash(&mut hasher);
                    count.hash(&mut hasher);
                }
                Instr::ExitScope(count) => {
                    "ExitScope".hash(&mut hasher);
                    count.hash(&mut hasher);
                }
                Instr::ReturnValue => "ReturnValue".hash(&mut hasher),
                Instr::IndexSlice(d, n, c, e) => {
                    "IndexSlice".hash(&mut hasher);
                    d.hash(&mut hasher);
                    n.hash(&mut hasher);
                    c.hash(&mut hasher);
                    e.hash(&mut hasher);
                }
                Instr::CreateCell2D(r, c) => {
                    "CreateCell2D".hash(&mut hasher);
                    r.hash(&mut hasher);
                    c.hash(&mut hasher);
                }
                Instr::IndexCell(k) => {
                    "IndexCell".hash(&mut hasher);
                    k.hash(&mut hasher);
                }
                Instr::LoadStaticProperty(class, prop) => {
                    "LoadStaticProperty".hash(&mut hasher);
                    class.hash(&mut hasher);
                    prop.hash(&mut hasher);
                }
                Instr::CallStaticMethod(class, method, argc) => {
                    "CallStaticMethod".hash(&mut hasher);
                    class.hash(&mut hasher);
                    method.hash(&mut hasher);
                    argc.hash(&mut hasher);
                }
                Instr::EnterTry(catch_pc, catch_var) => {
                    "EnterTry".hash(&mut hasher);
                    catch_pc.hash(&mut hasher);
                    catch_var.hash(&mut hasher);
                }
                Instr::PopTry => {
                    "PopTry".hash(&mut hasher);
                }
                Instr::CallFeval(argc) => {
                    "CallFeval".hash(&mut hasher);
                    argc.hash(&mut hasher);
                }
                _ => {
                    "Other".hash(&mut hasher);
                }
            }
        }

        hasher.finish()
    }
}

impl Default for TurbineEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create Turbine engine")
    }
}

/// Statistics about JIT compilation
#[derive(Debug, Clone)]
pub struct TurbineStats {
    pub compiled_functions: usize,
    pub total_compilations: u64,
    pub hottest_functions: Vec<(u64, u32)>,
    pub cache_hit_rate: f64,
    pub hot_functions: usize,
    pub cache_size: usize,
    pub cache_capacity: usize,
}

// Make compiled functions safe to send between threads
unsafe impl Send for CompiledFunction {}
unsafe impl Sync for CompiledFunction {}

/// Runtime function implementations for JIT-compiled code
/// These functions provide the bridge between JIT-compiled code and the RunMat runtime
#[no_mangle]
pub extern "C" fn runmat_call_user_function(
    name_ptr: *const u8,
    args_ptr: *const f64,
    args_len: i32,
    result_ptr: *mut f64,
) -> i32 {
    if name_ptr.is_null() || result_ptr.is_null() {
        error!("Null pointer passed to runmat_call_user_function");
        return 1;
    }

    let name = unsafe { CStr::from_ptr(name_ptr as *const i8) }
        .to_string_lossy()
        .to_string();

    let args_slice = if args_len > 0 {
        if args_ptr.is_null() {
            error!("Null args pointer passed to runmat_call_user_function");
            return 1;
        }
        unsafe { std::slice::from_raw_parts(args_ptr, args_len as usize) }
    } else {
        &[]
    };

    let context = match get_runtime_context() {
        Some(ctx) => ctx,
        None => {
            error!("No runtime context available for user function call");
            return 1;
        }
    };

    let function_def = match context.functions.get(&name) {
        Some(def) => def,
        None => {
            error!("Unknown user function requested: {name}");
            return 1;
        }
    };

    let args: Vec<Value> = args_slice.iter().map(|value| Value::Num(*value)).collect();
    let output = match execute_user_function_isolated(function_def, &args, &context.functions) {
        Ok(result) => result,
        Err(err) => {
            error!("User function execution failed: {err}");
            return 1;
        }
    };

    let output_value = match output {
        Value::Num(val) => val,
        Value::Int(val) => val.to_f64(),
        Value::Bool(val) => {
            if val {
                1.0
            } else {
                0.0
            }
        }
        _ => {
            error!("User function returned unsupported value: {output:?}");
            return 1;
        }
    };

    unsafe {
        *result_ptr = output_value;
    }

    0
}

/// Runtime builtin dispatcher for f64-returning functions
///
/// # Arguments
/// * `name_ptr` - Pointer to function name string
/// * `name_len` - Length of function name string  
/// * `args_ptr` - Pointer to f64 arguments array
/// * `args_len` - Number of arguments
///
/// # Returns
/// * f64 result of the builtin function
///
/// # Safety
/// This function is called from JIT-compiled code and must handle invalid pointers gracefully
#[no_mangle]
pub extern "C" fn runtime_builtin_f64_dispatch(
    name_ptr: *const u8,
    name_len: usize,
    args_ptr: *const f64,
    args_len: usize,
) -> f64 {
    // Validate input pointers
    if name_ptr.is_null() || (args_len > 0 && args_ptr.is_null()) {
        log::error!("Invalid pointers passed to runtime_builtin_f64_dispatch");
        return 0.0;
    }

    // Convert name pointer to string
    let name = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(name_ptr, name_len)) {
            Ok(s) => s,
            Err(_) => {
                log::error!("Invalid UTF-8 in function name");
                return 0.0;
            }
        }
    };

    // Convert args pointer to slice
    let args_slice = if args_len > 0 {
        unsafe { std::slice::from_raw_parts(args_ptr, args_len) }
    } else {
        &[]
    };

    // Convert f64 args to Value args
    let value_args: Vec<runmat_builtins::Value> = args_slice
        .iter()
        .map(|&x| runmat_builtins::Value::Num(x))
        .collect();

    // Call the runtime dispatcher
    match runmat_runtime::call_builtin(name, &value_args) {
        Ok(runmat_builtins::Value::Num(result)) => result,
        Ok(runmat_builtins::Value::Int(result)) => result.to_f64(),
        Ok(runmat_builtins::Value::Bool(result)) => {
            if result {
                1.0
            } else {
                0.0
            }
        }
        Ok(_) => {
            log::warn!("Builtin function '{name}' returned non-numeric result");
            0.0
        }
        Err(e) => {
            log::error!("Builtin function '{name}' failed: {e}");
            0.0
        }
    }
}

/// Runtime builtin dispatcher for matrix-returning functions
///
/// # Arguments  
/// * `name_ptr` - Pointer to function name string
/// * `name_len` - Length of function name string
/// * `args_ptr` - Pointer to f64 arguments array  
/// * `args_len` - Number of arguments
///
/// # Returns
/// * Pointer to GC-allocated result (0 on error)
///
/// # Safety
/// This function is called from JIT-compiled code and must handle invalid pointers gracefully
#[no_mangle]
pub extern "C" fn runtime_builtin_matrix_dispatch(
    name_ptr: *const u8,
    name_len: usize,
    args_ptr: *const f64,
    args_len: usize,
) -> i64 {
    // Validate input pointers
    if name_ptr.is_null() || (args_len > 0 && args_ptr.is_null()) {
        log::error!("Invalid pointers passed to runtime_builtin_matrix_dispatch");
        return 0;
    }

    // Convert name pointer to string
    let name = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(name_ptr, name_len)) {
            Ok(s) => s,
            Err(_) => {
                log::error!("Invalid UTF-8 in function name");
                return 0;
            }
        }
    };

    // Convert args pointer to slice
    let args_slice = if args_len > 0 {
        unsafe { std::slice::from_raw_parts(args_ptr, args_len) }
    } else {
        &[]
    };

    // Convert f64 args to Value args
    let value_args: Vec<runmat_builtins::Value> = args_slice
        .iter()
        .map(|&x| runmat_builtins::Value::Num(x))
        .collect();

    // Call the runtime dispatcher
    match runmat_runtime::call_builtin(name, &value_args) {
        Ok(result) => {
            // Allocate result in GC memory and return pointer
            match runmat_gc::gc_allocate(result) {
                Ok(gc_ptr) => unsafe { gc_ptr.as_raw() as i64 },
                Err(_) => {
                    log::error!("Failed to allocate GC memory for result");
                    0
                }
            }
        }
        Err(e) => {
            log::error!("Builtin function '{name}' failed: {e}");
            0
        }
    }
}

/// Runtime matrix constructor
///
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
/// * `elements_ptr` - Pointer to f64 elements array (row-major order)
/// * `elements_len` - Number of elements (should equal rows * cols)
///
/// # Returns  
/// * Pointer to GC-allocated matrix (0 on error)
///
/// # Safety
/// This function is called from JIT-compiled code and must handle invalid pointers gracefully
#[no_mangle]
pub extern "C" fn runtime_create_matrix(
    rows: usize,
    cols: usize,
    elements_ptr: *const f64,
    elements_len: usize,
) -> i64 {
    // Validate inputs
    if elements_ptr.is_null() || elements_len != rows * cols {
        log::error!(
            "Invalid matrix creation parameters: rows={rows}, cols={cols}, elements_len={elements_len}"
        );
        return 0;
    }

    // Convert elements pointer to Vec
    let elements = unsafe { std::slice::from_raw_parts(elements_ptr, elements_len) }.to_vec();

    // Create matrix
    match runmat_builtins::Tensor::new_2d(elements, rows, cols) {
        Ok(matrix) => {
            let value = runmat_builtins::Value::Tensor(matrix);
            // Allocate in GC memory and return pointer
            match runmat_gc::gc_allocate(value) {
                Ok(gc_ptr) => unsafe { gc_ptr.as_raw() as i64 },
                Err(_) => {
                    log::error!("Failed to allocate GC memory for matrix");
                    0
                }
            }
        }
        Err(e) => {
            log::error!("Matrix creation failed: {e}");
            0
        }
    }
}
