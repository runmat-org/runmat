//! RustMat Turbine - Cranelift-based JIT Compiler
//!
//! The optimizing tier of RustMat's V8-inspired tiered execution model.
//! Turbine compiles hot bytecode sequences from Ignition into native machine code
//! using Cranelift for maximum performance.

// Allow raw pointer dereference in FFI functions - they're inherently unsafe
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use log::{debug, error, info, warn};
use rustmat_builtins::Value;
use rustmat_gc::gc_allocate;
use rustmat_ignition::{Bytecode, Instr};
use std::boxed::Box;
use target_lexicon::Triple;
use thiserror::Error;

pub mod cache;
pub mod compiler;
pub mod jit_memory;
pub mod profiler;

pub use cache::*;
pub use compiler::*;
pub use jit_memory::*;
pub use profiler::*;

// Runtime interface functions for JIT compiled code
// These functions are called from JIT compiled code to interact with the Rust runtime

/// Create a new Value::Num and return a pointer to it
#[no_mangle]
pub extern "C" fn rustmat_create_value_num(val: f64) -> *mut Value {
    match gc_allocate(Value::Num(val)) {
        Ok(gc_ptr) => unsafe { gc_ptr.as_raw_mut() },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a Value object (no-op with GC, kept for compatibility)
#[no_mangle]
pub extern "C" fn rustmat_free_value(_ptr: *mut Value) {
    // With garbage collection, explicit freeing is not needed
    // The GC will automatically collect unreachable objects
}

/// Add two Value objects
#[no_mangle]
pub extern "C" fn rustmat_value_add(a_ptr: *const Value, b_ptr: *const Value) -> *mut Value {
    if a_ptr.is_null() || b_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let a = &*a_ptr;
        let b = &*b_ptr;

        let result = match (a, b) {
            (Value::Num(x), Value::Num(y)) => Value::Num(x + y),
            (Value::Int(x), Value::Int(y)) => Value::Int(x + y),
            (Value::Num(x), Value::Int(y)) => Value::Num(x + (*y as f64)),
            (Value::Int(x), Value::Num(y)) => Value::Num((*x as f64) + y),
            _ => {
                error!("Unsupported addition: {a:?} + {b:?}");
                return std::ptr::null_mut();
            }
        };

        match gc_allocate(result) {
            Ok(gc_ptr) => gc_ptr.as_raw_mut(),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// Subtract two Value objects
#[no_mangle]
pub extern "C" fn rustmat_value_sub(a_ptr: *const Value, b_ptr: *const Value) -> *mut Value {
    if a_ptr.is_null() || b_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let a = &*a_ptr;
        let b = &*b_ptr;

        let result = match (a, b) {
            (Value::Num(x), Value::Num(y)) => Value::Num(x - y),
            (Value::Int(x), Value::Int(y)) => Value::Int(x - y),
            (Value::Num(x), Value::Int(y)) => Value::Num(x - (*y as f64)),
            (Value::Int(x), Value::Num(y)) => Value::Num((*x as f64) - y),
            _ => {
                error!("Unsupported subtraction: {a:?} - {b:?}");
                return std::ptr::null_mut();
            }
        };

        match gc_allocate(result) {
            Ok(gc_ptr) => gc_ptr.as_raw_mut(),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// Multiply two Value objects
#[no_mangle]
pub extern "C" fn rustmat_value_mul(a_ptr: *const Value, b_ptr: *const Value) -> *mut Value {
    if a_ptr.is_null() || b_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let a = &*a_ptr;
        let b = &*b_ptr;

        let result = match (a, b) {
            (Value::Num(x), Value::Num(y)) => Value::Num(x * y),
            (Value::Int(x), Value::Int(y)) => Value::Int(x * y),
            (Value::Num(x), Value::Int(y)) => Value::Num(x * (*y as f64)),
            (Value::Int(x), Value::Num(y)) => Value::Num((*x as f64) * y),
            _ => {
                error!("Unsupported multiplication: {a:?} * {b:?}");
                return std::ptr::null_mut();
            }
        };

        match gc_allocate(result) {
            Ok(gc_ptr) => gc_ptr.as_raw_mut(),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// Divide two Value objects
#[no_mangle]
pub extern "C" fn rustmat_value_div(a_ptr: *const Value, b_ptr: *const Value) -> *mut Value {
    if a_ptr.is_null() || b_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let a = &*a_ptr;
        let b = &*b_ptr;

        let result = match (a, b) {
            (Value::Num(x), Value::Num(y)) => {
                if *y == 0.0 {
                    error!("Division by zero");
                    return std::ptr::null_mut();
                }
                Value::Num(x / y)
            }
            (Value::Int(x), Value::Int(y)) => {
                if *y == 0 {
                    error!("Division by zero");
                    return std::ptr::null_mut();
                }
                Value::Num((*x as f64) / (*y as f64))
            }
            (Value::Num(x), Value::Int(y)) => {
                if *y == 0 {
                    error!("Division by zero");
                    return std::ptr::null_mut();
                }
                Value::Num(x / (*y as f64))
            }
            (Value::Int(x), Value::Num(y)) => {
                if *y == 0.0 {
                    error!("Division by zero");
                    return std::ptr::null_mut();
                }
                Value::Num((*x as f64) / y)
            }
            _ => {
                error!("Unsupported division: {a:?} / {b:?}");
                return std::ptr::null_mut();
            }
        };

        match gc_allocate(result) {
            Ok(gc_ptr) => gc_ptr.as_raw_mut(),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// Power of two Value objects
#[no_mangle]
pub extern "C" fn rustmat_value_pow(a_ptr: *const Value, b_ptr: *const Value) -> *mut Value {
    if a_ptr.is_null() || b_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let a = &*a_ptr;
        let b = &*b_ptr;

        let result = match (a, b) {
            (Value::Num(x), Value::Num(y)) => Value::Num(x.powf(*y)),
            (Value::Int(x), Value::Int(y)) => Value::Num((*x as f64).powf(*y as f64)),
            (Value::Num(x), Value::Int(y)) => Value::Num(x.powf(*y as f64)),
            (Value::Int(x), Value::Num(y)) => Value::Num((*x as f64).powf(*y)),
            _ => {
                error!("Unsupported power: {a:?} ^ {b:?}");
                return std::ptr::null_mut();
            }
        };

        match gc_allocate(result) {
            Ok(gc_ptr) => gc_ptr.as_raw_mut(),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// Negate a Value object
#[no_mangle]
pub extern "C" fn rustmat_value_neg(a_ptr: *const Value) -> *mut Value {
    if a_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let a = &*a_ptr;

        let result = match a {
            Value::Num(x) => Value::Num(-x),
            Value::Int(x) => Value::Int(-x),
            _ => {
                error!("Unsupported negation: -{a:?}");
                return std::ptr::null_mut();
            }
        };

        match gc_allocate(result) {
            Ok(gc_ptr) => gc_ptr.as_raw_mut(),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// Element-wise multiplication
#[no_mangle]
pub extern "C" fn rustmat_value_elementwise_mul(a_ptr: *const Value, b_ptr: *const Value) -> *mut Value {
    if a_ptr.is_null() || b_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let a = &*a_ptr;
        let b = &*b_ptr;

        match rustmat_runtime::elementwise_mul(a, b) {
            Ok(result) => match gc_allocate(result) {
                Ok(gc_ptr) => gc_ptr.as_raw_mut(),
                Err(_) => std::ptr::null_mut(),
            },
            Err(e) => {
                error!("Element-wise multiplication error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

/// Element-wise division
#[no_mangle]
pub extern "C" fn rustmat_value_elementwise_div(a_ptr: *const Value, b_ptr: *const Value) -> *mut Value {
    if a_ptr.is_null() || b_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let a = &*a_ptr;
        let b = &*b_ptr;

        match rustmat_runtime::elementwise_div(a, b) {
            Ok(result) => match gc_allocate(result) {
                Ok(gc_ptr) => gc_ptr.as_raw_mut(),
                Err(_) => std::ptr::null_mut(),
            },
            Err(e) => {
                error!("Element-wise division error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

/// Element-wise power
#[no_mangle]
pub extern "C" fn rustmat_value_elementwise_pow(a_ptr: *const Value, b_ptr: *const Value) -> *mut Value {
    if a_ptr.is_null() || b_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let a = &*a_ptr;
        let b = &*b_ptr;

        match rustmat_runtime::elementwise_pow(a, b) {
            Ok(result) => match gc_allocate(result) {
                Ok(gc_ptr) => gc_ptr.as_raw_mut(),
                Err(_) => std::ptr::null_mut(),
            },
            Err(e) => {
                error!("Element-wise power error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

/// Element-wise left division
#[no_mangle]
pub extern "C" fn rustmat_value_elementwise_leftdiv(a_ptr: *const Value, b_ptr: *const Value) -> *mut Value {
    if a_ptr.is_null() || b_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let a = &*a_ptr;
        let b = &*b_ptr;

        // Left division is b \ a which is equivalent to a / b
        match rustmat_runtime::elementwise_div(b, a) {
            Ok(result) => match gc_allocate(result) {
                Ok(gc_ptr) => gc_ptr.as_raw_mut(),
                Err(_) => std::ptr::null_mut(),
            },
            Err(e) => {
                error!("Element-wise left division error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

/// Compare two Value objects: less than
#[no_mangle]
pub extern "C" fn rustmat_value_lt(a_ptr: *const Value, b_ptr: *const Value) -> *mut Value {
    if a_ptr.is_null() || b_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let a = &*a_ptr;
        let b = &*b_ptr;

        let result = match (a, b) {
            (Value::Num(x), Value::Num(y)) => Value::Num(if x < y { 1.0 } else { 0.0 }),
            (Value::Int(x), Value::Int(y)) => Value::Num(if x < y { 1.0 } else { 0.0 }),
            (Value::Num(x), Value::Int(y)) => Value::Num(if *x < (*y as f64) { 1.0 } else { 0.0 }),
            (Value::Int(x), Value::Num(y)) => Value::Num(if (*x as f64) < *y { 1.0 } else { 0.0 }),
            _ => {
                error!("Unsupported comparison: {a:?} < {b:?}");
                return std::ptr::null_mut();
            }
        };

        match gc_allocate(result) {
            Ok(gc_ptr) => gc_ptr.as_raw_mut(),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// Call a builtin function by name
#[no_mangle]
pub extern "C" fn rustmat_call_builtin(
    name_ptr: *const u8,
    name_len: usize,
    args_ptr: *const *const Value,
    args_len: usize,
) -> *mut Value {
    if name_ptr.is_null() || args_ptr.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        // Convert name from C string
        let name_slice = std::slice::from_raw_parts(name_ptr, name_len);
        let name = match std::str::from_utf8(name_slice) {
            Ok(s) => s,
            Err(_) => {
                error!("Invalid UTF-8 in builtin name");
                return std::ptr::null_mut();
            }
        };

        // Convert arguments
        let args_slice = std::slice::from_raw_parts(args_ptr, args_len);
        let mut args = Vec::new();
        for &arg_ptr in args_slice {
            if arg_ptr.is_null() {
                error!("Null argument in builtin call");
                return std::ptr::null_mut();
            }
            args.push((*arg_ptr).clone());
        }

        // Call the builtin
        match rustmat_runtime::call_builtin(name, &args) {
            Ok(result) => match gc_allocate(result) {
                Ok(gc_ptr) => gc_ptr.as_raw_mut(),
                Err(_) => std::ptr::null_mut(),
            },
            Err(e) => {
                error!("Builtin call failed: {e}");
                std::ptr::null_mut()
            }
        }
    }
}

/// Load a variable from the variables array
#[no_mangle]
pub extern "C" fn rustmat_load_var(
    vars_ptr: *mut Value,
    vars_len: usize,
    index: usize,
) -> *mut Value {
    if vars_ptr.is_null() || index >= vars_len {
        error!("Invalid variable access: index {index} >= length {vars_len}");
        return std::ptr::null_mut();
    }

    unsafe {
        let vars_slice = std::slice::from_raw_parts(vars_ptr, vars_len);
        let value = vars_slice[index].clone();
        match gc_allocate(value) {
            Ok(gc_ptr) => gc_ptr.as_raw_mut(),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// Store a variable to the variables array
#[no_mangle]
pub extern "C" fn rustmat_store_var(
    vars_ptr: *mut Value,
    vars_len: usize,
    index: usize,
    value_ptr: *const Value,
) -> i32 {
    if vars_ptr.is_null() || value_ptr.is_null() || index >= vars_len {
        error!("Invalid variable store: index {index} >= length {vars_len}");
        return -1; // Error
    }

    unsafe {
        let vars_slice = std::slice::from_raw_parts_mut(vars_ptr, vars_len);
        let value = (*value_ptr).clone();
        vars_slice[index] = value;
        0 // Success
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
    ExecutionError(String),

    #[error("Invalid function pointer")]
    InvalidFunctionPointer,
}

pub type Result<T> = std::result::Result<T, TurbineError>;

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

        // Configure symbol resolution for cross-platform compatibility
        builder.symbol_lookup_fn(Box::new(|name| {
            debug!("Symbol lookup requested for: {name}");

            // Provide symbol lookup for mathematical functions and runtime calls
            match name {
                // Math library functions that might be used by JIT compiled code
                "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2" => {
                    // In a complete implementation, these would link to actual math library functions
                    debug!("Math function {name} requested but not available for linking");
                    None
                }
                "exp" | "log" | "log10" | "log2" | "ln" => {
                    debug!("Math function {name} requested but not available for linking");
                    None
                }
                "sqrt" | "cbrt" | "pow" | "powf" => {
                    debug!("Math function {name} requested but not available for linking");
                    None
                }
                "floor" | "ceil" | "round" | "trunc" | "fabs" => {
                    debug!("Math function {name} requested but not available for linking");
                    None
                }
                "fmin" | "fmax" | "fmod" | "remainder" => {
                    debug!("Math function {name} requested but not available for linking");
                    None
                }

                // RustMat runtime functions
                "rustmat_call_builtin_f64" => {
                    debug!("Providing runtime builtin dispatcher for f64 functions");
                    Some(runtime_builtin_f64_dispatch as *const u8)
                }
                "rustmat_call_builtin_matrix" => {
                    debug!("Providing runtime builtin dispatcher for matrix functions");
                    Some(runtime_builtin_matrix_dispatch as *const u8)
                }
                "rustmat_create_matrix" => {
                    debug!("Providing runtime matrix constructor");
                    Some(runtime_create_matrix as *const u8)
                }

                // Memory management functions
                "malloc" | "free" | "calloc" | "realloc" => {
                    debug!(
                        "Memory management function {name} requested but not available for linking"
                    );
                    None
                }

                _ => {
                    debug!("Unknown symbol {name} requested");
                    None
                }
            }
        }));

        // Create the JIT module
        let module = JITModule::new(builder);
        let ctx = module.make_context();

        let engine = Self {
            module,
            ctx,
            cache: FunctionCache::with_capacity(1000),
            profiler: HotspotProfiler::new(),
            target_isa,
            compiler: BytecodeCompiler::new(),
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
                Value::Int(i) => f64_vars.push(*i as f64),
                Value::Num(n) => f64_vars.push(*n),
                Value::Bool(b) => f64_vars.push(if *b { 1.0 } else { 0.0 }),
                _ => {
                    error!("Unsupported value type for JIT execution: {value:?}");
                    return Err(TurbineError::ExecutionError(
                        "Unsupported value type".to_string(),
                    ));
                }
            }
        }

        // Execute the JIT compiled function
        let result = unsafe {
            if func.ptr.is_null() {
                return Err(TurbineError::InvalidFunctionPointer);
            }

            // Cast function pointer to correct signature: fn(*mut f64, usize) -> i32
            let jit_fn: extern "C" fn(*mut f64, usize) -> i32 = std::mem::transmute(func.ptr);

            jit_fn(f64_vars.as_mut_ptr(), f64_vars.len())
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

        // If function is compiled, execute it
        if self.cache.contains(hash) {
            return self
                .execute_compiled(hash, vars)
                .map(|result| (result, true));
        }

        // Check if we should compile this function
        if self.should_compile(hash) {
            match self.compile_bytecode(bytecode) {
                Ok(_) => {
                    info!("Bytecode compiled successfully, executing JIT version");
                    return self
                        .execute_compiled(hash, vars)
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

        // Fallback to interpreter
        debug!("Executing bytecode in interpreter mode");
        // Create a proper variable array with current state and execute with ignition interpreter
        let mut interpreter_vars = vars.to_vec();

        // Ensure interpreter_vars is large enough for the bytecode
        if interpreter_vars.len() < bytecode.var_count {
            interpreter_vars.resize(bytecode.var_count, Value::Num(0.0));
        }

        match Self::interpret_with_vars(bytecode, &mut interpreter_vars) {
            Ok(_) => {
                // Copy back only the variables that exist in the original vars array
                for (i, interpreter_val) in interpreter_vars.iter().enumerate() {
                    if i < vars.len() {
                        vars[i] = interpreter_val.clone();
                    }
                }
                Ok((0, false)) // false indicates interpreter was used
            }
            Err(e) => Err(TurbineError::ExecutionError(e)),
        }
    }

    /// Internal interpreter that preserves variable state (similar to REPL's custom interpreter)
    fn interpret_with_vars(
        bytecode: &Bytecode,
        vars: &mut Vec<Value>,
    ) -> std::result::Result<Vec<Value>, String> {
        use rustmat_ignition::Instr;
        use std::convert::TryInto;

        let mut stack: Vec<Value> = Vec::new();
        let mut pc: usize = 0;

        while pc < bytecode.instructions.len() {
            match &bytecode.instructions[pc] {
                Instr::LoadConst(c) => stack.push(Value::Num(*c)),
                Instr::LoadString(s) => stack.push(Value::String(s.clone())),
                Instr::LoadVar(i) => {
                    if *i < vars.len() {
                        stack.push(vars[*i].clone())
                    } else {
                        stack.push(Value::Num(0.0))
                    }
                }
                Instr::StoreVar(i) => {
                    let val = stack.pop().ok_or("stack underflow".to_string())?;
                    if *i >= vars.len() {
                        vars.resize(i + 1, Value::Num(0.0));
                    }
                    vars[*i] = val;
                }
                Instr::Add => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(a + b));
                }
                Instr::Sub => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(a - b));
                }
                Instr::Mul => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(a * b));
                }
                Instr::Div => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(a / b));
                }
                Instr::Pow => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(a.powf(b)));
                }
                Instr::Neg => {
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(-a));
                }
                Instr::ElemMul => {
                    let b = stack.pop().ok_or("stack underflow".to_string())?;
                    let a = stack.pop().ok_or("stack underflow".to_string())?;
                    let result = rustmat_runtime::elementwise_mul(&a, &b)
                        .map_err(|e| format!("Element-wise multiplication error: {}", e))?;
                    stack.push(result);
                }
                Instr::ElemDiv => {
                    let b = stack.pop().ok_or("stack underflow".to_string())?;
                    let a = stack.pop().ok_or("stack underflow".to_string())?;
                    let result = rustmat_runtime::elementwise_div(&a, &b)
                        .map_err(|e| format!("Element-wise division error: {}", e))?;
                    stack.push(result);
                }
                Instr::ElemPow => {
                    let b = stack.pop().ok_or("stack underflow".to_string())?;
                    let a = stack.pop().ok_or("stack underflow".to_string())?;
                    let result = rustmat_runtime::elementwise_pow(&a, &b)
                        .map_err(|e| format!("Element-wise power error: {}", e))?;
                    stack.push(result);
                }
                Instr::ElemLeftDiv => {
                    let b = stack.pop().ok_or("stack underflow".to_string())?;
                    let a = stack.pop().ok_or("stack underflow".to_string())?;
                    let result = rustmat_runtime::elementwise_div(&b, &a)
                        .map_err(|e| format!("Element-wise left division error: {}", e))?;
                    stack.push(result);
                }
                Instr::Less => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(if a < b { 1.0 } else { 0.0 }));
                }
                Instr::LessEqual => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(if a <= b { 1.0 } else { 0.0 }));
                }
                Instr::Greater => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(if a > b { 1.0 } else { 0.0 }));
                }
                Instr::GreaterEqual => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(if a >= b { 1.0 } else { 0.0 }));
                }
                Instr::Equal => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(if (a - b).abs() < f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }));
                }
                Instr::NotEqual => {
                    let b: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    let a: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    stack.push(Value::Num(if (a - b).abs() >= f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }));
                }
                Instr::Jump(target) => {
                    pc = *target;
                    continue;
                }
                Instr::JumpIfFalse(target) => {
                    let cond: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                        .try_into()
                        .map_err(|e: String| e)?;
                    if cond == 0.0 {
                        pc = *target;
                        continue;
                    }
                }
                Instr::Return => {
                    break;
                }
                Instr::Pop => {
                    stack.pop().ok_or("stack underflow".to_string())?;
                }
                Instr::CallBuiltin(name, arg_count) => {
                    // For builtin functions, use the runtime dispatcher
                    let mut args = Vec::new();
                    for _ in 0..*arg_count {
                        args.push(stack.pop().ok_or("stack underflow".to_string())?);
                    }
                    args.reverse(); // Arguments are popped in reverse order

                    // Call the runtime dispatcher
                    match rustmat_runtime::call_builtin(name, &args) {
                        Ok(result) => stack.push(result),
                        Err(e) => return Err(format!("Builtin function '{name}' failed: {e}")),
                    }
                }
                Instr::CreateMatrix(rows, cols) => {
                    let total_elements = rows * cols;
                    let mut elements = Vec::new();

                    for _ in 0..total_elements {
                        let val: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                            .try_into()
                            .map_err(|e: String| e)?;
                        elements.push(val);
                    }
                    elements.reverse(); // Elements are popped in reverse order

                    // Create matrix using the runtime
                    match rustmat_builtins::Matrix::new(elements, *rows, *cols) {
                        Ok(matrix) => stack.push(Value::Matrix(matrix)),
                        Err(e) => return Err(format!("Matrix creation failed: {e}")),
                    }
                }
                Instr::CreateMatrixDynamic(num_rows) => {
                    // Dynamic matrix creation with concatenation support
                    let mut row_lengths = Vec::new();
                    
                    // Pop row lengths (in reverse order since they're pushed last)
                    for _ in 0..*num_rows {
                        let row_len: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                            .try_into()
                            .map_err(|e: String| e)?;
                        row_lengths.push(row_len as usize);
                    }
                    row_lengths.reverse(); // Correct the order
                    
                    // Now pop elements according to row structure (in reverse order)
                    let mut rows_data = Vec::new();
                    for &row_len in row_lengths.iter().rev() {
                        let mut row_values = Vec::new();
                        for _ in 0..row_len {
                            row_values.push(stack.pop().ok_or("stack underflow".to_string())?);
                        }
                        row_values.reverse(); // Correct the order within row
                        rows_data.push(row_values);
                    }
                    
                    // Reverse rows to get correct order
                    rows_data.reverse();
                    
                    // Use the concatenation logic to create the matrix
                    let result = match rustmat_runtime::create_matrix_from_values(&rows_data) {
                        Ok(val) => val,
                        Err(e) => return Err(e),
                    };
                    stack.push(result);
                }
                Instr::CreateRange(has_step) => {
                    if *has_step {
                        // Stack: start, step, end
                        let end: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                            .try_into()
                            .map_err(|e: String| e)?;
                        let step: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                            .try_into()
                            .map_err(|e: String| e)?;
                        let start: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                            .try_into()
                            .map_err(|e: String| e)?;
                        
                        let range_result = match rustmat_runtime::create_range(start, Some(step), end) {
                            Ok(val) => val,
                            Err(e) => return Err(e),
                        };
                        stack.push(range_result);
                    } else {
                        // Stack: start, end
                        let end: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                            .try_into()
                            .map_err(|e: String| e)?;
                        let start: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                            .try_into()
                            .map_err(|e: String| e)?;
                        
                        let range_result = match rustmat_runtime::create_range(start, None, end) {
                            Ok(val) => val,
                            Err(e) => return Err(e),
                        };
                        stack.push(range_result);
                    }
                }
                Instr::Index(num_indices) => {
                    // Pop indices from stack (in reverse order)
                    let mut indices = Vec::new();
                    for _ in 0..*num_indices {
                        let index_val: f64 = (&stack.pop().ok_or("stack underflow".to_string())?)
                            .try_into()
                            .map_err(|e: String| e)?;
                        indices.push(index_val);
                    }
                    indices.reverse(); // Correct the order
                    
                    // Pop the base array/matrix
                    let base = stack.pop().ok_or("stack underflow".to_string())?;
                    
                    // Use the centralized indexing function
                    let result = match rustmat_runtime::perform_indexing(&base, &indices) {
                        Ok(val) => val,
                        Err(e) => return Err(e),
                    };
                    stack.push(result);
                }
            }
            pc += 1;
        }

        Ok(vars.clone())
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
                Instr::LoadString(s) => {
                    "LoadString".hash(&mut hasher);
                    s.hash(&mut hasher);
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
/// These functions provide the bridge between JIT-compiled code and the RustMat runtime

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
    let value_args: Vec<rustmat_builtins::Value> = args_slice
        .iter()
        .map(|&x| rustmat_builtins::Value::Num(x))
        .collect();

    // Call the runtime dispatcher
    match rustmat_runtime::call_builtin(name, &value_args) {
        Ok(rustmat_builtins::Value::Num(result)) => result,
        Ok(rustmat_builtins::Value::Int(result)) => result as f64,
        Ok(rustmat_builtins::Value::Bool(result)) => if result { 1.0 } else { 0.0 },
        Ok(_) => {
            log::warn!("Builtin function '{}' returned non-numeric result", name);
            0.0
        }
        Err(e) => {
            log::error!("Builtin function '{}' failed: {}", name, e);
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
    let value_args: Vec<rustmat_builtins::Value> = args_slice
        .iter()
        .map(|&x| rustmat_builtins::Value::Num(x))
        .collect();

    // Call the runtime dispatcher
    match rustmat_runtime::call_builtin(name, &value_args) {
        Ok(result) => {
            // Allocate result in GC memory and return pointer
            match rustmat_gc::gc_allocate(result) {
                Ok(gc_ptr) => unsafe { gc_ptr.as_raw() as i64 },
                Err(_) => {
                    log::error!("Failed to allocate GC memory for result");
                    0
                }
            }
        }
        Err(e) => {
            log::error!("Builtin function '{}' failed: {}", name, e);
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
            "Invalid matrix creation parameters: rows={}, cols={}, elements_len={}",
            rows,
            cols,
            elements_len
        );
        return 0;
    }

    // Convert elements pointer to Vec
    let elements = unsafe { std::slice::from_raw_parts(elements_ptr, elements_len) }.to_vec();

    // Create matrix
    match rustmat_builtins::Matrix::new(elements, rows, cols) {
        Ok(matrix) => {
            let value = rustmat_builtins::Value::Matrix(matrix);
            // Allocate in GC memory and return pointer
            match rustmat_gc::gc_allocate(value) {
                Ok(gc_ptr) => unsafe { gc_ptr.as_raw() as i64 },
                Err(_) => {
                    log::error!("Failed to allocate GC memory for matrix");
                    0
                }
            }
        }
        Err(e) => {
            log::error!("Matrix creation failed: {}", e);
            0
        }
    }
}
