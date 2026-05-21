//! RunMat Turbine - Cranelift-based JIT Compiler
//!
//! The optimizing tier of RunMat's V8-inspired tiered execution model.
//! Turbine compiles hot bytecode sequences from the VM into native machine code
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
use runmat_builtins::Value;
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_vm::{ArgSpec, Bytecode, Instr, InterpreterOutcome, SemanticFunctionRegistry};
use std::cell::Cell;
use std::env;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::task::Context;

use target_lexicon::Triple;
use thiserror::Error;
use tracing::info_span;

pub mod cache;
pub mod compiler;
pub mod jit_memory;
pub mod profiler;
pub mod value_abi;

pub use cache::*;
pub use compiler::*;
pub use jit_memory::*;
pub use profiler::HotspotProfiler;
pub use value_abi::{TurbineArgSpec, TurbineValue, TurbineValueTag};

const JIT_FALLBACK_STACK_BYTES: usize = 16 * 1024 * 1024;
const JIT_FALLBACK_STACK_ENV: &str = "RUNMAT_TURBINE_STACK_MB";

fn run_immediate<F: Future>(mut future: Pin<Box<F>>) -> Result<F::Output> {
    let stack_bytes = env::var(JIT_FALLBACK_STACK_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|mb| *mb > 0)
        .map(|mb| mb.saturating_mul(1024 * 1024))
        .unwrap_or(JIT_FALLBACK_STACK_BYTES);
    stacker::grow(stack_bytes, || {
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
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
    })
}

struct RuntimeContext {
    semantic_registry: SemanticFunctionRegistry,
}

impl RuntimeContext {
    fn new(semantic_registry: SemanticFunctionRegistry) -> Self {
        Self { semantic_registry }
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

fn declare_host_semantic_call_in_module(module: &mut JITModule) -> FuncId {
    let mut sig = module.make_signature();
    let pointer_type = module.isa().pointer_type();
    sig.params.push(AbiParam::new(types::I64)); // function id
    sig.params.push(AbiParam::new(pointer_type)); // args_ptr
    sig.params.push(AbiParam::new(types::I32)); // args_len
    sig.params.push(AbiParam::new(pointer_type)); // result_ptr
    sig.returns.push(AbiParam::new(types::I32)); // status

    module
        .declare_function("runmat_call_semantic_function", Linkage::Import, &sig)
        .expect("Failed to declare runmat_call_semantic_function")
}

fn declare_host_semantic_call_outputs_in_module(module: &mut JITModule) -> FuncId {
    let mut sig = module.make_signature();
    let pointer_type = module.isa().pointer_type();
    sig.params.push(AbiParam::new(types::I64)); // function id
    sig.params.push(AbiParam::new(pointer_type)); // args_ptr
    sig.params.push(AbiParam::new(types::I32)); // args_len
    sig.params.push(AbiParam::new(types::I32)); // out_count
    sig.params.push(AbiParam::new(pointer_type)); // results_ptr
    sig.returns.push(AbiParam::new(types::I32)); // status

    module
        .declare_function(
            "runmat_call_semantic_function_outputs",
            Linkage::Import,
            &sig,
        )
        .expect("Failed to declare runmat_call_semantic_function_outputs")
}

fn declare_host_semantic_value_call_in_module(module: &mut JITModule) -> FuncId {
    let mut sig = module.make_signature();
    let pointer_type = module.isa().pointer_type();
    sig.params.push(AbiParam::new(types::I64)); // function id
    sig.params.push(AbiParam::new(pointer_type)); // args_ptr: TurbineValue[]
    sig.params.push(AbiParam::new(types::I32)); // args_len
    sig.params.push(AbiParam::new(pointer_type)); // result_ptr: TurbineValue*
    sig.returns.push(AbiParam::new(types::I32)); // status

    module
        .declare_function("runmat_call_semantic_function_value", Linkage::Import, &sig)
        .expect("Failed to declare runmat_call_semantic_function_value")
}

fn declare_host_semantic_value_outputs_in_module(module: &mut JITModule) -> FuncId {
    let mut sig = module.make_signature();
    let pointer_type = module.isa().pointer_type();
    sig.params.push(AbiParam::new(types::I64)); // function id
    sig.params.push(AbiParam::new(pointer_type)); // args_ptr: TurbineValue[]
    sig.params.push(AbiParam::new(types::I32)); // args_len
    sig.params.push(AbiParam::new(types::I32)); // out_count
    sig.params.push(AbiParam::new(pointer_type)); // results_ptr: TurbineValue[]
    sig.returns.push(AbiParam::new(types::I32)); // status

    module
        .declare_function(
            "runmat_call_semantic_function_values",
            Linkage::Import,
            &sig,
        )
        .expect("Failed to declare runmat_call_semantic_function_values")
}

fn declare_host_semantic_expanded_value_call_in_module(module: &mut JITModule) -> FuncId {
    let mut sig = module.make_signature();
    let pointer_type = module.isa().pointer_type();
    sig.params.push(AbiParam::new(types::I64)); // function id
    sig.params.push(AbiParam::new(pointer_type)); // args_ptr: TurbineValue[]
    sig.params.push(AbiParam::new(types::I32)); // args_len
    sig.params.push(AbiParam::new(pointer_type)); // specs_ptr: TurbineArgSpec[]
    sig.params.push(AbiParam::new(types::I32)); // specs_len
    sig.params.push(AbiParam::new(pointer_type)); // result_ptr: TurbineValue*
    sig.returns.push(AbiParam::new(types::I32)); // status

    module
        .declare_function(
            "runmat_call_semantic_function_expanded_value",
            Linkage::Import,
            &sig,
        )
        .expect("Failed to declare runmat_call_semantic_function_expanded_value")
}

fn declare_host_semantic_expanded_value_outputs_in_module(module: &mut JITModule) -> FuncId {
    let mut sig = module.make_signature();
    let pointer_type = module.isa().pointer_type();
    sig.params.push(AbiParam::new(types::I64)); // function id
    sig.params.push(AbiParam::new(pointer_type)); // args_ptr: TurbineValue[]
    sig.params.push(AbiParam::new(types::I32)); // args_len
    sig.params.push(AbiParam::new(pointer_type)); // specs_ptr: TurbineArgSpec[]
    sig.params.push(AbiParam::new(types::I32)); // specs_len
    sig.params.push(AbiParam::new(types::I32)); // out_count
    sig.params.push(AbiParam::new(pointer_type)); // results_ptr: TurbineValue[]
    sig.returns.push(AbiParam::new(types::I32)); // status

    module
        .declare_function(
            "runmat_call_semantic_function_expanded_values",
            Linkage::Import,
            &sig,
        )
        .expect("Failed to declare runmat_call_semantic_function_expanded_values")
}

fn declare_host_feval_expanded_value_outputs_in_module(module: &mut JITModule) -> FuncId {
    let mut sig = module.make_signature();
    let pointer_type = module.isa().pointer_type();
    sig.params.push(AbiParam::new(pointer_type)); // args_ptr: TurbineValue[] (func + raw args)
    sig.params.push(AbiParam::new(types::I32)); // args_len
    sig.params.push(AbiParam::new(pointer_type)); // specs_ptr: TurbineArgSpec[]
    sig.params.push(AbiParam::new(types::I32)); // specs_len
    sig.params.push(AbiParam::new(types::I32)); // out_count
    sig.params.push(AbiParam::new(pointer_type)); // results_ptr: TurbineValue[]
    sig.returns.push(AbiParam::new(types::I32)); // status

    module
        .declare_function("runmat_call_feval_expanded_values", Linkage::Import, &sig)
        .expect("Failed to declare runmat_call_feval_expanded_values")
}

fn declare_host_builtin_expanded_value_outputs_in_module(module: &mut JITModule) -> FuncId {
    let mut sig = module.make_signature();
    let pointer_type = module.isa().pointer_type();
    sig.params.push(AbiParam::new(pointer_type)); // name_ptr: u8*
    sig.params.push(AbiParam::new(types::I32)); // name_len
    sig.params.push(AbiParam::new(pointer_type)); // args_ptr: TurbineValue[]
    sig.params.push(AbiParam::new(types::I32)); // args_len
    sig.params.push(AbiParam::new(pointer_type)); // specs_ptr: TurbineArgSpec[]
    sig.params.push(AbiParam::new(types::I32)); // specs_len
    sig.params.push(AbiParam::new(types::I32)); // out_count
    sig.params.push(AbiParam::new(pointer_type)); // results_ptr: TurbineValue[]
    sig.returns.push(AbiParam::new(types::I32)); // status

    module
        .declare_function("runmat_call_builtin_expanded_values", Linkage::Import, &sig)
        .expect("Failed to declare runmat_call_builtin_expanded_values")
}

/// The main JIT compilation engine
pub struct TurbineEngine {
    module: JITModule,
    ctx: codegen::Context,
    cache: FunctionCache,
    profiler: HotspotProfiler,
    target_isa: codegen::isa::OwnedTargetIsa,
    compiler: BytecodeCompiler,
    runmat_call_semantic_function_id: FuncId,
    runmat_call_semantic_function_outputs_id: FuncId,
    runmat_call_semantic_function_value_id: FuncId,
    runmat_call_semantic_function_values_id: FuncId,
    runmat_call_semantic_function_expanded_value_id: FuncId,
    runmat_call_semantic_function_expanded_values_id: FuncId,
    runmat_call_feval_expanded_values_id: FuncId,
    runmat_call_builtin_expanded_values_id: FuncId,
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
            "runmat_call_semantic_function",
            runmat_call_semantic_function as *const u8,
        );
        builder.symbol(
            "runmat_call_semantic_function_outputs",
            runmat_call_semantic_function_outputs as *const u8,
        );
        builder.symbol(
            "runmat_call_semantic_function_value",
            runmat_call_semantic_function_value as *const u8,
        );
        builder.symbol(
            "runmat_call_semantic_function_values",
            runmat_call_semantic_function_values as *const u8,
        );
        builder.symbol(
            "runmat_call_semantic_function_expanded_value",
            runmat_call_semantic_function_expanded_value as *const u8,
        );
        builder.symbol(
            "runmat_call_semantic_function_expanded_values",
            runmat_call_semantic_function_expanded_values as *const u8,
        );
        builder.symbol(
            "runmat_call_feval_expanded_values",
            runmat_call_feval_expanded_values as *const u8,
        );
        builder.symbol(
            "runmat_call_builtin_expanded_values",
            runmat_call_builtin_expanded_values as *const u8,
        );

        // Create the JIT module
        let mut module = JITModule::new(builder);

        // Declare the external function on the module using the expert's pattern
        let runmat_call_semantic_function_id = declare_host_semantic_call_in_module(&mut module);
        let runmat_call_semantic_function_outputs_id =
            declare_host_semantic_call_outputs_in_module(&mut module);
        let runmat_call_semantic_function_value_id =
            declare_host_semantic_value_call_in_module(&mut module);
        let runmat_call_semantic_function_values_id =
            declare_host_semantic_value_outputs_in_module(&mut module);
        let runmat_call_semantic_function_expanded_value_id =
            declare_host_semantic_expanded_value_call_in_module(&mut module);
        let runmat_call_semantic_function_expanded_values_id =
            declare_host_semantic_expanded_value_outputs_in_module(&mut module);
        let runmat_call_feval_expanded_values_id =
            declare_host_feval_expanded_value_outputs_in_module(&mut module);
        let runmat_call_builtin_expanded_values_id =
            declare_host_builtin_expanded_value_outputs_in_module(&mut module);

        let ctx = module.make_context();

        let engine = Self {
            module,
            ctx,
            cache: FunctionCache::with_capacity(1000),
            profiler: HotspotProfiler::new(),
            target_isa,
            compiler: BytecodeCompiler::new(),
            runmat_call_semantic_function_id,
            runmat_call_semantic_function_outputs_id,
            runmat_call_semantic_function_value_id,
            runmat_call_semantic_function_values_id,
            runmat_call_semantic_function_expanded_value_id,
            runmat_call_semantic_function_expanded_values_id,
            runmat_call_feval_expanded_values_id,
            runmat_call_builtin_expanded_values_id,
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
            &bytecode.semantic_registry(),
            &mut self.module,
            self.runmat_call_semantic_function_id,
            self.runmat_call_semantic_function_outputs_id,
            self.runmat_call_semantic_function_value_id,
            self.runmat_call_semantic_function_values_id,
            self.runmat_call_semantic_function_expanded_value_id,
            self.runmat_call_semantic_function_expanded_values_id,
            self.runmat_call_feval_expanded_values_id,
            self.runmat_call_builtin_expanded_values_id,
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
        self.execute_compiled_with_registry(hash, vars, &SemanticFunctionRegistry::default())
    }

    /// Execute compiled function with access to semantic function identities for user calls.
    pub fn execute_compiled_with_registry(
        &mut self,
        hash: u64,
        vars: &mut [Value],
        semantic_registry: &SemanticFunctionRegistry,
    ) -> Result<i32> {
        self.execute_compiled_with_function_products(hash, vars, semantic_registry)
    }

    fn execute_compiled_with_function_products(
        &mut self,
        hash: u64,
        vars: &mut [Value],
        semantic_registry: &SemanticFunctionRegistry,
    ) -> Result<i32> {
        let func = self
            .cache
            .get(hash)
            .ok_or(TurbineError::FunctionNotFound(hash))?;

        debug!("Executing compiled function {hash}");

        let mut turbine_vars: Vec<TurbineValue> = vars
            .iter()
            .cloned()
            .map(TurbineValue::from_runtime_value)
            .collect::<Result<Vec<_>>>()?;

        // Set up runtime context for user function calls
        let runtime_context = RuntimeContext::new(semantic_registry.clone());
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

            // Cast function pointer to correct signature: fn(*mut TurbineValue, usize) -> i32
            let jit_fn: extern "C" fn(*mut TurbineValue, usize) -> i32 =
                std::mem::transmute(func.ptr);

            let exec_result = jit_fn(turbine_vars.as_mut_ptr(), turbine_vars.len());

            // Clear runtime context after execution
            clear_runtime_context();

            exec_result
        };

        for (i, turbine_value) in turbine_vars.iter().copied().enumerate() {
            if i < vars.len() {
                vars[i] = turbine_value.to_runtime_value()?;
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
        let semantic_registry = bytecode.semantic_registry();

        // If function is compiled, execute it with function definitions
        if self.cache.contains(hash) {
            return self
                .execute_compiled_with_function_products(hash, vars, &semantic_registry)
                .map(|result| (result, true));
        }

        // Check if we should compile this function
        if self.should_compile(hash) {
            match self.compile_bytecode(bytecode) {
                Ok(_) => {
                    info!("Bytecode compiled successfully, executing JIT version");
                    return self
                        .execute_compiled_with_function_products(hash, vars, &semantic_registry)
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

        // Fallback to the main VM interpreter which supports all features
        debug!("Executing bytecode in VM interpreter mode (supports user functions)");

        // Use the main VM interpreter which has full feature support
        match run_immediate(Box::pin(runmat_vm::interpret_with_vars(
            bytecode,
            vars,
            Some("<main>"),
        )))? {
            Ok(InterpreterOutcome::Completed(_)) => Ok((0, false)),

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

    fn hash_named_function_call<H: Hasher>(
        hasher: &mut H,
        discriminator: &str,
        identity: &runmat_hir::CallableIdentity,
        fallback_policy: runmat_hir::CallableFallbackPolicy,
        argc: usize,
        out_count: Option<usize>,
    ) {
        discriminator.hash(hasher);
        identity.hash(hasher);
        fallback_policy.hash(hasher);
        argc.hash(hasher);
        if let Some(out_count) = out_count {
            out_count.hash(hasher);
        }
    }

    fn hash_arg_specs<H: Hasher>(hasher: &mut H, specs: &[ArgSpec]) {
        specs.len().hash(hasher);
        for spec in specs {
            spec.is_expand.hash(hasher);
            spec.num_indices.hash(hasher);
            spec.expand_all.hash(hasher);
        }
    }

    /// Calculate a hash for bytecode instructions
    pub fn calculate_bytecode_hash(&self, bytecode: &Bytecode) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();

        // Hash the instructions and variable count
        bytecode.var_count.hash(&mut hasher);
        if let Some(layout) = &bytecode.layout {
            let mut functions: Vec<_> = layout.functions.iter().collect();
            functions.sort_by_key(|(id, _)| id.0);
            for (id, function) in functions {
                "layout_function".hash(&mut hasher);
                id.0.hash(&mut hasher);
                function.local_count.hash(&mut hasher);
                function.display_name.hash(&mut hasher);

                let mut binding_slots: Vec<_> = function.binding_slots.iter().collect();
                binding_slots.sort_by_key(|(binding, _)| binding.0);
                for (binding, slot) in binding_slots {
                    binding.0.hash(&mut hasher);
                    slot.0.hash(&mut hasher);
                }

                let mut mir_local_slots: Vec<_> = function.mir_local_slots.iter().collect();
                mir_local_slots.sort_by_key(|(local, _)| local.0);
                for (local, slot) in mir_local_slots {
                    local.0.hash(&mut hasher);
                    slot.0.hash(&mut hasher);
                }
            }

            let mut entrypoints: Vec<_> = layout.entrypoints.iter().collect();
            entrypoints.sort_by_key(|(id, _)| id.0);
            for (id, entrypoint) in entrypoints {
                "layout_entrypoint".hash(&mut hasher);
                id.0.hash(&mut hasher);
                entrypoint.target.0.hash(&mut hasher);
                for export in &entrypoint.exports {
                    export.binding.0.hash(&mut hasher);
                    export.name.hash(&mut hasher);
                    export.slot.0.hash(&mut hasher);
                }
            }
        }
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
                Instr::RightDiv => "RightDiv".hash(&mut hasher),
                Instr::LeftDiv => "LeftDiv".hash(&mut hasher),
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
                Instr::LogicalNot => "LogicalNot".hash(&mut hasher),
                Instr::LogicalAnd => "LogicalAnd".hash(&mut hasher),
                Instr::LogicalOr => "LogicalOr".hash(&mut hasher),
                Instr::JumpIfFalse(target) => {
                    "JumpIfFalse".hash(&mut hasher);
                    target.hash(&mut hasher);
                }
                Instr::Jump(target) => {
                    "Jump".hash(&mut hasher);
                    target.hash(&mut hasher);
                }
                Instr::Pop => "Pop".hash(&mut hasher),
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
                Instr::CallFunctionMulti {
                    identity,
                    fallback_policy,
                    arg_count,
                    out_count,
                } => {
                    Self::hash_named_function_call(
                        &mut hasher,
                        "CallFunctionMulti",
                        identity,
                        *fallback_policy,
                        *arg_count,
                        Some(*out_count),
                    );
                }
                Instr::CallSemanticFunctionMulti(function, argc, out_count) => {
                    "CallSemanticFunctionMulti".hash(&mut hasher);
                    function.0.hash(&mut hasher);
                    argc.hash(&mut hasher);
                    out_count.hash(&mut hasher);
                }
                Instr::CallSemanticFunctionExpandMultiOutput(function, specs, out_count) => {
                    "CallSemanticFunctionExpandMultiOutput".hash(&mut hasher);
                    function.0.hash(&mut hasher);
                    Self::hash_arg_specs(&mut hasher, specs);
                    out_count.hash(&mut hasher);
                }
                Instr::Unpack(count) => {
                    "Unpack".hash(&mut hasher);
                    count.hash(&mut hasher);
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
                Instr::IndexCell { num_indices: k, .. } => {
                    "IndexCell".hash(&mut hasher);
                    k.hash(&mut hasher);
                }
                Instr::LoadStaticProperty(class, prop) => {
                    "LoadStaticProperty".hash(&mut hasher);
                    class.hash(&mut hasher);
                    prop.hash(&mut hasher);
                }
                Instr::EnterTry(catch_pc, catch_var) => {
                    "EnterTry".hash(&mut hasher);
                    catch_pc.hash(&mut hasher);
                    catch_var.hash(&mut hasher);
                }
                Instr::PopTry => {
                    "PopTry".hash(&mut hasher);
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

/// Runtime function implementations for JIT-compiled code.
/// These functions bridge semantic bytecode identities into the RunMat runtime.
#[no_mangle]
pub extern "C" fn runmat_call_semantic_function(
    function_id: i64,
    args_ptr: *const f64,
    args_len: i32,
    result_ptr: *mut f64,
) -> i32 {
    if result_ptr.is_null() {
        error!("Null result pointer passed to runmat_call_semantic_function");
        return 1;
    }

    let args_slice = if args_len > 0 {
        if args_ptr.is_null() {
            error!("Null args pointer passed to runmat_call_semantic_function");
            return 1;
        }
        unsafe { std::slice::from_raw_parts(args_ptr, args_len as usize) }
    } else {
        &[]
    };

    let context = match get_runtime_context() {
        Some(ctx) => ctx,
        None => {
            error!("No runtime context available for semantic function call");
            return 1;
        }
    };

    let function_id = match usize::try_from(function_id) {
        Ok(function_id) => function_id,
        Err(_) => {
            error!("Invalid semantic function id: {function_id}");
            return 1;
        }
    };
    let args: Vec<Value> = args_slice.iter().map(|value| Value::Num(*value)).collect();
    let output = run_immediate(Box::pin(runmat_vm::invoke_semantic_function_value(
        function_id,
        &args,
        1,
        &context.semantic_registry,
    )))
    .and_then(|result| result.map_err(TurbineError::ExecutionError));

    let output = match output {
        Ok(result) => result,
        Err(err) => {
            error!("Semantic function execution failed: {err}");
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
            error!("Semantic function returned unsupported value: {output:?}");
            return 1;
        }
    };

    unsafe {
        *result_ptr = output_value;
    }

    0
}

#[no_mangle]
pub extern "C" fn runmat_call_semantic_function_outputs(
    function_id: i64,
    args_ptr: *const f64,
    args_len: i32,
    out_count: i32,
    results_ptr: *mut f64,
) -> i32 {
    if results_ptr.is_null() {
        error!("Null results pointer passed to runmat_call_semantic_function_outputs");
        return 1;
    }
    if out_count < 0 {
        error!("Invalid output count passed to runmat_call_semantic_function_outputs");
        return 1;
    }

    let args_slice = if args_len > 0 {
        if args_ptr.is_null() {
            error!("Null args pointer passed to runmat_call_semantic_function_outputs");
            return 1;
        }
        unsafe { std::slice::from_raw_parts(args_ptr, args_len as usize) }
    } else {
        &[]
    };

    let context = match get_runtime_context() {
        Some(ctx) => ctx,
        None => {
            error!("No runtime context available for semantic function outputs call");
            return 1;
        }
    };

    let function_id = match usize::try_from(function_id) {
        Ok(function_id) => function_id,
        Err(_) => {
            error!("Invalid semantic function id: {function_id}");
            return 1;
        }
    };
    let args: Vec<Value> = args_slice.iter().map(|value| Value::Num(*value)).collect();
    let out_count = out_count as usize;
    let output = run_immediate(Box::pin(runmat_vm::invoke_semantic_function_value(
        function_id,
        &args,
        out_count,
        &context.semantic_registry,
    )))
    .and_then(|result| result.map_err(TurbineError::ExecutionError));

    let outputs = match output {
        Ok(Value::OutputList(values)) => values,
        Ok(value) => vec![value],
        Err(err) => {
            error!("Semantic function outputs execution failed: {err}");
            return 1;
        }
    };

    for i in 0..out_count {
        let output_value = match outputs.get(i).cloned().unwrap_or(Value::Num(0.0)) {
            Value::Num(val) => val,
            Value::Int(val) => val.to_f64(),
            Value::Bool(val) => {
                if val {
                    1.0
                } else {
                    0.0
                }
            }
            other => {
                error!("Semantic function returned unsupported output value: {other:?}");
                return 1;
            }
        };

        unsafe {
            *results_ptr.add(i) = output_value;
        }
    }

    0
}

fn read_turbine_value_args(args_ptr: *const TurbineValue, args_len: i32) -> Result<Vec<Value>> {
    if args_len < 0 {
        return Err(execution_error("negative TurbineValue argument count"));
    }
    if args_len == 0 {
        return Ok(Vec::new());
    }
    if args_ptr.is_null() {
        return Err(execution_error("null TurbineValue args pointer"));
    }

    unsafe { std::slice::from_raw_parts(args_ptr, args_len as usize) }
        .iter()
        .map(|value| value.to_runtime_value())
        .collect()
}

fn read_turbine_arg_specs(
    specs_ptr: *const TurbineArgSpec,
    specs_len: i32,
) -> Result<Vec<ArgSpec>> {
    if specs_len < 0 {
        return Err(execution_error("negative TurbineArgSpec count"));
    }
    if specs_len == 0 {
        return Ok(Vec::new());
    }
    if specs_ptr.is_null() {
        return Err(execution_error("null TurbineArgSpec pointer"));
    }

    unsafe { std::slice::from_raw_parts(specs_ptr, specs_len as usize) }
        .iter()
        .map(|spec| {
            Ok(ArgSpec {
                is_expand: spec.is_expand != 0,
                num_indices: spec.num_indices as usize,
                expand_all: spec.expand_all != 0,
            })
        })
        .collect()
}

fn row_major_pos_from_linear(cell: &runmat_builtins::CellArray, idx: usize) -> Result<usize> {
    if idx == 0 || idx > cell.data.len() {
        return Err(execution_error("Cell index out of bounds"));
    }
    if cell.rows <= 1 || cell.cols <= 1 {
        return Ok(idx - 1);
    }
    let zero = idx - 1;
    let row = zero % cell.rows;
    let col = zero / cell.rows;
    Ok(row * cell.cols + col)
}

fn turbine_index_cell_value(cell: &runmat_builtins::CellArray, indices: &[usize]) -> Result<Value> {
    match indices.len() {
        1 => Ok((*cell.data[row_major_pos_from_linear(cell, indices[0])?]).clone()),
        2 => {
            let row = indices[0];
            let col = indices[1];
            if row == 0 || row > cell.rows || col == 0 || col > cell.cols {
                return Err(execution_error("Cell subscript out of bounds"));
            }
            Ok((*cell.data[(row - 1) * cell.cols + (col - 1)]).clone())
        }
        _ => Err(execution_error("Unsupported number of cell indices")),
    }
}

fn turbine_expand_cell_indices(
    cell: &runmat_builtins::CellArray,
    indices: &[Value],
) -> Result<Vec<Value>> {
    match indices.len() {
        1 => match &indices[0] {
            Value::Num(n) if *n == 0.0 && n.is_sign_negative() => {
                Ok(vec![turbine_index_cell_value(cell, &[cell.data.len()])?])
            }
            Value::Num(n) if *n < 0.0 => {
                let idx = cell.data.len() as isize + *n as isize;
                if idx < 1 || idx as usize > cell.data.len() {
                    return Err(execution_error("Cell index out of bounds"));
                }
                Ok(vec![turbine_index_cell_value(cell, &[idx as usize])?])
            }
            Value::Num(n) => Ok(vec![turbine_index_cell_value(cell, &[*n as usize])?]),
            Value::Int(i) => Ok(vec![turbine_index_cell_value(
                cell,
                &[i.to_i64() as usize],
            )?]),
            Value::Tensor(t) => t
                .data
                .iter()
                .map(|&value| turbine_index_cell_value(cell, &[value as usize]))
                .collect(),
            _ => Err(execution_error("Unsupported cell index type")),
        },
        2 => {
            let row = value_to_usize_index(&indices[0])?;
            let col = value_to_usize_index(&indices[1])?;
            Ok(vec![turbine_index_cell_value(cell, &[row, col])?])
        }
        _ => Err(execution_error("Unsupported cell index type")),
    }
}

fn value_to_usize_index(value: &Value) -> Result<usize> {
    match value {
        Value::Num(value) => Ok(*value as usize),
        Value::Int(value) => Ok(value.to_i64() as usize),
        other => Err(execution_error(format!(
            "Unsupported cell index value: {other:?}"
        ))),
    }
}

fn expand_turbine_args(args: Vec<Value>, specs: &[ArgSpec]) -> Result<Vec<Value>> {
    let expected_args = specs.iter().fold(0usize, |count, spec| {
        count + 1 + if spec.is_expand { spec.num_indices } else { 0 }
    });
    if expected_args != args.len() {
        return Err(execution_error(format!(
            "expanded Turbine argument count mismatch: expected {expected_args}, got {}",
            args.len()
        )));
    }

    let mut cursor = 0;
    let mut expanded = Vec::new();
    for spec in specs {
        let value = args[cursor].clone();
        cursor += 1;
        if !spec.is_expand {
            expanded.push(value);
            continue;
        }

        let indices = args[cursor..cursor + spec.num_indices].to_vec();
        cursor += spec.num_indices;
        let values = if spec.expand_all {
            match value {
                Value::OutputList(values) => values,
                Value::Cell(cell) => (1..=cell.data.len())
                    .map(|idx| turbine_index_cell_value(&cell, &[idx]))
                    .collect::<Result<Vec<_>>>()?,
                other => {
                    return Err(execution_error(format!(
                        "expanded Turbine call requires cell or output list for expand_all, got {other:?}"
                    )))
                }
            }
        } else {
            match value {
                Value::Cell(cell) => turbine_expand_cell_indices(&cell, &indices)?,
                other => {
                    return Err(execution_error(format!(
                        "expanded Turbine call requires cell for indexed expansion, got {other:?}"
                    )))
                }
            }
        };
        expanded.extend(values);
    }
    Ok(expanded)
}

fn write_turbine_value(result_ptr: *mut TurbineValue, value: Value) -> Result<()> {
    if result_ptr.is_null() {
        return Err(execution_error("null TurbineValue result pointer"));
    }
    unsafe {
        *result_ptr = TurbineValue::from_runtime_value(value)?;
    }
    Ok(())
}

#[no_mangle]
pub extern "C" fn runmat_call_semantic_function_value(
    function_id: i64,
    args_ptr: *const TurbineValue,
    args_len: i32,
    result_ptr: *mut TurbineValue,
) -> i32 {
    let output = (|| -> Result<Value> {
        let context = get_runtime_context().ok_or_else(|| {
            execution_error("No runtime context available for TurbineValue semantic call")
        })?;
        let function_id = usize::try_from(function_id)
            .map_err(|_| execution_error(format!("Invalid semantic function id: {function_id}")))?;
        let args = read_turbine_value_args(args_ptr, args_len)?;
        run_immediate(Box::pin(runmat_vm::invoke_semantic_function_value(
            function_id,
            &args,
            1,
            &context.semantic_registry,
        )))?
        .map_err(TurbineError::ExecutionError)
    })();

    match output.and_then(|value| write_turbine_value(result_ptr, value)) {
        Ok(()) => 0,
        Err(err) => {
            error!("TurbineValue semantic function call failed: {err}");
            1
        }
    }
}

#[no_mangle]
pub extern "C" fn runmat_call_semantic_function_values(
    function_id: i64,
    args_ptr: *const TurbineValue,
    args_len: i32,
    out_count: i32,
    results_ptr: *mut TurbineValue,
) -> i32 {
    let requested_out_count = out_count;
    let output = (|| -> Result<Vec<Value>> {
        if out_count < 0 {
            return Err(execution_error("negative TurbineValue output count"));
        }
        if out_count > 0 && results_ptr.is_null() {
            return Err(execution_error("null TurbineValue results pointer"));
        }
        let context = get_runtime_context().ok_or_else(|| {
            execution_error("No runtime context available for TurbineValue semantic outputs call")
        })?;
        let function_id = usize::try_from(function_id)
            .map_err(|_| execution_error(format!("Invalid semantic function id: {function_id}")))?;
        let args = read_turbine_value_args(args_ptr, args_len)?;
        let out_count = out_count as usize;
        let output = run_immediate(Box::pin(runmat_vm::invoke_semantic_function_value(
            function_id,
            &args,
            out_count,
            &context.semantic_registry,
        )))?
        .map_err(TurbineError::ExecutionError)?;
        Ok(match output {
            Value::OutputList(values) => values,
            value => vec![value],
        })
    })();

    let outputs = match output {
        Ok(outputs) => outputs,
        Err(err) => {
            error!("TurbineValue semantic function outputs call failed: {err}");
            return 1;
        }
    };

    for index in 0..requested_out_count as usize {
        let value = outputs.get(index).cloned().unwrap_or(Value::Num(0.0));
        if let Err(err) = write_turbine_value(unsafe { results_ptr.add(index) }, value) {
            error!("TurbineValue semantic output write failed: {err}");
            return 1;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn runmat_call_semantic_function_expanded_value(
    function_id: i64,
    args_ptr: *const TurbineValue,
    args_len: i32,
    specs_ptr: *const TurbineArgSpec,
    specs_len: i32,
    result_ptr: *mut TurbineValue,
) -> i32 {
    let output = (|| -> Result<Value> {
        let context = get_runtime_context().ok_or_else(|| {
            execution_error("No runtime context available for expanded TurbineValue semantic call")
        })?;
        let function_id = usize::try_from(function_id)
            .map_err(|_| execution_error(format!("Invalid semantic function id: {function_id}")))?;
        let args = read_turbine_value_args(args_ptr, args_len)?;
        let specs = read_turbine_arg_specs(specs_ptr, specs_len)?;
        let expanded_args = expand_turbine_args(args, &specs)?;
        run_immediate(Box::pin(runmat_vm::invoke_semantic_function_value(
            function_id,
            &expanded_args,
            1,
            &context.semantic_registry,
        )))?
        .map_err(TurbineError::ExecutionError)
    })();

    match output.and_then(|value| write_turbine_value(result_ptr, value)) {
        Ok(()) => 0,
        Err(err) => {
            error!("Expanded TurbineValue semantic function call failed: {err}");
            1
        }
    }
}

#[no_mangle]
pub extern "C" fn runmat_call_semantic_function_expanded_values(
    function_id: i64,
    args_ptr: *const TurbineValue,
    args_len: i32,
    specs_ptr: *const TurbineArgSpec,
    specs_len: i32,
    out_count: i32,
    results_ptr: *mut TurbineValue,
) -> i32 {
    let requested_out_count = out_count;
    let output = (|| -> Result<Vec<Value>> {
        if out_count < 0 {
            return Err(execution_error("negative TurbineValue output count"));
        }
        if out_count > 0 && results_ptr.is_null() {
            return Err(execution_error("null TurbineValue results pointer"));
        }
        let context = get_runtime_context().ok_or_else(|| {
            execution_error(
                "No runtime context available for expanded TurbineValue semantic outputs call",
            )
        })?;
        let function_id = usize::try_from(function_id)
            .map_err(|_| execution_error(format!("Invalid semantic function id: {function_id}")))?;
        let args = read_turbine_value_args(args_ptr, args_len)?;
        let specs = read_turbine_arg_specs(specs_ptr, specs_len)?;
        let expanded_args = expand_turbine_args(args, &specs)?;
        let out_count = out_count as usize;
        let output = run_immediate(Box::pin(runmat_vm::invoke_semantic_function_value(
            function_id,
            &expanded_args,
            out_count,
            &context.semantic_registry,
        )))?
        .map_err(TurbineError::ExecutionError)?;
        Ok(match output {
            Value::OutputList(values) => values,
            value => vec![value],
        })
    })();

    let outputs = match output {
        Ok(outputs) => outputs,
        Err(err) => {
            error!("Expanded TurbineValue semantic function outputs call failed: {err}");
            return 1;
        }
    };

    for index in 0..requested_out_count as usize {
        let value = outputs.get(index).cloned().unwrap_or(Value::Num(0.0));
        if let Err(err) = write_turbine_value(unsafe { results_ptr.add(index) }, value) {
            error!("Expanded TurbineValue semantic output write failed: {err}");
            return 1;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn runmat_call_feval_expanded_values(
    args_ptr: *const TurbineValue,
    args_len: i32,
    specs_ptr: *const TurbineArgSpec,
    specs_len: i32,
    out_count: i32,
    results_ptr: *mut TurbineValue,
) -> i32 {
    let requested_out_count = out_count;
    let output = (|| -> Result<Vec<Value>> {
        if out_count < 0 {
            return Err(execution_error("negative TurbineValue output count"));
        }
        if out_count > 0 && results_ptr.is_null() {
            return Err(execution_error("null TurbineValue results pointer"));
        }

        let mut args = read_turbine_value_args(args_ptr, args_len)?;
        if args.is_empty() {
            return Err(execution_error(
                "expanded Turbine feval call requires callable argument",
            ));
        }
        let callable = args.remove(0);
        let specs = read_turbine_arg_specs(specs_ptr, specs_len)?;
        let expanded_args = expand_turbine_args(args, &specs)?;
        let output = run_immediate(Box::pin(runmat_runtime::call_feval_async_with_outputs(
            callable,
            &expanded_args,
            out_count as usize,
        )))?
        .map_err(TurbineError::ExecutionError)?;
        Ok(match output {
            Value::OutputList(values) => values,
            value => vec![value],
        })
    })();

    let outputs = match output {
        Ok(outputs) => outputs,
        Err(err) => {
            error!("Expanded Turbine feval outputs call failed: {err}");
            return 1;
        }
    };

    for index in 0..requested_out_count as usize {
        let value = outputs.get(index).cloned().unwrap_or(Value::Num(0.0));
        if let Err(err) = write_turbine_value(unsafe { results_ptr.add(index) }, value) {
            error!("Expanded Turbine feval output write failed: {err}");
            return 1;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn runmat_call_builtin_expanded_values(
    name_ptr: *const u8,
    name_len: i32,
    args_ptr: *const TurbineValue,
    args_len: i32,
    specs_ptr: *const TurbineArgSpec,
    specs_len: i32,
    out_count: i32,
    results_ptr: *mut TurbineValue,
) -> i32 {
    let requested_out_count = out_count;
    let output = (|| -> Result<Vec<Value>> {
        if name_len < 0 {
            return Err(execution_error("negative builtin name length"));
        }
        if name_ptr.is_null() {
            return Err(execution_error("null builtin name pointer"));
        }
        if out_count < 0 {
            return Err(execution_error("negative TurbineValue output count"));
        }
        if out_count > 0 && results_ptr.is_null() {
            return Err(execution_error("null TurbineValue results pointer"));
        }

        let name = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(name_ptr, name_len as usize))
                .map_err(|_| execution_error("invalid UTF-8 in builtin name"))?
        };
        let args = read_turbine_value_args(args_ptr, args_len)?;
        let specs = read_turbine_arg_specs(specs_ptr, specs_len)?;
        let expanded_args = expand_turbine_args(args, &specs)?;
        let output = run_immediate(Box::pin(runmat_runtime::call_builtin_async_with_outputs(
            name,
            &expanded_args,
            out_count as usize,
        )))?
        .map_err(TurbineError::ExecutionError)?;
        Ok(match output {
            Value::OutputList(values) => values,
            value => vec![value],
        })
    })();

    let outputs = match output {
        Ok(outputs) => outputs,
        Err(err) => {
            error!("Expanded Turbine builtin outputs call failed: {err}");
            return 1;
        }
    };

    for index in 0..requested_out_count as usize {
        let value = outputs.get(index).cloned().unwrap_or(Value::Num(0.0));
        if let Err(err) = write_turbine_value(unsafe { results_ptr.add(index) }, value) {
            error!("Expanded Turbine builtin output write failed: {err}");
            return 1;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_hir::FunctionId;
    use runmat_vm::SemanticFunctionBytecode;
    use std::collections::HashMap;

    fn install_semantic_context(functions: Vec<SemanticFunctionBytecode>) {
        let mut semantic_functions = HashMap::new();
        for function in functions {
            semantic_functions.insert(function.function, function);
        }
        let registry = SemanticFunctionRegistry::new(semantic_functions);
        let context = Box::leak(Box::new(RuntimeContext::new(registry)));
        set_runtime_context(context);
    }

    fn semantic_function(
        function: FunctionId,
        display_name: &str,
        instructions: Vec<Instr>,
        var_count: usize,
        input_slots: Vec<usize>,
        output_slots: Vec<usize>,
    ) -> SemanticFunctionBytecode {
        SemanticFunctionBytecode {
            function,
            display_name: display_name.to_string(),
            source_id: None,
            instructions,
            instr_spans: Vec::new(),
            call_arg_spans: Vec::new(),
            var_count,
            input_slots,
            varargin_slot: None,
            output_slots,
            varargout_slot: None,
            capture_slots: Vec::new(),
        }
    }

    #[test]
    fn named_function_hashing_stays_centralized() {
        let source = include_str!("lib.rs");
        let call_function_hash = ["\"CallFunction\"", ".hash"].concat();
        let call_function_multi_hash = ["\"CallFunctionMulti\"", ".hash"].concat();

        assert_eq!(source.matches(&call_function_hash).count(), 0);
        assert_eq!(source.matches(&call_function_multi_hash).count(), 0);
        assert_eq!(
            source
                .matches(&["Self::", "hash_named_function_call("].concat())
                .count(),
            1
        );
    }

    #[test]
    fn semantic_function_value_host_call_round_trips_scalar() {
        let function = FunctionId(1);
        install_semantic_context(vec![semantic_function(
            function,
            "inc",
            vec![
                Instr::LoadVar(0),
                Instr::LoadConst(1.0),
                Instr::Add,
                Instr::StoreVar(1),
            ],
            2,
            vec![0],
            vec![1],
        )]);

        let args = [TurbineValue::from_runtime_value(Value::Num(41.0)).unwrap()];
        let mut result = TurbineValue::empty();
        let status = runmat_call_semantic_function_value(
            function.0 as i64,
            args.as_ptr(),
            args.len() as i32,
            &mut result,
        );
        clear_runtime_context();

        assert_eq!(status, 0);
        assert_eq!(result.to_runtime_value().unwrap(), Value::Num(42.0));
    }

    #[test]
    fn semantic_function_value_host_call_round_trips_handle_value() {
        let function = FunctionId(2);
        install_semantic_context(vec![semantic_function(
            function,
            "label",
            vec![Instr::LoadString("ok".to_string()), Instr::StoreVar(0)],
            1,
            Vec::new(),
            vec![0],
        )]);

        let mut result = TurbineValue::empty();
        let status = runmat_call_semantic_function_value(
            function.0 as i64,
            std::ptr::null(),
            0,
            &mut result,
        );
        clear_runtime_context();

        assert_eq!(status, 0);
        assert_eq!(
            result.to_runtime_value().unwrap(),
            Value::String("ok".to_string())
        );
    }

    #[test]
    fn semantic_function_values_host_call_writes_multiple_outputs() {
        let function = FunctionId(3);
        install_semantic_context(vec![semantic_function(
            function,
            "pair",
            vec![
                Instr::LoadVar(0),
                Instr::StoreVar(1),
                Instr::LoadString("done".to_string()),
                Instr::StoreVar(2),
            ],
            3,
            vec![0],
            vec![1, 2],
        )]);

        let args = [TurbineValue::from_runtime_value(Value::Num(7.0)).unwrap()];
        let mut results = [TurbineValue::empty(), TurbineValue::empty()];
        let status = runmat_call_semantic_function_values(
            function.0 as i64,
            args.as_ptr(),
            args.len() as i32,
            results.len() as i32,
            results.as_mut_ptr(),
        );
        clear_runtime_context();

        assert_eq!(status, 0);
        assert_eq!(results[0].to_runtime_value().unwrap(), Value::Num(7.0));
        assert_eq!(
            results[1].to_runtime_value().unwrap(),
            Value::String("done".to_string())
        );
    }
}
