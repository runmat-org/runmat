//! RustMat Turbine - Cranelift-based JIT Compiler
//!
//! The optimizing tier of RustMat's V8-inspired tiered execution model.
//! Turbine compiles hot bytecode sequences from Ignition into native machine code
//! using Cranelift for maximum performance.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use cranelift_native;
use log::{debug, info};
use rustmat_builtins::Value;
use rustmat_ignition::{Bytecode, Instr};
use std::collections::HashMap;
use target_lexicon;
use thiserror::Error;

pub mod compiler;
pub mod profiler;
pub mod cache;

pub use compiler::*;
pub use profiler::*;
pub use cache::*;

/// The main JIT compilation engine
pub struct TurbineEngine {
    module: JITModule,
    ctx: codegen::Context,
    compiled_functions: HashMap<u64, CompiledFunction>,
    profiler: HotspotProfiler,
    target_isa: codegen::isa::OwnedTargetIsa,
}

/// A compiled function ready for execution
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
        
        // Get the native target triple and ISA
        let target_triple = target_lexicon::Triple::host();
        
        info!("Target triple: {}", target_triple);
        
        // Create ISA with proper flags for the target
        let mut flags_builder = cranelift_codegen::settings::builder();
        
        // Configure optimization level
        match config.optimization_level {
            OptimizationLevel::None => {
                flags_builder.set("opt_level", "none").unwrap();
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
        flags_builder.set("use_colocated_libcalls", "false").unwrap();
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
            debug!("Symbol lookup requested for: {}", name);
            None // Return None for now - can be extended for specific runtime needs
        }));
        
        // Create the JIT module
        let module = JITModule::new(builder);
        let ctx = module.make_context();
        
        let engine = Self {
            module,
            ctx,
            compiled_functions: HashMap::new(),
            profiler: HotspotProfiler::new(),
            target_isa,
        };
        
        info!("Turbine JIT engine initialized successfully for {}", target_triple);
        Ok(engine)
    }
    
    /// Check if the current platform supports JIT compilation
    pub fn is_jit_supported() -> bool {
        let triple = target_lexicon::Triple::host();
        // Check if we support this target
        matches!(triple.architecture, 
            target_lexicon::Architecture::X86_64 | 
            target_lexicon::Architecture::Aarch64(_))
    }
    
    /// Get target information
    pub fn target_info(&self) -> String {
        format!("Target ISA: {}", self.target_isa.triple())
    }
    
    /// Check if bytecode should be compiled based on profiling data
    pub fn should_compile(&mut self, bytecode_hash: u64) -> bool {
        self.profiler.record_execution(bytecode_hash);
        self.profiler.is_hot(bytecode_hash)
    }
    
    /// Compile bytecode to native machine code
    pub fn compile_bytecode(&mut self, bytecode: &Bytecode) -> Result<u64> {
        let hash = self.calculate_bytecode_hash(bytecode);
        
        if self.compiled_functions.contains_key(&hash) {
            debug!("Function already compiled: {}", hash);
            return Ok(hash);
        }
        
        info!("Compiling hot bytecode to native code: {}", hash);
        
        // Create function signature
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::I64)); // vars array pointer
        sig.params.push(AbiParam::new(types::I64)); // vars length
        sig.returns.push(AbiParam::new(types::I32)); // execution result
        
        // Create function
        let func_id = self.module
            .declare_function("jit_func", Linkage::Local, &sig)
            .map_err(|e| TurbineError::ModuleError(e.to_string()))?;
        
        // Compile bytecode to Cranelift IR
        let mut func = codegen::ir::Function::with_name_signature(
            codegen::ir::UserFuncName::user(0, 0),
            sig.clone(),
        );
        
        self.compile_bytecode_to_ir(bytecode, &mut func)?;
        
        // Compile to machine code
        self.ctx.func = func;
        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| TurbineError::ModuleError(e.to_string()))?;
        
        self.module.finalize_definitions()
            .map_err(|e| TurbineError::ModuleError(e.to_string()))?;
        
        // Get function pointer
        let ptr = self.module.get_finalized_function(func_id);
        
        let compiled_func = CompiledFunction {
            ptr,
            signature: sig,
            hotness: self.profiler.get_hotness(hash),
        };
        
        self.compiled_functions.insert(hash, compiled_func);
        
        info!("Successfully compiled function {}", hash);
        Ok(hash)
    }
    
    /// Execute compiled function
    pub fn execute_compiled(&self, hash: u64, _vars: &mut [Value]) -> Result<i32> {
        let _func = self.compiled_functions
            .get(&hash)
            .ok_or(TurbineError::FunctionNotFound(hash))?;
        
        // For now, this is a placeholder - actual execution would involve
        // converting Value array to appropriate native representation
        debug!("Executing compiled function {}", hash);
        
        // This would be the actual JIT function call:
        // let result = unsafe { std::mem::transmute::<_, fn(*mut Value, usize) -> i32>(func.ptr)(vars.as_mut_ptr(), vars.len()) };
        
        Ok(0) // Placeholder
    }
    
    /// Get compilation statistics
    pub fn stats(&self) -> TurbineStats {
        TurbineStats {
            compiled_functions: self.compiled_functions.len(),
            total_compilations: self.profiler.total_executions(),
            hottest_functions: self.profiler.get_hottest_functions(5),
        }
    }
    
    pub fn calculate_bytecode_hash(&self, bytecode: &Bytecode) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        format!("{:?}", bytecode.instructions).hash(&mut hasher);
        hasher.finish()
    }
    
    fn compile_bytecode_to_ir(&mut self, _bytecode: &Bytecode, func: &mut codegen::ir::Function) -> Result<()> {
        // This is where we'll translate bytecode instructions to Cranelift IR  
        // For now, just create a basic function that returns 0
        let mut builder_context = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(func, &mut builder_context);
        
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        
        // Return 0 for now
        let zero = builder.ins().iconst(types::I32, 0);
        builder.ins().return_(&[zero]);
        
        builder.seal_all_blocks();
        builder.finalize();
        
        Ok(())
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
}

// Make compiled functions safe to send between threads
unsafe impl Send for CompiledFunction {}
unsafe impl Sync for CompiledFunction {} 