//! Bytecode to Cranelift IR Compiler
//!
//! Translates RustMat bytecode instructions into Cranelift intermediate representation
//! for subsequent compilation to native machine code.
//!
//! The JIT compiler generates code that works with boxed Value objects to maintain
//! compatibility with the runtime type system.

use cranelift::prelude::*;
use rustmat_ignition::Instr;
use crate::{Result, TurbineError};
use std::collections::HashMap;

/// Stack simulation for tracking values during compilation  
/// Values are represented as pointers to boxed Value objects
#[derive(Debug)]
struct StackSimulator {
    values: Vec<Value>,
    variables: HashMap<usize, Variable>,
}

impl StackSimulator {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            variables: HashMap::new(),
        }
    }

    fn push(&mut self, value: Value) {
        self.values.push(value);
    }

    fn pop(&mut self) -> Result<Value> {
        self.values.pop()
            .ok_or_else(|| TurbineError::ModuleError("Stack underflow during compilation".to_string()))
    }

    fn pop_two(&mut self) -> Result<(Value, Value)> {
        let b = self.pop()?;
        let a = self.pop()?;
        Ok((a, b))
    }

    fn get_variable(&self, idx: usize) -> Option<Variable> {
        self.variables.get(&idx).copied()
    }

    fn set_variable(&mut self, idx: usize, var: Variable) {
        self.variables.insert(idx, var);
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.values.len()
    }
}

/// Compiles bytecode instructions to Cranelift IR using runtime Value objects
pub struct BytecodeCompiler {
    builder_context: FunctionBuilderContext,
}

impl BytecodeCompiler {
    pub fn new() -> Self {
        Self {
            builder_context: FunctionBuilderContext::new(),
        }
    }
    
    /// Compile a sequence of bytecode instructions to Cranelift IR
    /// Function signature: fn(*mut Value, usize) -> i32
    /// Args: variables_array, var_count
    pub fn compile_instructions(
        &mut self,
        instructions: &[Instr],
        func: &mut codegen::ir::Function,
        var_count: usize,
    ) -> Result<()> {
        let mut builder = FunctionBuilder::new(func, &mut self.builder_context);
        
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        
        // Function parameters:
        // - variables: *mut Value (pointer to variables array)
        // - var_count: usize
        let vars_ptr = builder.block_params(entry_block)[0]; // *mut Value
        let _vars_len = builder.block_params(entry_block)[1]; // usize
        
        // Initialize stack simulator and variables
        let mut stack = StackSimulator::new();
        Self::initialize_variables_static(&mut builder, &mut stack, var_count);
        
        // For now, implement a simplified linear execution model
        // TODO: Add proper control flow support
        Self::compile_instruction_sequence_static(&mut builder, &mut stack, instructions, vars_ptr)?;
        
        builder.seal_all_blocks();
        builder.finalize();
        
        Ok(())
    }

    fn initialize_variables_static(builder: &mut FunctionBuilder, stack: &mut StackSimulator, var_count: usize) {
        // Create variables for each RustMat variable
        // Variables are pointers to Value objects
        for i in 0..var_count {
            let var = Variable::new(i);
            builder.declare_var(var, types::I64); // Pointer type
            
            // Initialize with null pointer (will be set from runtime)
            let null_ptr = builder.ins().iconst(types::I64, 0);
            builder.def_var(var, null_ptr);
            
            stack.set_variable(i, var);
        }
    }
    
    fn compile_instruction_sequence_static(
        builder: &mut FunctionBuilder,
        stack: &mut StackSimulator,
        instructions: &[Instr],
        _vars_ptr: Value,
    ) -> Result<bool> {
        // Simplified approach: JIT compile straight-line code, fall back for control flow
        
        // Check if this bytecode contains control flow - if so, indicate we should fall back
        for instr in instructions {
            match instr {
                Instr::Jump(_) | Instr::JumpIfFalse(_) => {
                    // For now, don't JIT compile control flow - fall back to interpreter
                    return Err(TurbineError::ModuleError("Control flow not yet supported in JIT".to_string()));
                }
                _ => {}
            }
        }
        
        // Process straight-line code (no control flow)
        for instr in instructions {
            match instr {
                Instr::LoadConst(val) => {
                    let const_ptr = Self::create_value_num(builder, *val)?;
                    stack.push(const_ptr);
                }
                Instr::LoadVar(idx) => {
                    if let Some(var) = stack.get_variable(*idx) {
                        let val = builder.use_var(var);
                        stack.push(val);
                    } else {
                        return Err(TurbineError::ModuleError(format!("Variable {} not found", idx)));
                    }
                }
                Instr::StoreVar(idx) => {
                    let val = stack.pop()?;
                    if let Some(var) = stack.get_variable(*idx) {
                        builder.def_var(var, val);
                    } else {
                        return Err(TurbineError::ModuleError(format!("Variable {} not found", idx)));
                    }
                }
                Instr::Add => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_add(builder, a, b)?;
                    stack.push(result);
                }
                Instr::Sub => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_sub(builder, a, b)?;
                    stack.push(result);
                }
                Instr::Mul => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_mul(builder, a, b)?;
                    stack.push(result);
                }
                Instr::Div => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_div(builder, a, b)?;
                    stack.push(result);
                }
                Instr::Pow => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_pow(builder, a, b)?;
                    stack.push(result);
                }
                Instr::Neg => {
                    let val = stack.pop()?;
                    let result = Self::call_runtime_neg(builder, val)?;
                    stack.push(result);
                }
                // Comparison operations  
                Instr::LessEqual => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_le(builder, a, b)?;
                    stack.push(result);
                }
                Instr::Less => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_lt(builder, a, b)?;
                    stack.push(result);
                }
                Instr::Greater => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_gt(builder, a, b)?;
                    stack.push(result);
                }
                Instr::GreaterEqual => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_ge(builder, a, b)?;
                    stack.push(result);
                }
                Instr::Equal => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_eq(builder, a, b)?;
                    stack.push(result);
                }
                Instr::NotEqual => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_ne(builder, a, b)?;
                    stack.push(result);
                }
                Instr::CallBuiltin(name, arg_count) => {
                    let mut args = Vec::new();
                    for _ in 0..*arg_count {
                        args.push(stack.pop()?);
                    }
                    args.reverse();
                    
                    let result = Self::call_runtime_builtin(builder, name, &args)?;
                    stack.push(result);
                }
                Instr::CreateMatrix(rows, cols) => {
                    let total_elements = rows * cols;
                    let mut elements = Vec::new();
                    
                    for _ in 0..total_elements {
                        elements.push(stack.pop()?);
                    }
                    elements.reverse();
                    
                    let result = Self::call_runtime_create_matrix(builder, *rows, *cols, &elements)?;
                    stack.push(result);
                }
                Instr::Pop => {
                    stack.pop()?;
                }
                Instr::Return => {
                    let zero = builder.ins().iconst(types::I32, 0);
                    builder.ins().return_(&[zero]);
                    return Ok(true); // Terminated early
                }
                // Control flow should have been caught above
                Instr::Jump(_) | Instr::JumpIfFalse(_) => {
                    unreachable!("Control flow should have been caught in pre-check");
                }
            }
        }
        
        // If we reach here without explicit return, return success
        let zero = builder.ins().iconst(types::I32, 0);
        builder.ins().return_(&[zero]);
        
        Ok(false)
    }


    // Runtime interface functions - these will call into the Rust runtime
    
    fn create_value_num(builder: &mut FunctionBuilder, val: f64) -> Result<Value> {
        // Call runtime function to create Value::Num(val)
        // For now, return a dummy pointer - this needs runtime support
        let dummy_ptr = builder.ins().iconst(types::I64, val.to_bits() as i64);
        Ok(dummy_ptr)
    }
    
    fn call_runtime_add(builder: &mut FunctionBuilder, a: Value, b: Value) -> Result<Value> {
        // Call runtime addition function
        // For now, return dummy - needs runtime support
        let dummy_ptr = builder.ins().iadd(a, b);
        Ok(dummy_ptr)
    }
    
    fn call_runtime_sub(builder: &mut FunctionBuilder, a: Value, b: Value) -> Result<Value> {
        let dummy_ptr = builder.ins().isub(a, b);
        Ok(dummy_ptr)
    }
    
    fn call_runtime_mul(builder: &mut FunctionBuilder, a: Value, b: Value) -> Result<Value> {
        let dummy_ptr = builder.ins().imul(a, b);
        Ok(dummy_ptr)
    }
    
    fn call_runtime_div(builder: &mut FunctionBuilder, a: Value, b: Value) -> Result<Value> {
        // Integer division for now - needs proper runtime call
        let dummy_ptr = builder.ins().udiv(a, b);
        Ok(dummy_ptr)
    }
    
    fn call_runtime_pow(builder: &mut FunctionBuilder, a: Value, b: Value) -> Result<Value> {
        // Power operation - needs runtime support
        let dummy_ptr = builder.ins().imul(a, b); // Placeholder
        Ok(dummy_ptr)
    }
    
    fn call_runtime_neg(builder: &mut FunctionBuilder, val: Value) -> Result<Value> {
        let dummy_ptr = builder.ins().ineg(val);
        Ok(dummy_ptr)
    }
    
    fn call_runtime_le(builder: &mut FunctionBuilder, a: Value, b: Value) -> Result<Value> {
        let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, a, b);
        let result = builder.ins().uextend(types::I64, cmp);
        Ok(result)
    }
    
    fn call_runtime_lt(builder: &mut FunctionBuilder, a: Value, b: Value) -> Result<Value> {
        let cmp = builder.ins().icmp(IntCC::SignedLessThan, a, b);
        let result = builder.ins().uextend(types::I64, cmp);
        Ok(result)
    }
    
    fn call_runtime_gt(builder: &mut FunctionBuilder, a: Value, b: Value) -> Result<Value> {
        let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, a, b);
        let result = builder.ins().uextend(types::I64, cmp);
        Ok(result)
    }
    
    fn call_runtime_ge(builder: &mut FunctionBuilder, a: Value, b: Value) -> Result<Value> {
        let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, a, b);
        let result = builder.ins().uextend(types::I64, cmp);
        Ok(result)
    }
    
    fn call_runtime_eq(builder: &mut FunctionBuilder, a: Value, b: Value) -> Result<Value> {
        let cmp = builder.ins().icmp(IntCC::Equal, a, b);
        let result = builder.ins().uextend(types::I64, cmp);
        Ok(result)
    }
    
    fn call_runtime_ne(builder: &mut FunctionBuilder, a: Value, b: Value) -> Result<Value> {
        let cmp = builder.ins().icmp(IntCC::NotEqual, a, b);
        let result = builder.ins().uextend(types::I64, cmp);
        Ok(result)
    }
    
    fn call_runtime_builtin(builder: &mut FunctionBuilder, _name: &str, _args: &[Value]) -> Result<Value> {
        // Call runtime builtin function by name
        // For now, return dummy - needs runtime support
        let dummy_ptr = builder.ins().iconst(types::I64, 0);
        Ok(dummy_ptr)
    }
    
    fn call_runtime_create_matrix(builder: &mut FunctionBuilder, _rows: usize, _cols: usize, _elements: &[Value]) -> Result<Value> {
        // Call runtime matrix creation function
        // For now, return dummy - needs runtime support
        let dummy_ptr = builder.ins().iconst(types::I64, 0);
        Ok(dummy_ptr)
    }
}

impl Default for BytecodeCompiler {
    fn default() -> Self {
        Self::new()
    }
}



/// Optimization level for JIT compilation
#[derive(Debug, Clone, Copy, Default)]
pub enum OptimizationLevel {
    None,
    #[default]
    Fast,
    Aggressive,
}

/// Configuration for JIT compilation
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    pub optimization_level: OptimizationLevel,
    pub enable_profiling: bool,
    pub max_inline_depth: u32,
    pub enable_bounds_checking: bool,
    pub enable_overflow_checks: bool,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Fast,
            enable_profiling: true,
            max_inline_depth: 3,
            enable_bounds_checking: true,
            enable_overflow_checks: true,
        }
    }
} 