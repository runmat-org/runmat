//! Bytecode to Cranelift IR Compiler
//!
//! Translates RustMat bytecode instructions into Cranelift intermediate representation
//! for subsequent compilation to native machine code.

use cranelift::prelude::*;
use rustmat_ignition::Instr;
use crate::{Result, TurbineError};
use std::collections::{HashMap, BTreeSet};

/// Stack simulation for tracking values during compilation  
/// Values are represented as f64 values
#[derive(Debug, Clone)]
struct StackSimulator {
    values: Vec<Value>,
}

impl StackSimulator {
    fn new() -> Self {
        Self {
            values: Vec::new(),
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
}

/// Basic block information for control flow analysis
#[derive(Debug, Clone)]
struct BasicBlock {
    start_pc: usize,
    end_pc: usize,
    block: Block,
    predecessors: Vec<usize>,
    successors: Vec<usize>,
}

/// Control flow graph for bytecode
#[derive(Debug)]
struct ControlFlowGraph {
    blocks: HashMap<usize, BasicBlock>,
}

impl ControlFlowGraph {
    fn analyze(instructions: &[Instr]) -> Self {
        let mut block_starts = BTreeSet::new();
        block_starts.insert(0); // Entry point
        
        // Find all jump targets and block boundaries
        for (pc, instr) in instructions.iter().enumerate() {
            match instr {
                Instr::Jump(target) => {
                    block_starts.insert(*target);
                    // Instruction after jump starts new block (if reachable)
                    if pc + 1 < instructions.len() {
                        block_starts.insert(pc + 1);
                    }
                }
                Instr::JumpIfFalse(target) => {
                    block_starts.insert(*target);
                    // Fallthrough after conditional jump starts new block
                    if pc + 1 < instructions.len() {
                        block_starts.insert(pc + 1);
                    }
                }
                Instr::Return => {
                    // Instruction after return starts new block (if reachable)
                    if pc + 1 < instructions.len() {
                        block_starts.insert(pc + 1);
                    }
                }
                _ => {}
            }
        }
        
        // Create basic blocks
        let mut blocks = HashMap::new();
        let block_starts: Vec<usize> = block_starts.into_iter().collect();
        
        for (i, &start_pc) in block_starts.iter().enumerate() {
            let end_pc = block_starts.get(i + 1).copied().unwrap_or(instructions.len());
            
            blocks.insert(start_pc, BasicBlock {
                start_pc,
                end_pc,
                block: Block::new(0), // Will be set later
                predecessors: Vec::new(),
                successors: Vec::new(),
            });
        }
        
        // Analyze successors first
        let mut successors_map = HashMap::new();
        for (&start_pc, block) in blocks.iter() {
            let mut successors = Vec::new();
            let mut pc = start_pc;
            while pc < block.end_pc && pc < instructions.len() {
                match &instructions[pc] {
                    Instr::Jump(target) => {
                        if blocks.contains_key(target) {
                            successors.push(*target);
                        }
                        break; // End of block
                    }
                    Instr::JumpIfFalse(target) => {
                        if blocks.contains_key(target) {
                            successors.push(*target);
                        }
                        // Fallthrough
                        if pc + 1 < instructions.len() && blocks.contains_key(&(pc + 1)) {
                            successors.push(pc + 1);
                        }
                        break; // End of block
                    }
                    Instr::Return => {
                        break; // End of block
                    }
                    _ => {}
                }
                pc += 1;
            }
            
            // Implicit fallthrough to next block
            if pc < instructions.len() && pc == block.end_pc {
                if let Some(next_start) = block_starts.iter().find(|&&s| s > start_pc) {
                    successors.push(*next_start);
                }
            }
            
            successors_map.insert(start_pc, successors);
        }
        
        // Update blocks with successors
        for (&start_pc, block) in blocks.iter_mut() {
            if let Some(successors) = successors_map.get(&start_pc) {
                block.successors = successors.clone();
            }
        }
        
        // Set predecessors based on successors
        let successor_map: HashMap<usize, Vec<usize>> = blocks.iter()
            .map(|(&start, block)| (start, block.successors.clone()))
            .collect();
            
        for (&start_pc, successors) in successor_map.iter() {
            for &succ_pc in successors {
                if let Some(succ_block) = blocks.get_mut(&succ_pc) {
                    succ_block.predecessors.push(start_pc);
                }
            }
        }
        
        Self {
            blocks,
        }
    }
}

/// Compiles bytecode instructions to Cranelift IR
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
    pub fn compile_instructions(
        &mut self,
        instructions: &[Instr],
        func: &mut codegen::ir::Function,
        _var_count: usize,
    ) -> Result<()> {
        let mut builder = FunctionBuilder::new(func, &mut self.builder_context);
        
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        
        // Function parameters  
        let vars_ptr = builder.block_params(entry_block)[0]; // *mut f64 (not Value!)
        let vars_len = builder.block_params(entry_block)[1]; // usize
        
        // Initialize stack (no need for Cranelift variables since we use direct memory access)
        let mut stack = StackSimulator::new();
        
        // Analyze control flow
        let mut cfg = ControlFlowGraph::analyze(instructions);
        
        // Create Cranelift blocks for each basic block
        for (_, basic_block) in cfg.blocks.iter_mut() {
            if basic_block.start_pc == 0 {
                basic_block.block = entry_block;
            } else {
                basic_block.block = builder.create_block();
            }
        }
        
        // Compile with proper control flow
        let created_blocks = Self::compile_with_control_flow(&mut builder, &mut stack, instructions, vars_ptr, vars_len)?;
        
        // Seal all dynamically created blocks
        for block in created_blocks {
            builder.seal_block(block);
        }
        
        // Seal the entry block
        builder.seal_block(entry_block);
        
        builder.finalize();
        Ok(())
    }


    
    fn compile_with_control_flow(
        builder: &mut FunctionBuilder,
        stack: &mut StackSimulator,
        instructions: &[Instr],
        vars_ptr: Value,
        _vars_len: Value,
    ) -> Result<Vec<Block>> {
        let mut created_blocks = Vec::new();

        // Simple linear compilation - just execute instructions in order
        for instr in instructions {
            match instr {
                Instr::LoadConst(val) => {
                    // Create f64 constant and push to stack
                    let const_val = builder.ins().f64const(*val);
                    stack.push(const_val);
                }
                Instr::LoadVar(idx) => {
                    // Load f64 from vars_ptr[idx]
                    let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                    let element_size = builder.ins().iconst(types::I64, 8); // size of f64
                    let offset = builder.ins().imul(idx_val, element_size);
                    let var_addr = builder.ins().iadd(vars_ptr, offset);
                    
                    // Load f64 from memory
                    let val = builder.ins().load(types::F64, MemFlags::new(), var_addr, 0);
                    stack.push(val);
                }
                Instr::StoreVar(idx) => {
                    let val = stack.pop()?;
                    
                    // Store f64 to vars_ptr[idx]
                    let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                    let element_size = builder.ins().iconst(types::I64, 8); // size of f64
                    let offset = builder.ins().imul(idx_val, element_size);
                    let var_addr = builder.ins().iadd(vars_ptr, offset);
                    
                    // Store f64 to memory
                    builder.ins().store(MemFlags::new(), val, var_addr, 0);
                }
                Instr::Add => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_add_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Sub => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_sub_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Mul => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_mul_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Div => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_div_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Pow => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_pow_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Neg => {
                    let val = stack.pop()?;
                    let result = Self::call_runtime_neg_static(builder, val);
                    stack.push(result);
                }
                Instr::LessEqual => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_le_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Less => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_lt_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Greater => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_gt_static(builder, a, b);
                    stack.push(result);
                }
                Instr::GreaterEqual => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_ge_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Equal => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_eq_static(builder, a, b);
                    stack.push(result);
                }
                Instr::NotEqual => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_ne_static(builder, a, b);
                    stack.push(result);
                }
                Instr::CallBuiltin(name, arg_count) => {
                    let mut args = Vec::new();
                    for _ in 0..*arg_count {
                        args.push(stack.pop()?);
                    }
                    args.reverse();
                    
                    let result = Self::call_runtime_builtin_static(builder, name, &args);
                    stack.push(result);
                }
                Instr::CreateMatrix(rows, cols) => {
                    let total_elements = rows * cols;
                    let mut elements = Vec::new();
                    
                    for _ in 0..total_elements {
                        elements.push(stack.pop()?);
                    }
                    elements.reverse();
                    
                    let result = Self::call_runtime_create_matrix_static(builder, *rows, *cols, &elements);
                    stack.push(result);
                }
                Instr::Pop => {
                    stack.pop()?;
                }
                Instr::Return => {
                    let zero = builder.ins().iconst(types::I32, 0);
                    builder.ins().return_(&[zero]);
                    return Ok(created_blocks);
                }
                Instr::Jump(target) => {
                    // Create a block for the jump target
                    let target_block = builder.create_block();
                    created_blocks.push(target_block);
                    builder.ins().jump(target_block, &[]);
                    
                    // Switch to the target block and continue compilation from there
                    builder.switch_to_block(target_block);
                    
                    // Compile remaining instructions starting from the target
                    if *target < instructions.len() {
                        let mut remaining_blocks = Self::compile_remaining_from_with_blocks(builder, stack, instructions, *target, vars_ptr)?;
                        created_blocks.append(&mut remaining_blocks);
                        return Ok(created_blocks);
                    } else {
                        // Jump past end - just return
                        let zero = builder.ins().iconst(types::I32, 0);
                        builder.ins().return_(&[zero]);
                        return Ok(created_blocks);
                    }
                }
                Instr::JumpIfFalse(target) => {
                    let condition = stack.pop()?;
                    let zero = builder.ins().f64const(0.0);
                    let is_false = builder.ins().fcmp(FloatCC::Equal, condition, zero);
                    
                    // Create blocks for both branches
                    let false_block = builder.create_block();
                    let true_block = builder.create_block();
                    created_blocks.push(false_block);
                    created_blocks.push(true_block);
                    
                    // Conditional branch
                    builder.ins().brif(is_false, false_block, &[], true_block, &[]);
                    
                    // Compile false branch
                    builder.switch_to_block(false_block);
                    if *target < instructions.len() {
                        let mut false_blocks = Self::compile_remaining_from_with_blocks(builder, &mut stack.clone(), instructions, *target, vars_ptr)?;
                        created_blocks.append(&mut false_blocks);
                    } else {
                        let zero = builder.ins().iconst(types::I32, 0);
                        builder.ins().return_(&[zero]);
                    }
                    
                    // Compile true branch (fallthrough)
                    builder.switch_to_block(true_block);
                    // Continue with the next instruction after the current one
                    // This will be handled by the loop continuation
                }
            }
        }
        
        // If no explicit return, add one
        let zero = builder.ins().iconst(types::I32, 0);
        builder.ins().return_(&[zero]);
        
        Ok(created_blocks)
    }
    
    fn compile_remaining_from_with_blocks(
        builder: &mut FunctionBuilder,
        stack: &mut StackSimulator,
        instructions: &[Instr],
        start_index: usize,
        vars_ptr: Value,
    ) -> Result<Vec<Block>> {
        let mut created_blocks = Vec::new();
        // Compile instructions starting from start_index
        for (_i, instr) in instructions.iter().enumerate().skip(start_index) {
            match instr {
                Instr::LoadConst(val) => {
                    // Create f64 constant and push to stack
                    let const_val = builder.ins().f64const(*val);
                    stack.push(const_val);
                }
                Instr::LoadVar(idx) => {
                    // Load f64 from vars_ptr[idx]
                    let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                    let element_size = builder.ins().iconst(types::I64, 8); // size of f64
                    let offset = builder.ins().imul(idx_val, element_size);
                    let var_addr = builder.ins().iadd(vars_ptr, offset);
                    
                    // Load f64 from memory
                    let val = builder.ins().load(types::F64, MemFlags::new(), var_addr, 0);
                    stack.push(val);
                }
                Instr::StoreVar(idx) => {
                    let val = stack.pop()?;
                    
                    // Store f64 to vars_ptr[idx]
                    let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                    let element_size = builder.ins().iconst(types::I64, 8); // size of f64
                    let offset = builder.ins().imul(idx_val, element_size);
                    let var_addr = builder.ins().iadd(vars_ptr, offset);
                    
                    // Store f64 to memory
                    builder.ins().store(MemFlags::new(), val, var_addr, 0);
                }
                Instr::Add => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_add_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Sub => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_sub_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Mul => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_mul_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Div => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_div_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Pow => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_pow_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Neg => {
                    let val = stack.pop()?;
                    let result = Self::call_runtime_neg_static(builder, val);
                    stack.push(result);
                }
                Instr::LessEqual => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_le_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Less => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_lt_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Greater => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_gt_static(builder, a, b);
                    stack.push(result);
                }
                Instr::GreaterEqual => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_ge_static(builder, a, b);
                    stack.push(result);
                }
                Instr::Equal => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_eq_static(builder, a, b);
                    stack.push(result);
                }
                Instr::NotEqual => {
                    let (a, b) = stack.pop_two()?;
                    let result = Self::call_runtime_ne_static(builder, a, b);
                    stack.push(result);
                }
                Instr::CallBuiltin(name, arg_count) => {
                    let mut args = Vec::new();
                    for _ in 0..*arg_count {
                        args.push(stack.pop()?);
                    }
                    args.reverse();
                    
                    let result = Self::call_runtime_builtin_static(builder, name, &args);
                    stack.push(result);
                }
                Instr::CreateMatrix(rows, cols) => {
                    let total_elements = rows * cols;
                    let mut elements = Vec::new();
                    
                    for _ in 0..total_elements {
                        elements.push(stack.pop()?);
                    }
                    elements.reverse();
                    
                    let result = Self::call_runtime_create_matrix_static(builder, *rows, *cols, &elements);
                    stack.push(result);
                }
                Instr::Pop => {
                    stack.pop()?;
                }
                Instr::Return => {
                    let zero = builder.ins().iconst(types::I32, 0);
                    builder.ins().return_(&[zero]);
                    return Ok(created_blocks);
                }
                Instr::Jump(target) => {
                    let target_block = builder.create_block();
                    created_blocks.push(target_block);
                    builder.ins().jump(target_block, &[]);
                    builder.switch_to_block(target_block);
                    
                    if *target < instructions.len() {
                        let mut remaining_blocks = Self::compile_remaining_from_with_blocks(builder, stack, instructions, *target, vars_ptr)?;
                        created_blocks.append(&mut remaining_blocks);
                        return Ok(created_blocks);
                    } else {
                        let zero = builder.ins().iconst(types::I32, 0);
                        builder.ins().return_(&[zero]);
                        return Ok(created_blocks);
                    }
                }
                Instr::JumpIfFalse(target) => {
                    let condition = stack.pop()?;
                    let zero = builder.ins().f64const(0.0);
                    let is_false = builder.ins().fcmp(FloatCC::Equal, condition, zero);
                    
                    let false_block = builder.create_block();
                    let true_block = builder.create_block();
                    created_blocks.push(false_block);
                    created_blocks.push(true_block);
                    
                    builder.ins().brif(is_false, false_block, &[], true_block, &[]);
                    
                    // Compile false branch
                    builder.switch_to_block(false_block);
                    if *target < instructions.len() {
                        let mut false_blocks = Self::compile_remaining_from_with_blocks(builder, &mut stack.clone(), instructions, *target, vars_ptr)?;
                        created_blocks.append(&mut false_blocks);
                    } else {
                        let zero = builder.ins().iconst(types::I32, 0);
                        builder.ins().return_(&[zero]);
                    }
                    
                    // Compile true branch (continue with next instruction)
                    builder.switch_to_block(true_block);
                    // Continue with next iteration
                }
            }
        }
        
        // If we reach the end without return, add one
        let zero = builder.ins().iconst(types::I32, 0);
        builder.ins().return_(&[zero]);
        Ok(created_blocks)
    }

    
    // Runtime interface functions for f64 operations
    
    fn call_runtime_add_static(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        builder.ins().fadd(a, b)
    }
    
    fn call_runtime_sub_static(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        builder.ins().fsub(a, b)
    }
    
    fn call_runtime_mul_static(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        builder.ins().fmul(a, b)
    }
    
    fn call_runtime_div_static(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        builder.ins().fdiv(a, b)
    }
    
    fn call_runtime_pow_static(_builder: &mut FunctionBuilder, a: Value, _b: Value) -> Value {
        // Placeholder - needs proper implementation (could use libm pow)
        a
    }
    
    fn call_runtime_neg_static(builder: &mut FunctionBuilder, val: Value) -> Value {
        builder.ins().fneg(val)
    }
    
    fn call_runtime_le_static(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let cmp = builder.ins().fcmp(FloatCC::LessThanOrEqual, a, b);
        let one = builder.ins().f64const(1.0);
        let zero = builder.ins().f64const(0.0);
        builder.ins().select(cmp, one, zero)
    }
    
    fn call_runtime_lt_static(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let cmp = builder.ins().fcmp(FloatCC::LessThan, a, b);
        let one = builder.ins().f64const(1.0);
        let zero = builder.ins().f64const(0.0);
        builder.ins().select(cmp, one, zero)
    }
    
    fn call_runtime_gt_static(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let cmp = builder.ins().fcmp(FloatCC::GreaterThan, a, b);
        let one = builder.ins().f64const(1.0);
        let zero = builder.ins().f64const(0.0);
        builder.ins().select(cmp, one, zero)
    }
    
    fn call_runtime_ge_static(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let cmp = builder.ins().fcmp(FloatCC::GreaterThanOrEqual, a, b);
        let one = builder.ins().f64const(1.0);
        let zero = builder.ins().f64const(0.0);
        builder.ins().select(cmp, one, zero)
    }
    
    fn call_runtime_eq_static(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let cmp = builder.ins().fcmp(FloatCC::Equal, a, b);
        let one = builder.ins().f64const(1.0);
        let zero = builder.ins().f64const(0.0);
        builder.ins().select(cmp, one, zero)
    }
    
    fn call_runtime_ne_static(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let cmp = builder.ins().fcmp(FloatCC::NotEqual, a, b);
        let one = builder.ins().f64const(1.0);
        let zero = builder.ins().f64const(0.0);
        builder.ins().select(cmp, one, zero)
    }
    
    fn call_runtime_builtin_static(builder: &mut FunctionBuilder, _name: &str, _args: &[Value]) -> Value {
        builder.ins().iconst(types::I64, 0)
    }
    
    fn call_runtime_create_matrix_static(builder: &mut FunctionBuilder, _rows: usize, _cols: usize, _elements: &[Value]) -> Value {
        builder.ins().iconst(types::I64, 0)
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