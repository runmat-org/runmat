//! SSA Builder: Convert Ignition bytecode to SSA IR
//!
//! Simulates the stack symbolically to reconstruct SSA values from
//! stack-based bytecode.

use crate::ssa::{
    BlockId, CmpOp, EffectKind, SsaFunc, SsaInstr, SsaOp, SsaType, SsaValue, Terminator,
};
use runmat_ignition::Instr;
use std::collections::{HashMap, HashSet};

/// Build SSA IR from Ignition bytecode
pub struct SsaBuilder {
    func: SsaFunc,
    /// Symbolic stack (values, not runtime data)
    stack: Vec<SsaValue>,
    /// Current block being built
    current_block: BlockId,
    /// PC -> BlockId mapping for jump targets
    pc_to_block: HashMap<usize, BlockId>,
    /// Variable slot -> current SSA value (for load/store tracking)
    var_values: HashMap<usize, SsaValue>,
}

impl SsaBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        let mut func = SsaFunc::new(name);
        let entry = func.new_block();
        func.entry = entry;

        SsaBuilder {
            func,
            stack: Vec::new(),
            current_block: entry,
            pc_to_block: HashMap::new(),
            var_values: HashMap::new(),
        }
    }

    /// Build SSA from bytecode instructions
    pub fn build(mut self, instructions: &[Instr], _var_count: usize) -> SsaFunc {
        // Phase 1: Find basic block leaders (jump targets)
        let leaders = find_leaders(instructions);

        // Create blocks for all leaders
        for &pc in &leaders {
            if pc != 0 {
                let block_id = self.func.new_block();
                self.pc_to_block.insert(pc, block_id);
            }
        }
        self.pc_to_block.insert(0, self.func.entry);

        // Phase 2: Translate instructions
        for (pc, instr) in instructions.iter().enumerate() {
            // Start new block if this is a leader
            if leaders.contains(&pc) && pc != 0 {
                // Terminate previous block with fall-through
                let target = *self.pc_to_block.get(&pc).unwrap();
                self.terminate_with_br(target);
                self.current_block = target;
                self.stack.clear(); // Stack doesn't persist across blocks in our model
            }

            self.translate_instr(pc, instr, instructions.len());
        }

        // Terminate final block if not already terminated
        self.ensure_terminated();

        self.func
    }

    fn translate_instr(&mut self, pc: usize, instr: &Instr, _total_len: usize) {
        match instr {
            // Constants
            Instr::LoadConst(val) => {
                let v = self.emit(SsaOp::ConstF64(*val), SsaType::F64);
                self.stack.push(v);
            }
            Instr::LoadBool(val) => {
                let v = self.emit(SsaOp::ConstBool(*val), SsaType::Bool);
                self.stack.push(v);
            }
            Instr::LoadString(_s) => {
                // Strings go through runtime
                let v = self.emit(
                    SsaOp::CallRuntime {
                        func: "__load_string".to_string(),
                        args: vec![], // String data handled separately
                    },
                    SsaType::Ptr,
                );
                self.stack.push(v);
            }

            // Variable access
            Instr::LoadVar(idx) => {
                // Get pointer to variable slot
                let ptr = self.emit(SsaOp::VarPtr(*idx), SsaType::Ptr);
                let v = self.emit(SsaOp::Load(ptr), SsaType::F64);
                self.stack.push(v);
            }
            Instr::StoreVar(idx) => {
                let val = self.pop();
                let ptr = self.emit(SsaOp::VarPtr(*idx), SsaType::Ptr);
                self.emit(SsaOp::Store(ptr, val), SsaType::F64);
                self.var_values.insert(*idx, val);
            }

            // Arithmetic (binary)
            Instr::Add => self.binary_op(SsaOp::Add),
            Instr::Sub => self.binary_op(SsaOp::Sub),
            Instr::Mul => self.binary_op(SsaOp::Mul),
            Instr::Div => self.binary_op(SsaOp::Div),
            Instr::Pow => self.binary_op(SsaOp::Pow),

            // Unary
            Instr::Neg => self.unary_op(SsaOp::Neg),
            Instr::UPlus => {
                // No-op, value stays on stack
            }

            // Element-wise
            Instr::ElemMul => self.binary_op(SsaOp::ElemMul),
            Instr::ElemDiv => self.binary_op(SsaOp::ElemDiv),
            Instr::ElemPow => self.binary_op(SsaOp::ElemPow),
            Instr::ElemLeftDiv => {
                // a .\ b = b ./ a
                self.binary_op(|a, b| SsaOp::ElemDiv(b, a));
            }

            // Comparisons
            Instr::Less => self.binary_op(|a, b| SsaOp::Cmp(CmpOp::Lt, a, b)),
            Instr::LessEqual => self.binary_op(|a, b| SsaOp::Cmp(CmpOp::Le, a, b)),
            Instr::Greater => self.binary_op(|a, b| SsaOp::Cmp(CmpOp::Gt, a, b)),
            Instr::GreaterEqual => self.binary_op(|a, b| SsaOp::Cmp(CmpOp::Ge, a, b)),
            Instr::Equal => self.binary_op(|a, b| SsaOp::Cmp(CmpOp::Eq, a, b)),
            Instr::NotEqual => self.binary_op(|a, b| SsaOp::Cmp(CmpOp::Ne, a, b)),

            // Control flow
            Instr::Jump(target) => {
                let target_block = self.get_or_create_block(*target);
                self.set_terminator(Terminator::Br {
                    target: target_block,
                    args: vec![],
                });
            }
            Instr::JumpIfFalse(target) => {
                let cond = self.pop();
                let else_block = self.get_or_create_block(*target);
                let then_block = self.get_or_create_block(pc + 1);
                self.set_terminator(Terminator::Cbr {
                    cond,
                    then_block,
                    then_args: vec![],
                    else_block,
                    else_args: vec![],
                });
            }

            // Short-circuit logicals
            Instr::AndAnd(target) => {
                // If top is false, jump to target; else continue
                let lhs = self.pop();
                let else_block = self.get_or_create_block(*target);
                let then_block = self.get_or_create_block(pc + 1);
                self.set_terminator(Terminator::Cbr {
                    cond: lhs,
                    then_block,
                    then_args: vec![],
                    else_block,
                    else_args: vec![],
                });
            }
            Instr::OrOr(target) => {
                // If top is true, jump to target; else continue
                let lhs = self.pop();
                let else_block = self.get_or_create_block(pc + 1);
                let then_block = self.get_or_create_block(*target);
                self.set_terminator(Terminator::Cbr {
                    cond: lhs,
                    then_block,
                    then_args: vec![],
                    else_block,
                    else_args: vec![],
                });
            }

            // Stack manipulation
            Instr::Pop => {
                self.pop();
            }
            Instr::Swap => {
                if self.stack.len() >= 2 {
                    let len = self.stack.len();
                    self.stack.swap(len - 1, len - 2);
                }
            }

            // Return
            Instr::Return => {
                self.set_terminator(Terminator::Ret(None));
            }
            Instr::ReturnValue => {
                let val = self.pop();
                self.set_terminator(Terminator::Ret(Some(val)));
            }

            // Builtin calls
            Instr::CallBuiltin(name, argc) => {
                let args = self.pop_n(*argc);
                let effect = classify_builtin_effect(name);
                let v = self.emit(
                    SsaOp::Call {
                        func: name.clone(),
                        args,
                        effect,
                    },
                    SsaType::Ptr,
                );
                self.stack.push(v);
            }

            // User function calls
            Instr::CallFunction(name, argc) => {
                let args = self.pop_n(*argc);
                let v = self.emit(
                    SsaOp::CallRuntime {
                        func: format!("__call_user_{}", name),
                        args,
                    },
                    SsaType::Ptr,
                );
                self.stack.push(v);
            }

            // Matrix creation
            Instr::CreateMatrix(rows, cols) => {
                let count = rows * cols;
                let args = self.pop_n(count);
                let v = self.emit(
                    SsaOp::CallRuntime {
                        func: "__create_matrix".to_string(),
                        args,
                    },
                    SsaType::Ptr,
                );
                self.stack.push(v);
            }

            // Range creation
            Instr::CreateRange(has_step) => {
                let args = if *has_step {
                    self.pop_n(3) // start, step, end
                } else {
                    self.pop_n(2) // start, end
                };
                let v = self.emit(
                    SsaOp::CallRuntime {
                        func: "__create_range".to_string(),
                        args,
                    },
                    SsaType::Ptr,
                );
                self.stack.push(v);
            }

            // Indexing
            Instr::Index(num_indices) => {
                let indices = self.pop_n(*num_indices);
                let base = self.pop();
                let mut args = vec![base];
                args.extend(indices);
                let v = self.emit(
                    SsaOp::CallRuntime {
                        func: "__index".to_string(),
                        args,
                    },
                    SsaType::Ptr,
                );
                self.stack.push(v);
            }

            // Transpose
            Instr::Transpose => {
                let val = self.pop();
                let v = self.emit(
                    SsaOp::Call {
                        func: "transpose".to_string(),
                        args: vec![val],
                        effect: EffectKind::Pure,
                    },
                    SsaType::Ptr,
                );
                self.stack.push(v);
            }

            // Scoping (no-op for SSA, handled by variable naming)
            Instr::EnterScope(_) | Instr::ExitScope(_) => {}

            // Local variable access (function-local)
            Instr::LoadLocal(offset) => {
                let ptr = self.emit(SsaOp::VarPtr(1000 + offset), SsaType::Ptr); // Offset locals
                let v = self.emit(SsaOp::Load(ptr), SsaType::F64);
                self.stack.push(v);
            }
            Instr::StoreLocal(offset) => {
                let val = self.pop();
                let ptr = self.emit(SsaOp::VarPtr(1000 + offset), SsaType::Ptr);
                self.emit(SsaOp::Store(ptr, val), SsaType::F64);
            }

            // Try/catch (simplified: treat as barriers)
            Instr::EnterTry(_, _) | Instr::PopTry => {
                // These are control flow barriers; emit as runtime calls
                self.emit(
                    SsaOp::CallRuntime {
                        func: "__try_barrier".to_string(),
                        args: vec![],
                    },
                    SsaType::Ptr,
                );
            }

            // Everything else: fall back to runtime call
            _ => {
                // Generic fallback for unhandled instructions
                let v = self.emit(
                    SsaOp::CallRuntime {
                        func: format!("__unhandled_{:?}", std::mem::discriminant(instr)),
                        args: vec![],
                    },
                    SsaType::Ptr,
                );
                self.stack.push(v);
            }
        }
    }

    /// Emit an instruction in the current block
    fn emit(&mut self, op: SsaOp, ty: SsaType) -> SsaValue {
        let dst = self.func.new_value();
        if let Some(block) = self.func.block_mut(self.current_block) {
            block.instrs.push(SsaInstr { dst, op, ty });
        }
        dst
    }

    /// Pop a value from the symbolic stack
    fn pop(&mut self) -> SsaValue {
        self.stack.pop().unwrap_or_else(|| {
            // Stack underflow: emit a placeholder
            self.emit(SsaOp::ConstF64(0.0), SsaType::F64)
        })
    }

    /// Pop N values (in order: first popped is last in vec)
    fn pop_n(&mut self, n: usize) -> Vec<SsaValue> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            result.push(self.pop());
        }
        result.reverse();
        result
    }

    /// Binary operation helper
    fn binary_op<F>(&mut self, make_op: F)
    where
        F: FnOnce(SsaValue, SsaValue) -> SsaOp,
    {
        let b = self.pop();
        let a = self.pop();
        let v = self.emit(make_op(a, b), SsaType::F64);
        self.stack.push(v);
    }

    /// Unary operation helper
    fn unary_op<F>(&mut self, make_op: F)
    where
        F: FnOnce(SsaValue) -> SsaOp,
    {
        let a = self.pop();
        let v = self.emit(make_op(a), SsaType::F64);
        self.stack.push(v);
    }

    /// Get or create block for a PC
    fn get_or_create_block(&mut self, pc: usize) -> BlockId {
        if let Some(&block) = self.pc_to_block.get(&pc) {
            block
        } else {
            let block = self.func.new_block();
            self.pc_to_block.insert(pc, block);
            block
        }
    }

    /// Set terminator for current block
    fn set_terminator(&mut self, term: Terminator) {
        if let Some(block) = self.func.block_mut(self.current_block) {
            block.term = term;
        }
    }

    /// Terminate current block with branch to target
    fn terminate_with_br(&mut self, target: BlockId) {
        if let Some(block) = self.func.block_mut(self.current_block) {
            if matches!(block.term, Terminator::Unreachable) {
                block.term = Terminator::Br {
                    target,
                    args: vec![],
                };
            }
        }
    }

    /// Ensure current block has a terminator
    fn ensure_terminated(&mut self) {
        if let Some(block) = self.func.block_mut(self.current_block) {
            if matches!(block.term, Terminator::Unreachable) {
                block.term = Terminator::Ret(None);
            }
        }
    }
}

/// Find basic block leaders (targets of jumps + instruction after conditional)
fn find_leaders(instructions: &[Instr]) -> HashSet<usize> {
    let mut leaders = HashSet::new();
    leaders.insert(0); // Entry is always a leader

    for (pc, instr) in instructions.iter().enumerate() {
        match instr {
            Instr::Jump(target)
            | Instr::JumpIfFalse(target)
            | Instr::AndAnd(target)
            | Instr::OrOr(target) => {
                leaders.insert(*target);
                // Instruction after conditional is also a leader
                if !matches!(instr, Instr::Jump(_)) {
                    leaders.insert(pc + 1);
                }
            }
            Instr::EnterTry(catch_pc, _) => {
                leaders.insert(*catch_pc);
            }
            _ => {}
        }
    }

    leaders
}

/// Classify builtin effect for CSE purposes
fn classify_builtin_effect(name: &str) -> EffectKind {
    // Pure math functions
    const PURE_BUILTINS: &[&str] = &[
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "log10",
        "log2",
        "sqrt",
        "abs",
        "floor",
        "ceil",
        "round",
        "sign",
        "mod",
        "rem",
        "power",
        "plus",
        "minus",
        "times",
        "rdivide",
        "ldivide",
        "uminus",
        "uplus",
        "transpose",
        "ctranspose",
        "size",
        "length",
        "numel",
        "ndims",
        "isempty",
        "isscalar",
        "isvector",
        "ismatrix",
        "isrow",
        "iscolumn",
        "isreal",
        "isnan",
        "isinf",
        "isfinite",
        "real",
        "imag",
        "conj",
        "angle",
        "min",
        "max",
        "sum",
        "prod",
        "mean",
        "std",
        "var",
        "norm",
        "dot",
        "cross",
        "eye",
        "zeros",
        "ones",
        "linspace",
        "logspace",
        "reshape",
        "repmat",
        "cat",
        "horzcat",
        "vertcat",
        "fliplr",
        "flipud",
        "rot90",
        "tril",
        "triu",
        "diag",
        "trace",
    ];

    // Read-only (depend on global state but don't modify)
    const READONLY_BUILTINS: &[&str] = &["rand", "randn", "randi", "clock", "now", "tic", "toc"];

    if PURE_BUILTINS.contains(&name) {
        EffectKind::Pure
    } else if READONLY_BUILTINS.contains(&name) {
        EffectKind::ReadOnly
    } else {
        EffectKind::SideEffect
    }
}

/// Convert bytecode to SSA (convenience function)
pub fn bytecode_to_ssa(
    instructions: &[Instr],
    var_count: usize,
    name: impl Into<String>,
) -> SsaFunc {
    let builder = SsaBuilder::new(name);
    builder.build(instructions, var_count)
}

/// Check if bytecode is safe for SSA compilation.
///
/// Returns `false` if the bytecode has stack values crossing block boundaries
/// (e.g., values pushed in branches and consumed at join points).
/// This is a conservative check - it rejects patterns that require phi nodes
/// which the current SSA builder doesn't support.
pub fn is_ssa_safe(instructions: &[Instr]) -> bool {
    let leaders = find_leaders(instructions);

    // Simulate stack depth at each block boundary
    // If any block ends with non-zero stack depth (excluding terminators that consume the stack),
    // and the next instruction is a leader, we have values crossing boundaries.
    let mut stack_depth: i32 = 0;

    for (pc, instr) in instructions.iter().enumerate() {
        // At block boundaries, stack must be empty (or we're jumping with values)
        if leaders.contains(&pc) && pc != 0 {
            if stack_depth != 0 {
                return false;
            }
            stack_depth = 0;
        }

        // Track stack depth changes
        match instr {
            // Push operations
            Instr::LoadConst(_)
            | Instr::LoadBool(_)
            | Instr::LoadString(_)
            | Instr::LoadCharRow(_)
            | Instr::LoadVar(_)
            | Instr::LoadLocal(_)
            | Instr::LoadMember(_)
            | Instr::LoadMemberDynamic
            | Instr::LoadMethod(_)
            | Instr::LoadStaticProperty(_, _) => {
                stack_depth += 1;
            }

            // Pop 1, push 1 (net 0)
            Instr::Neg | Instr::UPlus | Instr::Transpose => {}

            // Pop 2, push 1 (net -1)
            Instr::Add
            | Instr::Sub
            | Instr::Mul
            | Instr::Div
            | Instr::Pow
            | Instr::ElemMul
            | Instr::ElemDiv
            | Instr::ElemPow
            | Instr::ElemLeftDiv
            | Instr::Less
            | Instr::Greater
            | Instr::LessEqual
            | Instr::GreaterEqual
            | Instr::Equal
            | Instr::NotEqual => {
                stack_depth -= 1;
            }

            // Store operations: pop 1
            Instr::StoreVar(_) | Instr::StoreLocal(_) => {
                stack_depth -= 1;
            }

            // Pop without push
            Instr::Pop => {
                stack_depth -= 1;
            }

            // Control flow: JumpIfFalse pops condition, AndAnd/OrOr pop condition
            Instr::JumpIfFalse(_) | Instr::AndAnd(_) | Instr::OrOr(_) => {
                stack_depth -= 1;
                // After a conditional branch, check the outgoing stack depth
                // If non-zero, there are values that would need to be carried to successors
                if stack_depth != 0 {
                    return false;
                }
            }

            // Jump: no stack effect, but check outgoing
            Instr::Jump(_) => {
                if stack_depth != 0 {
                    return false;
                }
            }

            // Return: pops if value present, ends block
            Instr::Return => {}
            Instr::ReturnValue => {
                stack_depth -= 1;
            }

            // Function calls: CallBuiltin(name, argc) pops args, pushes result
            Instr::CallBuiltin(_, argc) => {
                stack_depth -= *argc as i32;
                stack_depth += 1;
            }

            // CallFunction: pops args, pushes 1 result
            Instr::CallFunction(_, argc) => {
                stack_depth -= *argc as i32;
                stack_depth += 1;
            }

            // CallFunctionMulti: pops argc, pushes out_count
            Instr::CallFunctionMulti(_, argc, out_count) => {
                stack_depth -= *argc as i32;
                stack_depth += *out_count as i32;
            }

            Instr::CallBuiltinMulti(_, argc, out_count) => {
                stack_depth -= *argc as i32;
                stack_depth += *out_count as i32;
            }

            // Matrix creation: pops rows*cols, pushes 1
            Instr::CreateMatrix(rows, cols) => {
                stack_depth -= (*rows * *cols) as i32;
                stack_depth += 1;
            }

            Instr::CreateMatrixDynamic(rows) => {
                // Conservative: assume at least `rows` elements
                stack_depth -= *rows as i32;
                stack_depth += 1;
            }

            // CreateRange: pops 2 or 3, pushes 1
            Instr::CreateRange(has_step) => {
                stack_depth -= if *has_step { 3 } else { 2 };
                stack_depth += 1;
            }

            // Index: pops base + indices, pushes 1
            Instr::Index(num_indices) => {
                stack_depth -= 1 + *num_indices as i32;
                stack_depth += 1;
            }

            // IndexSlice: pops base + numeric indices, pushes 1
            Instr::IndexSlice(_, numeric, _, _) => {
                stack_depth -= 1 + *numeric as i32;
                stack_depth += 1;
            }

            // Cell arrays
            Instr::CreateCell2D(rows, cols) => {
                stack_depth -= (*rows * *cols) as i32;
                stack_depth += 1;
            }

            Instr::IndexCell(num_indices) => {
                stack_depth -= 1 + *num_indices as i32;
                stack_depth += 1;
            }

            // Store with index: pops base + indices + rhs, pushes updated base
            Instr::StoreIndex(num_indices) => {
                stack_depth -= 1 + *num_indices as i32 + 1;
                stack_depth += 1;
            }

            Instr::StoreIndexCell(num_indices) => {
                stack_depth -= 1 + *num_indices as i32 + 1;
                stack_depth += 1;
            }

            // Member store: pops base + rhs, pushes updated base
            Instr::StoreMember(_) => {
                stack_depth -= 1;
            }

            Instr::StoreMemberDynamic => {
                stack_depth -= 2;
            }

            // Method call: pops base + args, pushes result
            Instr::CallMethod(_, argc) => {
                stack_depth -= 1 + *argc as i32;
                stack_depth += 1;
            }

            // Static method: pops args, pushes result
            Instr::CallStaticMethod(_, _, argc) => {
                stack_depth -= *argc as i32;
                stack_depth += 1;
            }

            // Swap: no depth change
            Instr::Swap => {}

            // Try/catch control flow
            Instr::EnterTry(_, _) | Instr::PopTry => {}

            // Scope management: no stack effect
            Instr::EnterScope(_) | Instr::ExitScope(_) => {}

            // Closures: pops captures, pushes closure
            Instr::CreateClosure(_, capture_count) => {
                stack_depth -= *capture_count as i32;
                stack_depth += 1;
            }

            // feval: pops args, pushes result
            Instr::CallFeval(argc) => {
                stack_depth -= *argc as i32;
                stack_depth += 1;
            }

            // Pack operations
            Instr::PackToRow(n) | Instr::PackToCol(n) => {
                stack_depth -= *n as i32;
                stack_depth += 1;
            }

            // Global/persistent declarations: no stack effect
            Instr::DeclareGlobal(_)
            | Instr::DeclarePersistent(_)
            | Instr::DeclareGlobalNamed(_, _)
            | Instr::DeclarePersistentNamed(_, _)
            | Instr::RegisterImport { .. }
            | Instr::RegisterClass { .. } => {}

            // Stochastic evolution: varies, be conservative
            Instr::StochasticEvolution => {}

            // Complex indexing operations - be conservative and reject
            Instr::IndexSliceEx(_, _, _, _, _)
            | Instr::IndexRangeEnd { .. }
            | Instr::Index1DRangeEnd { .. }
            | Instr::StoreRangeEnd { .. }
            | Instr::StoreSlice(_, _, _, _)
            | Instr::StoreSliceEx(_, _, _, _, _)
            | Instr::StoreSlice1DRangeEnd { .. }
            | Instr::IndexCellExpand(_, _)
            | Instr::CallFunctionExpandAt(_, _, _, _)
            | Instr::CallBuiltinExpandLast(_, _, _)
            | Instr::CallBuiltinExpandAt(_, _, _, _)
            | Instr::CallFunctionExpandMulti(_, _)
            | Instr::CallBuiltinExpandMulti(_, _)
            | Instr::CallFevalExpandMulti(_) => {
                // These are complex and may not be safe
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_arithmetic() {
        // x = 1 + 2
        let instructions = vec![
            Instr::LoadConst(1.0),
            Instr::LoadConst(2.0),
            Instr::Add,
            Instr::StoreVar(0),
            Instr::Return,
        ];

        let func = bytecode_to_ssa(&instructions, 1, "test");
        let dump = func.dump();

        assert!(dump.contains("ConstF64(1.0)"));
        assert!(dump.contains("ConstF64(2.0)"));
        assert!(dump.contains("Add"));
    }

    #[test]
    fn test_conditional() {
        // if x < 10 then y = 1 else y = 2
        let instructions = vec![
            Instr::LoadVar(0),      // 0
            Instr::LoadConst(10.0), // 1
            Instr::Less,            // 2
            Instr::JumpIfFalse(7),  // 3 -> else
            Instr::LoadConst(1.0),  // 4 (then)
            Instr::StoreVar(1),     // 5
            Instr::Jump(9),         // 6 -> end
            Instr::LoadConst(2.0),  // 7 (else)
            Instr::StoreVar(1),     // 8
            Instr::Return,          // 9
        ];

        let func = bytecode_to_ssa(&instructions, 2, "test");

        // Should have multiple blocks
        assert!(func.blocks.len() >= 3);
    }

    #[test]
    fn test_find_leaders() {
        let instructions = vec![
            Instr::LoadConst(1.0), // 0 - leader (entry)
            Instr::JumpIfFalse(4), // 1
            Instr::LoadConst(2.0), // 2 - leader (after conditional)
            Instr::Jump(5),        // 3
            Instr::LoadConst(3.0), // 4 - leader (jump target)
            Instr::Return,         // 5 - leader (jump target)
        ];

        let leaders = find_leaders(&instructions);
        assert!(leaders.contains(&0));
        assert!(leaders.contains(&2));
        assert!(leaders.contains(&4));
        assert!(leaders.contains(&5));
    }
}
