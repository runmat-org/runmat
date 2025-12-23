//! SSA IR for Turbine JIT optimization
//!
//! A lightweight SSA representation using block arguments (no φ-nodes).
//! Enables CSE, constant folding, LICM, and loop optimizations before
//! lowering to Cranelift.

use std::fmt;

/// A unique SSA value identifier (immutable, defined exactly once)
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct SsaValue(pub u32);

impl fmt::Debug for SsaValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl fmt::Display for SsaValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Block identifier
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BlockId(pub u32);

impl fmt::Debug for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// SSA types (simplified for JIT)
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum SsaType {
    #[default]
    F64,
    I64,
    Bool,
    Ptr, // Pointer to Value (for runtime calls)
}

/// Comparison predicates
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Effect annotation for calls
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum EffectKind {
    /// No side effects, result depends only on inputs (CSE-able)
    Pure,
    /// Reads memory but doesn't write (CSE-able if no intervening stores)
    ReadOnly,
    /// Has side effects (barrier for CSE/LICM)
    SideEffect,
}

/// SSA operation (the "opcode")
#[derive(Debug, Clone, PartialEq)]
pub enum SsaOp {
    // Constants
    ConstF64(f64),
    ConstI64(i64),
    ConstBool(bool),

    // Copy (used after CSE to replace redundant expressions)
    Copy(SsaValue),

    // Block argument (parameter from predecessor)
    BlockArg(usize),

    // Arithmetic (pure, CSE-able)
    Add(SsaValue, SsaValue),
    Sub(SsaValue, SsaValue),
    Mul(SsaValue, SsaValue),
    Div(SsaValue, SsaValue),
    Neg(SsaValue),
    Pow(SsaValue, SsaValue),

    // Element-wise ops
    ElemMul(SsaValue, SsaValue),
    ElemDiv(SsaValue, SsaValue),
    ElemPow(SsaValue, SsaValue),

    // Comparisons (pure)
    Cmp(CmpOp, SsaValue, SsaValue),

    // Logical
    And(SsaValue, SsaValue),
    Or(SsaValue, SsaValue),
    Not(SsaValue),

    // Conversions
    F64ToI64(SsaValue),
    I64ToF64(SsaValue),
    BoolToF64(SsaValue),

    // Memory operations
    Load(SsaValue),            // Load from pointer
    Store(SsaValue, SsaValue), // Store val to pointer (ptr, val)

    // Variable access (pointer to variable slot)
    VarPtr(usize), // Get pointer to variable slot

    // Function calls
    Call {
        func: String,
        args: Vec<SsaValue>,
        effect: EffectKind,
    },

    // Runtime helper calls (return Ptr to Value)
    CallRuntime {
        func: String,
        args: Vec<SsaValue>,
    },
}

impl SsaOp {
    /// Check if this operation is pure (no side effects)
    pub fn is_pure(&self) -> bool {
        match self {
            SsaOp::ConstF64(_)
            | SsaOp::ConstI64(_)
            | SsaOp::ConstBool(_)
            | SsaOp::Copy(_)
            | SsaOp::BlockArg(_)
            | SsaOp::Add(_, _)
            | SsaOp::Sub(_, _)
            | SsaOp::Mul(_, _)
            | SsaOp::Div(_, _)
            | SsaOp::Neg(_)
            | SsaOp::Pow(_, _)
            | SsaOp::ElemMul(_, _)
            | SsaOp::ElemDiv(_, _)
            | SsaOp::ElemPow(_, _)
            | SsaOp::Cmp(_, _, _)
            | SsaOp::And(_, _)
            | SsaOp::Or(_, _)
            | SsaOp::Not(_)
            | SsaOp::F64ToI64(_)
            | SsaOp::I64ToF64(_)
            | SsaOp::BoolToF64(_)
            | SsaOp::VarPtr(_) => true,

            SsaOp::Call { effect, .. } => *effect == EffectKind::Pure,

            SsaOp::Load(_) | SsaOp::Store(_, _) | SsaOp::CallRuntime { .. } => false,
        }
    }

    /// Get operand values (for use counting, CSE key building)
    pub fn operands(&self) -> Vec<SsaValue> {
        match self {
            SsaOp::ConstF64(_)
            | SsaOp::ConstI64(_)
            | SsaOp::ConstBool(_)
            | SsaOp::BlockArg(_)
            | SsaOp::VarPtr(_) => vec![],

            SsaOp::Copy(v)
            | SsaOp::Neg(v)
            | SsaOp::Not(v)
            | SsaOp::F64ToI64(v)
            | SsaOp::I64ToF64(v)
            | SsaOp::BoolToF64(v)
            | SsaOp::Load(v) => vec![*v],

            SsaOp::Add(a, b)
            | SsaOp::Sub(a, b)
            | SsaOp::Mul(a, b)
            | SsaOp::Div(a, b)
            | SsaOp::Pow(a, b)
            | SsaOp::ElemMul(a, b)
            | SsaOp::ElemDiv(a, b)
            | SsaOp::ElemPow(a, b)
            | SsaOp::Cmp(_, a, b)
            | SsaOp::And(a, b)
            | SsaOp::Or(a, b)
            | SsaOp::Store(a, b) => vec![*a, *b],

            SsaOp::Call { args, .. } | SsaOp::CallRuntime { args, .. } => args.clone(),
        }
    }

    /// Check if this op is commutative (for CSE canonicalization)
    pub fn is_commutative(&self) -> bool {
        matches!(
            self,
            SsaOp::Add(_, _)
                | SsaOp::Mul(_, _)
                | SsaOp::ElemMul(_, _)
                | SsaOp::And(_, _)
                | SsaOp::Or(_, _)
        )
    }
}

/// SSA instruction (defines exactly one value)
#[derive(Debug, Clone)]
pub struct SsaInstr {
    pub dst: SsaValue,
    pub op: SsaOp,
    pub ty: SsaType,
}

/// Block terminator
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Unconditional branch with block arguments
    Br {
        target: BlockId,
        args: Vec<SsaValue>,
    },
    /// Conditional branch
    Cbr {
        cond: SsaValue,
        then_block: BlockId,
        then_args: Vec<SsaValue>,
        else_block: BlockId,
        else_args: Vec<SsaValue>,
    },
    /// Return from function
    Ret(Option<SsaValue>),
    /// Unreachable (for error paths)
    Unreachable,
}

impl Terminator {
    /// Get all successor block IDs
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Br { target, .. } => vec![*target],
            Terminator::Cbr {
                then_block,
                else_block,
                ..
            } => vec![*then_block, *else_block],
            Terminator::Ret(_) | Terminator::Unreachable => vec![],
        }
    }

    /// Get all values used by the terminator
    pub fn args(&self) -> Vec<SsaValue> {
        match self {
            Terminator::Br { args, .. } => args.clone(),
            Terminator::Cbr {
                cond,
                then_args,
                else_args,
                ..
            } => {
                let mut all = vec![*cond];
                all.extend(then_args.iter().cloned());
                all.extend(else_args.iter().cloned());
                all
            }
            Terminator::Ret(Some(v)) => vec![*v],
            Terminator::Ret(None) | Terminator::Unreachable => vec![],
        }
    }
}

/// Basic block (sequence of instructions + terminator)
#[derive(Debug, Clone)]
pub struct SsaBlock {
    pub id: BlockId,
    /// Block parameters (like φ-node results, but cleaner)
    pub params: Vec<(SsaValue, SsaType)>,
    /// Instructions in order
    pub instrs: Vec<SsaInstr>,
    /// How the block ends
    pub term: Terminator,
}

impl SsaBlock {
    pub fn new(id: BlockId) -> Self {
        SsaBlock {
            id,
            params: Vec::new(),
            instrs: Vec::new(),
            term: Terminator::Unreachable,
        }
    }
}

/// SSA function
#[derive(Debug, Clone)]
pub struct SsaFunc {
    pub name: String,
    pub blocks: Vec<SsaBlock>,
    pub entry: BlockId,
    /// Next available value ID
    next_value: u32,
    /// Next available block ID
    next_block: u32,
}

impl SsaFunc {
    pub fn new(name: impl Into<String>) -> Self {
        SsaFunc {
            name: name.into(),
            blocks: Vec::new(),
            entry: BlockId(0),
            next_value: 0,
            next_block: 0,
        }
    }

    /// Allocate a new SSA value
    pub fn new_value(&mut self) -> SsaValue {
        let v = SsaValue(self.next_value);
        self.next_value += 1;
        v
    }

    /// Allocate a new block
    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        self.blocks.push(SsaBlock::new(id));
        id
    }

    /// Get block by ID
    pub fn block(&self, id: BlockId) -> Option<&SsaBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    /// Get mutable block by ID
    pub fn block_mut(&mut self, id: BlockId) -> Option<&mut SsaBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    /// Find which block defines a value
    pub fn def_block(&self, val: SsaValue) -> Option<BlockId> {
        for block in &self.blocks {
            // Check params
            if block.params.iter().any(|(v, _)| *v == val) {
                return Some(block.id);
            }
            // Check instructions
            if block.instrs.iter().any(|i| i.dst == val) {
                return Some(block.id);
            }
        }
        None
    }

    /// Get predecessors of a block
    pub fn predecessors(&self, target: BlockId) -> Vec<BlockId> {
        self.blocks
            .iter()
            .filter(|b| b.term.successors().contains(&target))
            .map(|b| b.id)
            .collect()
    }

    /// Pretty-print the function
    pub fn dump(&self) -> String {
        let mut out = format!("func {}:\n", self.name);
        for block in &self.blocks {
            let params: Vec<String> = block
                .params
                .iter()
                .map(|(v, ty)| format!("{v}: {ty:?}"))
                .collect();
            if params.is_empty() {
                out.push_str(&format!("  {}:\n", block.id));
            } else {
                out.push_str(&format!("  {}({}):\n", block.id, params.join(", ")));
            }
            for instr in &block.instrs {
                out.push_str(&format!(
                    "    {} = {:?} : {:?}\n",
                    instr.dst, instr.op, instr.ty
                ));
            }
            match &block.term {
                Terminator::Br { target, args } => {
                    let args_str: Vec<String> = args.iter().map(|v| format!("{v}")).collect();
                    out.push_str(&format!("    br {}({})\n", target, args_str.join(", ")));
                }
                Terminator::Cbr {
                    cond,
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                } => {
                    let then_str: Vec<String> = then_args.iter().map(|v| format!("{v}")).collect();
                    let else_str: Vec<String> = else_args.iter().map(|v| format!("{v}")).collect();
                    out.push_str(&format!(
                        "    cbr {}, {}({}), {}({})\n",
                        cond,
                        then_block,
                        then_str.join(", "),
                        else_block,
                        else_str.join(", ")
                    ));
                }
                Terminator::Ret(Some(v)) => {
                    out.push_str(&format!("    ret {v}\n"));
                }
                Terminator::Ret(None) => {
                    out.push_str("    ret\n");
                }
                Terminator::Unreachable => {
                    out.push_str("    unreachable\n");
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssa_value_display() {
        let v = SsaValue(42);
        assert_eq!(format!("{v}"), "%42");
    }

    #[test]
    fn test_block_id_display() {
        let b = BlockId(3);
        assert_eq!(format!("{b}"), "bb3");
    }

    #[test]
    fn test_op_is_pure() {
        assert!(SsaOp::Add(SsaValue(0), SsaValue(1)).is_pure());
        assert!(SsaOp::ConstF64(1.0).is_pure());
        assert!(!SsaOp::Load(SsaValue(0)).is_pure());
        assert!(!SsaOp::Store(SsaValue(0), SsaValue(1)).is_pure());
    }

    #[test]
    fn test_op_operands() {
        assert!(SsaOp::ConstF64(1.0).operands().is_empty());
        assert_eq!(SsaOp::Neg(SsaValue(5)).operands(), vec![SsaValue(5)]);
        assert_eq!(
            SsaOp::Add(SsaValue(1), SsaValue(2)).operands(),
            vec![SsaValue(1), SsaValue(2)]
        );
    }

    #[test]
    fn test_func_building() {
        let mut func = SsaFunc::new("test");
        let entry = func.new_block();
        func.entry = entry;

        let v0 = func.new_value();
        let v1 = func.new_value();
        let v2 = func.new_value();

        let block = func.block_mut(entry).unwrap();
        block.instrs.push(SsaInstr {
            dst: v0,
            op: SsaOp::ConstF64(1.0),
            ty: SsaType::F64,
        });
        block.instrs.push(SsaInstr {
            dst: v1,
            op: SsaOp::ConstF64(2.0),
            ty: SsaType::F64,
        });
        block.instrs.push(SsaInstr {
            dst: v2,
            op: SsaOp::Add(v0, v1),
            ty: SsaType::F64,
        });
        block.term = Terminator::Ret(Some(v2));

        let dump = func.dump();
        assert!(dump.contains("bb0"));
        assert!(dump.contains("ConstF64(1.0)"));
        assert!(dump.contains("Add"));
        assert!(dump.contains("ret %2"));
    }
}
