//! SSA Optimization Passes
//!
//! Implements CSE, constant folding, DCE, and LICM for the SSA IR.

use crate::dominators::DomTree;
use crate::loop_analysis::{ensure_preheader, LoopInfo};
use crate::ssa::{BlockId, CmpOp, SsaFunc, SsaInstr, SsaOp, SsaValue};
use std::collections::{HashMap, HashSet};

/// Optimization level (mirrors config::JitOptLevel)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No SSA optimization (legacy path)
    None,
    /// Minimal: simplify + DCE
    Size,
    /// Standard: simplify + DCE + CSE
    Speed,
    /// Full: all passes including LICM
    Aggressive,
}

/// Pass flags for fine-grained control via RUNMAT_SSA_PASSES env var
pub mod pass_flags {
    pub const SIMPLIFY: u32 = 1;
    pub const DCE: u32 = 2;
    pub const CSE: u32 = 4;
    pub const LOAD_CSE: u32 = 8;
    pub const LICM: u32 = 16;
}

/// Get pass mask from environment or return None to use opt level defaults
fn get_pass_mask_from_env() -> Option<u32> {
    std::env::var("RUNMAT_SSA_PASSES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
}

/// Run optimization passes based on opt level (or RUNMAT_SSA_PASSES override)
pub fn optimize(func: &mut SsaFunc, level: OptLevel) {
    // Check for env var override
    if let Some(mask) = get_pass_mask_from_env() {
        run_passes_by_mask(func, mask);
        return;
    }

    // Default behavior based on opt level
    match level {
        OptLevel::None => {
            // No optimization
        }
        OptLevel::Size => {
            simplify_all(func);
            dce(func);
        }
        OptLevel::Speed => {
            simplify_all(func);
            dce(func);
            let dom = DomTree::compute(func);
            cse(func, &dom);
            load_cse(func);
            dce(func);
        }
        OptLevel::Aggressive => {
            simplify_all(func);
            dce(func);
            let dom = DomTree::compute(func);
            cse(func, &dom);
            load_cse(func);
            let mut loop_info = LoopInfo::compute(func, &dom);
            licm(func, &mut loop_info, &dom);
            dce(func);
        }
    }
}

/// Run passes according to bitmask (for experimentation)
fn run_passes_by_mask(func: &mut SsaFunc, mask: u32) {
    use pass_flags::*;

    log::debug!("SSA passes override: mask={} (binary {:05b})", mask, mask);

    if mask & SIMPLIFY != 0 {
        simplify_all(func);
    }
    if mask & DCE != 0 {
        dce(func);
    }
    if mask & (CSE | LOAD_CSE | LICM) != 0 {
        let dom = DomTree::compute(func);
        if mask & CSE != 0 {
            cse(func, &dom);
        }
        if mask & LOAD_CSE != 0 {
            load_cse(func);
        }
        if mask & LICM != 0 {
            let mut loop_info = LoopInfo::compute(func, &dom);
            licm(func, &mut loop_info, &dom);
        }
    }
    // Final DCE cleanup if DCE is enabled and we ran other passes
    if mask & DCE != 0 && mask & (CSE | LOAD_CSE | LICM) != 0 {
        dce(func);
    }
}

// ============================================================================
// Constant Folding / Simplification
// ============================================================================

/// Simplify all instructions in a function
pub fn simplify_all(func: &mut SsaFunc) {
    // Build value -> constant map
    let mut constants: HashMap<SsaValue, Const> = HashMap::new();

    for block in &mut func.blocks {
        for instr in &mut block.instrs {
            // Try to simplify
            if let Some(simplified) = simplify(&instr.op, &constants) {
                instr.op = simplified;
            }

            // Track constants
            match &instr.op {
                SsaOp::ConstF64(v) => {
                    constants.insert(instr.dst, Const::F64(*v));
                }
                SsaOp::ConstI64(v) => {
                    constants.insert(instr.dst, Const::I64(*v));
                }
                SsaOp::ConstBool(v) => {
                    constants.insert(instr.dst, Const::Bool(*v));
                }
                _ => {}
            }
        }
    }
}

/// Constant value for folding
#[derive(Debug, Clone, Copy)]
enum Const {
    F64(f64),
    I64(i64),
    Bool(bool),
}

impl Const {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Const::F64(v) => Some(*v),
            Const::I64(v) => Some(*v as f64),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Const::Bool(v) => Some(*v),
            Const::F64(v) => Some(*v != 0.0),
            Const::I64(v) => Some(*v != 0),
        }
    }
}

/// Try to simplify an operation
fn simplify(op: &SsaOp, constants: &HashMap<SsaValue, Const>) -> Option<SsaOp> {
    match op {
        // Constant folding for arithmetic
        SsaOp::Add(a, b) => {
            let ca = constants.get(a).and_then(|c| c.as_f64());
            let cb = constants.get(b).and_then(|c| c.as_f64());
            match (ca, cb) {
                (Some(va), Some(vb)) => Some(SsaOp::ConstF64(va + vb)),
                (Some(0.0), _) => Some(SsaOp::Copy(*b)), // 0 + x = x
                (_, Some(0.0)) => Some(SsaOp::Copy(*a)), // x + 0 = x
                _ => None,
            }
        }
        SsaOp::Sub(a, b) => {
            let ca = constants.get(a).and_then(|c| c.as_f64());
            let cb = constants.get(b).and_then(|c| c.as_f64());
            match (ca, cb) {
                (Some(va), Some(vb)) => Some(SsaOp::ConstF64(va - vb)),
                (_, Some(0.0)) => Some(SsaOp::Copy(*a)), // x - 0 = x
                _ if a == b => Some(SsaOp::ConstF64(0.0)), // x - x = 0
                _ => None,
            }
        }
        SsaOp::Mul(a, b) => {
            let ca = constants.get(a).and_then(|c| c.as_f64());
            let cb = constants.get(b).and_then(|c| c.as_f64());
            match (ca, cb) {
                (Some(va), Some(vb)) => Some(SsaOp::ConstF64(va * vb)),
                (Some(0.0), _) | (_, Some(0.0)) => Some(SsaOp::ConstF64(0.0)), // 0 * x = 0
                (Some(1.0), _) => Some(SsaOp::Copy(*b)),                       // 1 * x = x
                (_, Some(1.0)) => Some(SsaOp::Copy(*a)),                       // x * 1 = x
                _ => None,
            }
        }
        SsaOp::Div(a, b) => {
            let ca = constants.get(a).and_then(|c| c.as_f64());
            let cb = constants.get(b).and_then(|c| c.as_f64());
            match (ca, cb) {
                (Some(va), Some(vb)) if vb != 0.0 => Some(SsaOp::ConstF64(va / vb)),
                (Some(0.0), _) => Some(SsaOp::ConstF64(0.0)), // 0 / x = 0
                (_, Some(1.0)) => Some(SsaOp::Copy(*a)),      // x / 1 = x
                _ => None,
            }
        }
        SsaOp::Neg(a) => {
            if let Some(Const::F64(v)) = constants.get(a) {
                Some(SsaOp::ConstF64(-v))
            } else {
                None
            }
        }
        SsaOp::Pow(a, b) => {
            let ca = constants.get(a).and_then(|c| c.as_f64());
            let cb = constants.get(b).and_then(|c| c.as_f64());
            match (ca, cb) {
                (Some(va), Some(vb)) => Some(SsaOp::ConstF64(va.powf(vb))),
                (_, Some(0.0)) => Some(SsaOp::ConstF64(1.0)), // x^0 = 1
                (_, Some(1.0)) => Some(SsaOp::Copy(*a)),      // x^1 = x
                (Some(1.0), _) => Some(SsaOp::ConstF64(1.0)), // 1^x = 1
                _ => None,
            }
        }

        // Comparison folding
        SsaOp::Cmp(cmp_op, a, b) => {
            let ca = constants.get(a).and_then(|c| c.as_f64());
            let cb = constants.get(b).and_then(|c| c.as_f64());
            if let (Some(va), Some(vb)) = (ca, cb) {
                let result = match cmp_op {
                    CmpOp::Eq => va == vb,
                    CmpOp::Ne => va != vb,
                    CmpOp::Lt => va < vb,
                    CmpOp::Le => va <= vb,
                    CmpOp::Gt => va > vb,
                    CmpOp::Ge => va >= vb,
                };
                Some(SsaOp::ConstBool(result))
            } else {
                None
            }
        }

        // Logical folding
        SsaOp::And(a, b) => {
            let ca = constants.get(a).and_then(|c| c.as_bool());
            let cb = constants.get(b).and_then(|c| c.as_bool());
            match (ca, cb) {
                (Some(va), Some(vb)) => Some(SsaOp::ConstBool(va && vb)),
                (Some(false), _) | (_, Some(false)) => Some(SsaOp::ConstBool(false)),
                (Some(true), _) => Some(SsaOp::Copy(*b)),
                (_, Some(true)) => Some(SsaOp::Copy(*a)),
                _ => None,
            }
        }
        SsaOp::Or(a, b) => {
            let ca = constants.get(a).and_then(|c| c.as_bool());
            let cb = constants.get(b).and_then(|c| c.as_bool());
            match (ca, cb) {
                (Some(va), Some(vb)) => Some(SsaOp::ConstBool(va || vb)),
                (Some(true), _) | (_, Some(true)) => Some(SsaOp::ConstBool(true)),
                (Some(false), _) => Some(SsaOp::Copy(*b)),
                (_, Some(false)) => Some(SsaOp::Copy(*a)),
                _ => None,
            }
        }
        SsaOp::Not(a) => {
            if let Some(Const::Bool(v)) = constants.get(a) {
                Some(SsaOp::ConstBool(!v))
            } else {
                None
            }
        }

        // Copy of copy
        SsaOp::Copy(v) => {
            if constants.contains_key(v) {
                // Propagate constant
                match constants.get(v) {
                    Some(Const::F64(x)) => Some(SsaOp::ConstF64(*x)),
                    Some(Const::I64(x)) => Some(SsaOp::ConstI64(*x)),
                    Some(Const::Bool(x)) => Some(SsaOp::ConstBool(*x)),
                    None => None,
                }
            } else {
                None
            }
        }

        _ => None,
    }
}

// ============================================================================
// Dead Code Elimination
// ============================================================================

/// Remove instructions with no uses
pub fn dce(func: &mut SsaFunc) {
    loop {
        let mut use_counts: HashMap<SsaValue, usize> = HashMap::new();

        // Count uses in instructions
        for block in &func.blocks {
            for instr in &block.instrs {
                for operand in instr.op.operands() {
                    *use_counts.entry(operand).or_default() += 1;
                }
            }
            // Count uses in terminators
            for arg in block.term.args() {
                *use_counts.entry(arg).or_default() += 1;
            }
        }

        // Remove dead instructions
        let mut changed = false;
        for block in &mut func.blocks {
            let original_len = block.instrs.len();
            block.instrs.retain(|instr| {
                // Keep if: has uses, or has side effects
                let has_uses = use_counts.get(&instr.dst).copied().unwrap_or(0) > 0;
                let has_effects = !instr.op.is_pure();
                has_uses || has_effects
            });
            if block.instrs.len() != original_len {
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
}

// ============================================================================
// Common Subexpression Elimination
// ============================================================================

/// CSE key for expression equality
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct ExprKey {
    opcode: u8,
    operands: Vec<SsaValue>,
}

impl ExprKey {
    fn from_op(op: &SsaOp) -> Option<Self> {
        // Only CSE pure operations
        if !op.is_pure() {
            return None;
        }

        let opcode = match op {
            SsaOp::Add(_, _) => 1,
            SsaOp::Sub(_, _) => 2,
            SsaOp::Mul(_, _) => 3,
            SsaOp::Div(_, _) => 4,
            SsaOp::Neg(_) => 5,
            SsaOp::Pow(_, _) => 6,
            SsaOp::ElemMul(_, _) => 7,
            SsaOp::ElemDiv(_, _) => 8,
            SsaOp::ElemPow(_, _) => 9,
            SsaOp::Cmp(cmp, _, _) => 10 + *cmp as u8,
            SsaOp::And(_, _) => 20,
            SsaOp::Or(_, _) => 21,
            SsaOp::Not(_) => 22,
            SsaOp::Call {
                func,
                effect: crate::ssa::EffectKind::Pure,
                ..
            } => {
                // Hash function name into opcode
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                func.hash(&mut h);
                (100 + (h.finish() % 100)) as u8
            }
            // VarPtr is pure - CSE redundant pointer computations
            SsaOp::VarPtr(slot) => {
                // Encode slot in operands since VarPtr has no SSA operands
                return Some(ExprKey {
                    opcode: 200,
                    operands: vec![SsaValue(*slot as u32)],
                });
            }
            _ => return None,
        };

        let mut operands = op.operands();

        // Canonicalize commutative ops
        if op.is_commutative() && operands.len() == 2 && operands[0] > operands[1] {
            operands.swap(0, 1);
        }

        Some(ExprKey { opcode, operands })
    }
}

/// Common subexpression elimination
pub fn cse(func: &mut SsaFunc, dom: &DomTree) {
    // Map from expression key to (defining value, defining block)
    let mut available: HashMap<ExprKey, (SsaValue, BlockId)> = HashMap::new();

    // Replacement map: value -> replacement value
    let mut replacements: HashMap<SsaValue, SsaValue> = HashMap::new();

    // Walk blocks in dominator tree preorder
    for &block_id in dom.preorder() {
        // Scope available expressions to this block's dominators
        let mut block_available = available.clone();

        if let Some(block) = func.block(block_id) {
            for instr in &block.instrs {
                if let Some(key) = ExprKey::from_op(&instr.op) {
                    if let Some(&(existing_val, existing_block)) = block_available.get(&key) {
                        // Check if existing definition dominates this use
                        if dom.dominates(existing_block, block_id) {
                            replacements.insert(instr.dst, existing_val);
                        }
                    } else {
                        // Record this expression
                        block_available.insert(key, (instr.dst, block_id));
                    }
                }
            }
        }

        // Update available for dominated blocks
        available = block_available;
    }

    // Apply replacements
    if !replacements.is_empty() {
        log::debug!(
            "CSE: eliminating {} redundant expressions",
            replacements.len()
        );
        apply_replacements(func, &replacements);
    }
}

/// Load CSE: Eliminate redundant loads from the same pointer within a basic block.
///
/// Within a single basic block, if we load from the same pointer twice without
/// an intervening store to that pointer, we can reuse the first loaded value.
/// This is safe because VarPtr slots don't alias each other.
pub fn load_cse(func: &mut SsaFunc) {
    let mut replacements: HashMap<SsaValue, SsaValue> = HashMap::new();

    for block in &mut func.blocks {
        // Map from pointer SSA value -> loaded value (within this block)
        let mut available_loads: HashMap<SsaValue, SsaValue> = HashMap::new();

        for instr in &block.instrs {
            match &instr.op {
                SsaOp::Load(ptr) => {
                    // Check if we've already loaded from this pointer
                    if let Some(&existing_val) = available_loads.get(ptr) {
                        // Reuse the existing load
                        replacements.insert(instr.dst, existing_val);
                    } else {
                        // Record this load for future reuse
                        available_loads.insert(*ptr, instr.dst);
                    }
                }
                SsaOp::Store(ptr, _val) => {
                    // Invalidate any load from this pointer
                    // This is safe because VarPtr(x) and VarPtr(y) never alias when x != y
                    available_loads.remove(ptr);
                }
                SsaOp::VarPtr(_) => {
                    // VarPtr is pure and already handled by regular CSE
                }
                SsaOp::Call { .. } | SsaOp::CallRuntime { .. } => {
                    // Conservative: function calls may modify memory
                    // Invalidate all available loads
                    available_loads.clear();
                }
                _ => {}
            }
        }
    }

    if !replacements.is_empty() {
        log::debug!(
            "Load CSE: eliminating {} redundant loads",
            replacements.len()
        );
        apply_replacements(func, &replacements);
    }
}

/// Apply value replacements throughout the function
fn apply_replacements(func: &mut SsaFunc, replacements: &HashMap<SsaValue, SsaValue>) {
    fn replace(val: SsaValue, replacements: &HashMap<SsaValue, SsaValue>) -> SsaValue {
        // Chase replacement chain
        let mut current = val;
        while let Some(&replacement) = replacements.get(&current) {
            if replacement == current {
                break;
            }
            current = replacement;
        }
        current
    }

    for block in &mut func.blocks {
        // Replace in instructions
        for instr in &mut block.instrs {
            // If this instruction is replaced, convert to Copy
            if let Some(&replacement) = replacements.get(&instr.dst) {
                instr.op = SsaOp::Copy(replacement);
                continue;
            }

            // Replace operands
            match &mut instr.op {
                SsaOp::Copy(v) => *v = replace(*v, replacements),
                SsaOp::Neg(v) | SsaOp::Not(v) | SsaOp::Load(v) => {
                    *v = replace(*v, replacements);
                }
                SsaOp::F64ToI64(v) | SsaOp::I64ToF64(v) | SsaOp::BoolToF64(v) => {
                    *v = replace(*v, replacements);
                }
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
                | SsaOp::Store(a, b) => {
                    *a = replace(*a, replacements);
                    *b = replace(*b, replacements);
                }
                SsaOp::Call { args, .. } | SsaOp::CallRuntime { args, .. } => {
                    for arg in args.iter_mut() {
                        *arg = replace(*arg, replacements);
                    }
                }
                _ => {}
            }
        }

        // Replace in terminator
        match &mut block.term {
            crate::ssa::Terminator::Br { args, .. } => {
                for arg in args.iter_mut() {
                    *arg = replace(*arg, replacements);
                }
            }
            crate::ssa::Terminator::Cbr {
                cond,
                then_args,
                else_args,
                ..
            } => {
                *cond = replace(*cond, replacements);
                for arg in then_args.iter_mut() {
                    *arg = replace(*arg, replacements);
                }
                for arg in else_args.iter_mut() {
                    *arg = replace(*arg, replacements);
                }
            }
            crate::ssa::Terminator::Ret(Some(v)) => {
                *v = replace(*v, replacements);
            }
            _ => {}
        }
    }
}

// ============================================================================
// Loop-Invariant Code Motion (LICM)
// ============================================================================

/// Hoist loop-invariant instructions to preheader
pub fn licm(func: &mut SsaFunc, loop_info: &mut LoopInfo, dom: &DomTree) {
    // Process loops from innermost to outermost (reverse depth order)
    let mut loops: Vec<_> = loop_info.loops().map(|l| (l.header, l.depth)).collect();
    loops.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by depth descending

    for (header, _depth) in loops {
        licm_loop(func, loop_info, dom, header);
    }
}

/// Hoist invariant code from a single loop
fn licm_loop(func: &mut SsaFunc, loop_info: &mut LoopInfo, _dom: &DomTree, header: BlockId) {
    // Ensure we have a preheader to hoist to
    let preheader = ensure_preheader(func, loop_info, header);

    // Get loop blocks (clone to avoid borrow issues)
    let loop_blocks: HashSet<BlockId> = loop_info
        .loop_by_header(header)
        .map(|l| l.blocks.clone())
        .unwrap_or_default();

    if loop_blocks.is_empty() {
        return;
    }

    // Build definition map: value -> (block, instr_index)
    let mut def_map: HashMap<SsaValue, BlockId> = HashMap::new();
    for block in &func.blocks {
        for instr in &block.instrs {
            def_map.insert(instr.dst, block.id);
        }
    }

    // Collect invariant instructions to hoist
    let mut to_hoist: Vec<(BlockId, usize, SsaInstr)> = Vec::new();
    let mut invariant_values: HashSet<SsaValue> = HashSet::new();

    // Iterate until no more invariants found
    let mut changed = true;
    while changed {
        changed = false;

        for &block_id in &loop_blocks {
            if let Some(block) = func.block(block_id) {
                for (idx, instr) in block.instrs.iter().enumerate() {
                    // Skip if already marked for hoisting
                    if invariant_values.contains(&instr.dst) {
                        continue;
                    }

                    // Check if instruction is loop-invariant and hoistable
                    if is_loop_invariant_instr(instr, &loop_blocks, &def_map, &invariant_values)
                        && is_hoistable(&instr.op)
                    {
                        invariant_values.insert(instr.dst);
                        to_hoist.push((block_id, idx, instr.clone()));
                        changed = true;
                    }
                }
            }
        }
    }

    // Actually hoist the instructions
    if !to_hoist.is_empty() {
        // Insert hoisted instructions into preheader (before terminator)
        if let Some(ph_block) = func.block_mut(preheader) {
            for (_src_block, _idx, instr) in &to_hoist {
                ph_block.instrs.push(instr.clone());
            }
        }

        // Remove hoisted instructions from their original blocks
        // We need to track which instructions to remove by value (dst)
        let hoisted_values: HashSet<SsaValue> = to_hoist.iter().map(|(_, _, i)| i.dst).collect();

        for block in &mut func.blocks {
            if loop_blocks.contains(&block.id) {
                block
                    .instrs
                    .retain(|instr| !hoisted_values.contains(&instr.dst));
            }
        }
    }
}

/// Check if an instruction is loop-invariant
fn is_loop_invariant_instr(
    instr: &SsaInstr,
    loop_blocks: &HashSet<BlockId>,
    def_map: &HashMap<SsaValue, BlockId>,
    invariant_values: &HashSet<SsaValue>,
) -> bool {
    // An instruction is loop-invariant if all its operands are:
    // 1. Defined outside the loop, OR
    // 2. Already proven invariant (for iterative discovery)

    for operand in instr.op.operands() {
        // Check if operand is defined in the loop
        if let Some(&def_block) = def_map.get(&operand) {
            if loop_blocks.contains(&def_block) {
                // Defined inside loop - only okay if already proven invariant
                if !invariant_values.contains(&operand) {
                    return false;
                }
            }
        }
        // If not in def_map, it's a parameter or external - that's fine
    }

    true
}

/// Check if an operation can be safely hoisted
fn is_hoistable(op: &SsaOp) -> bool {
    // Only hoist pure operations (no side effects, deterministic)
    // Also skip loads/stores which depend on memory state
    match op {
        SsaOp::Load(_) | SsaOp::Store(_, _) => false, // Memory operations aren't hoistable
        SsaOp::VarPtr(_) => true,                     // VarPtr is just address computation
        SsaOp::CallRuntime { .. } => false,           // Runtime calls have side effects
        SsaOp::Call { effect, .. } => {
            // Only hoist pure calls
            matches!(effect, crate::ssa::EffectKind::Pure)
        }
        _ => op.is_pure(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::{SsaType, Terminator};

    #[test]
    fn test_constant_folding() {
        let mut func = SsaFunc::new("test");
        let entry = func.new_block();
        func.entry = entry;

        let v0 = func.new_value();
        let v1 = func.new_value();
        let v2 = func.new_value();

        let block = func.block_mut(entry).unwrap();
        block.instrs.push(SsaInstr {
            dst: v0,
            op: SsaOp::ConstF64(2.0),
            ty: SsaType::F64,
        });
        block.instrs.push(SsaInstr {
            dst: v1,
            op: SsaOp::ConstF64(3.0),
            ty: SsaType::F64,
        });
        block.instrs.push(SsaInstr {
            dst: v2,
            op: SsaOp::Add(v0, v1),
            ty: SsaType::F64,
        });
        block.term = Terminator::Ret(Some(v2));

        simplify_all(&mut func);

        // The Add should be folded to ConstF64(5.0)
        let block = func.block(entry).unwrap();
        assert!(matches!(block.instrs[2].op, SsaOp::ConstF64(v) if v == 5.0));
    }

    #[test]
    fn test_dce() {
        let mut func = SsaFunc::new("test");
        let entry = func.new_block();
        func.entry = entry;

        let v0 = func.new_value();
        let v1 = func.new_value(); // Dead
        let v2 = func.new_value();

        let block = func.block_mut(entry).unwrap();
        block.instrs.push(SsaInstr {
            dst: v0,
            op: SsaOp::ConstF64(1.0),
            ty: SsaType::F64,
        });
        block.instrs.push(SsaInstr {
            dst: v1,
            op: SsaOp::ConstF64(2.0), // Not used
            ty: SsaType::F64,
        });
        block.instrs.push(SsaInstr {
            dst: v2,
            op: SsaOp::Add(v0, v0), // Uses v0 twice
            ty: SsaType::F64,
        });
        block.term = Terminator::Ret(Some(v2));

        dce(&mut func);

        // v1 should be removed
        let block = func.block(entry).unwrap();
        assert_eq!(block.instrs.len(), 2);
        assert!(matches!(block.instrs[0].op, SsaOp::ConstF64(1.0)));
        assert!(matches!(block.instrs[1].op, SsaOp::Add(_, _)));
    }

    #[test]
    fn test_cse() {
        let mut func = SsaFunc::new("test");
        let entry = func.new_block();
        func.entry = entry;

        let v0 = func.new_value();
        let v1 = func.new_value();
        let v2 = func.new_value();
        let v3 = func.new_value(); // Same as v2

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
        block.instrs.push(SsaInstr {
            dst: v3,
            op: SsaOp::Add(v0, v1), // Duplicate
            ty: SsaType::F64,
        });
        block.term = Terminator::Ret(Some(v3));

        let dom = DomTree::compute(&func);
        cse(&mut func, &dom);

        // v3 should be replaced with Copy(v2)
        let block = func.block(entry).unwrap();
        assert!(matches!(block.instrs[3].op, SsaOp::Copy(v) if v == v2));
    }

    #[test]
    fn test_optimize_speed() {
        let mut func = SsaFunc::new("test");
        let entry = func.new_block();
        func.entry = entry;

        let v0 = func.new_value();
        let v1 = func.new_value();
        let v2 = func.new_value();
        let v3 = func.new_value();
        let _v4 = func.new_value(); // Dead

        let block = func.block_mut(entry).unwrap();
        block.instrs.push(SsaInstr {
            dst: v0,
            op: SsaOp::ConstF64(2.0),
            ty: SsaType::F64,
        });
        block.instrs.push(SsaInstr {
            dst: v1,
            op: SsaOp::ConstF64(3.0),
            ty: SsaType::F64,
        });
        block.instrs.push(SsaInstr {
            dst: v2,
            op: SsaOp::Add(v0, v1), // Will fold to 5.0
            ty: SsaType::F64,
        });
        block.instrs.push(SsaInstr {
            dst: v3,
            op: SsaOp::Add(v0, v1), // Will CSE to v2
            ty: SsaType::F64,
        });
        block.term = Terminator::Ret(Some(v3));

        optimize(&mut func, OptLevel::Speed);

        // After optimization: should have minimal instructions
        let block = func.block(entry).unwrap();
        // Const fold + CSE + DCE should simplify significantly
        assert!(block.instrs.len() <= 4);
    }

    #[test]
    fn test_licm() {
        // Build a simple loop with invariant code:
        //   entry: c = 10
        //          br header
        //   header: cond = ...
        //           cbr cond, body, exit
        //   body: x = c * 2   <- loop-invariant!
        //         br header
        //   exit: ret x

        let mut func = SsaFunc::new("licm_test");
        let entry = func.new_block();
        let header = func.new_block();
        let body = func.new_block();
        let exit = func.new_block();
        func.entry = entry;

        let c = func.new_value();
        let cond = func.new_value();
        let two = func.new_value();
        let x = func.new_value();

        // Entry block: define constant c = 10
        {
            let block = func.block_mut(entry).unwrap();
            block.instrs.push(SsaInstr {
                dst: c,
                op: SsaOp::ConstF64(10.0),
                ty: SsaType::F64,
            });
            block.term = Terminator::Br {
                target: header,
                args: vec![],
            };
        }

        // Header: cbr to body or exit
        {
            let block = func.block_mut(header).unwrap();
            block.instrs.push(SsaInstr {
                dst: cond,
                op: SsaOp::ConstBool(true),
                ty: SsaType::Bool,
            });
            block.term = Terminator::Cbr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: exit,
                else_args: vec![],
            };
        }

        // Body: x = c * 2 (loop-invariant: c and 2 are defined outside loop)
        {
            let block = func.block_mut(body).unwrap();
            block.instrs.push(SsaInstr {
                dst: two,
                op: SsaOp::ConstF64(2.0),
                ty: SsaType::F64,
            });
            block.instrs.push(SsaInstr {
                dst: x,
                op: SsaOp::Mul(c, two),
                ty: SsaType::F64,
            });
            block.term = Terminator::Br {
                target: header,
                args: vec![],
            };
        }

        // Exit
        {
            let block = func.block_mut(exit).unwrap();
            block.term = Terminator::Ret(Some(x));
        }

        // Count instructions in body BEFORE LICM
        let body_instrs_before = func.block(body).unwrap().instrs.len();
        assert_eq!(body_instrs_before, 2); // two, x

        // Run LICM via aggressive optimization
        optimize(&mut func, OptLevel::Aggressive);

        // After LICM, the loop body should have fewer instructions
        // because the invariants (ConstF64(2.0) and Mul) were hoisted
        let body_instrs_after = func.block(body).unwrap().instrs.len();

        // Invariant instructions should have been hoisted to preheader
        // The body should now have 0 instructions (both were invariant)
        assert!(
            body_instrs_after < body_instrs_before,
            "LICM should hoist invariant instructions (before: {}, after: {})",
            body_instrs_before,
            body_instrs_after
        );
    }
}
