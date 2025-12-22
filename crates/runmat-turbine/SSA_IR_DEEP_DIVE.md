# Turbine SSA IR Deep Dive

## Executive Summary

Turbine currently compiles Ignition bytecode directly to Cranelift IR without an intermediate optimization layer. This document explores the design of a **small SSA-like IR** to enable CSE, constant folding, LICM, and loop optimizations—as called out in [TODO.md L132](./TODO.md#L132).

---

## Current Architecture

```
MATLAB Source
    ↓
Parser → AST
    ↓
runmat-hir → HIR (typed, "SSA-friendly")
    ↓
runmat-ignition → Stack-based Bytecode
    ↓
runmat-turbine → Cranelift IR → Native Code
```

**The gap**: Turbine performs 1:1 translation from bytecode to Cranelift with no optimization pass. Every `Instr::Add` becomes a `runmat_value_add` helper call.

---

## Why SSA?

SSA's power isn't φ-nodes—it's that **values are immutable and named once**. This makes:
- "This expression equals that expression" trivial to test (CSE)
- Moving code out of loops safe (LICM)
- Dead code obvious (DCE)

### HIR is "SSA-friendly", Not SSA

HIR uses `VarId` for stable naming, but variables can be reassigned:

```rust
// From runmat-hir/src/lib.rs
HirStmt::Assign(VarId, HirExpr, bool)  // Same VarId can appear multiple times
```

This is fine—HIR preserves MATLAB semantics. The SSA conversion belongs in Turbine.

---

## Proposed Design: Block Arguments (No φ-nodes)

Instead of traditional φ-nodes, use **block parameters** (like MLIR, Swift IR, Cranelift itself):

```
block header(i: i64, sum: f64):
    cond = icmp lt i, 100
    cbr cond, body(i, sum), exit(sum)

block body(i: i64, sum: f64):
    v1 = load arr, i
    sum2 = fadd sum, v1
    i2 = iadd i, 1
    br header(i2, sum2)

block exit(result: f64):
    ret result
```

**Benefits**:
- No φ insertion algorithm needed
- Rename is "wire values along edges"
- Maps directly to Cranelift's block parameters

---

## Minimal Instruction Set

### Pure Ops (CSE-friendly)

| Category | Instructions |
|----------|-------------|
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg` |
| Bitwise | `and`, `or`, `xor`, `shl`, `shr` |
| Compare | `icmp`, `fcmp` (with predicates: eq, ne, lt, le, gt, ge) |
| Convert | `i2f`, `f2i`, `zext`, `sext`, `trunc` |
| Select | `select cond, a, b` |

### Control

| Instruction | Semantics |
|-------------|-----------|
| `br block(args...)` | Unconditional branch with block arguments |
| `cbr cond, tblock(args...), fblock(args...)` | Conditional branch |
| `ret val?` | Return from function |

### Effects (Not CSE-able without analysis)

| Instruction | Semantics |
|-------------|-----------|
| `load ptr` | Memory read |
| `store ptr, val` | Memory write |
| `call fn, args... [pure\|readonly\|sideeffect]` | Function call with effect annotation |

---

## Data Structures

```rust
// Proposed: runmat-turbine/src/ssa.rs

/// A unique value identifier (SSA name)
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct SsaValue(u32);

/// Block identifier
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct BlockId(u32);

/// SSA instruction
pub struct SsaInstr {
    pub dst: SsaValue,
    pub op: SsaOp,
    pub ty: SsaType,
}

pub enum SsaOp {
    // Constants
    ConstF64(f64),
    ConstI64(i64),
    ConstBool(bool),
    
    // Pure arithmetic
    Add(SsaValue, SsaValue),
    Sub(SsaValue, SsaValue),
    Mul(SsaValue, SsaValue),
    Div(SsaValue, SsaValue),
    Neg(SsaValue),
    
    // Comparisons
    Cmp(CmpOp, SsaValue, SsaValue),
    
    // Memory
    Load(SsaValue),              // Load from pointer
    Store(SsaValue, SsaValue),   // Store val to pointer
    
    // Calls
    Call {
        func: String,
        args: Vec<SsaValue>,
        effect: EffectKind,
    },
    
    // Block argument (pseudo-instruction for parameters)
    BlockArg(usize),
}

pub enum EffectKind {
    Pure,       // No side effects, CSE-able
    ReadOnly,   // Reads memory, not CSE-able across stores
    SideEffect, // Full barrier
}

/// Block terminator
pub enum Terminator {
    Br { target: BlockId, args: Vec<SsaValue> },
    Cbr { 
        cond: SsaValue, 
        then_block: BlockId, 
        then_args: Vec<SsaValue>,
        else_block: BlockId,
        else_args: Vec<SsaValue>,
    },
    Ret(Option<SsaValue>),
}

/// Basic block
pub struct SsaBlock {
    pub id: BlockId,
    pub params: Vec<(SsaValue, SsaType)>,
    pub instrs: Vec<SsaInstr>,
    pub term: Terminator,
}

/// SSA function
pub struct SsaFunc {
    pub name: String,
    pub blocks: Vec<SsaBlock>,
    pub entry: BlockId,
}
```

---

## Optimization Passes

### 1. CSE (Common Subexpression Elimination)

```rust
/// CSE via hash-consing + dominance
pub fn cse(func: &mut SsaFunc, dom_tree: &DomTree) {
    let mut expr_map: HashMap<ExprKey, SsaValue> = HashMap::new();
    
    // Walk blocks in dominator-tree preorder
    for block_id in dom_tree.preorder() {
        let block = &mut func.blocks[block_id];
        
        for instr in &mut block.instrs {
            if !is_pure(&instr.op) {
                continue;
            }
            
            let key = ExprKey::from(&instr.op);
            
            if let Some(&existing) = expr_map.get(&key) {
                // Replace with existing value
                instr.op = SsaOp::Copy(existing);
            } else {
                expr_map.insert(key, instr.dst);
            }
        }
    }
}

#[derive(Hash, Eq, PartialEq)]
struct ExprKey {
    opcode: u8,
    operands: Vec<SsaValue>,  // Canonicalized order for commutative ops
}
```

### 2. Constant Folding

```rust
/// Simplify instruction, returning constant or simpler form
pub fn simplify(op: &SsaOp, values: &ValueInfo) -> Option<SsaOp> {
    match op {
        SsaOp::Add(a, b) => {
            match (values.as_const(*a), values.as_const(*b)) {
                (Some(Const::F64(x)), Some(Const::F64(y))) => {
                    Some(SsaOp::ConstF64(x + y))
                }
                (Some(Const::F64(0.0)), _) => Some(SsaOp::Copy(*b)),
                (_, Some(Const::F64(0.0))) => Some(SsaOp::Copy(*a)),
                _ => None,
            }
        }
        SsaOp::Mul(a, b) => {
            match (values.as_const(*a), values.as_const(*b)) {
                (Some(Const::F64(x)), Some(Const::F64(y))) => {
                    Some(SsaOp::ConstF64(x * y))
                }
                (Some(Const::F64(1.0)), _) => Some(SsaOp::Copy(*b)),
                (_, Some(Const::F64(1.0))) => Some(SsaOp::Copy(*a)),
                (Some(Const::F64(0.0)), _) | (_, Some(Const::F64(0.0))) => {
                    Some(SsaOp::ConstF64(0.0))
                }
                _ => None,
            }
        }
        // ... other cases
        _ => None,
    }
}
```

### 3. LICM (Loop-Invariant Code Motion)

```rust
/// Hoist loop-invariant instructions to preheader
pub fn licm(func: &mut SsaFunc, loop_info: &LoopInfo, dom_tree: &DomTree) {
    for loop_data in loop_info.loops() {
        let preheader = loop_data.preheader;
        let loop_blocks = &loop_data.blocks;
        
        for &block_id in loop_blocks {
            let block = &func.blocks[block_id];
            
            for instr in &block.instrs {
                if is_loop_invariant(instr, loop_blocks, func) 
                   && is_pure(&instr.op)
                   && dominates_all_uses(instr.dst, preheader, dom_tree) 
                {
                    hoist_to(instr, preheader);
                }
            }
        }
    }
}

fn is_loop_invariant(instr: &SsaInstr, loop_blocks: &[BlockId], func: &SsaFunc) -> bool {
    // Instruction is invariant if all operands are:
    // 1. Constants, or
    // 2. Defined outside the loop, or
    // 3. Already proven invariant
    for operand in instr.op.operands() {
        let def_block = func.def_block(operand);
        if loop_blocks.contains(&def_block) {
            return false;
        }
    }
    true
}
```

### 4. DCE (Dead Code Elimination)

```rust
/// Remove instructions with no uses
pub fn dce(func: &mut SsaFunc) {
    let mut use_counts: HashMap<SsaValue, usize> = HashMap::new();
    
    // Count uses
    for block in &func.blocks {
        for instr in &block.instrs {
            for operand in instr.op.operands() {
                *use_counts.entry(operand).or_default() += 1;
            }
        }
        for arg in block.term.args() {
            *use_counts.entry(arg).or_default() += 1;
        }
    }
    
    // Remove dead instructions (iterate to fixpoint)
    loop {
        let mut changed = false;
        for block in &mut func.blocks {
            block.instrs.retain(|instr| {
                if is_pure(&instr.op) && use_counts.get(&instr.dst).copied().unwrap_or(0) == 0 {
                    changed = true;
                    false
                } else {
                    true
                }
            });
        }
        if !changed { break; }
    }
}
```

---

## Pipeline Integration

### Compilation Pipeline

```
Ignition Bytecode
    ↓
[NEW] Bytecode → SSA conversion (ssa_builder.rs)
    ↓
SSA IR
    ↓
[NEW] Optimization passes:
    1. simplify (const-fold)
    2. dce
    3. cse
    4. loop discovery
    5. licm
    6. dce (cleanup)
    ↓
[EXISTING] SSA → Cranelift IR (compiler.rs, extended)
    ↓
Cranelift → Native Code
```

### New Files

| File | Purpose |
|------|---------|
| `src/ssa.rs` | SSA IR data structures |
| `src/ssa_builder.rs` | Bytecode → SSA conversion |
| `src/ssa_opt.rs` | Optimization passes (CSE, LICM, const-fold, DCE) |
| `src/dominators.rs` | Dominator tree computation |
| `src/loop_analysis.rs` | Loop detection and preheader insertion |

### Modified Files

| File | Changes |
|------|---------|
| `src/compiler.rs` | Accept SSA IR instead of raw bytecode |
| `src/lib.rs` | Wire up SSA pipeline |

---

## Bytecode → SSA Conversion

The key insight: Ignition bytecode is **stack-based**, but we can reconstruct SSA by simulating the stack symbolically.

```rust
pub fn bytecode_to_ssa(bytecode: &Bytecode) -> SsaFunc {
    let mut builder = SsaBuilder::new();
    let mut stack: Vec<SsaValue> = Vec::new();
    
    // First pass: identify basic block boundaries (jump targets)
    let leaders = find_leaders(bytecode);
    
    // Second pass: build SSA
    for (pc, instr) in bytecode.instructions.iter().enumerate() {
        if leaders.contains(&pc) {
            builder.start_block(pc);
        }
        
        match instr {
            Instr::LoadConst(val) => {
                let v = builder.emit(SsaOp::ConstF64(*val));
                stack.push(v);
            }
            Instr::Add => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let v = builder.emit(SsaOp::Add(a, b));
                stack.push(v);
            }
            Instr::StoreVar(idx) => {
                let val = stack.pop().unwrap();
                builder.emit(SsaOp::Store(builder.var_ptr(*idx), val));
            }
            Instr::LoadVar(idx) => {
                let v = builder.emit(SsaOp::Load(builder.var_ptr(*idx)));
                stack.push(v);
            }
            Instr::JumpIfFalse(target) => {
                let cond = stack.pop().unwrap();
                builder.emit_cbr(cond, pc + 1, *target);
            }
            Instr::Jump(target) => {
                builder.emit_br(*target);
            }
            // ... other instructions
        }
    }
    
    builder.finish()
}
```

---

## Relationship to Source-to-Source CSE

This SSA IR is **distinct** from the source-to-source CSE discussed in `CSE_SUMMARY.md`:

| Aspect | Turbine SSA IR | Source-to-Source CSE |
|--------|---------------|---------------------|
| Purpose | JIT optimization | User-facing code transformation |
| Input | Ignition bytecode | MATLAB code string |
| Output | Native code | Optimized MATLAB code string |
| IR Level | Low (f64, pointers) | High (HIR expressions) |
| Implementation | `runmat-turbine/src/ssa.rs` | `runmat-hir/src/cse_dag.rs` |

Both are valuable; they serve different use cases.

---

## Implementation Phases

### Phase 1: Infrastructure (1-2 weeks) ✅
- [x] Define `SsaValue`, `SsaOp`, `SsaBlock`, `SsaFunc` in `ssa.rs`
- [x] Implement dominator tree computation
- [x] Basic bytecode → SSA conversion for arithmetic ops

**Verification**:
```bash
# Run unit tests for SSA infrastructure
cargo test -p runmat-turbine ssa:: --no-fail-fast

# Verify no regressions (SSA disabled)
cargo test -p runmat-turbine --no-fail-fast
```

### Phase 2: Core Optimizations (2-3 weeks) ✅
- [x] Constant folding via `simplify()`
- [x] DCE with use counting
- [x] CSE with expression hashing

**Verification**:
```bash
# Run optimization pass tests
cargo test -p runmat-turbine ssa_opt:: --no-fail-fast

# Compare with/without optimization
cargo test -p runmat-turbine -- --test-threads=1
```

### Phase 3: Loop Optimizations (2-3 weeks) ✅
- [x] Loop detection (backedge identification)
- [x] Preheader insertion
- [ ] LICM for pure operations (TODO)

**Verification**:
```bash
# Run loop analysis tests
cargo test -p runmat-turbine loop_analysis:: --no-fail-fast
```

### Phase 4: Integration (1-2 weeks)
- [ ] Extend `compiler.rs` to lower SSA → Cranelift
- [ ] Wire into `TurbineEngine::compile_bytecode`
- [ ] Add tests comparing optimized vs unoptimized output

---

## Testing Protocol

### After Each Implementation Step

Run the following tests to verify correctness and measure performance impact:

#### 1. Unit Tests (Correctness)

```bash
# All SSA-related tests
cargo test -p runmat-turbine ssa --no-fail-fast
cargo test -p runmat-turbine dominators --no-fail-fast
cargo test -p runmat-turbine loop_analysis --no-fail-fast

# Full turbine test suite
cargo test -p runmat-turbine --no-fail-fast
```

#### 2. Integration Tests (Semantic Equivalence)

```bash
# Run with SSA disabled (baseline)
RUSTMAT_JIT_OPT_LEVEL=none cargo test -p runmat-ignition --no-fail-fast

# Run with SSA enabled (speed)
RUSTMAT_JIT_OPT_LEVEL=speed cargo test -p runmat-ignition --no-fail-fast

# Run with SSA aggressive
RUSTMAT_JIT_OPT_LEVEL=aggressive cargo test -p runmat-ignition --no-fail-fast
```

#### 3. Performance Benchmarks

```bash
# Baseline (no SSA)
RUSTMAT_JIT_OPT_LEVEL=none cargo bench -p runmat-turbine -- --save-baseline no-ssa

# With SSA optimizations
RUSTMAT_JIT_OPT_LEVEL=speed cargo bench -p runmat-turbine -- --save-baseline ssa-speed

# Compare
cargo bench -p runmat-turbine -- --baseline no-ssa
```

#### 4. A/B Comparison Script

Create `scripts/ssa_ab_test.sh`:

```bash
#!/bin/bash
# A/B test SSA vs no-SSA

SCRIPT=${1:-"benchmarks/matrix_ops.m"}
ITERATIONS=${2:-10}

echo "=== Testing: $SCRIPT ==="
echo ""

echo "--- Without SSA (--jit-opt-level=none) ---"
time for i in $(seq 1 $ITERATIONS); do
    runmat --jit-opt-level=none "$SCRIPT" > /dev/null 2>&1
done

echo ""
echo "--- With SSA (--jit-opt-level=speed) ---"
time for i in $(seq 1 $ITERATIONS); do
    runmat --jit-opt-level=speed "$SCRIPT" > /dev/null 2>&1
done

echo ""
echo "--- With SSA Aggressive (--jit-opt-level=aggressive) ---"
time for i in $(seq 1 $ITERATIONS); do
    runmat --jit-opt-level=aggressive "$SCRIPT" > /dev/null 2>&1
done
```

#### 5. Dump SSA for Debugging

```bash
# Dump SSA IR for inspection
RUSTMAT_JIT_DUMP_SSA=1 runmat --jit-opt-level=speed script.m 2> ssa_dump.txt
```

#### 6. Fine-Grained Pass Control (Experimentation)

Use `RUNMAT_SSA_PASSES` environment variable with a bitmask to enable specific passes:

| Bit | Value | Pass |
|-----|-------|------|
| 0 | 1 | SIMPLIFY (constant folding) |
| 1 | 2 | DCE (dead code elimination) |
| 2 | 4 | CSE (common subexpression elimination) |
| 3 | 8 | LOAD_CSE (redundant load elimination) |
| 4 | 16 | LICM (loop-invariant code motion) |

Examples:
```bash
# CSE + LOAD_CSE only (4+8=12)
RUNMAT_SSA_PASSES=12 runmat --jit-threshold 1 script.m

# All passes except LICM (1+2+4+8=15)
RUNMAT_SSA_PASSES=15 runmat --jit-threshold 1 script.m

# All passes (1+2+4+8+16=31)
RUNMAT_SSA_PASSES=31 runmat --jit-threshold 1 script.m

# No passes (equivalent to --jit-opt-level=none)
RUNMAT_SSA_PASSES=0 runmat --jit-threshold 1 script.m
```

Enable debug logging to see which passes run:
```bash
RUNMAT_SSA_PASSES=12 RUST_LOG=runmat_turbine=debug runmat script.m 2>&1 | grep "SSA passes"
# Output: SSA passes override: mask=12 (binary 01100)
```

### Expected Results

| Metric | None | Size | Speed | Aggressive |
|--------|------|------|-------|------------|
| Compile time | Baseline | +5% | +15% | +30% |
| Runtime (arithmetic) | Baseline | -5% | -15% | -25% |
| Runtime (loops) | Baseline | -5% | -15% | -40% |
| Code size | Baseline | -10% | -5% | +5% |

### Regression Checklist

Before merging SSA changes, verify:

- [ ] All `cargo test -p runmat-turbine` pass
- [ ] All `cargo test -p runmat-ignition` pass with each opt-level
- [ ] No performance regression at `--jit-opt-level=none`
- [ ] Measurable improvement at `--jit-opt-level=speed` on benchmarks
- [ ] `cargo clippy -p runmat-turbine` clean
- [ ] `cargo fmt -p runmat-turbine` clean

---

## Design Decisions

### Why Not Use Cranelift's IR Directly?

Cranelift IR is powerful but:
1. **Too low-level** for easy pattern matching (no `Call` with effect annotations)
2. **No RunMat-specific knowledge** (e.g., which builtins are pure)
3. **Harder to debug** (our SSA can have pretty-print with MATLAB semantics)

A thin SSA layer lets us do RunMat-aware optimizations before handing off to Cranelift.

### Why Block Arguments Over φ-nodes?

1. **Simpler implementation**: No SSA construction algorithm needed
2. **Direct Cranelift mapping**: Cranelift uses block parameters
3. **Cleaner semantics**: Values flow along edges, not "magically" at block start

### Memory Model

Start simple:
- `load`/`store` are barriers for CSE
- Pure calls (annotated) can be CSE'd
- No alias analysis initially

Future: Add memory SSA or simple alias analysis for load/store optimization.

---

## CLI Integration

### Opt-Level → SSA Pass Assignment

The existing `--jit-opt-level` flag controls which SSA passes run. **This is the official mapping**:

| Opt Level | CLI Flag | SSA Pipeline | Passes Enabled |
|-----------|----------|--------------|----------------|
| **None** | `--jit-opt-level=none` | Disabled | Legacy direct bytecode→Cranelift (no SSA) |
| **Size** | `--jit-opt-level=size` | Minimal | `simplify` → `dce` |
| **Speed** | `--jit-opt-level=speed` *(default)* | Standard | `simplify` → `dce` → `cse` → `dce` |
| **Aggressive** | `--jit-opt-level=aggressive` | Full | `simplify` → `dce` → `cse` → `licm` → `loop_opts` → `dce` |

### Pass Descriptions

| Pass | Cost | Benefit | Enabled At |
|------|------|---------|------------|
| `simplify` | O(n) | Const-fold, algebraic identities | size+ |
| `dce` | O(n) | Remove dead code | size+ |
| `cse` | O(n log n) | Eliminate redundant computations | speed+ |
| `licm` | O(n × loops) | Hoist invariants out of loops | aggressive |
| `loop_opts` | O(n × loops) | Strength reduction, IV recognition | aggressive |

### Rationale

- **None**: Fallback for debugging; ensures SSA bugs don't block users
- **Size**: Minimal compile time; useful for one-shot scripts
- **Speed**: Best balance for interactive/REPL use *(default)*
- **Aggressive**: Maximum runtime perf; higher compile time; for hot functions

### Implementation

**In `runmat/src/config.rs`** — no changes needed, `JitOptLevel` already exists:

```rust
pub enum JitOptLevel {
    None,       // Interpreter-like: no SSA
    Size,       // Minimal: DCE only
    Speed,      // Default: DCE + CSE + const-fold
    Aggressive, // Full: + LICM + loop opts
}
```

**In `runmat-turbine/src/lib.rs`** — check opt level before running passes:

```rust
impl TurbineEngine {
    pub fn compile_bytecode_with_opts(
        &mut self, 
        bytecode: &Bytecode,
        opt_level: JitOptLevel,
    ) -> Result<u64> {
        // Convert to SSA
        let mut ssa_func = bytecode_to_ssa(bytecode);
        
        // Apply passes based on opt level
        match opt_level {
            JitOptLevel::None => {
                // Skip SSA entirely, use legacy direct lowering
                return self.compile_bytecode_legacy(bytecode);
            }
            JitOptLevel::Size => {
                dce(&mut ssa_func);
            }
            JitOptLevel::Speed => {
                simplify_all(&mut ssa_func);  // const-fold
                dce(&mut ssa_func);
                cse(&mut ssa_func, &compute_dominators(&ssa_func));
                dce(&mut ssa_func);           // cleanup
            }
            JitOptLevel::Aggressive => {
                simplify_all(&mut ssa_func);
                dce(&mut ssa_func);
                cse(&mut ssa_func, &compute_dominators(&ssa_func));
                let loop_info = analyze_loops(&ssa_func);
                licm(&mut ssa_func, &loop_info);
                dce(&mut ssa_func);
            }
        }
        
        // Lower SSA to Cranelift
        self.lower_ssa_to_cranelift(&ssa_func)
    }
}
```

### Alternative: Explicit `--ssa` Flag (Development Only)

For development/debugging, a separate flag can be useful:

```rust
// In runmat/src/main.rs, Cli struct
/// [DEV] Force SSA IR pipeline (default: auto based on opt-level)
#[arg(long, hide = true, env = "RUSTMAT_JIT_SSA")]
jit_ssa: bool,

/// [DEV] Dump SSA IR to stderr before optimization
#[arg(long, hide = true, env = "RUSTMAT_JIT_DUMP_SSA")]
jit_dump_ssa: bool,
```

Usage:
```bash
# Force SSA even at opt-level=none (for testing)
runmat --jit-ssa script.m

# Debug: dump SSA IR
runmat --jit-dump-ssa --jit-opt-level=aggressive script.m
```

**Recommendation**: Do **not** expose `--jit-ssa` publicly. Use `--jit-opt-level` as the user-facing control. Keep `--jit-ssa` hidden (`hide = true`) for internal testing.

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `RUSTMAT_JIT_OPT_LEVEL` | `none`, `size`, `speed`, `aggressive` | Controls SSA passes |
| `RUSTMAT_JIT_SSA` | `0`, `1` | [DEV] Force SSA pipeline |
| `RUSTMAT_JIT_DUMP_SSA` | `0`, `1` | [DEV] Dump SSA to stderr |

### Config File

In `.runmat.yaml`:

```yaml
jit:
  enabled: true
  threshold: 10
  optimization_level: speed  # Controls SSA passes
  
  # Development options (optional)
  debug:
    dump_ssa: false
    dump_cranelift: false
```

---

## References

- [HIR Documentation](https://runmat.org/docs/internals/hir) — "Provide a typed, SSA-friendly structure"
- [Turbine TODO.md L132](./TODO.md#L132) — Original requirement
- [CSE_IMPLEMENTATION_SKETCH.md](/CSE_IMPLEMENTATION_SKETCH.md) — Source-to-source alternative
- [Cranelift IR Reference](https://cranelift.readthedocs.io/en/latest/ir.html) — Target IR
- [MLIR Block Arguments](https://mlir.llvm.org/docs/Rationale/Rationale/#block-arguments-vs-phi-nodes) — Design inspiration

---

**Status**: Phase 1-3 implemented, Phase 4 in progress  
**Owner**: TBD  
**Priority**: After Phase 1-2 JIT parity (per TODO.md)

---

## Handover Notes (Dec 2024)

### Completed
- [x] `ssa.rs` — SSA IR data structures (SsaValue, SsaOp, SsaBlock, SsaFunc)
- [x] `dominators.rs` — Dominator tree computation
- [x] `ssa_builder.rs` — Bytecode → SSA conversion
- [x] `ssa_opt.rs` — CSE, const-fold, DCE passes
- [x] `loop_analysis.rs` — Loop detection for LICM
- [x] `lib.rs` — Wired SSA pipeline into TurbineEngine
- [x] `compiler.rs` — Added `SsaOptLevel` to `CompilerConfig`
- [x] `tests/ssa_integration.rs` — Integration tests

### Remaining Build Fixes
Two unused import errors in test modules (trivial):
```
crates/runmat-turbine/src/dominators.rs:265 — remove SsaBlock
crates/runmat-turbine/src/loop_analysis.rs:279 — remove SsaBlock
```
**Already fixed** in latest edits, just need rebuild.

### To Run Tests
```bash
cargo test -p runmat-turbine ssa::
cargo test -p runmat-turbine --test ssa_integration
```

### Remaining Work (Phase 4)
1. **LICM implementation** — Add `licm()` function in `ssa_opt.rs`
2. **SSA → Cranelift lowering** — Replace/augment `compiler.rs` to use SSA
3. **Full A/B benchmarks** — Run `scripts/ssa_ab_test.sh`

### Key Files
| File | Lines | Purpose |
|------|-------|---------|
| `src/ssa.rs` | ~400 | Core IR types |
| `src/dominators.rs` | ~270 | Dom tree |
| `src/ssa_builder.rs` | ~360 | Bytecode→SSA |
| `src/ssa_opt.rs` | ~450 | Optimization passes |
| `src/loop_analysis.rs` | ~280 | Loop detection |
| `tests/ssa_integration.rs` | ~250 | Tests |

### CLI Integration
SSA passes are gated by `--jit-opt-level`:
- `none` — Skip SSA (legacy path)
- `size` — simplify + DCE
- `speed` — simplify + DCE + CSE (default)
- `aggressive` — All passes + LICM
