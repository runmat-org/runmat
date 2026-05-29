---
title: "Mid-Level IR (MIR)"
category: "Compilation Pipeline"
section: "2.3"
last_updated: "May 28, 2026"
---

# Mid-Level IR (MIR)

The Mid-Level IR (MIR) represents the stage in the RunMat compilation pipeline where High-Level IR (HIR) is lowered into a Control-Flow Graph (CFG) of Basic Blocks. While HIR maintains a structure close to the original MATLAB AST (nested loops, if-statements), MIR flattens these into explicit jumps, branch targets, and local variable slots (locals). This representation is used for dataflow analysis, type inference, and as the primary input for bytecode generation.

The process of lowering HIR to MIR is a process used in modern compiler such as the Rust compiler. It allows for the compiler to perform dataflow analysis, type inference, and as the primary input for bytecode generation, enabling static analysis and optimizations of runtime branches prior to execution.

---

## MIR Structure & CFG

A MIR program is organized into a `MirAssembly`, which contains a collection of `MirBody` objects indexed by their `FunctionId`. Each `MirBody` consists of a list of `MirLocal` declarations and a graph of `BasicBlock` nodes.

### Basic Blocks and Control Flow

Every `BasicBlock` contains a sequence of `MirStmt` and exactly one `MirTerminator`.

- `MirStmt`: Linear operations like assignments (`Assign`), multi-assignments (`MultiAssign`), or expressions evaluated for side effects (`Expr`).
- `MirTerminator`: Dictates the transfer of control. Kinds include `Goto`, `Branch` (conditional), `Switch`, `Return`, `Await`, and `TryCatch`.

### MIR Data Entity Mapping

The following diagram bridges the conceptual MATLAB control flow to the internal MIR representation.

```mermaid
flowchart LR
  %% Subgraph: MIR Code Entity Space (crates/runmat-mir)
  %% Subgraph: MATLAB Source Space
  S1["if condition"]
  S2["while loop"]
  S3["function y = f(x)"]
  B1["MirBody"]
  BB["BasicBlock"]
  T1["MirTerminatorKind::Branch"]
  T2["MirTerminatorKind::Goto"]
  T3["MirTerminatorKind::Return"]
  L1["MirLocal (MirLocalKind::Parameter)"]
  L2["MirLocal (MirLocalKind::Output)"]
  S3 --> B1
  B1 --> L1
  B1 --> L2
  B1 --> BB
  S1 --> T1
  S2 --> T2
  L2 --> T3
```

---

## MIR Lowering (HIR → MIR)

Lowering is performed by the `lower_assembly` function, which iterates through HIR functions and utilizes a `ControlFlowBuilder` to construct the CFG.

### ControlFlowBuilder & Continuation Passing

The `ControlFlowBuilder` handles the conversion of nested HIR structures into flat blocks using a continuation-passing approach. When encountering a branch or an `await` point, the builder:

1. Allocates a `fresh_block()` for the continuation.
2. Lowers the "current" block's terminator to point to the new block.
3. Recursively lowers the remaining statements into the continuation block.

### MirLocal Slots

MIR replaces HIR `BindingId` references with `MirLocalId` slots. Locals are categorized by `MirLocalKind`:

- `Parameter`: Input arguments to the function.
- `Output`: Variables that will be returned.
- `Binding`: Standard local variables.
- `Capture`: Variables captured from an outer scope (closures).
- `Temporary`: Compiler-generated slots for intermediate expression results.

---

## Rvalues and Indexing Plans

MIR expressions are represented as `MirRvalue`. Unlike HIR expressions, `MirRvalue` is shallow; its operands are usually `MirOperand::Local` or `MirOperand::Constant`.

### Indexing Operations

MATLAB indexing is complex (supporting `end`, `:`, and logical masks). MIR lowers these into a `MirIndexing` structure containing `MirIndexComponent`s.

- `MirIndexPlan`: Determines if the access is `Scalar`, `Slice`, or `Cell`.
- `MirRvalue::Index`: Represents a read operation.
- `MirStmtKind::Assign` with `MirPlace::Index`: Represents a write/mutation operation.

### Rvalue Kinds

| Kind | Description |
| --- | --- |
| Use | Simple move or copy of an operand. |
| Binary / Unary | Arithmetic and logical operations. |
| Call | Function invocation with MirCallee (Static or Dynamic). |
| Aggregate | Construction of Tensors or Cell arrays. |
| ShortCircuit | Logical && and ` |

---

## MIR Dataflow Analysis

Once lowered, the `AnalysisStore` tracks facts about `MirLocal` slots across the CFG using a fixed-point iteration engine in `compute_simple_local_facts`.

### Analysis Logic Flow

The diagram below illustrates how the analysis engine processes a `MirBody` to produce type and shape facts.

```mermaid
flowchart TD
  A["MirBody"]
  B["compute_simple_local_facts"]
  C["transfer_fact_block"]
  D["MirStmtKind::Assign"]
  E["simple_rvalue_fact"]
  F["SimpleValueFact (Type, Shape, Async)"]
  G["join_fact_state (Merge Paths)"]
  H["AnalysisStore"]
  A --> B
  B --> C
  C --> D
  D --> E
  E --> F
  F --> G
  G --> H
```

### Key Analysis Facts

- `InitFact`: Tracks if a local is `Unassigned`, `MaybeAssigned`, or `DefinitelyAssigned`. Used for definite assignment validation.
- `SimpleValueFact`: Aggregates `TypeFact` (e.g., Double, Struct), `ShapeFact` (dimensions), and `ValueFlowFact`.
- `SpawnSafetyFact`: Analyzes if a closure or function is safe to `spawn` on a background thread based on its captures and effects.