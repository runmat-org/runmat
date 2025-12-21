# Memo: Studying Symbolica Safely for RunMat Integration

**To:** RunMat Symbolic Engine Contributor  
**From:** RunMat Core Team  
**Subject:** How to Study Symbolica Without Infringement – Claims, Risks, and Safe Design Paths  
**Date:** _[today]_

---

## 1. Purpose of This Memo

You will study the Symbolica source code to understand *ideas* and *design principles* relevant to building a symbolic engine for RunMat. **You are not allowed to copy, translate, or closely mirror Symbolica code.**

This memo gives you:

- A **claim-oriented way of thinking** (similar to patent analysis)
- Likely **hypothetical claims** Symbolica could assert
- **Concrete avoidance strategies** for each claim area
- Guidance on **RunMat-native alternatives**, even if slower or more conventional

The goal is to ensure **independent creation**, **legal safety**, and **architectural differentiation**, while still learning from Symbolica at a conceptual level.

---

## 2. General Rules (Non-Negotiable)

1. **No code copying** (including “rewriting in your own words”).
2. **No parallel coding** while Symbolica code is open.
3. **No reuse of internal names**, module layouts, or APIs.
4. **All implementation must be justifiable without referencing Symbolica.**
5. Treat Symbolica as if it were a **paper, not a library**.

You *may*:
- Study high-level ideas
- Learn performance pitfalls
- Identify what *not* to do

---

## 3. How to Think: Hypothetical Claims Analysis

Assume Symbolica could assert claims like a patent holder would. Your task is to:

1. Identify **where claims could exist**
2. Ensure RunMat’s design **avoids or routes around** them

Below are the **main claim risk areas**.

---

## 4. Claim Area A: Expression Representation

### Possible Symbolica Claim (Hypothetical)

> A symbolic algebra system using a compact, hash-consed, arena-allocated DAG representation with canonicalized node ordering for fast equality, simplification, and pattern matching.

### Risk Signals
- Very specific node layouts
- Custom arena invariants
- Canonical ordering rules baked into construction

### How to Avoid

Use **conceptual similarity**, not structural similarity:

- DAG-based expressions are allowed (standard CAS practice)
- But:
  - Different node enums
  - Different ownership model
  - Different canonicalization timing

### RunMat-Native Alternative

- Use **IR-aligned symbolic nodes**:
  - Expression nodes map 1:1 (or close) to RunMat IR ops
  - Canonicalization happens at IR-lowering stage, not construction
- Accept higher memory usage initially

**Key argument:** "We designed expressions to integrate with RunMat IR, not to optimize symbolic algebra per se."

---

## 5. Claim Area B: Simplification & Rewriting Engine

### Possible Symbolica Claim

> A rule-based simplification engine using prioritized pattern matching with memoization over large expression DAGs.

### Risk Signals
- Sophisticated pattern DSLs
- Large built-in rewrite rule sets
- Highly tuned rewrite scheduling

### How to Avoid

- Avoid implementing a general rewrite engine early
- Prefer **deterministic normalization passes**:
  - flatten sums/products
  - constant folding
  - ordering terms lexicographically

### RunMat-Native Alternative

- Split simplification into phases:
  1. Local algebraic cleanup
  2. IR canonicalization
  3. Optional backend simplify call

This may be **slower** but is extremely defensible.

---

## 6. Claim Area C: Performance-Critical Memory Techniques

### Possible Symbolica Claim

> A memory-efficient symbolic engine using specialized arenas, pointer tagging, and cache-aware node layouts.

### Risk Signals
- Non-obvious memory tricks
- Bit-level encoding
- Tight coupling between algebra and allocator

### How to Avoid

- Use **standard Rust containers** or generic arenas
- Avoid clever pointer tagging
- Prefer clarity over peak performance

### RunMat-Native Alternative

- Leverage RunMat’s **existing allocation and GC strategy**
- Accept higher memory overhead

**Important:** Performance differences are a *feature*, not a bug, from a legal perspective.

---

## 7. Claim Area D: Substitution and Evaluation

### Possible Symbolica Claim

> A fast substitution mechanism using structural sharing and dependency tracking across large expressions.

### How to Avoid

- Implement substitution as:
  - recursive descent
  - optional memoization
- No global dependency graphs

### RunMat-Native Alternative

- Perform substitution during **IR lowering**
- Cache at the IR level, not the symbolic level

---

## 8. Claim Area E: Symbolic–Numeric Code Generation

### Possible Symbolica Claim

> A system that converts symbolic expressions into optimized numeric kernels.

### How to Avoid

- Do **not** generate standalone optimized C/LLVM code from symbolic trees

### RunMat-Native Alternative (Preferred)

- Lower symbolic expressions directly into **RunMat IR**
- Let the existing RunMat JIT:
  - do CSE
  - do fusion
  - do vectorization

This is a **clear differentiation point**.

---

## 9. Claim Area F: API & User Model

### Possible Symbolica Claim

> A user-facing symbolic API tightly coupled to internal algebra structures.

### How to Avoid

- MATLAB compatibility layer must be:
  - Separate
  - Behavior-driven

### RunMat-Native Alternative

- MATLAB API → compatibility shim → symbolic core
- Symbolica never exposed to user code

---

## 10. Practical Study Protocol (Very Important)

When studying Symbolica:

1. **Read only** – no coding
2. Write **conceptual notes only** (no pseudocode)
3. Close Symbolica repo
4. Wait at least one work session
5. Write a **fresh design proposal** from memory

All RunMat code must be written **only** from the design proposal.

---

## 11. What We Are Optimizing For (Explicitly)

We are **not** optimizing for:
- Maximum symbolic performance
- Beating Symbolica at its own game

We *are* optimizing for:
- MATLAB compatibility
- JIT integration
- Long-term maintainability
- Legal defensibility

---

## 12. Final Guiding Principle

> If someone familiar with Symbolica can say
> “this looks like Symbolica inside RunMat”,
> then we are too close.

> If they say
> “this is a MATLAB-compatible symbolic layer designed around a JIT IR”,
> then we are exactly where we want to be.

---

**If in doubt: choose the simpler, slower, more conventional solution.**

