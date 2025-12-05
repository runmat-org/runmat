# RunMat Fusion Roadmap: VM Ops and Loop Fusion

## Core Problem

One of the primary losses in performance within GPU-accelerated math libraries is the overhead of launching and synchronizing kernels. This overhead is particularly pronounced for small, stateful math idioms that are common in stochastic simulations, iterative solvers, and other numerical computations.

For example, the Monte Carlo pricing model is a classic example of a stateful math idiom that is common in financial simulations. The model computes the price of an option by simulating the random paths of the underlying asset and then averaging the results. The simulation is performed by a series of small, stateful math operations that are repeated for each path.

The overhead of launching and synchronizing kernels is particularly pronounced for this type of computation because the number of operations is large and the operations are small. This means that the overhead of launching and synchronizing kernels is a significant fraction of the total runtime.

## Why we need VM ops

VM ops are one way to add semantics to the bytecode instructions that can annotate the structure of the computation and allow the compiler to generate optimized provider hooks for stateful math idioms.

For example, Stochastic Evolution is a stateful math idiom that is fairly canonical pattern we expect to see anywhere someone is simulating geometric Brownian motion or lognormal asset paths (Monte Carlo pricing, stochastic volatility backtests, particle filters, etc.). The instruction takes the form of:

```matlab
for t = 1:T
    Z = randn(M, 1, 'single');              % draw standard normals
    S = S .* exp(drift + scale .* Z);
end
```

In RunMat, the compiler detects that recurrent structure (deterministic loop bounds, a randn draw, multiplicative update with constants) and replaces it with Instr::StochasticEvolution instruction.

The provider hook for this instruction is then responsible for generating the GPU kernel that implements the stochastic evolution, with a fallback to a host-based implementation when one is not available.

This lets the runtime encode the semantics of common stateful math idioms in single instructions that can be dispatched to single-dispatch kernels when a provider is available, with a fallback to a host-based implementation when one is not available.

We believe the tradeoff of having a small set of explicit VM ops for stateful math idioms is worth it in RunMat because:

1. **Algorithmically** — advancing state with S ← S · exp(drift + scale · Z) is the standard Euler/log-Euler step for GBM, which is exactly what most Monte Carlo pricing, stochastic volatility backtests, particle filters, etc. code contains.

2. **Runtime-design-wise** — it’s no different than having specialized instructions for GEMM, FFT, etc. We just codified a higher-level pattern so we can dispatch to a highly optimized kernel when we see it, rather than interpreting 256 loops worth of scalar ops.

The purpose of RunMat is to be the fastest runtime for array math. We believe that having a small set of explicit VM ops for stateful math idioms is worth it to achieve this goal.

## The Argument for General Loop Fusion

The obvious next question is why don't we just have the fusion planner in RunMat recognize and collapse "pure" math loops automatically?

E.g. the following loop:

```matlab
for i = 1:N
    x = x + a * x;
end
```

Could be collapsed into a single kernel that computes `x = x * (1 + a)^N`.

A loop-aware fusion system could, in principle, subsume instructions like StochasticEvolution, but only if it grows substantially beyond today’s DAG-based elementwise fusion:

- You’d need to represent loops/recurrences explicitly in the fusion IR (e.g. “apply this body N times with state carried between iterations”).
- You’d need to model RNG state (Philox counters, seed advancement) so that fused kernels can produce reproducible random draws.
- You’d need scheduling logic that decides when the loop body is simple enough to be collapsed into one dispatch versus multiple dispatches/reductions.

If we eventually build that level of loop fusion, it could auto-detect many of the same patterns without bespoke VM instructions. However, it’s a large investment: new IR, new dependency analysis, new codegen templates for scan-like patterns, and RNG semantics in the planner. Until then, a targeted bytecode instruction is the pragmatic way to graft a high-value optimization into the pipeline without rewriting the fusion stack.

Even with a richer loop-fusion system, we’d likely still keep the VM opcode (or an equivalent “semantic marker”) for a while, for three reasons:

- **Back-compatibility:** CPU fallbacks can keep implementing the instruction directly without understanding the new fusion machinery.
- **Fast path:** when we know it’s exactly the GBM pattern, we can dispatch to a hand-tuned kernel immediately rather than relying on heuristics to rediscover it.
- **Fallback:** if the loop-fusion pass can’t prove safety (e.g. someone tweaks the loop), the opcode path still gives us the optimized behavior when the pattern matches.

**Long-term vision:** long term with RunMat, we want to enrich fusion so it can express looped recurrences and RNG, and use that to generate the kernel. Then we can treat the bytecode instruction as a hint or a convenience wrapper around the same fused plan, rather than as a separate execution path. But we don’t need to block on that; the finite set of “stateful math idioms” is small, and encoding them as explicit VM ops is a reasonable stepping stone while the fusion system evolves.

## Plan for loop-aware fusion

- **Current fusion = straight-line DAGs.** `runmat-accelerate`’s planner only sees acyclic graphs between loads/stores. It fuses elementwise chains, reductions, matmul epilogues, etc., but it has *zero* notion of loops, recurrent state, or RNG counters.
- **Bytecode instructions add semantics cheaply.** When the compiler spots a well-known loop idiom (e.g., the GBM update in Monte Carlo), emitting `Instr::StochasticEvolution` lets us drive an optimized provider hook without inventing a new fusion IR or proof system. CPU fallbacks remain simple and parity is preserved.
- **Loop-aware fusion is still desirable.** Eventually we want fusion to recognize and collapse “pure” loops automatically (scan/reduce operators, recurrent updates). That unlocks broader classes of math without adding new VM opcodes each time.
- **Complementary strategy.** Until loop fusion exists—*and even after it does*—a small, finite set of explicit VM ops for “stateful math idioms” gives us deterministic semantics, backward compatibility, and precise optimization hints. Think of them as semantic markers akin to ISA extensions (AES, SHA) for high-value patterns.

## Roadmap to loop-aware fusion

1. **IR improvements**
   - Extend the fusion IR with loop constructs (`Loop`, `Scan`, carrying state tensors between iterations).
   - Treat RNG state as explicit inputs/outputs (Philox counters, seeds) so fused kernels can advance them deterministically.
   - Track side effects and aliasing so we can prove a loop body is safe to fuse.
2. **Detection pipeline**
   - Enhance the HIR → fusion extraction to identify eligible loops (fixed bounds, no early exits, writes limited to the state tensor).
   - Annotate loops with metadata: iteration count, strides, RNG usage, reduction axes.
3. **Code generation**
   - Add WGPU/CPU templates for loop kernels (Euler/GBM updates, scans, rolling reductions).
   - Develop cost models that pick between “single fused loop kernel” vs “multiple kernels” based on T, tensor size, and occupancy.
4. **Debugging & parity**
   - Provide env toggles (e.g., `RUNMAT_DISABLE_LOOP_FUSION=stochastic`) to fall back to VM ops when needed.
   - Emit telemetry summarizing fused loops (iterations, bytes touched, RNG seeds) for diagnostics and marketing.

## Canonical VM opcode set

| Instruction            | Pattern                                                | Typical workloads                                         | Provider hook / status                              |
|------------------------|--------------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------|
| `StochasticEvolution`  | GBM-style loop `S = S .* exp(drift + scale .* randn)`  | Monte Carlo pricing, stochastic sims, particle filters    | `AccelProvider::stochastic_evolution` (implemented) |
| `PrefixScan` *(future)*| Inclusive/exclusive scan with optional custom op       | CDFs, cumulative sums/products, segmented reductions      | `AccelProvider::prefix_scan`                        |
| `IterativeReduce` *(future)* | Fixed-iteration solver (CG step, Jacobi update) | Sparse linear solvers, iterative refinement               | Solver-specific hooks (to design)                   |
| `RandomizedMatmul` *(future?)* | Matmul with RNG-driven sketching               | Randomized SVD/PCA, sketching pipelines                   | TBD                                                 |

## Putting it together

1. **Now:** continue adding VM opcodes when they unlock large wins otherwise impossible with DAG fusion (e.g., `StochasticEvolution`). Each opcode must ship with compiler detection, VM dispatch, provider API, GPU kernel, and tests.
2. **Mid-term:** instrument workloads to see which idioms dominate; prioritize them both for opcodes and for the future loop-fusion work.
3. **Long-term:** evolve the fusion planner to generate loop kernels automatically. When loop fusion can faithfully reproduce a VM opcode’s behavior, we can make the opcode a thin wrapper around the fused plan or retire it entirely. In some cases, keeping the opcode 

This layered approach lets RunMat deliver “best possible” performance today (with targeted VM ops) while we build the more general loop-fusion substrate needed to keep scaling toward “fastest runtime for array math.” The opcode list stays finite because it only covers the handful of stateful idioms whose semantics are richer than “elementwise DAG,” and everything else continues to flow through the fusion planner. 

## Monte Carlo case study

We re-ran the Monte Carlo computation chain with the `StochasticEvolution` opcode enabled and disabled to evaluate the performance delta from single kernel launches for the StochasticEvolution instruction pattern, versus the performance of the unfused loop. The results are shown in the following table:

| `M` paths | RunMat (fused opcode) | RunMat w/o opcode | Slowdown |
|-----------|----------------------:|------------------:|---------:|
| 250 K     | 7.9e1                 | 3.25e3            | 41×      |
| 500 K     | 8.6e1                 | 4.06e3            | 48×      |
| 1 M       | 1.02e2                | 5.66e3            | 55×      |
| 2 M       | 1.23e2                | 9.22e3            | 75×      |
| 5 M       | 2.24e2                | 2.18e4            | 97×      |

The results show that a fused StochasticEvolution opcode results in a 97x speedup over the unfused loop. Given how there is a relatively finite set of stateful math idioms, we expect that a layer of explicit VM ops will allow a large amount of math to be fused in RunMat.

Combined with loop fusion, we expect that a large amount of math will be fused in RunMat.