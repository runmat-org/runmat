# Cooperative cancellation microbenchmark

Run this standalone Rust harness on a quiet native machine, outside concurrent product builds:

```sh
cargo bench -p runmat-execution --bench cancellation -- 1000000 9
```

Cargo's bench profile supplies release optimization. The two optional positional arguments are iterations per sample and samples per case. Output is CSV with nanoseconds per operation for every sample; compilation is outside the timed region. Record the commit, Rust version, target, CPU, power mode, profile overrides and command alongside results. Repeat the complete run before drawing conclusions. No numbers are supplied here because this harness has not been executed for qualification.

The single-token uncancelled check is compared with an `Arc<AtomicBool>` acquire load using the same loop and black-box structure. This is a fair baseline for the earlier polling check, not an equivalent replacement for wakeable cancellation. The two-parent check additionally measures composed-token traversal.

Listener-cycle measurements include constructing the cancellation future, first polling/registration, and dropping/deregistering it. Source generation and waker construction are excluded; registry capacity is primed. The two-parent variant registers against both authorities. The retained-listener case measures repolling an already registered listener with the same waker. The pending-future baseline uses the same poll/context machinery without cancellation registration: it indicates harness overhead, but does not supply equivalent cancellation semantics. Do not present its difference as a measured engine speedup.

Inspect all samples and summarize their median and spread per case. Case order rotates after an untimed warmup. A noisy or timer-resolution-limited result requires longer runs or a quieter machine, not an acceptance threshold adjustment. There is no pass/fail timing threshold and no performance assertion in functional tests.

This measures uncontended authority overhead, not signal-to-resume latency, scheduler contention, cold-generation allocation, actual VM/native throughput, GPU completion, or noncooperative foreign interruption. The counting/reentrant-waker and register/cancel race tests remain separate correctness evidence. Engine-level performance needs a separately controlled workload; no end-to-end improvement is claimed by this microbenchmark.
