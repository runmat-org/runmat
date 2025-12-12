//! MATLAB-compatible `tic` builtin with precise stopwatch semantics for RunMat.

use once_cell::sync::Lazy;
use runmat_macros::runtime_builtin;
use runmat_time::Instant;
use std::sync::Mutex;
use std::time::Duration;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "tic", builtin_path = "crate::builtins::timing::tic")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "tic"
category: "timing"
keywords: ["tic", "timer", "profile", "benchmark", "performance"]
summary: "Start a high-resolution stopwatch and optionally return a handle for toc."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Stopwatch helpers always run on the host CPU; GPU providers are not consulted."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::timing::tic::tests"
  integration: "runmat_runtime::io::tests::test_tic_toc"
---

# What does the `tic` function do in MATLAB / RunMat?
`tic` starts a high-resolution stopwatch. Calls to `toc` report the elapsed time in seconds. When you assign
the return value (for example, `t = tic;`), the resulting handle can be passed to `toc(t)` to measure a
different code region while keeping the global stopwatch untouched.

## How does the `tic` function behave in MATLAB / RunMat?
- Uses the host's monotonic clock for nanosecond-resolution timing.
- Supports nested timers: each call pushes a new start time on an internal stack. `toc` without inputs always
  reads the most recent `tic` and removes it, leaving earlier timers intact so outer scopes continue measuring.
- Returns an opaque scalar handle (a `double`) that encodes the monotonic timestamp. The handle can be stored
  or passed explicitly to `toc`.
- Executes entirely on the CPU. There are no GPU variants because `tic` interacts with wall-clock state.
- Calling `toc` before `tic` raises the MATLAB-compatible error `MATLAB:toc:NoMatchingTic`.

## How does `tic` behave with RunMat Accelerate?
`tic` never leaves the CPU. When called while tensors reside on the GPU, the stopwatch state stays on the
host. There are no acceleration-provider hooks for timers, so the runtime neither uploads nor gathers data.
Fusion plans skip the builtin entirely because it has no numeric inputs.

## Examples of using the `tic` function in MATLAB / RunMat

### Measuring a simple loop

```matlab
tic;
for k = 1:1e5
    sqrt(k);
end
elapsed = toc;
```

`elapsed` reports the seconds since the matching `tic`.

### Capturing and reusing the tic handle

```matlab
t = tic;
heavyComputation();
elapsed = toc(t);
```

Using the handle lets you insert additional timing regions without resetting the default stopwatch.

### Nesting timers for staged profiling

```matlab
tic;              % Outer stopwatch
stage1();         % Work you want to measure once
inner = tic;      % Nested stopwatch
stage2();
innerT = toc(inner);  % Elapsed time for stage2 only
outerT = toc;         % Elapsed time for everything since the first tic
```

`toc` without inputs reads the most recent `tic`, so nested regions work naturally.

### Measuring asynchronous work

```matlab
token = tic;
future = backgroundTask();
wait(future);
elapsed = toc(token);
```

Handles can be stored in structures or passed to callbacks while asynchronous work completes.

### Resetting the stopwatch after a measurement

```matlab
elapsed1 = toc(tic);  % Equivalent to separate tic/toc calls
pause(0.1);
elapsed2 = toc(tic);  % Starts a new timer immediately
```

Calling `toc(tic)` starts a new stopwatch and immediately measures it, mirroring MATLAB idioms.

## FAQ

### Does `tic` print anything when called without a semicolon?
No. `tic` is marked as a sink builtin, so scripts do not display the returned handle unless you assign it or
explicitly request output.

### Is the returned handle portable across sessions?
No. The handle encodes a monotonic timestamp that is only meaningful within the current RunMat process. Passing
it to another session or saving it to disk is undefined behaviour, matching MATLAB.

### Can I run `tic` on a worker thread?
Yes. Each thread shares the same stopwatch stack. Nested `tic`/`toc` pairs remain well-defined, but you should
serialise access at the script level to avoid interleaving unrelated timings.

### How accurate is the measurement?
`tic` relies on a monotonic clock (via `runmat_time::Instant`), typically providing microsecond or better precision. The actual resolution
depends on your operating system. There is no artificial jitter or throttling introduced by RunMat.

### Does `tic` participate in GPU fusion?
No. Timer builtins are tagged as CPU-only. Expressions containing `tic` are always executed on the host, and
any GPU-resident tensors are gathered automatically by surrounding code when necessary.

## See Also
[toc](./toc), [timeit](./timeit), [profile](../diagnostics/profile)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/timing/tic.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/tic.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::timing::tic")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tic",
    op_kind: GpuOpKind::Custom("timer"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Stopwatch state lives on the host. Providers are never consulted for tic/toc.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::timing::tic")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tic",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Timing builtins are executed eagerly on the host and do not participate in fusion.",
};

static MONOTONIC_ORIGIN: Lazy<Instant> = Lazy::new(Instant::now);
static STOPWATCH: Lazy<Mutex<StopwatchState>> = Lazy::new(|| Mutex::new(StopwatchState::default()));

#[cfg(test)]
pub(crate) static TEST_GUARD: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[derive(Default)]
struct StopwatchState {
    stack: Vec<Instant>,
}

impl StopwatchState {
    fn push(&mut self, instant: Instant) {
        self.stack.push(instant);
    }

    fn pop(&mut self) -> Option<Instant> {
        self.stack.pop()
    }
}

const LOCK_ERR: &str = "tic: failed to acquire stopwatch state";

/// Start a stopwatch timer and return a handle suitable for `toc`.
#[runtime_builtin(
    name = "tic",
    category = "timing",
    summary = "Start a stopwatch timer and optionally return a handle for toc.",
    keywords = "tic,timing,profiling,benchmark",
    sink = true,
    builtin_path = "crate::builtins::timing::tic"
)]
pub fn tic_builtin() -> Result<f64, String> {
    record_tic()
}

/// Record a `tic` start time and return the encoded handle.
pub(crate) fn record_tic() -> Result<f64, String> {
    let now = Instant::now();
    {
        let mut guard = STOPWATCH.lock().map_err(|_| LOCK_ERR.to_string())?;
        guard.push(now);
    }
    Ok(encode_instant(now))
}

/// Remove and return the most recently recorded `tic`, if any.
pub(crate) fn take_latest_start() -> Result<Option<Instant>, String> {
    let mut guard = STOPWATCH.lock().map_err(|_| LOCK_ERR.to_string())?;
    Ok(guard.pop())
}

/// Encode an `Instant` into the scalar handle returned by `tic`.
pub(crate) fn encode_instant(instant: Instant) -> f64 {
    instant.duration_since(*MONOTONIC_ORIGIN).as_secs_f64()
}

/// Decode a scalar handle into an `Instant`.
pub(crate) fn decode_handle(handle: f64) -> Result<Instant, String> {
    if !handle.is_finite() || handle.is_sign_negative() {
        return Err("MATLAB:toc:InvalidTimerHandle".to_string());
    }
    let duration = Duration::from_secs_f64(handle);
    Ok((*MONOTONIC_ORIGIN) + duration)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    use crate::builtins::common::test_support;

    fn reset_stopwatch() {
        let mut guard = STOPWATCH.lock().unwrap();
        guard.stack.clear();
    }

    #[test]
    fn tic_returns_monotonic_handle() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_stopwatch();
        let handle = tic_builtin().expect("tic");
        assert!(handle >= 0.0);
        assert!(take_latest_start().expect("take").is_some());
    }

    #[test]
    fn tic_handles_increase_over_time() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_stopwatch();
        let first = tic_builtin().expect("tic");
        thread::sleep(Duration::from_millis(5));
        let second = tic_builtin().expect("tic");
        assert!(second > first);
    }

    #[test]
    fn decode_roundtrip_matches_handle() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_stopwatch();
        let handle = tic_builtin().expect("tic");
        let decoded = decode_handle(handle).expect("decode");
        let round_trip = encode_instant(decoded);
        let delta = (round_trip - handle).abs();
        assert!(delta < 1e-9, "delta {delta}");
    }

    #[test]
    fn take_latest_start_pops_stack() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_stopwatch();
        tic_builtin().expect("tic");
        assert!(take_latest_start().expect("take").is_some());
        assert!(take_latest_start().expect("second take").is_none());
    }

    #[test]
    fn decode_handle_rejects_invalid_values() {
        let _guard = TEST_GUARD.lock().unwrap();
        assert!(decode_handle(f64::NAN).is_err());
        assert!(decode_handle(-1.0).is_err());
    }

    #[test]
    fn doc_examples_present() {
        let _guard = TEST_GUARD.lock().unwrap();
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
