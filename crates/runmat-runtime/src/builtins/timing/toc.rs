//! MATLAB-compatible `toc` builtin that reports elapsed stopwatch time.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_time::Instant;
use std::convert::TryFrom;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::timing::tic::{decode_handle, take_latest_start};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "toc", wasm_path = "crate::builtins::timing::toc")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "toc"
category: "timing"
keywords: ["toc", "timer", "elapsed time", "profiling", "benchmark"]
summary: "Read the elapsed time since the most recent tic or an explicit handle."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "toc always runs on the host CPU. GPU providers are not consulted."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::timing::toc::tests"
  integration: "builtins::timing::toc::tests"
---

# What does the `toc` function do in MATLAB / RunMat?
`toc` returns the elapsed wall-clock time in seconds since the last matching `tic`, or since the `tic`
handle you pass as an argument. It mirrors the stopwatch utilities that MATLAB users rely on for ad-hoc
profiling and benchmarking.

## How does the `toc` function behave in MATLAB / RunMat?
- `toc` without inputs pops the most recent `tic` from the stopwatch stack and returns the elapsed seconds.
- `toc(t)` accepts a handle previously produced by `tic` and measures the time since that handle without
  altering the stack.
- Calling `toc` before `tic` raises the MATLAB-compatible error identifier `MATLAB:toc:NoMatchingTic`.
- Passing anything other than a finite, non-negative scalar handle raises `MATLAB:toc:InvalidTimerHandle`.
- The stopwatch uses a monotonic host clock, so measurements are immune to wall-clock adjustments.

## `toc` Function GPU Execution Behaviour
The stopwatch lives entirely on the host. `toc` never transfers tensors or consults acceleration providers,
so there are no GPU hooks to implement. Expressions that combine `toc` with GPU-resident data gather any
numeric operands back to the CPU before evaluating the timer logic, and the builtin is excluded from fusion
plans entirely.

## Examples of using the `toc` function in MATLAB / RunMat

### Measuring elapsed time since the last tic

```matlab
tic;
pause(0.25);
elapsed = toc;
```

`elapsed` contains the seconds since the `tic`. The matching stopwatch entry is removed automatically.

### Using toc with an explicit tic handle

```matlab
token = tic;
heavyComputation();
elapsed = toc(token);
```

Passing the handle makes `toc` leave the global stopwatch stack untouched, so earlier timers keep running.

### Timing nested stages with toc

```matlab
tic;          % Outer stopwatch
stage1();
inner = tic;  % Nested stopwatch
stage2();
stage2Time = toc(inner);
totalTime = toc;
```

`stage2Time` measures only the inner section, while `totalTime` spans the entire outer region.

### Printing elapsed time without capturing output

```matlab
tic;
longRunningTask();
toc;   % Displays the elapsed seconds because the result is not assigned
```

When you omit an output argument, RunMat displays the elapsed seconds in the console. Add a semicolon or
capture the result to suppress the text, mirroring MATLAB's default command-window behaviour.

### Measuring immediately with toc(tic)

```matlab
elapsed = toc(tic);  % Starts a timer and reads it right away
```

This idiom is equivalent to separate `tic`/`toc` calls, and the stopwatch entry created by the inner `tic`
remains on the stack for later use.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. Timing utilities never touch GPU memory. You can freely combine `toc` with code that produces or consumes
`gpuArray` values—the stopwatch itself still executes on the CPU.

## FAQ

### What happens if I call `toc` before `tic`?
The builtin raises `MATLAB:toc:NoMatchingTic`, matching MATLAB's behaviour when no stopwatch start exists.

### Does `toc` remove the matching `tic`?
Yes when called without arguments. The most-recent stopwatch entry is popped so nested timers unwind in order.
When you pass a handle (`toc(t)`), the stack remains unchanged and you may reuse the handle multiple times.

### Can I reuse a `tic` handle after calling `toc(t)`?
Yes. Handles are deterministic timestamps, so you can call `toc(handle)` multiple times or store the handle in
structures for later inspection.

### Does `toc` print output?
When you do not capture the result, the interpreter shows the elapsed seconds. Assigning the return value (or
ending the statement with a semicolon) suppresses the display, just like in MATLAB.

### Is `toc` affected by GPU execution or fusion?
No. The stopwatch uses the host's monotonic clock. GPU acceleration, fusion, and pipeline residency do not
change the measured interval.

### How accurate is the reported time?
`toc` relies on the same monotonic clock (`runmat_time::Instant`), typically offering microsecond precision on modern platforms. The actual
resolution depends on your operating system.

## See Also
[tic](./tic)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/timing/toc.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/toc.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::timing::toc")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "toc",
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
    notes: "Stopwatch state lives on the host. Providers are never consulted for toc.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::timing::toc")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "toc",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Timing builtins execute eagerly on the host and do not participate in fusion.",
};

const ERR_NO_MATCHING_TIC: &str = "MATLAB:toc:NoMatchingTic";
const ERR_INVALID_HANDLE: &str = "MATLAB:toc:InvalidTimerHandle";
const ERR_TOO_MANY_INPUTS: &str = "MATLAB:toc:TooManyInputs";

/// Read elapsed time from the stopwatch stack or a specific handle.
#[runtime_builtin(
    name = "toc",
    category = "timing",
    summary = "Read the elapsed time since the most recent tic or an explicit handle.",
    keywords = "toc,timing,profiling,benchmark",
    wasm_path = "crate::builtins::timing::toc"
)]
pub fn toc_builtin(args: Vec<Value>) -> Result<f64, String> {
    match args.len() {
        0 => latest_elapsed(),
        1 => elapsed_from_value(&args[0]),
        _ => Err(ERR_TOO_MANY_INPUTS.to_string()),
    }
}

fn latest_elapsed() -> Result<f64, String> {
    let start = take_latest_start()?.ok_or_else(|| ERR_NO_MATCHING_TIC.to_string())?;
    Ok(start.elapsed().as_secs_f64())
}

fn elapsed_from_value(value: &Value) -> Result<f64, String> {
    let handle = f64::try_from(value).map_err(|_| ERR_INVALID_HANDLE.to_string())?;
    let instant = decode_handle(handle)?;
    let now = Instant::now();
    let elapsed = now
        .checked_duration_since(instant)
        .ok_or_else(|| ERR_INVALID_HANDLE.to_string())?;
    Ok(elapsed.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::timing::tic::{encode_instant, record_tic, take_latest_start, TEST_GUARD};
    use std::time::Duration;

    use crate::builtins::common::test_support;

    fn clear_tic_stack() {
        while let Ok(Some(_)) = take_latest_start() {}
    }

    #[test]
    fn toc_requires_matching_tic() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let err = toc_builtin(Vec::new()).unwrap_err();
        assert_eq!(err, ERR_NO_MATCHING_TIC);
    }

    #[test]
    fn toc_reports_elapsed_for_latest_start() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        record_tic().expect("tic");
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = toc_builtin(Vec::new()).expect("toc");
        assert!(elapsed >= 0.0);
        assert!(take_latest_start().unwrap().is_none());
    }

    #[test]
    fn toc_with_handle_measures_without_popping_stack() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let handle = record_tic().expect("tic");
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = toc_builtin(vec![Value::Num(handle)]).expect("toc(handle)");
        assert!(elapsed >= 0.0);
        // Stack still contains the entry so a subsequent toc pops it.
        let later = toc_builtin(Vec::new()).expect("second toc");
        assert!(later >= elapsed);
    }

    #[test]
    fn toc_rejects_invalid_handle() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let err = toc_builtin(vec![Value::Num(f64::NAN)]).unwrap_err();
        assert_eq!(err, ERR_INVALID_HANDLE);
    }

    #[test]
    fn toc_rejects_future_handle() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let future_handle = encode_instant(Instant::now()) + 10_000.0;
        let err = toc_builtin(vec![Value::Num(future_handle)]).unwrap_err();
        assert_eq!(err, ERR_INVALID_HANDLE);
    }

    #[test]
    fn toc_rejects_string_handle() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let err = toc_builtin(vec![Value::from("not a timer")]).unwrap_err();
        assert_eq!(err, ERR_INVALID_HANDLE);
    }

    #[test]
    fn toc_rejects_extra_arguments() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let err = toc_builtin(vec![Value::Num(0.0), Value::Num(0.0)]).unwrap_err();
        assert_eq!(err, ERR_TOO_MANY_INPUTS);
    }

    #[test]
    fn toc_nested_timers() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        record_tic().expect("outer");
        std::thread::sleep(Duration::from_millis(2));
        record_tic().expect("inner");
        std::thread::sleep(Duration::from_millis(4));
        let inner = toc_builtin(Vec::new()).expect("inner toc");
        assert!(inner >= 0.0);
        std::thread::sleep(Duration::from_millis(2));
        let outer = toc_builtin(Vec::new()).expect("outer toc");
        assert!(outer >= inner);
    }

    #[test]
    fn doc_examples_present() {
        let _guard = TEST_GUARD.lock().unwrap();
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn toc_ignores_wgpu_provider() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        record_tic().expect("tic");
        std::thread::sleep(Duration::from_millis(1));
        let elapsed = toc_builtin(Vec::new()).expect("toc");
        assert!(elapsed >= 0.0);
    }
}
