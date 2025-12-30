//! MATLAB-compatible `timeit` builtin for RunMat.
//!
//! Measures the execution time of zero-input function handles by running them
//! repeatedly and returning the median per-invocation runtime in seconds.

use runmat_time::Instant;
use std::cmp::Ordering;

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

const TARGET_BATCH_SECONDS: f64 = 0.005;
const MAX_BATCH_SECONDS: f64 = 0.25;
const LOOP_COUNT_LIMIT: usize = 1 << 20;
const MIN_SAMPLE_COUNT: usize = 7;
const MAX_SAMPLE_COUNT: usize = 21;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "timeit",
        builtin_path = "crate::builtins::timing::timeit"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "timeit"
category: "timing"
keywords: ["timeit", "benchmark", "timing", "performance", "gpu"]
summary: "Measure the execution time of a zero-argument function handle."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs purely on the host CPU; GPU work inside the timed function executes through the active provider."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::timing::timeit::tests"
  integration: "builtins::timing::timeit::tests::timeit_measures_time"
---

# Measure execution time of a function handle
`t = timeit(f)` evaluates the zero-argument function handle `f` repeatedly and returns the median runtime (in seconds).
RunMat accepts the optional `numOutputs` argument for MATLAB compatibility; today the handle executes with its default output arity (or none when you pass `0`) and all returned values are discarded.

## Syntax
```matlab
t = timeit(f)
t = timeit(f, numOutputs)
```

- `f` is a zero-argument function handle (for example, `@() myOp(A)`).
- `numOutputs` is an optional nonnegative integer kept for MATLAB compatibility. Passing `0` suppresses outputs entirely; any other value currently executes the handle with its default output arity while discarding the result.

## How `timeit` works
- Executes `f` repeatedly, adjusting the inner loop count until a single batch takes at least a few milliseconds or the function is slow enough.
- Collects multiple samples (at least seven batches) and returns the median per-invocation time, which is robust against outliers.
- Drops the outputs produced by `f`; you should perform any validation that depends on those outputs inside the handle. Multi-output dispatch will route through this helper once the runtime exposes multi-return `feval`.
- Leaves GPU residency untouched—if `f` launches GPU kernels, they execute on the active provider. Insert `wait(gpuDevice)` inside the handle if you need explicit synchronisation.

## Examples

### Timing a simple anonymous function
```matlab
f = @() sin(rand(1000, 1));
t = timeit(f);
```

### Comparing two implementations
```matlab
A = rand(1e5, 1);
slow = @() sum(A .* A);
fast = @() sumsq(A);

slowTime = timeit(slow);
fastTime = timeit(fast);
```

### Timing a function that returns no outputs
```matlab
logMessage = @() fprintf("Iteration complete\n");
t = timeit(logMessage, 0);
```

### Timing a multiple-output function
```matlab
svdTime = timeit(@() svd(rand(256)), 3);
```
This records the runtime while discarding any outputs produced by `svd`.

### Measuring GPU-bound work
```matlab
gfun = @() gather(sin(gpuArray.rand(4096, 1)));
tgpu = timeit(gfun);
```

### Timing a preallocation helper
```matlab
makeMatrix = @() zeros(2048, 2048);
t = timeit(makeMatrix);
```

## FAQ

1. **What does `timeit` return?** — A scalar double containing the median runtime per invocation in seconds.
2. **How many runs does `timeit` perform?** — It automatically selects a loop count so each batch lasts a few milliseconds, collecting at least seven batches.
3. **Does `timeit` synchronise GPU kernels?** — No. Insert `wait(gpuDevice)` inside the handle when you need explicit synchronisation.
4. **Can I time functions that require inputs?** — Yes. Capture them in the handle, for example `timeit(@() myfun(A, B))`.
5. **How do I time a function with multiple outputs?** — Pass `timeit(@() svd(A), 3)` to mirror MATLAB’s call signature. RunMat currently ignores values greater than zero until multi-output dispatch lands, but the handle still executes.
6. **Why do successive runs differ slightly?** — Normal system jitter, cache effects, and GPU scheduling can change runtimes slightly; the median mitigates outliers.
7. **Can `timeit` time scripts?** — Wrap the script body in a function handle so it becomes zero-argument, then call `timeit` on that handle.
8. **Does `timeit` participate in fusion or JIT tiers?** — It simply executes the provided handle; any tiering or fusion happens inside the timed function.
9. **What happens if the function errors?** — The error is propagated immediately and timing stops, matching MATLAB behaviour.
10. **Is there a limit on runs?** — Yes. `timeit` caps the inner loop at about one million iterations to avoid runaway measurements.

## See Also
[tic](./tic), [toc](./toc), [feval](../introspection/feval)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/timing/timeit.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/timeit.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::timing::timeit")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "timeit",
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
    notes: "Host-side helper; GPU kernels execute only if invoked by the timed function.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::timing::timeit")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "timeit",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Timing helper; excluded from fusion planning.",
};

#[runtime_builtin(
    name = "timeit",
    category = "timing",
    summary = "Measure the execution time of a zero-argument function handle.",
    keywords = "timeit,benchmark,timing,performance,gpu",
    accel = "helper",
    builtin_path = "crate::builtins::timing::timeit"
)]
fn timeit_builtin(func: Value, rest: Vec<Value>) -> Result<Value, String> {
    let requested_outputs = parse_num_outputs(&rest)?;
    let callable = prepare_callable(func, requested_outputs)?;

    // Warm-up once to catch early errors and pay one-time JIT costs.
    callable.invoke()?;

    let loop_count = determine_loop_count(&callable)?;
    let samples = collect_samples(&callable, loop_count)?;
    if samples.is_empty() {
        return Ok(Value::Num(0.0));
    }

    Ok(Value::Num(compute_median(samples)))
}

fn parse_num_outputs(rest: &[Value]) -> Result<Option<usize>, String> {
    match rest.len() {
        0 => Ok(None),
        1 => parse_non_negative_integer(&rest[0]).map(Some),
        _ => Err("timeit: too many input arguments".to_string()),
    }
}

fn parse_non_negative_integer(value: &Value) -> Result<usize, String> {
    match value {
        Value::Int(iv) => {
            let raw = iv.to_i64();
            if raw < 0 {
                Err("timeit: numOutputs must be a nonnegative integer".to_string())
            } else {
                Ok(raw as usize)
            }
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("timeit: numOutputs must be finite".to_string());
            }
            if *n < 0.0 {
                return Err("timeit: numOutputs must be a nonnegative integer".to_string());
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err("timeit: numOutputs must be an integer value".to_string());
            }
            Ok(rounded as usize)
        }
        _ => Err("timeit: numOutputs must be a scalar numeric value".to_string()),
    }
}

fn determine_loop_count(callable: &TimeitCallable) -> Result<usize, String> {
    let mut loops = 1usize;
    loop {
        let elapsed = run_batch(callable, loops)?;
        if elapsed >= TARGET_BATCH_SECONDS
            || elapsed >= MAX_BATCH_SECONDS
            || loops >= LOOP_COUNT_LIMIT
        {
            return Ok(loops);
        }
        loops = loops.saturating_mul(2);
        if loops == 0 {
            return Ok(LOOP_COUNT_LIMIT);
        }
    }
}

fn collect_samples(callable: &TimeitCallable, loop_count: usize) -> Result<Vec<f64>, String> {
    let mut samples = Vec::with_capacity(MIN_SAMPLE_COUNT);
    while samples.len() < MIN_SAMPLE_COUNT {
        let elapsed = run_batch(callable, loop_count)?;
        let per_iter = elapsed / loop_count as f64;
        samples.push(per_iter);
        if samples.len() >= MAX_SAMPLE_COUNT || elapsed >= MAX_BATCH_SECONDS {
            break;
        }
    }
    Ok(samples)
}

fn run_batch(callable: &TimeitCallable, loop_count: usize) -> Result<f64, String> {
    let start = Instant::now();
    for _ in 0..loop_count {
        let value = callable.invoke()?;
        drop(value);
    }
    Ok(start.elapsed().as_secs_f64())
}

fn compute_median(mut samples: Vec<f64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or_else(|| {
            if a < b {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }),
    });
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        (samples[mid - 1] + samples[mid]) * 0.5
    }
}

#[derive(Clone)]
struct TimeitCallable {
    handle: Value,
    num_outputs: Option<usize>,
}

impl TimeitCallable {
    fn invoke(&self) -> Result<Value, String> {
        // The runtime currently treats all builtin invocations as returning a single `Value`.
        // The optional `num_outputs` flag is stored so future multi-output support can
        // request the correct number of outputs when dispatching through `feval`.
        // For now, we invoke the handle normally and drop whatever value is produced.
        if let Some(0) = self.num_outputs {
            let value = crate::call_builtin("feval", std::slice::from_ref(&self.handle))?;
            drop(value);
            Ok(Value::Num(0.0))
        } else {
            crate::call_builtin("feval", std::slice::from_ref(&self.handle))
        }
    }
}

fn prepare_callable(func: Value, num_outputs: Option<usize>) -> Result<TimeitCallable, String> {
    match func {
        Value::String(text) => parse_handle_string(&text).map(|handle| TimeitCallable {
            handle: Value::String(handle),
            num_outputs,
        }),
        Value::CharArray(arr) => {
            if arr.rows != 1 {
                Err(
                    "timeit: function handle must be a string scalar or function handle"
                        .to_string(),
                )
            } else {
                let text: String = arr.data.iter().collect();
                parse_handle_string(&text).map(|handle| TimeitCallable {
                    handle: Value::String(handle),
                    num_outputs,
                })
            }
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                parse_handle_string(&sa.data[0]).map(|handle| TimeitCallable {
                    handle: Value::String(handle),
                    num_outputs,
                })
            } else {
                Err(
                    "timeit: function handle must be a string scalar or function handle"
                        .to_string(),
                )
            }
        }
        Value::FunctionHandle(name) => Ok(TimeitCallable {
            handle: Value::String(format!("@{name}")),
            num_outputs,
        }),
        Value::Closure(closure) => Ok(TimeitCallable {
            handle: Value::Closure(closure),
            num_outputs,
        }),
        other => Err(format!(
            "timeit: first argument must be a function handle, got {other:?}"
        )),
    }
}

fn parse_handle_string(text: &str) -> Result<String, String> {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix('@') {
        if rest.trim().is_empty() {
            Err("timeit: empty function handle string".to_string())
        } else {
            Ok(format!("@{}", rest.trim()))
        }
    } else {
        Err("timeit: expected a function handle string beginning with '@'".to_string())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::IntValue;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::builtins::common::test_support;

    static COUNTER_DEFAULT: AtomicUsize = AtomicUsize::new(0);
    static COUNTER_NUM_OUTPUTS: AtomicUsize = AtomicUsize::new(0);
    static COUNTER_INVALID: AtomicUsize = AtomicUsize::new(0);
    static COUNTER_ZERO_OUTPUTS: AtomicUsize = AtomicUsize::new(0);

    #[runtime_builtin(
        name = "__timeit_helper_counter_default",
        builtin_path = "crate::builtins::timing::timeit::tests"
    )]
    fn helper_counter_default() -> Result<Value, String> {
        COUNTER_DEFAULT.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Num(1.0))
    }

    #[runtime_builtin(
        name = "__timeit_helper_counter_outputs",
        builtin_path = "crate::builtins::timing::timeit::tests"
    )]
    fn helper_counter_outputs() -> Result<Value, String> {
        COUNTER_NUM_OUTPUTS.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Num(1.0))
    }

    #[runtime_builtin(
        name = "__timeit_helper_counter_invalid",
        builtin_path = "crate::builtins::timing::timeit::tests"
    )]
    fn helper_counter_invalid() -> Result<Value, String> {
        COUNTER_INVALID.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Num(1.0))
    }

    #[runtime_builtin(
        name = "__timeit_helper_zero_outputs",
        builtin_path = "crate::builtins::timing::timeit::tests"
    )]
    fn helper_counter_zero_outputs() -> Result<Value, String> {
        COUNTER_ZERO_OUTPUTS.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Num(0.0))
    }

    fn default_handle() -> Value {
        Value::String("@__timeit_helper_counter_default".to_string())
    }

    fn outputs_handle() -> Value {
        Value::String("@__timeit_helper_counter_outputs".to_string())
    }

    fn invalid_handle() -> Value {
        Value::String("@__timeit_helper_counter_invalid".to_string())
    }

    fn zero_outputs_handle() -> Value {
        Value::String("@__timeit_helper_zero_outputs".to_string())
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_measures_time() {
        COUNTER_DEFAULT.store(0, Ordering::SeqCst);
        let result = timeit_builtin(default_handle(), Vec::new()).expect("timeit");
        match result {
            Value::Num(v) => assert!(v >= 0.0),
            other => panic!("expected numeric result, got {other:?}"),
        }
        assert!(
            COUNTER_DEFAULT.load(Ordering::SeqCst) >= MIN_SAMPLE_COUNT,
            "expected at least {} invocations",
            MIN_SAMPLE_COUNT
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_accepts_num_outputs_argument() {
        COUNTER_NUM_OUTPUTS.store(0, Ordering::SeqCst);
        let args = vec![Value::Int(IntValue::I32(3))];
        let _ = timeit_builtin(outputs_handle(), args).expect("timeit numOutputs");
        assert!(
            COUNTER_NUM_OUTPUTS.load(Ordering::SeqCst) >= MIN_SAMPLE_COUNT,
            "expected at least {} invocations",
            MIN_SAMPLE_COUNT
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_supports_zero_outputs() {
        COUNTER_ZERO_OUTPUTS.store(0, Ordering::SeqCst);
        let args = vec![Value::Int(IntValue::I32(0))];
        let _ = timeit_builtin(zero_outputs_handle(), args).expect("timeit zero outputs");
        assert!(
            COUNTER_ZERO_OUTPUTS.load(Ordering::SeqCst) >= MIN_SAMPLE_COUNT,
            "expected at least {} invocations",
            MIN_SAMPLE_COUNT
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn timeit_runs_with_wgpu_provider_registered() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let result = timeit_builtin(default_handle(), Vec::new()).expect("timeit with wgpu");
        match result {
            Value::Num(v) => assert!(v >= 0.0),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_rejects_non_function_input() {
        let err = timeit_builtin(Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(
            err.to_ascii_lowercase().contains("function"),
            "unexpected error text: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_rejects_invalid_num_outputs() {
        COUNTER_INVALID.store(0, Ordering::SeqCst);
        let err = timeit_builtin(invalid_handle(), vec![Value::Num(-1.0)]).unwrap_err();
        assert!(err.to_ascii_lowercase().contains("nonnegative"));
        assert_eq!(COUNTER_INVALID.load(Ordering::SeqCst), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn timeit_rejects_extra_arguments() {
        let err =
            timeit_builtin(default_handle(), vec![Value::from(1.0), Value::from(2.0)]).unwrap_err();
        assert!(err.to_ascii_lowercase().contains("too many"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
