//! MATLAB-compatible `pause` builtin that temporarily suspends execution.

use once_cell::sync::Lazy;
use runmat_builtins::{CharArray, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::sync::RwLock;
use std::thread;
use std::time::Duration;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{runtime_error, BuiltinResult, RuntimeControlFlow};
#[cfg(not(test))]
use crate::interaction;
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "pause",
        builtin_path = "crate::builtins::timing::pause"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "pause"
category: "timing"
keywords: ["pause", "sleep", "wait", "delay", "press any key", "execution"]
summary: "Suspend execution until the user presses a key or a specified time elapses."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "pause executes entirely on the host CPU. GPU providers are never consulted and no residency changes occur."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::timing::pause::tests"
  integration: "builtins::timing::pause::tests::pause_gpu_duration_gathered"
---

# What does the `pause` function do in MATLAB / RunMat?
`pause` suspends execution and mirrors MATLAB's timing semantics:

- `pause` with no inputs waits for keyboard input (press any key) while pause mode is `on`.
- `pause(t)` delays execution for `t` seconds (non-negative numeric scalar). `t = Inf` behaves like `pause` with no arguments.
- `pause('on')` and `pause('off')` enable or disable pausing globally, returning the previous state (`'on'` or `'off'`).
- `pause('query')` reports the current state (`'on'` or `'off'`).
- `pause([])` is treated as `pause` with no arguments.
- When pause mode is `off`, delays and key waits complete immediately.

Invalid usages (negative times, non-scalar numeric inputs, or unknown strings) raise `MATLAB:pause:InvalidInputArgument`, matching MATLAB diagnostics.

## GPU Execution and Residency
`pause` never runs on the GPU. When you pass GPU-resident values (for example, `pause(gpuArray(0.5))`), RunMat automatically gathers them to the host before evaluating the delay. No residency changes occur otherwise, and acceleration providers do not receive any callbacks.

## Examples of using the `pause` function in MATLAB / RunMat

### Pausing for a fixed duration
```matlab
tic;
pause(0.05);     % wait 50 milliseconds
elapsed = toc;
```

### Waiting for user input mid-script
```matlab
disp("Press any key to continue the demo...");
pause;           % waits until the user presses a key (while pause is 'on')
```

### Temporarily disabling pauses in automated runs
```matlab
state = pause('off');   % returns previous state so it can be restored
cleanup = onCleanup(@() pause(state));  % ensure state is restored
pause(1.0);             % returns immediately because pause is disabled
```

### Querying the current pause mode
```matlab
current = pause('query');   % returns 'on' or 'off'
```

### Using empty input to rely on the default behaviour
```matlab
pause([]);   % equivalent to calling pause with no arguments
```

## FAQ

1. **Does `pause` block forever when standard input is not interactive?** No. When RunMat detects a non-interactive standard input (for example, during automated tests), `pause` completes immediately even in `'on'` mode.
2. **What happens if I call `pause` with a negative duration?** RunMat raises `MATLAB:pause:InvalidInputArgument`, matching MATLAB.
3. **Does `pause` accept logical or integer values?** Yes. Logical and integer inputs are converted to doubles before evaluating the delay.
4. **Can I force pausing off globally?** Use `pause('off')` to disable pauses. Record the return value so you can restore the prior state with `pause(previousState)`.
5. **Does `pause('query')` change the pause state?** No. It simply reports the current mode (`'on'` or `'off'`).
6. **Is `pause` affected by GPU fusion or auto-offload?** No. The builtin runs on the host regardless of fusion plans or acceleration providers.
7. **What is the default pause state?** `'on'`. Every RunMat session starts with pausing enabled.
8. **Can I pass a gpuArray as the duration?** Yes. RunMat gathers the scalar duration to the host before evaluating the delay.

## See Also
[tic](./tic), [toc](./toc), [timeit](./timeit)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/timing/pause.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/pause.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::timing::pause")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pause",
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
    notes: "pause executes entirely on the host. Acceleration providers are never queried.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::timing::pause")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pause",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "pause suspends host execution and is excluded from fusion pipelines.",
};

static PAUSE_STATE: Lazy<RwLock<PauseState>> = Lazy::new(|| RwLock::new(PauseState::default()));

#[cfg(test)]
use std::sync::Mutex;
#[cfg(test)]
pub(crate) static TEST_GUARD: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[derive(Debug, Clone, Copy)]
struct PauseState {
    enabled: bool,
}

impl Default for PauseState {
    fn default() -> Self {
        Self { enabled: true }
    }
}

const ERR_INVALID_ARG: &str = "MATLAB:pause:InvalidInputArgument";
const ERR_TOO_MANY_INPUTS: &str = "MATLAB:pause:TooManyInputs";
const ERR_STATE_LOCK: &str = "pause: failed to acquire pause state";

#[derive(Debug, Clone, Copy)]
enum PauseArgument {
    Wait(PauseWait),
    SetState(bool),
    Query,
}

#[derive(Debug, Clone, Copy)]
enum PauseWait {
    Default,
    Seconds(f64),
}

/// Suspend execution according to MATLAB-compatible pause semantics.
#[runtime_builtin(
    name = "pause",
    category = "timing",
    summary = "Suspend execution until a key press or specified duration.",
    keywords = "pause,sleep,wait,delay",
    accel = "metadata",
    sink = true,
    builtin_path = "crate::builtins::timing::pause"
)]
fn pause_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.len() {
        0 => {
            perform_wait(PauseWait::Default)?;
            Ok(empty_return_value())
        }
        1 => match classify_argument(&args[0])? {
            PauseArgument::Wait(wait) => {
                perform_wait(wait)?;
                Ok(empty_return_value())
            }
            PauseArgument::SetState(next_state) => {
                let previous = set_pause_enabled(next_state).map_err(Into::into)?;
                Ok(state_value(previous))
            }
            PauseArgument::Query => {
                let current =
                    pause_enabled().map_err(Into::into)?;
                Ok(state_value(current))
            }
        },
        _ => Err(runtime_error(ERR_TOO_MANY_INPUTS).build().into()),
    }
}

fn perform_wait(wait: PauseWait) -> Result<(), RuntimeControlFlow> {
    if !pause_enabled().map_err(Into::into)? {
        return Ok(());
    }

    match wait {
        PauseWait::Default => wait_for_key_press(),
        PauseWait::Seconds(seconds) => {
            if seconds == 0.0 {
                return Ok(());
            }
            // from_secs_f64 rejects NaN/±Inf; classify_argument filters those earlier.
            let duration = Duration::from_secs_f64(seconds);
            thread::sleep(duration);
            Ok(())
        }
    }
}

fn wait_for_key_press() -> Result<(), RuntimeControlFlow> {
    #[cfg(test)]
    {
        Ok(())
    }
    #[cfg(not(test))]
    {
        interaction::wait_for_key("")
    }
}

fn classify_argument(arg: &Value) -> Result<PauseArgument, RuntimeControlFlow> {
    let host_value = gpu_helpers::gather_value(arg)
        .map_err(|e| runtime_error(format!("pause: {e}")).build().into())?;
    match host_value {
        Value::String(text) => parse_command(&text).map_err(Into::into),
        Value::CharArray(ca) => {
            if ca.rows == 0 || ca.data.is_empty() {
                Ok(PauseArgument::Wait(PauseWait::Default))
            } else if ca.rows == 1 {
                let text: String = ca.data.iter().collect();
                parse_command(&text).map_err(Into::into)
            } else {
                Err(runtime_error(ERR_INVALID_ARG).build().into())
            }
        }
        Value::StringArray(sa) => {
            if sa.data.is_empty() {
                Ok(PauseArgument::Wait(PauseWait::Default))
            } else if sa.data.len() == 1 {
                parse_command(&sa.data[0]).map_err(Into::into)
            } else {
                Err(runtime_error(ERR_INVALID_ARG).build().into())
            }
        }
        Value::Num(value) => parse_numeric(value).map_err(Into::into),
        Value::Int(int_value) => parse_numeric(int_value.to_f64()).map_err(Into::into),
        Value::Bool(flag) => parse_numeric(if flag { 1.0 } else { 0.0 }).map_err(Into::into),
        Value::Tensor(tensor) => parse_tensor(tensor).map_err(Into::into),
        Value::LogicalArray(logical) => parse_logical(logical).map_err(Into::into),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            parse_tensor(tensor)
        }
        Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(runtime_error(ERR_INVALID_ARG).build().into()),
    }
}

fn parse_command(raw: &str) -> Result<PauseArgument, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(PauseArgument::Wait(PauseWait::Default));
    }
    let lower = trimmed.to_ascii_lowercase();
    match lower.as_str() {
        "on" => Ok(PauseArgument::SetState(true)),
        "off" => Ok(PauseArgument::SetState(false)),
        "query" => Ok(PauseArgument::Query),
        _ => Err(ERR_INVALID_ARG.to_string()),
    }
}

fn parse_numeric(value: f64) -> Result<PauseArgument, String> {
    if !value.is_finite() {
        if value.is_sign_positive() {
            return Ok(PauseArgument::Wait(PauseWait::Default));
        }
        return Err(ERR_INVALID_ARG.to_string());
    }
    if value < 0.0 {
        return Err(ERR_INVALID_ARG.to_string());
    }
    Ok(PauseArgument::Wait(PauseWait::Seconds(value)))
}

fn parse_tensor(tensor: Tensor) -> Result<PauseArgument, String> {
    if tensor.data.is_empty() {
        return Ok(PauseArgument::Wait(PauseWait::Default));
    }
    if tensor.data.len() != 1 {
        return Err(ERR_INVALID_ARG.to_string());
    }
    parse_numeric(tensor.data[0])
}

fn parse_logical(logical: LogicalArray) -> Result<PauseArgument, String> {
    if logical.data.is_empty() {
        return Ok(PauseArgument::Wait(PauseWait::Default));
    }
    if logical.data.len() != 1 {
        return Err(ERR_INVALID_ARG.to_string());
    }
    let scalar = if logical.data[0] != 0 { 1.0 } else { 0.0 };
    parse_numeric(scalar)
}

fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn state_value(enabled: bool) -> Value {
    let text = if enabled { "on" } else { "off" };
    Value::CharArray(CharArray::new_row(text))
}

fn pause_enabled() -> Result<bool, String> {
    PAUSE_STATE
        .read()
        .map(|guard| guard.enabled)
        .map_err(|_| ERR_STATE_LOCK.to_string())
}

fn set_pause_enabled(next: bool) -> Result<bool, String> {
    let mut guard = PAUSE_STATE
        .write()
        .map_err(|_| ERR_STATE_LOCK.to_string())?;
    let previous = guard.enabled;
    guard.enabled = next;
    Ok(previous)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, LogicalArray, Tensor};

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;

    fn reset_state(enabled: bool) {
        let mut guard = PAUSE_STATE.write().unwrap_or_else(|e| e.into_inner());
        guard.enabled = enabled;
    }

    fn char_array_to_string(value: Value) -> String {
        match value {
            Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect(),
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn query_returns_on_by_default() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let result = pause_builtin(vec![Value::from("query")]).expect("pause query");
        assert_eq!(char_array_to_string(result), "on");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pause_off_returns_previous_state() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let previous = pause_builtin(vec![Value::from("off")]).expect("pause off");
        assert_eq!(char_array_to_string(previous), "on");
        assert!(!pause_enabled().unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pause_on_restores_state() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(false);
        let previous = pause_builtin(vec![Value::from("on")]).expect("pause on");
        assert_eq!(char_array_to_string(previous), "off");
        assert!(pause_enabled().unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pause_default_returns_empty_tensor() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let result = pause_builtin(Vec::new()).expect("pause()");
        match result {
            Value::Tensor(t) => assert_eq!(t.data.len(), 0),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_zero_is_accepted() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let result = pause_builtin(vec![Value::Num(0.0)]).expect("pause(0)");
        match result {
            Value::Tensor(t) => assert_eq!(t.data.len(), 0),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integer_scalar_is_accepted() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let result = pause_builtin(vec![Value::Int(IntValue::I32(0))]).expect("pause(int)");
        match result {
            Value::Tensor(t) => assert_eq!(t.data.len(), 0),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_negative_zero_is_treated_as_zero() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let result = pause_builtin(vec![Value::Num(-0.0)]).expect("pause(-0)");
        match result {
            Value::Tensor(t) => assert_eq!(t.data.len(), 0),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn negative_duration_raises_error() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let err = pause_builtin(vec![Value::Num(-0.1)]).unwrap_err();
        assert_eq!(err, ERR_INVALID_ARG);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_scalar_tensor_is_rejected() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = pause_builtin(vec![Value::Tensor(tensor)]).unwrap_err();
        assert_eq!(err, ERR_INVALID_ARG);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_tensor_behaves_like_default_pause() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_state(true);
        let empty = Tensor::zeros(vec![0, 0]);
        let result = pause_builtin(vec![Value::Tensor(empty)]).expect("pause([])");
        match result {
            Value::Tensor(t) => assert_eq!(t.data.len(), 0),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_scalar_is_accepted() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_state(true);
        let logical = LogicalArray::new(vec![1u8], vec![1, 1]).unwrap();
        let result = pause_builtin(vec![Value::LogicalArray(logical)]).expect("pause(true)");
        match result {
            Value::Tensor(t) => assert_eq!(t.data.len(), 0),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn infinite_duration_behaves_like_default() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_state(true);
        let result = pause_builtin(vec![Value::Num(f64::INFINITY)]).expect("pause(Inf)");
        match result {
            Value::Tensor(t) => assert_eq!(t.data.len(), 0),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pause_gpu_duration_gathered() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_state(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = pause_builtin(vec![Value::GpuTensor(handle)]).expect("pause(gpuScalar)");
            match result {
                Value::Tensor(t) => assert_eq!(t.data.len(), 0),
                other => panic!("expected empty tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pause_wgpu_duration_gathered() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_state(true);
        if wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default())
            .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = pause_builtin(vec![Value::GpuTensor(handle)]).expect("pause(gpuScalar)");
        match result {
            Value::Tensor(t) => assert_eq!(t.data.len(), 0),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_command_raises_error() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_state(true);
        let err = pause_builtin(vec![Value::from("invalid")]).unwrap_err();
        assert_eq!(err, ERR_INVALID_ARG);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let _guard = TEST_GUARD.lock().unwrap();
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
