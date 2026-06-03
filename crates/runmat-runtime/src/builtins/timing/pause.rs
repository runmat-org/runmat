//! MATLAB-compatible `pause` builtin that temporarily suspends execution.

use once_cell::sync::Lazy;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, LogicalArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use std::sync::RwLock;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
use crate::builtins::plotting;
use crate::builtins::timing::type_resolvers::pause_type;
#[cfg(not(test))]
use crate::interaction;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

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

const BUILTIN_NAME: &str = "pause";

const PAUSE_OUTPUT_EMPTY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty array when pausing or changing state.",
}];

const PAUSE_OUTPUT_STATE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "state",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Previous pause state ('on' or 'off').",
}];

const PAUSE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const PAUSE_INPUTS_DURATION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "duration",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("0"),
    description: "Duration scalar or command-like scalar value accepted by pause.",
}];
const PAUSE_INPUTS_COMMAND: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "command",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One of 'on', 'off', or 'query'.",
}];

const PAUSE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "out = pause()",
        inputs: &PAUSE_INPUTS_NONE,
        outputs: &PAUSE_OUTPUT_EMPTY,
    },
    BuiltinSignatureDescriptor {
        label: "out = pause(duration)",
        inputs: &PAUSE_INPUTS_DURATION,
        outputs: &PAUSE_OUTPUT_EMPTY,
    },
    BuiltinSignatureDescriptor {
        label: "state = pause(command)",
        inputs: &PAUSE_INPUTS_COMMAND,
        outputs: &PAUSE_OUTPUT_STATE,
    },
];

const PAUSE_ERROR_INVALID_ARG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAUSE.INVALID_ARG",
    identifier: Some("RunMat:pause:InvalidInputArgument"),
    when: "Input argument is malformed, unsupported, non-scalar where scalar is required, or a negative/non-finite duration.",
    message: "pause: invalid input argument",
};

const PAUSE_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAUSE.TOO_MANY_INPUTS",
    identifier: Some("RunMat:pause:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "pause: too many input arguments",
};

const PAUSE_ERROR_STATE_LOCK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAUSE.STATE_LOCK",
    identifier: Some("RunMat:pause:StateLockFailed"),
    when: "Internal pause-state lock cannot be acquired.",
    message: "pause: failed to acquire pause state",
};

const PAUSE_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAUSE.GPU_GATHER_FAILED",
    identifier: Some("RunMat:pause:GpuGatherFailed"),
    when: "Gathering a GPU argument to host fails during argument classification.",
    message: "pause: failed to gather gpu input",
};

const PAUSE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    PAUSE_ERROR_INVALID_ARG,
    PAUSE_ERROR_TOO_MANY_INPUTS,
    PAUSE_ERROR_STATE_LOCK,
    PAUSE_ERROR_GATHER_FAILED,
];

pub const PAUSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PAUSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PAUSE_ERRORS,
};

fn pause_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

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
    summary = "Pause execution until keypress or specified duration elapses.",
    keywords = "pause,sleep,wait,delay",
    accel = "metadata",
    sink = true,
    type_resolver(pause_type),
    descriptor(crate::builtins::timing::pause::PAUSE_DESCRIPTOR),
    builtin_path = "crate::builtins::timing::pause"
)]
async fn pause_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.len() {
        0 => {
            perform_wait(PauseWait::Default).await?;
            Ok(empty_return_value())
        }
        1 => match classify_argument(&args[0]).await? {
            PauseArgument::Wait(wait) => {
                perform_wait(wait).await?;
                Ok(empty_return_value())
            }
            PauseArgument::SetState(next_state) => {
                let previous = set_pause_enabled(next_state)?;
                Ok(state_value(previous))
            }
            PauseArgument::Query => {
                let current = pause_enabled()?;
                Ok(state_value(current))
            }
        },
        _ => Err(pause_error_with_message(
            PAUSE_ERROR_TOO_MANY_INPUTS.message,
            &PAUSE_ERROR_TOO_MANY_INPUTS,
        )),
    }
}

async fn perform_wait(wait: PauseWait) -> Result<(), RuntimeError> {
    if !pause_enabled()? {
        return Ok(());
    }

    #[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
    {
        // MATLAB semantics: `pause` gives the UI a chance to update.
        // In RunMat Web/WASM this is an explicit flush boundary for plotting.
        let handle = plotting::current_figure_handle();
        // Present before the wait.
        let _ = plotting::render_current_scene(handle.as_u32());
    }

    match wait {
        PauseWait::Default => wait_for_key_press().await,
        PauseWait::Seconds(seconds) => {
            if seconds == 0.0 {
                // `pause(0)` is a useful yield point in simulation loops.
                #[cfg(target_arch = "wasm32")]
                {
                    return wasm_sleep_seconds(0.0).await;
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    return Ok(());
                }
            }
            sleep_seconds(seconds).await?;
            #[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
            {
                // Present again after the wait to ensure the compositor sees the most recent frame.
                // Some browser/driver combinations appear to delay presentation unless we yield across
                // a timer boundary.
                let handle = plotting::current_figure_handle();
                let _ = plotting::render_current_scene(handle.as_u32());
            }
            Ok(())
        }
    }
}

async fn wait_for_key_press() -> Result<(), RuntimeError> {
    #[cfg(test)]
    {
        Ok(())
    }
    #[cfg(not(test))]
    {
        interaction::wait_for_key_async("").await
    }
}

async fn sleep_seconds(seconds: f64) -> Result<(), RuntimeError> {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_sleep_seconds(seconds).await
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // from_secs_f64 rejects NaN/±Inf; classify_argument filters those earlier.
        let duration = std::time::Duration::from_secs_f64(seconds);
        std::thread::sleep(duration);
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
async fn wasm_sleep_seconds(seconds: f64) -> Result<(), RuntimeError> {
    use js_sys::{Function, Promise, Reflect};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    // `pause` runs in both Window and WebWorker contexts; workers do not have `window`.
    // Use the global `setTimeout` function instead.
    let global = js_sys::global();
    let set_timeout = Reflect::get(&global, &wasm_bindgen::JsValue::from_str("setTimeout"))
        .map_err(|_| build_runtime_error("pause: setTimeout unavailable").build())?
        .dyn_into::<Function>()
        .map_err(|_| build_runtime_error("pause: setTimeout unavailable").build())?;

    let millis = (seconds * 1000.0).max(0.0).round();
    let millis_i32 = if millis > i32::MAX as f64 {
        i32::MAX
    } else {
        millis as i32
    };

    let promise = Promise::new(&mut |resolve, _reject| {
        let resolve: Function = resolve.unchecked_into();
        let _ = set_timeout.call2(
            &global,
            &resolve.into(),
            &wasm_bindgen::JsValue::from_f64(millis_i32 as f64),
        );
    });

    let _ = JsFuture::from(promise)
        .await
        .map_err(|err| build_runtime_error(format!("pause: timer failed ({err:?})")).build())?;
    Ok(())
}

async fn classify_argument(arg: &Value) -> Result<PauseArgument, RuntimeError> {
    let host_value = gpu_helpers::gather_value_async(arg)
        .await
        .map_err(|e| pause_error_with_message(format!("pause: {e}"), &PAUSE_ERROR_GATHER_FAILED))?;
    match host_value {
        Value::String(text) => parse_command(&text),
        Value::CharArray(ca) => {
            if ca.rows == 0 || ca.data.is_empty() {
                Ok(PauseArgument::Wait(PauseWait::Default))
            } else if ca.rows == 1 {
                let text: String = ca.data.iter().collect();
                parse_command(&text)
            } else {
                Err(pause_error_with_message(
                    PAUSE_ERROR_INVALID_ARG.message,
                    &PAUSE_ERROR_INVALID_ARG,
                ))
            }
        }
        Value::StringArray(sa) => {
            if sa.data.is_empty() {
                Ok(PauseArgument::Wait(PauseWait::Default))
            } else if sa.data.len() == 1 {
                parse_command(&sa.data[0])
            } else {
                Err(pause_error_with_message(
                    PAUSE_ERROR_INVALID_ARG.message,
                    &PAUSE_ERROR_INVALID_ARG,
                ))
            }
        }
        Value::Num(value) => parse_numeric(value),
        Value::Int(int_value) => parse_numeric(int_value.to_f64()),
        Value::Bool(flag) => parse_numeric(if flag { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) => parse_tensor(tensor),
        Value::LogicalArray(logical) => parse_logical(logical),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
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
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(pause_error_with_message(
            PAUSE_ERROR_INVALID_ARG.message,
            &PAUSE_ERROR_INVALID_ARG,
        )),
    }
}

fn parse_command(raw: &str) -> Result<PauseArgument, RuntimeError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(PauseArgument::Wait(PauseWait::Default));
    }
    let lower = trimmed.to_ascii_lowercase();
    match lower.as_str() {
        "on" => Ok(PauseArgument::SetState(true)),
        "off" => Ok(PauseArgument::SetState(false)),
        "query" => Ok(PauseArgument::Query),
        _ => Err(pause_error_with_message(
            PAUSE_ERROR_INVALID_ARG.message,
            &PAUSE_ERROR_INVALID_ARG,
        )),
    }
}

fn parse_numeric(value: f64) -> Result<PauseArgument, RuntimeError> {
    if !value.is_finite() {
        if value.is_sign_positive() {
            return Ok(PauseArgument::Wait(PauseWait::Default));
        }
        return Err(pause_error_with_message(
            PAUSE_ERROR_INVALID_ARG.message,
            &PAUSE_ERROR_INVALID_ARG,
        ));
    }
    if value < 0.0 {
        return Err(pause_error_with_message(
            PAUSE_ERROR_INVALID_ARG.message,
            &PAUSE_ERROR_INVALID_ARG,
        ));
    }
    Ok(PauseArgument::Wait(PauseWait::Seconds(value)))
}

fn parse_tensor(tensor: Tensor) -> Result<PauseArgument, RuntimeError> {
    if tensor.data.is_empty() {
        return Ok(PauseArgument::Wait(PauseWait::Default));
    }
    if tensor.data.len() != 1 {
        return Err(pause_error_with_message(
            PAUSE_ERROR_INVALID_ARG.message,
            &PAUSE_ERROR_INVALID_ARG,
        ));
    }
    parse_numeric(tensor.data[0])
}

fn parse_logical(logical: LogicalArray) -> Result<PauseArgument, RuntimeError> {
    if logical.data.is_empty() {
        return Ok(PauseArgument::Wait(PauseWait::Default));
    }
    if logical.data.len() != 1 {
        return Err(pause_error_with_message(
            PAUSE_ERROR_INVALID_ARG.message,
            &PAUSE_ERROR_INVALID_ARG,
        ));
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

fn pause_enabled() -> Result<bool, RuntimeError> {
    PAUSE_STATE.read().map(|guard| guard.enabled).map_err(|_| {
        pause_error_with_message(PAUSE_ERROR_STATE_LOCK.message, &PAUSE_ERROR_STATE_LOCK)
    })
}

fn set_pause_enabled(next: bool) -> Result<bool, RuntimeError> {
    let mut guard = PAUSE_STATE.write().map_err(|_| {
        pause_error_with_message(PAUSE_ERROR_STATE_LOCK.message, &PAUSE_ERROR_STATE_LOCK)
    })?;
    let previous = guard.enabled;
    guard.enabled = next;
    Ok(previous)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
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

    fn assert_pause_error_identifier(err: crate::RuntimeError, identifier: &str) {
        assert_eq!(
            err.identifier(),
            Some(identifier),
            "message: {}",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn query_returns_on_by_default() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let result = block_on(pause_builtin(vec![Value::from("query")])).expect("pause query");
        assert_eq!(char_array_to_string(result), "on");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pause_off_returns_previous_state() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let previous = block_on(pause_builtin(vec![Value::from("off")])).expect("pause off");
        assert_eq!(char_array_to_string(previous), "on");
        assert!(!pause_enabled().unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pause_on_restores_state() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(false);
        let previous = block_on(pause_builtin(vec![Value::from("on")])).expect("pause on");
        assert_eq!(char_array_to_string(previous), "off");
        assert!(pause_enabled().unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pause_default_returns_empty_tensor() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let result = block_on(pause_builtin(Vec::new())).expect("pause()");
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
        let result = block_on(pause_builtin(vec![Value::Num(0.0)])).expect("pause(0)");
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
        let result =
            block_on(pause_builtin(vec![Value::Int(IntValue::I32(0))])).expect("pause(int)");
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
        let result = block_on(pause_builtin(vec![Value::Num(-0.0)])).expect("pause(-0)");
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
        let err = block_on(pause_builtin(vec![Value::Num(-0.1)])).unwrap_err();
        assert_pause_error_identifier(err, PAUSE_ERROR_INVALID_ARG.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_scalar_tensor_is_rejected() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = block_on(pause_builtin(vec![Value::Tensor(tensor)])).unwrap_err();
        assert_pause_error_identifier(err, PAUSE_ERROR_INVALID_ARG.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_tensor_behaves_like_default_pause() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_state(true);
        let empty = Tensor::zeros(vec![0, 0]);
        let result = block_on(pause_builtin(vec![Value::Tensor(empty)])).expect("pause([])");
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
        let result =
            block_on(pause_builtin(vec![Value::LogicalArray(logical)])).expect("pause(true)");
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
        let result = block_on(pause_builtin(vec![Value::Num(f64::INFINITY)])).expect("pause(Inf)");
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
            let result =
                block_on(pause_builtin(vec![Value::GpuTensor(handle)])).expect("pause(gpuScalar)");
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
        let result =
            block_on(pause_builtin(vec![Value::GpuTensor(handle)])).expect("pause(gpuScalar)");
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
        let err = block_on(pause_builtin(vec![Value::from("invalid")])).unwrap_err();
        assert_pause_error_identifier(err, PAUSE_ERROR_INVALID_ARG.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn too_many_inputs_raises_error() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_state(true);
        let err = block_on(pause_builtin(vec![Value::Num(0.0), Value::Num(0.0)])).unwrap_err();
        assert_pause_error_identifier(err, PAUSE_ERROR_TOO_MANY_INPUTS.identifier.unwrap());
    }
}
