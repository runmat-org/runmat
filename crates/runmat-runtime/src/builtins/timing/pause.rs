//! MATLAB-compatible `pause` builtin that temporarily suspends execution.

#[cfg(test)]
use once_cell::sync::Lazy;
use runmat_builtins::{
    BuiltinErrorDescriptor, PAUSE_ERROR_CONTEXT_UNAVAILABLE, PAUSE_ERROR_GATHER_FAILED,
    PAUSE_ERROR_INVALID_ARG, PAUSE_ERROR_TOO_MANY_INPUTS, PAUSE_GPU_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, LogicalArray, Tensor, Value};
use std::time::Duration;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor as tensor_utils;
#[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
use crate::builtins::plotting;
use crate::context::RuntimeContext;
#[cfg(not(test))]
use crate::interaction;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[cfg(test)]
use std::sync::Mutex;
#[cfg(test)]
pub(crate) static TEST_GUARD: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

const BUILTIN_NAME: &str = "pause";

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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::timing::pause"
)]
async fn pause_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let runtime = runtime.ok_or_else(|| {
        pause_error_with_message(
            PAUSE_ERROR_CONTEXT_UNAVAILABLE.message,
            &PAUSE_ERROR_CONTEXT_UNAVAILABLE,
        )
    })?;
    match args.len() {
        0 => {
            perform_wait(&runtime, PauseWait::Default).await?;
            Ok(empty_return_value())
        }
        1 => match classify_argument(&args[0]).await? {
            PauseArgument::Wait(wait) => {
                perform_wait(&runtime, wait).await?;
                Ok(empty_return_value())
            }
            PauseArgument::SetState(next_state) => {
                let previous = runtime
                    .state()
                    .timing
                    .borrow_mut()
                    .set_pause_enabled(next_state);
                Ok(state_value(previous))
            }
            PauseArgument::Query => {
                let current = runtime.state().timing.borrow().pause_enabled();
                Ok(state_value(current))
            }
        },
        _ => Err(pause_error_with_message(
            PAUSE_ERROR_TOO_MANY_INPUTS.message,
            &PAUSE_ERROR_TOO_MANY_INPUTS,
        )),
    }
}

async fn perform_wait(runtime: &RuntimeContext, wait: PauseWait) -> Result<(), RuntimeError> {
    if !runtime.state().timing.borrow().pause_enabled() {
        return Ok(());
    }

    #[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
    {
        // MATLAB semantics: `pause` gives the UI a chance to update.
        // In RunMat Web/WASM this is an explicit flush boundary for plotting.
        let handle = plotting::current_figure_handle();
        // Present before the wait.
        tracing::info_span!("runtime.pause.present", phase = "before_wait").in_scope(|| {
            let _ = plotting::render_current_scene(handle.as_u32());
        });
    }

    match wait {
        PauseWait::Default => wait_for_key_press(runtime).await,
        PauseWait::Seconds(seconds) => {
            if seconds == 0.0 {
                // `pause(0)` is a useful yield point in simulation loops.
                return runtime
                    .cpu_cooperation()
                    .await_cancellable(runtime.service_ports().timing()?.sleep(Duration::ZERO))
                    .await
                    .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
            }
            let duration = Duration::try_from_secs_f64(seconds).map_err(|_| {
                pause_error_with_message(PAUSE_ERROR_INVALID_ARG.message, &PAUSE_ERROR_INVALID_ARG)
            })?;
            #[cfg(not(target_arch = "wasm32"))]
            if runmat_time::Instant::now().checked_add(duration).is_none() {
                return Err(pause_error_with_message(
                    PAUSE_ERROR_INVALID_ARG.message,
                    &PAUSE_ERROR_INVALID_ARG,
                ));
            }
            tracing::Instrument::instrument(
                runtime
                    .cpu_cooperation()
                    .await_cancellable(runtime.service_ports().timing()?.sleep(duration)),
                tracing::info_span!("runtime.pause.wait", requested_seconds = seconds),
            )
            .await
            .map_err(crate::execution::ExecutionServiceError::into_runtime_error)??;
            #[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
            {
                // Present again after the wait to ensure the compositor sees the most recent frame.
                // Some browser/driver combinations appear to delay presentation unless we yield across
                // a timer boundary.
                let handle = plotting::current_figure_handle();
                tracing::info_span!("runtime.pause.present", phase = "after_wait").in_scope(|| {
                    let _ = plotting::render_current_scene(handle.as_u32());
                });
            }
            Ok(())
        }
    }
}

async fn wait_for_key_press(runtime: &RuntimeContext) -> Result<(), RuntimeError> {
    #[cfg(test)]
    {
        let _ = runtime;
        Ok(())
    }
    #[cfg(not(test))]
    {
        runtime
            .cpu_cooperation()
            .await_cancellable(interaction::wait_for_key_async(""))
            .await
            .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?
    }
}

async fn classify_argument(arg: &Value) -> Result<PauseArgument, RuntimeError> {
    if matches!(arg, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PAUSE_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let host_value = gpu_helpers::gather_value_async(arg)
        .await
        .map_err(|e| pause_error_with_message(format!("pause: {e}"), &PAUSE_ERROR_GATHER_FAILED))?;
    match host_value {
        Value::String(runmat_value::StringElement::Text(text)) => parse_command(&text),
        Value::CharArray(ca) => {
            if ca.rows == 0 || ca.data().is_empty() {
                Ok(PauseArgument::Wait(PauseWait::Default))
            } else if ca.rows == 1 {
                let text =
                    crate::builtins::common::text::decode_utf16_with_error(ca.data(), || {
                        pause_error_with_message(
                            "pause: argument is not valid UTF-16 text",
                            &PAUSE_ERROR_INVALID_ARG,
                        )
                    })?;
                parse_command(&text)
            } else {
                Err(pause_error_with_message(
                    PAUSE_ERROR_INVALID_ARG.message,
                    &PAUSE_ERROR_INVALID_ARG,
                ))
            }
        }
        Value::StringArray(sa) => {
            if sa.data().is_empty() {
                Ok(PauseArgument::Wait(PauseWait::Default))
            } else if sa.data().len() == 1 {
                parse_command(sa.data()[0].as_str().ok_or_else(|| {
                    pause_error_with_message(
                        PAUSE_ERROR_INVALID_ARG.message,
                        &PAUSE_ERROR_INVALID_ARG,
                    )
                })?)
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
        Value::String(runmat_value::StringElement::Missing)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::SparseTensor(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_)
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
        if value == f64::INFINITY {
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
    let len = tensor.len();
    if len == 0 {
        return Ok(PauseArgument::Wait(PauseWait::Default));
    }
    if len != 1 {
        return Err(pause_error_with_message(
            PAUSE_ERROR_INVALID_ARG.message,
            &PAUSE_ERROR_INVALID_ARG,
        ));
    }
    parse_numeric(tensor_utils::tensor_value_f64(&tensor, 0))
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::execution::RuntimeExecutionService;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntValue, IntegerStorage, LogicalArray, Tensor};
    use std::rc::Rc;

    thread_local! {
        static TEST_RUNTIME: RuntimeContext = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    }

    async fn pause_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        let runtime = TEST_RUNTIME.with(Clone::clone);
        super::pause_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;

    fn reset_state(enabled: bool) {
        TEST_RUNTIME.with(|runtime| {
            runtime
                .state()
                .timing
                .borrow_mut()
                .set_pause_enabled(enabled);
        });
    }

    fn pause_enabled() -> Result<bool, RuntimeError> {
        Ok(TEST_RUNTIME.with(|runtime| runtime.state().timing.borrow().pause_enabled()))
    }

    fn char_array_to_string(value: Value) -> String {
        match value {
            Value::CharArray(ca) if ca.rows == 1 => {
                runmat_value::decode_utf16_code_units(ca.data()).unwrap()
            }
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
            Value::Tensor(t) => assert_eq!(t.materialize_f64().len(), 0),
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
            Value::Tensor(t) => assert_eq!(t.materialize_f64().len(), 0),
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
            Value::Tensor(t) => assert_eq!(t.materialize_f64().len(), 0),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn pause_tensor_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1])
            .expect("typed pause tensor");

        match parse_tensor(tensor).expect("pause tensor") {
            PauseArgument::Wait(PauseWait::Seconds(seconds)) => assert_eq!(seconds, 2026.0),
            other => panic!("expected wait seconds, got {other:?}"),
        }
    }

    #[test]
    fn pause_accepts_every_integer_duration_class() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(false);
        for value in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ] {
            let result = block_on(pause_builtin(vec![Value::Int(value)]))
                .expect("documented integer duration");
            assert!(matches!(result, Value::Tensor(tensor) if tensor.is_empty()));
        }
        reset_state(true);
    }

    #[test]
    fn pause_resident_argument_is_gated_before_provider_access() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("unregistered owner");
        let explicit = allocation
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit view");
        let resident = Value::GpuTensor(explicit);
        let error = block_on(pause_builtin(vec![resident])).expect_err("resident pause input");

        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:PauseGpuInputExtension")
        );
        assert_eq!(allocation.probe.retirements(), 0);
        drop((error, allocation.handle));
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn pause_automatically_resident_duration_gathers_in_strict_mode() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(false);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor)
                .expect("resident duration")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("automatic view");
            let lease = handle.allocation_lease().downgrade();
            let result = {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                block_on(pause_builtin(vec![Value::GpuTensor(handle.clone())]))
                    .expect("automatic duration gathers transparently")
            };
            assert!(matches!(result, Value::Tensor(tensor) if tensor.is_empty()));
            drop(handle);
            assert!(lease.acquire().is_none());
        });
        reset_state(true);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_negative_zero_is_treated_as_zero() {
        let _guard = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        reset_state(true);
        let result = block_on(pause_builtin(vec![Value::Num(-0.0)])).expect("pause(-0)");
        match result {
            Value::Tensor(t) => assert_eq!(t.materialize_f64().len(), 0),
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

    #[test]
    fn nan_duration_is_invalid_not_a_keypress_wait() {
        let _guard = TEST_GUARD.lock().unwrap();
        reset_state(true);
        let error = block_on(pause_builtin(vec![Value::Num(f64::NAN)])).unwrap_err();
        assert_pause_error_identifier(error, PAUSE_ERROR_INVALID_ARG.identifier.unwrap());
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
            Value::Tensor(t) => assert_eq!(t.materialize_f64().len(), 0),
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
            Value::Tensor(t) => assert_eq!(t.materialize_f64().len(), 0),
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
            Value::Tensor(t) => assert_eq!(t.materialize_f64().len(), 0),
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
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                block_on(pause_builtin(vec![Value::GpuTensor(handle)])).expect("pause(gpuScalar)");
            match result {
                Value::Tensor(t) => assert_eq!(t.materialize_f64().len(), 0),
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
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result =
            block_on(pause_builtin(vec![Value::GpuTensor(handle)])).expect("pause(gpuScalar)");
        match result {
            Value::Tensor(t) => assert_eq!(t.materialize_f64().len(), 0),
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

    #[test]
    fn numeric_wait_is_pending_and_cancellation_drops_timer() {
        use crate::context::{RuntimeServiceFuture, RuntimeServicePorts, RuntimeTimingService};
        use runmat_execution::cancellation::CancellationSource;
        use runmat_execution::CancellationReason;
        use std::cell::Cell;
        use std::future::Future;
        use std::pin::Pin;
        use std::task::{Context, Poll};

        struct PendingTimer(Rc<Cell<usize>>);
        impl Future for PendingTimer {
            type Output = Result<(), RuntimeError>;
            fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
                Poll::Pending
            }
        }
        impl Drop for PendingTimer {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        struct TestTiming {
            drops: Rc<Cell<usize>>,
            calls: Rc<Cell<usize>>,
        }
        impl RuntimeTimingService for TestTiming {
            fn monotonic_elapsed(&self) -> Duration {
                Duration::ZERO
            }
            fn wall_time_since_epoch(&self) -> Duration {
                Duration::ZERO
            }
            fn local_datetime_to_instant(
                &self,
                local: chrono::NaiveDateTime,
            ) -> Option<crate::context::RuntimeCalendarInstant> {
                Some(crate::context::RuntimeCalendarInstant::from_naive_utc(
                    local,
                ))
            }
            fn local_datetime_at_instant(
                &self,
                instant: crate::context::RuntimeCalendarInstant,
            ) -> Option<chrono::NaiveDateTime> {
                instant.at_fixed_offset(0)
            }
            fn process_cpu_seconds(&self) -> Option<f64> {
                Some(0.0)
            }
            fn sleep(&self, _: Duration) -> RuntimeServiceFuture<Result<(), RuntimeError>> {
                self.calls.set(self.calls.get() + 1);
                Box::pin(PendingTimer(self.drops.clone()))
            }
        }

        let drops = Rc::new(Cell::new(0));
        let calls = Rc::new(Cell::new(0));
        let source = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            source.token(),
        )
        .with_service_ports(
            RuntimeServicePorts::default().with_timing(Rc::new(TestTiming {
                drops: drops.clone(),
                calls: calls.clone(),
            })),
        );
        let invalid = block_on(super::pause_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(f64::MAX)]),
        ))
        .unwrap_err();
        assert_pause_error_identifier(invalid, PAUSE_ERROR_INVALID_ARG.identifier.unwrap());
        assert_eq!(calls.get(), 0);
        let mut wait = Box::pin(super::pause_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(wait.as_mut().poll(&mut context).is_pending());
        assert_eq!(calls.get(), 1);
        assert_eq!(drops.get(), 0);
        source.cancel(CancellationReason::User);
        assert!(matches!(
            wait.as_mut().poll(&mut context),
            Poll::Ready(Err(_))
        ));
        assert_eq!(drops.get(), 1);
    }
}
