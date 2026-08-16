//! MATLAB-compatible `toc` builtin that reports elapsed stopwatch time.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerClass, BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, Value,
};
use runmat_macros::runtime_builtin;
use runmat_time::Instant;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::timing::tic::{decode_handle, elapsed_since, latest_start};
use crate::builtins::timing::type_resolvers::toc_type;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::timing::toc")]
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::timing::toc")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "toc",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Timing builtins execute eagerly on the host and do not participate in fusion.",
};

const BUILTIN_NAME: &str = "toc";

const LEGACY_DOUBLE_HANDLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "toc-legacy-double-timer-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "toc accepts an older RunMat double-valued timer handle",
    error_identifier: Some("RunMat:compatibility:TocLegacyDoubleTimerHandleExtension"),
};
pub const TOC_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [LEGACY_DOUBLE_HANDLE_EXTENSION];

const UINT64_TIMER_CLASS: [BuiltinIntegerClass; 1] = [BuiltinIntegerClass::Uint64];
const TOC_TIMER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "timerVal",
    classes: &UINT64_TIMER_CLASS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
    notes:
        "tic returns an opaque uint64 token; toc reads its exact bits without numeric conversion.",
}];
pub const TOC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "elapsed = toc(uint64_timerVal)",
        inputs: &TOC_TIMER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The uint64 value is an opaque stopwatch token rather than an elapsed-time count.",
    }];

const TOC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "elapsed",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elapsed time in seconds.",
}];

const TOC_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const TOC_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "timerVal",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Handle returned by tic.",
}];

const TOC_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "elapsed = toc()",
        inputs: &TOC_INPUTS_NONE,
        outputs: &TOC_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "elapsed = toc(timerVal)",
        inputs: &TOC_INPUTS_HANDLE,
        outputs: &TOC_OUTPUT,
    },
];

const TOC_ERROR_NO_MATCHING_TIC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOC.NO_MATCHING_TIC",
    identifier: Some("RunMat:toc:NoMatchingTic"),
    when: "toc() is called without a matching prior tic().",
    message: "toc: no matching tic",
};

const TOC_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOC.INVALID_HANDLE",
    identifier: Some("RunMat:toc:InvalidTimerHandle"),
    when: "The timer handle is missing, malformed, non-finite, negative, or points to a future instant.",
    message: "toc: invalid timer handle",
};

const TOC_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOC.TOO_MANY_INPUTS",
    identifier: Some("RunMat:toc:TooManyInputs"),
    when: "More than one input argument is supplied.",
    message: "toc: too many input arguments",
};

const TOC_ERRORS: [BuiltinErrorDescriptor; 3] = [
    TOC_ERROR_NO_MATCHING_TIC,
    TOC_ERROR_INVALID_HANDLE,
    TOC_ERROR_TOO_MANY_INPUTS,
];

pub const TOC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TOC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TOC_ERRORS,
};

fn toc_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

/// Read elapsed time from the stopwatch stack or a specific handle.
#[runtime_builtin(
    name = "toc",
    category = "timing",
    summary = "Return elapsed time since the latest tic or a specific tic handle.",
    keywords = "toc,timing,profiling,benchmark",
    type_resolver(toc_type),
    descriptor(crate::builtins::timing::toc::TOC_DESCRIPTOR),
    extensions(crate::builtins::timing::toc::TOC_EXTENSIONS),
    integer_capabilities(crate::builtins::timing::toc::TOC_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::timing::toc"
)]
pub async fn toc_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    match args.len() {
        0 => latest_elapsed(),
        1 => elapsed_from_value(&args[0]),
        _ => Err(toc_error_with_message(
            TOC_ERROR_TOO_MANY_INPUTS.message,
            &TOC_ERROR_TOO_MANY_INPUTS,
        )),
    }
}

fn latest_elapsed() -> Result<f64, crate::RuntimeError> {
    let start = latest_start(BUILTIN_NAME)?.ok_or_else(|| {
        toc_error_with_message(
            TOC_ERROR_NO_MATCHING_TIC.message,
            &TOC_ERROR_NO_MATCHING_TIC,
        )
    })?;
    Ok(elapsed_since(start).as_secs_f64())
}

fn elapsed_from_value(value: &Value) -> Result<f64, crate::RuntimeError> {
    let handle = match value {
        Value::Int(IntValue::U64(handle)) => *handle,
        Value::Tensor(tensor) if tensor.len() == 1 => match tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0))
        {
            Some(IntValue::U64(handle)) => handle,
            _ => legacy_or_invalid_handle(value)?,
        },
        _ => legacy_or_invalid_handle(value)?,
    };
    let instant = decode_handle(handle, BUILTIN_NAME, &TOC_ERROR_INVALID_HANDLE)?;
    let now = Instant::now();
    let elapsed = now.checked_duration_since(instant).ok_or_else(|| {
        toc_error_with_message(TOC_ERROR_INVALID_HANDLE.message, &TOC_ERROR_INVALID_HANDLE)
    })?;
    Ok(elapsed.as_secs_f64())
}

fn legacy_or_invalid_handle(value: &Value) -> Result<u64, crate::RuntimeError> {
    let seconds = match value {
        Value::Num(seconds) => *seconds,
        Value::Tensor(tensor) if tensor.len() == 1 && tensor.integer_storage().is_none() => {
            crate::builtins::common::tensor::tensor_value_f64(tensor, 0)
        }
        _ => {
            return Err(toc_error_with_message(
                TOC_ERROR_INVALID_HANDLE.message,
                &TOC_ERROR_INVALID_HANDLE,
            ))
        }
    };
    if !seconds.is_finite() || seconds.is_sign_negative() {
        return Err(toc_error_with_message(
            TOC_ERROR_INVALID_HANDLE.message,
            &TOC_ERROR_INVALID_HANDLE,
        ));
    }
    crate::compatibility::ensure_builtin_extension_enabled(
        &LEGACY_DOUBLE_HANDLE_EXTENSION,
        BUILTIN_NAME,
    )?;
    Ok(seconds.to_bits())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::timing::tic::{
        clear_stopwatch_for_test, encode_instant, record_tic, TEST_GUARD,
    };
    use futures::executor::block_on;
    use std::time::Duration;

    fn clear_tic_stack() {
        clear_stopwatch_for_test();
    }

    fn assert_toc_error_identifier(err: crate::RuntimeError, identifier: &str) {
        assert_eq!(
            err.identifier(),
            Some(identifier),
            "message: {}",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn toc_requires_matching_tic() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let err = block_on(toc_builtin(Vec::new())).unwrap_err();
        assert_toc_error_identifier(err, TOC_ERROR_NO_MATCHING_TIC.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn toc_reports_elapsed_for_latest_start() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        record_tic("tic").expect("tic");
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = block_on(toc_builtin(Vec::new())).expect("toc");
        assert!(elapsed >= 0.0);
        assert!(latest_start(BUILTIN_NAME).unwrap().is_some());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn toc_with_handle_measures_without_popping_stack() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let handle = record_tic("tic").expect("tic");
        std::thread::sleep(Duration::from_millis(5));
        let elapsed =
            block_on(toc_builtin(vec![Value::Int(IntValue::U64(handle))])).expect("toc(handle)");
        assert!(elapsed >= 0.0);
        // Stack still contains the entry, and no-argument toc reads it again.
        let later = block_on(toc_builtin(Vec::new())).expect("second toc");
        assert!(later >= elapsed);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn toc_rejects_invalid_handle() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let err = block_on(toc_builtin(vec![Value::Num(f64::NAN)])).unwrap_err();
        assert_toc_error_identifier(err, TOC_ERROR_INVALID_HANDLE.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn toc_rejects_overflowing_handle() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = block_on(toc_builtin(vec![Value::Num(f64::MAX)])).unwrap_err();
        assert_toc_error_identifier(err, TOC_ERROR_INVALID_HANDLE.identifier.unwrap());
    }

    #[test]
    fn toc_rejects_legacy_double_token_in_matlab_mode() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let handle = record_tic("tic").expect("tic");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(toc_builtin(vec![Value::Num(f64::from_bits(handle))])).unwrap_err();
        assert_toc_error_identifier(
            err,
            LEGACY_DOUBLE_HANDLE_EXTENSION.error_identifier.unwrap(),
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn toc_rejects_future_handle() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let future_seconds = f64::from_bits(encode_instant(Instant::now())) + 10_000.0;
        let err = block_on(toc_builtin(vec![Value::Int(IntValue::U64(
            future_seconds.to_bits(),
        ))]))
        .unwrap_err();
        assert_toc_error_identifier(err, TOC_ERROR_INVALID_HANDLE.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn toc_rejects_string_handle() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let err = block_on(toc_builtin(vec![Value::from("not a timer")])).unwrap_err();
        assert_toc_error_identifier(err, TOC_ERROR_INVALID_HANDLE.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn toc_rejects_extra_arguments() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        let err = block_on(toc_builtin(vec![Value::Num(0.0), Value::Num(0.0)])).unwrap_err();
        assert_toc_error_identifier(err, TOC_ERROR_TOO_MANY_INPUTS.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn toc_nested_timers() {
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        record_tic("tic").expect("outer");
        std::thread::sleep(Duration::from_millis(2));
        record_tic("tic").expect("inner");
        std::thread::sleep(Duration::from_millis(4));
        let inner = block_on(toc_builtin(Vec::new())).expect("inner toc");
        assert!(inner >= 0.0);
        std::thread::sleep(Duration::from_millis(2));
        let inner_again = block_on(toc_builtin(Vec::new())).expect("second inner toc");
        assert!(inner_again >= inner);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn toc_ignores_wgpu_provider() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let _guard = TEST_GUARD.lock().unwrap();
        clear_tic_stack();
        record_tic("tic").expect("tic");
        std::thread::sleep(Duration::from_millis(1));
        let elapsed = block_on(toc_builtin(Vec::new())).expect("toc");
        assert!(elapsed >= 0.0);
    }
}
