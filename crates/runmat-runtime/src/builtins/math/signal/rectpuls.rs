//! MATLAB-compatible `rectpuls` builtin for rectangular pulse samples.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, RECTPULS_ERROR_ARG_COUNT, RECTPULS_ERROR_INTERNAL,
    RECTPULS_ERROR_INVALID_INPUT, RECTPULS_ERROR_WIDTH_INVALID, RECTPULS_EXPLICIT_GPU_EXTENSION,
    RECTPULS_INTEGER_T_EXTENSION, RECTPULS_INTEGER_W_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor::tensor_into_value;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::signal::common::{
    map_real_waveform, PreparedWaveformControl, WaveformControlError, WaveformOutputMode,
    WindowCooperation,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "rectpuls";

fn rectpuls_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    rectpuls_error_with_message(error.message, error)
}

fn rectpuls_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    rectpuls_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn rectpuls_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rectpuls_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    let mut failure = builder.build();
    failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
    failure
}

pub(crate) fn rectpuls_scalar(t: f64, width: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let half_width = width / 2.0;
    if t >= -half_width && t < half_width {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn validate_width(width: f64) -> Result<f64, String> {
    if !width.is_finite() || width <= 0.0 {
        Err(format!("got {width}"))
    } else {
        Ok(width)
    }
}

#[runtime_builtin(
    name = "rectpuls",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::rectpuls"
)]
async fn rectpuls_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    t: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    rectpuls_in_runtime(runtime.as_ref(), t, rest).await
}

#[cfg(test)]
async fn rectpuls_builtin(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    rectpuls_in_runtime(None, t, rest).await
}

async fn rectpuls_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    t: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let mut cooperation = WindowCooperation::new(runtime, BUILTIN_NAME);
    cooperation.completed(0).await?;
    ensure_rectpuls_compatibility(&t, &rest)?;
    let width = prepare_width(&rest)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&t, BUILTIN_NAME)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &t,
        &RECTPULS_INTEGER_T_EXTENSION,
        BUILTIN_NAME,
        "T",
    )
    .await?;
    let gpu_source = rectpuls_gpu_source(&t, &rest)?;
    let width = resolve_width(width).await?;
    let tensor = rectpuls_input_tensor(t).await?;
    let output = map_real_waveform(
        tensor,
        WaveformOutputMode::Double,
        |value| rectpuls_scalar(value, width),
        &mut cooperation,
    )
    .await
    .map_err(|error| error.into_runtime_error(BUILTIN_NAME, &RECTPULS_ERROR_INTERNAL))?;
    let Some(source) = gpu_source else {
        return Ok(tensor_into_value(output));
    };
    let restored =
        gpu_helpers::restore_class_preserving_value(&source, Value::Tensor(output), BUILTIN_NAME)?;
    if runmat_accelerate_api::handle_is_explicit(&source)
        && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(rectpuls_error_with_detail(
            &RECTPULS_ERROR_INTERNAL,
            "provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

fn ensure_rectpuls_compatibility(t: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(t) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RECTPULS_INTEGER_T_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if std::iter::once(t)
        .chain(rest.iter())
        .any(|value| matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RECTPULS_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn rectpuls_gpu_source(t: &Value, rest: &[Value]) -> BuiltinResult<Option<GpuTensorHandle>> {
    gpu_helpers::select_resident_output_source(
        std::iter::once(t)
            .chain(rest.iter())
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
        BUILTIN_NAME,
    )
}

fn prepare_width(rest: &[Value]) -> BuiltinResult<Option<PreparedWaveformControl<'_>>> {
    match rest.len() {
        0 => Ok(None),
        1 => {
            let prepared = PreparedWaveformControl::admit(
                &rest[0],
                &RECTPULS_INTEGER_W_EXTENSION,
                BUILTIN_NAME,
                "W",
            )
            .map_err(width_control_error)?;
            if let Some(raw) = prepared.host_value() {
                validate_width(raw).map_err(|err| {
                    rectpuls_error_with_detail(&RECTPULS_ERROR_WIDTH_INVALID, err)
                })?;
            }
            Ok(Some(prepared))
        }
        _ => Err(rectpuls_error_with_detail(
            &RECTPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        )),
    }
}

async fn resolve_width(control: Option<PreparedWaveformControl<'_>>) -> BuiltinResult<f64> {
    let Some(control) = control else {
        return Ok(1.0);
    };
    let raw = control.resolve().await.map_err(width_control_error)?;
    validate_width(raw)
        .map_err(|err| rectpuls_error_with_detail(&RECTPULS_ERROR_WIDTH_INVALID, err))
}

fn width_control_error(error: WaveformControlError) -> RuntimeError {
    match error {
        WaveformControlError::Invalid(detail) => {
            rectpuls_error_with_detail(&RECTPULS_ERROR_WIDTH_INVALID, detail)
        }
        WaveformControlError::Boundary(source) => source,
        WaveformControlError::Provider(source) => {
            rectpuls_error_with_source(&RECTPULS_ERROR_INTERNAL, "control admission failed", source)
        }
    }
}

async fn rectpuls_input_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => {
            gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|source| {
                    rectpuls_error_with_source(
                        &RECTPULS_ERROR_INTERNAL,
                        "gpu gather failed",
                        source,
                    )
                })
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(rectpuls_error(&RECTPULS_ERROR_INVALID_INPUT))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(rectpuls_error(&RECTPULS_ERROR_INVALID_INPUT))
        }
        other => tensor::value_into_tensor_for(BUILTIN_NAME, other)
            .map_err(|err| rectpuls_error_with_detail(&RECTPULS_ERROR_INVALID_INPUT, err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::{CharArray, IntValue, IntegerStorage};

    fn call(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(rectpuls_builtin(t, rest))
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn integer_tensor(values: Vec<i16>, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(IntegerStorage::I16(values), shape).expect("typed integer tensor")
    }

    #[cfg(feature = "wgpu")]
    fn all_integer_storages() -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(vec![0, 1]),
            IntegerStorage::I16(vec![0, 1]),
            IntegerStorage::I32(vec![0, 1]),
            IntegerStorage::I64(vec![0, 1]),
            IntegerStorage::U8(vec![0, 1]),
            IntegerStorage::U16(vec![0, 1]),
            IntegerStorage::U32(vec![0, 1]),
            IntegerStorage::U64(vec![0, 1]),
        ]
    }

    #[test]
    fn rectpuls_default_width_samples_rectangle() {
        let input = Tensor::new(vec![-0.5, -0.25, 0.0, 0.25, 0.5, 0.75], vec![1, 6]).unwrap();
        let out = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("rectpuls"));
        assert_eq!(out.shape, vec![1, 6]);
        assert_eq!(out.materialize_f64(), vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn rectpuls_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = integer_tensor(vec![-1, 0, 1], vec![1, 3]);
        let out =
            expect_tensor(call(Value::Tensor(input), vec![Value::Num(2.0)]).expect("rectpuls"));
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(out.materialize_f64(), vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn rectpuls_custom_width_and_scalar_output() {
        let value = call(Value::Num(0.75), vec![Value::Num(2.0)]).expect("rectpuls");
        assert!(matches!(value, Value::Num(n) if n == 1.0));
    }

    #[test]
    fn rectpuls_rejects_bad_width_and_text_input() {
        let err = call(Value::Num(0.0), vec![Value::Num(0.0)]).expect_err("width");
        assert_eq!(err.identifier(), RECTPULS_ERROR_WIDTH_INVALID.identifier);

        let err =
            call(Value::CharArray(CharArray::new_row("abc")), Vec::new()).expect_err("text input");
        assert_eq!(err.identifier(), RECTPULS_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn invalid_host_width_precedes_resident_sample_access() {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
        };
        let owner = TestAllocationOwner::new();
        let input = owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        input.probe.mark_unavailable();
        let err =
            call(Value::GpuTensor(input.handle), vec![Value::Num(0.0)]).expect_err("host width");
        assert_eq!(err.identifier(), RECTPULS_ERROR_WIDTH_INVALID.identifier);
    }

    #[test]
    fn integer_width_requires_exact_binary64_conversion() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let rounded = call(
            Value::Num(0.0),
            vec![Value::Int(IntValue::U64((1u64 << 53) + 1))],
        )
        .expect_err("inexact width");
        assert!(rounded
            .to_string()
            .contains("exactly representable as double"));
        let exact = call(Value::Num(0.0), vec![Value::Int(IntValue::U64(1u64 << 54))])
            .expect("exact wide width");
        assert!(matches!(exact, Value::Num(value) if value == 1.0));
    }

    #[test]
    fn rectpuls_automatic_residency_is_transparent_and_explicit_residency_is_gated() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("input");
            let automatic = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let output = call(Value::GpuTensor(automatic), vec![Value::Num(2.0)])
                .expect("automatic resident input");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected automatic resident output");
            };
            assert!(!runmat_accelerate_api::handle_is_explicit(output_handle));
            assert_eq!(
                test_support::gather(output)
                    .expect("gather")
                    .materialize_f64(),
                vec![1.0, 0.0]
            );

            let explicit = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let explicit = explicit
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("publish provenance view");
            let error = call(Value::GpuTensor(explicit), vec![Value::Num(2.0)])
                .expect_err("strict explicit input");
            assert_eq!(
                error.identifier(),
                RECTPULS_EXPLICIT_GPU_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn rectpuls_wgpu_fallback_enforces_double_residency_for_all_integer_classes() {
        use crate::builtins::common::test_support;

        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in all_integer_storages() {
            let input = Tensor::new_integer(storage, vec![1, 2]).expect("integer input");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let handle = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("publish provenance view");
            let result = call(Value::GpuTensor(handle), vec![Value::Num(2.0)]);
            let output = result.expect("resident integer rectpuls");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected resident output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
            assert_eq!(
                runmat_accelerate_api::handle_precision(output_handle),
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            );
            assert_eq!(
                test_support::gather(output)
                    .expect("gather")
                    .materialize_f64(),
                vec![1.0, 0.0]
            );
        }
    }

    #[test]
    fn rectpuls_is_registered() {
        let entry = runmat_builtins::builtin_catalog_entry_by_name("rectpuls").unwrap();
        assert!(builtin_function_by_name("rectpuls").is_none());
        assert_eq!(entry.bindings.len(), 1);
        let bindings = crate::builtin::runtime_builtin_bindings_by_name("rectpuls");
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].identity,
            entry.binding_identity(&entry.bindings[0])
        );
    }
}
