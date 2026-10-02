//! MATLAB-compatible `tripuls` builtin for triangular pulse samples.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, TRIPULS_ERROR_ARG_COUNT, TRIPULS_ERROR_INTERNAL,
    TRIPULS_ERROR_INVALID_INPUT, TRIPULS_ERROR_SKEW_INVALID, TRIPULS_ERROR_WIDTH_INVALID,
    TRIPULS_GPU_INPUT_EXTENSION, TRIPULS_INTEGER_CONTROL_EXTENSION, TRIPULS_INTEGER_DATA_EXTENSION,
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

const BUILTIN_NAME: &str = "tripuls";

fn tripuls_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    tripuls_error_with_message(error.message, error)
}

fn tripuls_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    tripuls_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn tripuls_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tripuls_error_with_source(
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

pub(crate) fn tripuls_scalar(t: f64, width: f64, skew: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let left = -width / 2.0;
    let right = width / 2.0;
    if t < left || t > right {
        return 0.0;
    }
    let peak = skew * width / 2.0;
    if t == peak {
        return 1.0;
    }
    if t < peak {
        let span = peak - left;
        if span <= 0.0 {
            1.0
        } else {
            (t - left) / span
        }
    } else {
        let span = right - peak;
        if span <= 0.0 {
            1.0
        } else {
            (right - t) / span
        }
    }
}

pub(crate) fn validate_width(width: f64) -> Result<f64, String> {
    if !width.is_finite() || width <= 0.0 {
        Err(format!("got {width}"))
    } else {
        Ok(width)
    }
}

pub(crate) fn validate_skew(skew: f64) -> Result<f64, String> {
    if !skew.is_finite() || !(-1.0..=1.0).contains(&skew) {
        Err(format!("got {skew}"))
    } else {
        Ok(skew)
    }
}

#[runtime_builtin(
    name = "tripuls",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::tripuls"
)]
async fn tripuls_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    t: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    tripuls_in_runtime(runtime.as_ref(), t, rest).await
}

#[cfg(test)]
async fn tripuls_builtin(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    tripuls_in_runtime(None, t, rest).await
}

async fn tripuls_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    t: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let mut cooperation = WindowCooperation::new(runtime, BUILTIN_NAME);
    cooperation.completed(0).await?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(&t)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRIPULS_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if crate::builtins::common::validation::value_has_native_integer_class(&t) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRIPULS_INTEGER_DATA_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let controls = prepare_options(&rest)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &t,
        &TRIPULS_INTEGER_DATA_EXTENSION,
        BUILTIN_NAME,
        "sample times",
    )
    .await?;
    let (width, skew) = resolve_options(controls).await?;
    match t {
        Value::GpuTensor(handle) => tripuls_gpu(handle, width, skew, &mut cooperation).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(tripuls_error(&TRIPULS_ERROR_INVALID_INPUT))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(tripuls_error(&TRIPULS_ERROR_INVALID_INPUT))
        }
        other => tripuls_real(other, width, skew, &mut cooperation).await,
    }
}

type TripulsControls<'a> = (
    Option<PreparedWaveformControl<'a>>,
    Option<PreparedWaveformControl<'a>>,
);

fn prepare_options(rest: &[Value]) -> BuiltinResult<TripulsControls<'_>> {
    if rest.len() > 2 {
        return Err(tripuls_error_with_detail(
            &TRIPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        ));
    }
    let width = match rest.first() {
        Some(value) => {
            let control = PreparedWaveformControl::admit(
                value,
                &TRIPULS_INTEGER_CONTROL_EXTENSION,
                BUILTIN_NAME,
                "width or skew",
            )
            .map_err(|error| control_error(error, &TRIPULS_ERROR_WIDTH_INVALID))?;
            if let Some(raw) = control.host_value() {
                validate_width(raw)
                    .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_WIDTH_INVALID, err))?;
            }
            Some(control)
        }
        None => None,
    };
    let skew = match rest.get(1) {
        Some(value) => {
            let control = PreparedWaveformControl::admit(
                value,
                &TRIPULS_INTEGER_CONTROL_EXTENSION,
                BUILTIN_NAME,
                "width or skew",
            )
            .map_err(|error| control_error(error, &TRIPULS_ERROR_SKEW_INVALID))?;
            if let Some(raw) = control.host_value() {
                validate_skew(raw)
                    .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_SKEW_INVALID, err))?;
            }
            Some(control)
        }
        None => None,
    };
    Ok((width, skew))
}

async fn resolve_options(controls: TripulsControls<'_>) -> BuiltinResult<(f64, f64)> {
    let width = match controls.0 {
        Some(control) => {
            let raw = control
                .resolve()
                .await
                .map_err(|error| control_error(error, &TRIPULS_ERROR_WIDTH_INVALID))?;
            validate_width(raw)
                .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_WIDTH_INVALID, err))?
        }
        None => 1.0,
    };
    let skew = match controls.1 {
        Some(control) => {
            let raw = control
                .resolve()
                .await
                .map_err(|error| control_error(error, &TRIPULS_ERROR_SKEW_INVALID))?;
            validate_skew(raw)
                .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_SKEW_INVALID, err))?
        }
        None => 0.0,
    };
    Ok((width, skew))
}

fn control_error(
    error: WaveformControlError,
    invalid: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    match error {
        WaveformControlError::Invalid(detail) => tripuls_error_with_detail(invalid, detail),
        WaveformControlError::Boundary(source) => source,
        WaveformControlError::Provider(source) => {
            tripuls_error_with_source(&TRIPULS_ERROR_INTERNAL, "control download failed", source)
        }
    }
}

async fn tripuls_gpu(
    handle: GpuTensorHandle,
    width: f64,
    skew: f64,
    cooperation: &mut WindowCooperation,
) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|source| {
            tripuls_error_with_source(&TRIPULS_ERROR_INTERNAL, "gpu gather failed", source)
        })?;
    let result = map_real_waveform(
        tensor,
        WaveformOutputMode::PreserveSingle,
        |value| tripuls_scalar(value, width, skew),
        cooperation,
    )
    .await
    .map_err(|error| error.into_runtime_error(BUILTIN_NAME, &TRIPULS_ERROR_INTERNAL))?;
    let restored =
        gpu_helpers::restore_class_preserving_value(&handle, Value::Tensor(result), BUILTIN_NAME)?;
    if runmat_accelerate_api::handle_is_explicit(&handle)
        && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(tripuls_error_with_detail(
            &TRIPULS_ERROR_INTERNAL,
            "provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

async fn tripuls_real(
    value: Value,
    width: f64,
    skew: f64,
    cooperation: &mut WindowCooperation,
) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_INVALID_INPUT, err))?;
    map_real_waveform(
        tensor,
        WaveformOutputMode::PreserveSingle,
        |sample| tripuls_scalar(sample, width, skew),
        cooperation,
    )
    .await
    .map(tensor_into_value)
    .map_err(|error| error.into_runtime_error(BUILTIN_NAME, &TRIPULS_ERROR_INTERNAL))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::{CharArray, IntegerStorage, NumericStorage};

    fn call(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(tripuls_builtin(t, rest))
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

    #[test]
    fn tripuls_default_samples_triangle() {
        let input = Tensor::new(vec![-0.5, -0.25, 0.0, 0.25, 0.5], vec![1, 5]).unwrap();
        let out = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("tripuls"));
        assert_eq!(out.shape, vec![1, 5]);
        assert_eq!(out.materialize_f64(), vec![0.0, 0.5, 1.0, 0.5, 0.0]);
    }

    #[test]
    fn tripuls_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = integer_tensor(vec![-1, 0, 1], vec![1, 3]);
        let out =
            expect_tensor(call(Value::Tensor(input), vec![Value::Num(2.0)]).expect("tripuls"));
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn tripuls_preserves_documented_single_output_class() {
        let input = Tensor::from_f32(vec![-0.5, 0.0, 0.5], vec![1, 3]).expect("single input");
        let out = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("tripuls"));
        assert_eq!(
            out.into_numeric_storage().expect("single output"),
            NumericStorage::F32(vec![0.0, 1.0, 0.0])
        );
    }

    #[test]
    fn tripuls_integer_roles_are_independently_compatibility_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let input = integer_tensor(vec![-1, 0, 1], vec![1, 3]);
        let data_error = call(Value::Tensor(input), vec![Value::Num(2.0)])
            .expect_err("typed sample times are an extension");
        assert_eq!(
            data_error.identifier(),
            TRIPULS_INTEGER_DATA_EXTENSION.error_identifier
        );

        let control =
            Tensor::new_integer(IntegerStorage::U8(vec![2]), vec![1, 1]).expect("typed width");
        let control_error = call(Value::Num(0.0), vec![Value::Tensor(control)])
            .expect_err("typed width is an extension");
        assert_eq!(
            control_error.identifier(),
            TRIPULS_INTEGER_CONTROL_EXTENSION.error_identifier
        );
    }

    #[test]
    fn invalid_host_skew_precedes_resident_sample_access() {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
        };
        let owner = TestAllocationOwner::new();
        let input = owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        input.probe.mark_unavailable();
        let err = call(
            Value::GpuTensor(input.handle),
            vec![Value::Num(1.0), Value::Num(2.0)],
        )
        .expect_err("host skew");
        assert_eq!(err.identifier(), TRIPULS_ERROR_SKEW_INVALID.identifier);
    }

    #[test]
    fn tripuls_automatic_residency_is_transparent_and_explicit_residency_is_gated() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![-0.5, 0.0, 0.5], vec![1, 3]).expect("input");
            let automatic = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let _matlab_mode = crate::compatibility::push_runmat_extensions_enabled(false);
            let output =
                call(Value::GpuTensor(automatic), Vec::new()).expect("automatic resident input");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected automatic resident output");
            };
            assert!(!runmat_accelerate_api::handle_is_explicit(output_handle));
            assert_eq!(
                test_support::gather(output)
                    .expect("gather")
                    .materialize_f64(),
                vec![0.0, 1.0, 0.0]
            );

            let explicit = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let explicit = explicit
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("publish provenance view");
            let error = call(Value::GpuTensor(explicit), Vec::new())
                .expect_err("MATLAB mode rejects explicit interactive gpuArray input");
            assert_eq!(
                error.identifier(),
                TRIPULS_GPU_INPUT_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn tripuls_runmat_gpu_extension_preserves_explicit_provenance() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![-0.5, 0.0, 0.5], vec![1, 3]).expect("input");
            let explicit = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let explicit = explicit
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("publish provenance view");
            let _runmat_mode = crate::compatibility::push_runmat_extensions_enabled(true);
            let output = call(Value::GpuTensor(explicit), Vec::new())
                .expect("RunMat mode accepts explicit gpuArray input");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected explicit resident output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
            assert_eq!(
                test_support::gather(output)
                    .expect("gather")
                    .materialize_f64(),
                vec![0.0, 1.0, 0.0]
            );
        });
    }

    #[test]
    fn tripuls_skew_moves_peak() {
        let input = Tensor::new(vec![-1.0, 0.0, 1.0], vec![1, 3]).unwrap();
        let out = expect_tensor(
            call(Value::Tensor(input), vec![Value::Num(2.0), Value::Num(1.0)]).expect("tripuls"),
        );
        assert_eq!(out.materialize_f64(), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn tripuls_rejects_bad_options_and_text_input() {
        let err = call(Value::Num(0.0), vec![Value::Num(-1.0)]).expect_err("width");
        assert_eq!(err.identifier(), TRIPULS_ERROR_WIDTH_INVALID.identifier);

        let err = call(Value::Num(0.0), vec![Value::Num(1.0), Value::Num(2.0)]).expect_err("skew");
        assert_eq!(err.identifier(), TRIPULS_ERROR_SKEW_INVALID.identifier);

        let err =
            call(Value::CharArray(CharArray::new_row("abc")), Vec::new()).expect_err("text input");
        assert_eq!(err.identifier(), TRIPULS_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn tripuls_is_registered() {
        let entry = runmat_builtins::builtin_catalog_entry_by_name("tripuls").unwrap();
        assert!(builtin_function_by_name("tripuls").is_none());
        assert_eq!(entry.bindings.len(), 1);
        let bindings = crate::builtin::runtime_builtin_bindings_by_name("tripuls");
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].identity,
            entry.binding_identity(&entry.bindings[0])
        );
    }
}
