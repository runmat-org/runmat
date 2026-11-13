//! MATLAB-compatible `square` builtin for RunMat.
//!
//! `y = square(t)` evaluates a square wave with period `2*pi` at the sample
//! times in `t`, taking the value `+1` over the first half of each period and
//! `-1` over the second half. The optional second argument `duty ∈ [0, 100]`
//! is the duty cycle expressed as a percentage: the output is `+1` over the
//! first `duty/100 * 2*pi` of every period and `-1` over the remainder.

use std::f64::consts::PI;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, SQUARE_ERROR_ARG_COUNT, SQUARE_ERROR_COMPLEX_UNSUPPORTED,
    SQUARE_ERROR_DUTY_INVALID, SQUARE_ERROR_DUTY_RANGE, SQUARE_ERROR_INTERNAL,
    SQUARE_ERROR_INVALID_INPUT, SQUARE_GPU_INPUT_EXTENSION, SQUARE_NONFLOATING_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor::tensor_into_value;
use crate::builtins::math::signal::common::{
    admit_real_waveform_sample, admit_resident_waveform_operand, map_real_waveform,
    prepared_real_integer_tensor, waveform_input_tensor, PreparedWaveformControl,
    SignalCooperation, WaveformControlError, WaveformOutputMode, WaveformSampleError,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "square";
const TWO_PI: f64 = 2.0 * PI;

fn square_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    square_error_with_message(error.message, error)
}

fn square_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    square_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn square_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn square_error_with_source(
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

/// Element-wise scalar square wave.
///
/// Returns `+1` while the reduced phase is strictly less than
/// `duty/100 * 2π` and `-1` for the remainder of the period. `duty = 0`
/// therefore degenerates to a constant `-1` and `duty = 100` to a constant
/// `+1`, matching MATLAB's documented half-open interval convention.
#[inline]
fn square_scalar(t: f64, duty: f64) -> f64 {
    if !t.is_finite() {
        return f64::NAN;
    }
    if duty == 0.0 {
        return -1.0;
    }
    if duty == 100.0 {
        return 1.0;
    }
    let phi = t.rem_euclid(TWO_PI);
    // Keep the percentage comparison in its non-underflowing scale. A
    // positive subnormal duty still includes phase zero.
    if phi * 100.0 < duty * TWO_PI {
        1.0
    } else {
        -1.0
    }
}

#[runtime_builtin(
    name = "square",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::square"
)]
async fn square_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    t: Value,
    varargin: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    square_in_runtime(runtime.as_ref(), t, varargin).await
}

#[cfg(test)]
async fn square_builtin(t: Value, varargin: Vec<Value>) -> BuiltinResult<Value> {
    let varargin = crate::execution::RootedValues::new(varargin)
        .map_err(|error| error.into_runtime_error())?;
    square_in_runtime(None, t, varargin.into()).await
}

async fn square_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    t: Value,
    varargin: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime, BUILTIN_NAME)
        .with_integer_error_descriptor(&SQUARE_ERROR_INTERNAL);
    cooperation.completed(0).await?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(&t)
        || varargin
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SQUARE_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if varargin.len() <= 1
        && (is_supported_nonfloating_argument(&t)
            || varargin.iter().any(is_supported_nonfloating_argument))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SQUARE_NONFLOATING_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let mut duty = prepare_duty(varargin.view())?;
    admit_real_waveform_sample(&t).map_err(|error| match error {
        WaveformSampleError::Complex => square_error(&SQUARE_ERROR_COMPLEX_UNSUPPORTED),
        WaveformSampleError::Unsupported => square_error(&SQUARE_ERROR_INVALID_INPUT),
    })?;
    if let Some(control) = &mut duty {
        control
            .admit_host_boundary(&mut cooperation)
            .await
            .map_err(duty_control_error)?;
    }
    if let Some(control) = &duty {
        control
            .admit_resident_metadata(BUILTIN_NAME, &SQUARE_ERROR_INTERNAL)
            .map_err(duty_control_error)?;
    }
    admit_resident_waveform_operand(&t, BUILTIN_NAME, "sample", &SQUARE_ERROR_INTERNAL)?;
    let prepared = cooperation
        .prepare_integer_boundary(&t, &SQUARE_NONFLOATING_INPUT_EXTENSION, "sample-time")
        .await?;
    let prepared_payload = prepared.into_resident_payload();
    let duty = resolve_duty(duty, &mut cooperation).await?;
    match t {
        Value::GpuTensor(handle) => {
            square_gpu(handle, duty, prepared_payload, &mut cooperation).await
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(square_error(&SQUARE_ERROR_COMPLEX_UNSUPPORTED))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(square_error(&SQUARE_ERROR_INVALID_INPUT))
        }
        other => square_real(other, duty, &mut cooperation).await,
    }
}

fn is_supported_nonfloating_argument(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => !matches!(
            tensor.numeric_dtype(),
            NumericDType::F64 | NumericDType::F32
        ),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_integer_type(handle).is_some()
        }
        _ => false,
    }
}

fn prepare_duty<'a>(
    varargin: crate::call::FunctionArgumentView<'a>,
) -> BuiltinResult<Option<PreparedWaveformControl<'a>>> {
    match varargin.len() {
        0 => Ok(None),
        1 => {
            let control = PreparedWaveformControl::admit(
                varargin.first().expect("single duty operand was checked"),
                &SQUARE_NONFLOATING_INPUT_EXTENSION,
                BUILTIN_NAME,
                "duty-cycle",
            )
            .map_err(duty_control_error)?;
            if let Some(raw) = control.host_value() {
                validate_duty(raw)?;
            }
            Ok(Some(control))
        }
        _ => Err(square_error_with_detail(
            &SQUARE_ERROR_ARG_COUNT,
            format!("got {}", varargin.len() + 1),
        )),
    }
}

async fn resolve_duty(
    control: Option<PreparedWaveformControl<'_>>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<f64> {
    let Some(control) = control else {
        return Ok(50.0);
    };
    validate_duty(
        control
            .resolve(cooperation)
            .await
            .map_err(duty_control_error)?,
    )
}

fn validate_duty(raw: f64) -> BuiltinResult<f64> {
    if !raw.is_finite() || !(0.0..=100.0).contains(&raw) {
        return Err(square_error_with_detail(
            &SQUARE_ERROR_DUTY_RANGE,
            format!("got {raw}"),
        ));
    }
    Ok(raw)
}

fn duty_control_error(error: WaveformControlError) -> RuntimeError {
    match error {
        WaveformControlError::Invalid(detail) => {
            square_error_with_detail(&SQUARE_ERROR_DUTY_INVALID, detail)
        }
        WaveformControlError::Boundary(source) => source,
        WaveformControlError::Provider(source) => {
            square_error_with_source(&SQUARE_ERROR_INTERNAL, "control download failed", source)
        }
    }
}

async fn square_gpu(
    handle: GpuTensorHandle,
    duty: f64,
    prepared_payload: Option<crate::builtins::common::validation::PreparedIntegerPayload>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    let tensor = match prepared_real_integer_tensor(prepared_payload)
        .map_err(|detail| square_error_with_detail(&SQUARE_ERROR_INVALID_INPUT, detail))?
    {
        Some(tensor) => tensor,
        None => gpu_helpers::gather_tensor_async(&handle)
            .await
            .map_err(|source| {
                square_error_with_source(&SQUARE_ERROR_INTERNAL, "gpu gather failed", source)
            })?,
    };
    map_real_waveform(
        tensor,
        WaveformOutputMode::PreserveSingle,
        |value| square_scalar(value, duty),
        cooperation,
    )
    .await?
    .map(tensor_into_value)
    .map_err(|error| error.into_runtime_error(BUILTIN_NAME, &SQUARE_ERROR_INTERNAL))
}

async fn square_real(
    value: Value,
    duty: f64,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    let tensor = waveform_input_tensor(value, BUILTIN_NAME, cooperation)
        .await?
        .map_err(|error| {
            error.into_runtime_error(
                BUILTIN_NAME,
                &SQUARE_ERROR_INVALID_INPUT,
                &SQUARE_ERROR_INTERNAL,
            )
        })?;
    map_real_waveform(
        tensor,
        WaveformOutputMode::PreserveSingle,
        |sample| square_scalar(sample, duty),
        cooperation,
    )
    .await?
    .map(tensor_into_value)
    .map_err(|error| error.into_runtime_error(BUILTIN_NAME, &SQUARE_ERROR_INTERNAL))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingControlProvider {
        owner: runmat_accelerate_api::test_support::TestAllocationOwner,
        probe: runmat_accelerate_api::test_support::RetirementProbe,
        downloads: std::sync::atomic::AtomicUsize,
        fail_download: bool,
    }

    impl runmat_accelerate_api::AccelProvider for CountingControlProvider {
        fn device_id(&self) -> u32 {
            self.owner.device_id()
        }
        fn device_info(&self) -> String {
            "waveform-control-counting-provider".into()
        }
        fn upload(
            &self,
            _: &runmat_accelerate_api::HostTensorView<'_>,
        ) -> anyhow::Result<GpuTensorHandle> {
            anyhow::bail!("control-only fixture must not upload")
        }
        fn download<'a>(
            &'a self,
            handle: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            self.downloads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Box::pin(async move {
                if self.fail_download {
                    anyhow::bail!("control download device fault")
                }
                let data = self.probe.with_data(handle, |data| match data {
                    runmat_accelerate_api::HostNumericDataView::F64(data) => Ok(data.to_vec()),
                    _ => Err(anyhow::anyhow!("expected F64 control")),
                })??;
                Ok(runmat_accelerate_api::HostTensorOwned {
                    data,
                    shape: handle.shape().to_vec(),
                    storage: handle.physical().storage(),
                })
            })
        }
    }

    fn counted_control(fail_download: bool) -> (&'static CountingControlProvider, GpuTensorHandle) {
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let allocation = owner
            .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                data: runmat_accelerate_api::HostNumericDataOwned::F64(vec![25.0]),
                shape: vec![1, 1],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .unwrap();
        let provider = Box::leak(Box::new(CountingControlProvider {
            owner,
            probe: allocation.probe,
            downloads: std::sync::atomic::AtomicUsize::new(0),
            fail_download,
        }));
        (provider, allocation.handle)
    }

    #[test]
    fn resident_duty_downloads_once_and_preserves_operational_failure() {
        use std::sync::atomic::Ordering;
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        for fail in [false, true] {
            let (provider, control) = counted_control(fail);
            let borrowed = control.clone();
            let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
            let result = call_with_duty(Value::Num(0.0), Value::GpuTensor(control));
            assert_eq!(provider.downloads.load(Ordering::Relaxed), 1);
            assert!(borrowed.allocation_lease().validate_live().is_ok());
            if fail {
                let error = result.expect_err("operational download failure");
                assert_eq!(error.identifier(), SQUARE_ERROR_INTERNAL.identifier);
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert!(error.source.is_some());
            } else {
                assert!(matches!(result.unwrap(), Value::Num(value) if value == 1.0));
            }
        }
    }

    #[test]
    fn known_invalid_sample_does_not_download_resident_duty() {
        use std::sync::atomic::Ordering;
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let (provider, control) = counted_control(true);
        let borrowed = control.clone();
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let error = call_with_duty(Value::String("invalid".into()), Value::GpuTensor(control))
            .expect_err("known invalid sample");
        assert_eq!(error.identifier(), SQUARE_ERROR_INVALID_INPUT.identifier);
        assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
        assert!(borrowed.allocation_lease().validate_live().is_ok());
    }

    #[test]
    fn resident_sample_owner_metadata_keeps_square_context_before_download() {
        use std::sync::atomic::Ordering;
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let (provider, _) = counted_control(true);
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let foreign_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let allocation = foreign_owner
            .allocate_zeroed(
                vec![2, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let borrowed = allocation.handle.clone();
        let error = call(Value::GpuTensor(allocation.handle))
            .expect_err("resident sample without an active owner must be rejected");
        assert_eq!(error.identifier(), SQUARE_ERROR_INTERNAL.identifier);
        assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN_NAME));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        let original = error
            .source
            .as_deref()
            .and_then(|cause| cause.downcast_ref::<RuntimeError>())
            .expect("typed metadata failure retained as source");
        assert_ne!(original.identifier(), SQUARE_ERROR_INTERNAL.identifier);
        assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
        borrowed.allocation_lease().validate_live().unwrap();
        assert_eq!(allocation.probe.retirements(), 0);
    }

    #[test]
    fn invalid_host_duty_precedes_resident_sample_access() {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
        };
        let owner = TestAllocationOwner::new();
        let input = owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        input.probe.mark_unavailable();
        let err = call_with_duty(Value::GpuTensor(input.handle), Value::Num(101.0))
            .expect_err("host duty");
        assert_eq!(err.identifier(), SQUARE_ERROR_DUTY_RANGE.identifier);
    }

    #[test]
    fn integer_sample_literal_boundary_matches_catalog_fact() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error =
            call(Value::Int(IntValue::U64((1_u64 << 53) + 1))).expect_err("inexact sample time");
        assert!(error.message().contains("exactly representable"));
        assert!(matches!(
            call(Value::Int(IntValue::U64(1_u64 << 54))).unwrap(),
            Value::Num(_)
        ));
    }
    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::{IntValue, LogicalArray, NumericStorage, Tensor};

    fn call(t: Value) -> BuiltinResult<Value> {
        block_on(square_builtin(t, Vec::new()))
    }

    fn call_with_duty(t: Value, duty: Value) -> BuiltinResult<Value> {
        block_on(square_builtin(t, vec![duty]))
    }

    fn expect_num(value: Value) -> f64 {
        match value {
            Value::Num(v) => v,
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn square_descriptor_signatures_and_errors() {
        let entry = runmat_builtins::builtin_catalog_entry_by_name(BUILTIN_NAME).unwrap();
        assert!(builtin_function_by_name(BUILTIN_NAME).is_none());
        let bindings = crate::builtin::runtime_builtin_bindings_by_name(BUILTIN_NAME);
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].identity,
            entry.binding_identity(&entry.bindings[0])
        );
        let descriptor = entry.descriptor;
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = square(t)"));
        assert!(labels.contains(&"Y = square(t, duty)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.SQUARE.DUTY_RANGE"));
    }

    #[test]
    fn square_default_is_fifty_percent_duty() {
        assert_eq!(expect_num(call(Value::Num(0.0)).unwrap()), 1.0);
        assert_eq!(expect_num(call(Value::Num(PI / 2.0)).unwrap()), 1.0);
        // The boundary at phi == pi falls into the second half (half-open semantics).
        assert_eq!(expect_num(call(Value::Num(PI)).unwrap()), -1.0);
        assert_eq!(expect_num(call(Value::Num(3.0 * PI / 2.0)).unwrap()), -1.0);
        // A full period wraps back to +1.
        assert_eq!(expect_num(call(Value::Num(TWO_PI)).unwrap()), 1.0);
    }

    #[test]
    fn smallest_positive_duty_includes_only_representable_zero_phase() {
        let minimum = f64::from_bits(1);
        assert_eq!(square_scalar(0.0, minimum), 1.0);
        assert_eq!(square_scalar(TWO_PI, minimum), 1.0);
        assert_eq!(square_scalar(minimum, minimum), -1.0);
        assert_eq!(square_scalar(0.0, 0.0), -1.0);
        assert_eq!(square_scalar(0.0, 100.0), 1.0);
        assert_eq!(
            square_scalar(f64::from_bits(TWO_PI.to_bits() - 1), 100.0),
            1.0
        );
    }

    #[test]
    fn square_vector_only_contains_plus_or_minus_one() {
        let n: usize = 64;
        let step = TWO_PI / (n as f64 - 1.0);
        let data: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
        let tensor = Tensor::new(data.clone(), vec![1, n]).unwrap();
        let result = expect_tensor(call(Value::Tensor(tensor)).unwrap());
        assert_eq!(result.shape(), vec![1, n]);
        for (idx, &value) in result.materialize_f64().iter().enumerate() {
            assert!(
                value == 1.0 || value == -1.0,
                "index {idx}: expected only +-1 values, got {value}"
            );
        }
        // Each sample must agree with the elementwise formula.
        for (&t, &y) in data.iter().zip(result.materialize_f64().iter()) {
            assert_eq!(y, square_scalar(t, 50.0));
        }
        // The wave is +1 across the first half period, then -1 across the second
        // half. The final sample lands on the period boundary (t = 2*pi) and wraps
        // back to phi = 0 -> +1.
        assert_eq!(result.materialize_f64()[0], 1.0);
        let toggle_point = data.iter().position(|&t| t >= PI).unwrap();
        assert!(result.materialize_f64()[..toggle_point]
            .iter()
            .all(|&v| v == 1.0));
        assert!(result.materialize_f64()[toggle_point..n - 1]
            .iter()
            .all(|&v| v == -1.0));
        assert_eq!(*result.materialize_f64().last().unwrap(), 1.0);
    }

    #[test]
    fn square_negative_time_wraps_into_period() {
        // -pi/2 wraps to 3*pi/2 -> second half -> -1.
        assert_eq!(expect_num(call(Value::Num(-PI / 2.0)).unwrap()), -1.0);
        // -3*pi/2 wraps to pi/2 -> first half -> +1.
        assert_eq!(expect_num(call(Value::Num(-3.0 * PI / 2.0)).unwrap()), 1.0);
    }

    #[test]
    fn square_duty_zero_is_constant_minus_one() {
        for &t in &[0.0, PI / 2.0, PI, 3.0 * PI / 2.0, TWO_PI] {
            assert_eq!(
                expect_num(call_with_duty(Value::Num(t), Value::Num(0.0)).unwrap()),
                -1.0
            );
        }
    }

    #[test]
    fn square_duty_one_hundred_is_constant_plus_one() {
        for &t in &[0.0, PI / 2.0, PI, 3.0 * PI / 2.0, TWO_PI - 1e-9] {
            assert_eq!(
                expect_num(call_with_duty(Value::Num(t), Value::Num(100.0)).unwrap()),
                1.0
            );
        }
    }

    #[test]
    fn square_duty_twenty_five_only_first_quarter_is_positive() {
        let n: usize = 80;
        let step = TWO_PI / (n as f64);
        let data: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
        let tensor = Tensor::new(data.clone(), vec![1, n]).unwrap();
        let result =
            expect_tensor(call_with_duty(Value::Tensor(tensor), Value::Num(25.0)).unwrap());
        let threshold = PI / 2.0;
        for (idx, (&t, &y)) in data.iter().zip(result.materialize_f64().iter()).enumerate() {
            let expected = if t < threshold { 1.0 } else { -1.0 };
            assert_eq!(y, expected, "index {idx}, t={t}");
        }
    }

    #[test]
    fn square_duty_fifty_matches_default_call() {
        for &t in &[0.0, 0.7, PI - 1e-3, PI, PI + 1e-3, 5.5, -2.0, TWO_PI] {
            assert_eq!(
                expect_num(call(Value::Num(t)).unwrap()),
                expect_num(call_with_duty(Value::Num(t), Value::Num(50.0)).unwrap())
            );
        }
    }

    #[test]
    fn square_duty_rejects_out_of_range() {
        assert!(call_with_duty(Value::Num(0.0), Value::Num(-1.0)).is_err());
        assert!(call_with_duty(Value::Num(0.0), Value::Num(101.0)).is_err());
        assert!(call_with_duty(Value::Num(0.0), Value::Num(f64::NAN)).is_err());
        assert!(call_with_duty(Value::Num(0.0), Value::Num(f64::NEG_INFINITY)).is_err());
    }

    #[test]
    fn square_duty_rejects_non_scalar() {
        let tensor = Tensor::new(vec![25.0, 50.0], vec![1, 2]).unwrap();
        assert!(call_with_duty(Value::Num(0.0), Value::Tensor(tensor)).is_err());
    }

    #[test]
    fn square_int_and_logical_promote_to_double() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert_eq!(expect_num(call(Value::Int(IntValue::I32(0))).unwrap()), 1.0);
        assert_eq!(expect_num(call(Value::Bool(true)).unwrap()), 1.0);

        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let result = expect_tensor(call(Value::LogicalArray(logical)).unwrap());
        assert_eq!(result.shape(), vec![1, 2]);
        assert_eq!(result.materialize_f64(), vec![1.0, 1.0]);
    }

    #[test]
    fn square_extensions_follow_compatibility_mode() {
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error =
                call(Value::Int(IntValue::I16(0))).expect_err("MATLAB mode rejects integer input");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:SquareNonfloatingInputExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(call(Value::Int(IntValue::I16(0))).is_ok());
        }

        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("admit owned fixture")
            .handle;
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call(Value::GpuTensor(handle))
            .expect_err("MATLAB mode rejects explicit interactive gpuArray input before gather");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:SquareGpuInputExtension")
        );
    }

    #[test]
    fn square_preserves_native_single_storage() {
        let input =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0, 4.0]), vec![1, 2]).unwrap();
        let result = expect_tensor(call(Value::Tensor(input)).unwrap());
        assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(
            result.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, -1.0])
        );
    }

    #[test]
    fn square_nonfinite_inputs_return_nan() {
        assert!(expect_num(call(Value::Num(f64::NAN)).unwrap()).is_nan());
        assert!(expect_num(call(Value::Num(f64::INFINITY)).unwrap()).is_nan());
        assert!(expect_num(call(Value::Num(f64::NEG_INFINITY)).unwrap()).is_nan());
    }

    #[test]
    fn square_complex_input_errors() {
        let err = call(Value::Complex(1.0, 2.0)).expect_err("complex square should error");
        assert!(err
            .message()
            .contains("square: input must be real; complex values are not supported"));
    }

    #[test]
    fn square_text_input_errors() {
        let err = call(Value::String("0".into())).expect_err("text square should error");
        assert!(err.message().contains("square: expected numeric input"));
    }

    #[test]
    fn square_preserves_matrix_shape() {
        let tensor = Tensor::new(vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0], vec![2, 2]).unwrap();
        let result = expect_tensor(call(Value::Tensor(tensor)).unwrap());
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.materialize_f64(), vec![1.0, 1.0, -1.0, -1.0]);
    }
}
