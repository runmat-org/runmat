//! MATLAB-compatible `sawtooth` builtin for RunMat.
//!
//! `y = sawtooth(t)` evaluates a sawtooth waveform with period `2*pi` at the
//! sample times in `t`. The optional second argument `xmax ∈ [0, 1]` controls
//! the position of the peak inside each period: `xmax = 1` (default) produces
//! a rising sawtooth, `xmax = 0` produces a falling sawtooth, and any value
//! in between (e.g. `xmax = 0.5` for a triangle wave) interpolates between
//! the two via a piecewise-linear ramp.

use std::f64::consts::PI;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, SAWTOOTH_ERROR_ARG_COUNT, SAWTOOTH_ERROR_COMPLEX_UNSUPPORTED,
    SAWTOOTH_ERROR_INTERNAL, SAWTOOTH_ERROR_INVALID_INPUT, SAWTOOTH_ERROR_XMAX_INVALID,
    SAWTOOTH_ERROR_XMAX_RANGE, SAWTOOTH_GPU_INPUT_EXTENSION, SAWTOOTH_NONDOUBLE_INPUT_EXTENSION,
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

const BUILTIN_NAME: &str = "sawtooth";
const TWO_PI: f64 = 2.0 * PI;

fn sawtooth_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    sawtooth_error_with_message(error.message, error)
}

fn sawtooth_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    sawtooth_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn sawtooth_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sawtooth_error_with_source(
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

/// Element-wise scalar sawtooth.
///
/// `width = xmax * 2π` is the peak position within the period. For phase
/// values in `[0, width)` the output rises linearly from `-1` to `+1`. For
/// phase values in `[width, 2π)` the output falls linearly from `+1` back to
/// `-1`. The boundary cases `xmax == 0` (pure falling) and `xmax == 1` (pure
/// rising) reduce naturally because `phi ∈ [0, 2π)` never reaches the open
/// upper bound of the rising branch.
#[inline]
fn sawtooth_scalar(t: f64, xmax: f64) -> f64 {
    if !t.is_finite() {
        return f64::NAN;
    }
    let phi = t.rem_euclid(TWO_PI);
    let width = xmax * TWO_PI;
    if width <= 0.0 {
        1.0 - phi / PI
    } else if phi < width {
        -1.0 + 2.0 * phi / width
    } else {
        let falling_width = TWO_PI - width;
        if falling_width <= 0.0 {
            -1.0 + 2.0 * phi / width
        } else {
            1.0 - 2.0 * (phi - width) / falling_width
        }
    }
}

#[runtime_builtin(
    name = "sawtooth",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::sawtooth"
)]
async fn sawtooth_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    t: Value,
    varargin: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    sawtooth_in_runtime(runtime.as_ref(), t, varargin).await
}

#[cfg(test)]
async fn sawtooth_builtin(t: Value, varargin: Vec<Value>) -> BuiltinResult<Value> {
    let varargin = crate::execution::RootedValues::new(varargin)
        .map_err(|error| error.into_runtime_error())?;
    sawtooth_in_runtime(None, t, varargin.into()).await
}

async fn sawtooth_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    t: Value,
    varargin: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime, BUILTIN_NAME)
        .with_integer_error_descriptor(&SAWTOOTH_ERROR_INTERNAL);
    cooperation.completed(0).await?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(&t)
        || varargin
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SAWTOOTH_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if varargin.len() <= 1
        && (is_supported_nondouble_argument(&t)
            || varargin.iter().any(is_supported_nondouble_argument))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SAWTOOTH_NONDOUBLE_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let mut xmax = prepare_xmax(varargin.view())?;
    admit_real_waveform_sample(&t).map_err(|error| match error {
        WaveformSampleError::Complex => sawtooth_error(&SAWTOOTH_ERROR_COMPLEX_UNSUPPORTED),
        WaveformSampleError::Unsupported => sawtooth_error(&SAWTOOTH_ERROR_INVALID_INPUT),
    })?;
    if let Some(control) = &mut xmax {
        control
            .admit_host_boundary(&mut cooperation)
            .await
            .map_err(xmax_control_error)?;
    }
    if let Some(control) = &xmax {
        control
            .admit_resident_metadata(BUILTIN_NAME, &SAWTOOTH_ERROR_INTERNAL)
            .map_err(xmax_control_error)?;
    }
    admit_resident_waveform_operand(&t, BUILTIN_NAME, "sample", &SAWTOOTH_ERROR_INTERNAL)?;
    let prepared = cooperation
        .prepare_integer_boundary(&t, &SAWTOOTH_NONDOUBLE_INPUT_EXTENSION, "t")
        .await?;
    let prepared_payload = prepared.into_resident_payload();
    let xmax = resolve_xmax(xmax, &mut cooperation).await?;
    match t {
        Value::GpuTensor(handle) => {
            sawtooth_gpu(handle, xmax, prepared_payload, &mut cooperation).await
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(sawtooth_error(&SAWTOOTH_ERROR_COMPLEX_UNSUPPORTED))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(sawtooth_error(&SAWTOOTH_ERROR_INVALID_INPUT))
        }
        other => sawtooth_real(other, xmax, &mut cooperation).await,
    }
}

fn is_supported_nondouble_argument(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => tensor.numeric_dtype() != NumericDType::F64,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32)
        }
        _ => false,
    }
}

fn prepare_xmax<'a>(
    varargin: crate::call::FunctionArgumentView<'a>,
) -> BuiltinResult<Option<PreparedWaveformControl<'a>>> {
    match varargin.len() {
        0 => Ok(None),
        1 => {
            let control = PreparedWaveformControl::admit(
                varargin.first().expect("single xmax operand was checked"),
                &SAWTOOTH_NONDOUBLE_INPUT_EXTENSION,
                BUILTIN_NAME,
                "xmax",
            )
            .map_err(xmax_control_error)?;
            if let Some(raw) = control.host_value() {
                validate_xmax(raw)?;
            }
            Ok(Some(control))
        }
        _ => Err(sawtooth_error_with_detail(
            &SAWTOOTH_ERROR_ARG_COUNT,
            format!("got {}", varargin.len() + 1),
        )),
    }
}

async fn resolve_xmax(
    control: Option<PreparedWaveformControl<'_>>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<f64> {
    let Some(control) = control else {
        return Ok(1.0);
    };
    validate_xmax(
        control
            .resolve(cooperation)
            .await
            .map_err(xmax_control_error)?,
    )
}

fn validate_xmax(raw: f64) -> BuiltinResult<f64> {
    if !raw.is_finite() || !(0.0..=1.0).contains(&raw) {
        return Err(sawtooth_error_with_detail(
            &SAWTOOTH_ERROR_XMAX_RANGE,
            format!("got {raw}"),
        ));
    }
    Ok(raw)
}

fn xmax_control_error(error: WaveformControlError) -> RuntimeError {
    match error {
        WaveformControlError::Invalid(detail) => {
            sawtooth_error_with_detail(&SAWTOOTH_ERROR_XMAX_INVALID, detail)
        }
        WaveformControlError::Boundary(source) => source,
        WaveformControlError::Provider(source) => {
            sawtooth_error_with_source(&SAWTOOTH_ERROR_INTERNAL, "control download failed", source)
        }
    }
}

async fn sawtooth_gpu(
    handle: GpuTensorHandle,
    xmax: f64,
    prepared_payload: Option<crate::builtins::common::validation::PreparedIntegerPayload>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    let tensor = match prepared_real_integer_tensor(prepared_payload)
        .map_err(|detail| sawtooth_error_with_detail(&SAWTOOTH_ERROR_INVALID_INPUT, detail))?
    {
        Some(tensor) => tensor,
        None => gpu_helpers::gather_tensor_async(&handle)
            .await
            .map_err(|source| {
                sawtooth_error_with_source(&SAWTOOTH_ERROR_INTERNAL, "gpu gather failed", source)
            })?,
    };
    map_real_waveform(
        tensor,
        WaveformOutputMode::Double,
        |value| sawtooth_scalar(value, xmax),
        cooperation,
    )
    .await?
    .map(tensor_into_value)
    .map_err(|error| error.into_runtime_error(BUILTIN_NAME, &SAWTOOTH_ERROR_INTERNAL))
}

async fn sawtooth_real(
    value: Value,
    xmax: f64,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    let tensor = waveform_input_tensor(value, BUILTIN_NAME, cooperation)
        .await?
        .map_err(|error| {
            error.into_runtime_error(
                BUILTIN_NAME,
                &SAWTOOTH_ERROR_INVALID_INPUT,
                &SAWTOOTH_ERROR_INTERNAL,
            )
        })?;
    map_real_waveform(
        tensor,
        WaveformOutputMode::Double,
        |sample| sawtooth_scalar(sample, xmax),
        cooperation,
    )
    .await?
    .map(tensor_into_value)
    .map_err(|error| error.into_runtime_error(BUILTIN_NAME, &SAWTOOTH_ERROR_INTERNAL))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_host_xmax_precedes_resident_sample_access() {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
        };
        let owner = TestAllocationOwner::new();
        let input = owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        input.probe.mark_unavailable();
        let err =
            call_with_xmax(Value::GpuTensor(input.handle), Value::Num(2.0)).expect_err("host xmax");
        assert_eq!(err.identifier(), SAWTOOTH_ERROR_XMAX_RANGE.identifier);
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

    #[test]
    fn invalid_host_sample_precedes_resident_xmax_access() {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
        };
        let owner = TestAllocationOwner::new();
        let control = owner
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        control.probe.mark_unavailable();
        let error = call_with_xmax(
            Value::String("invalid".into()),
            Value::GpuTensor(control.handle),
        )
        .expect_err("invalid sample before control download");
        assert_eq!(error.identifier(), SAWTOOTH_ERROR_INVALID_INPUT.identifier);
    }

    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::{IntValue, LogicalArray, Tensor};

    fn call(t: Value) -> BuiltinResult<Value> {
        block_on(sawtooth_builtin(t, Vec::new()))
    }

    fn call_with_xmax(t: Value, xmax: Value) -> BuiltinResult<Value> {
        block_on(sawtooth_builtin(t, vec![xmax]))
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

    fn assert_close(got: f64, want: f64) {
        assert!(
            (got - want).abs() < 1e-12,
            "got {got}, expected {want} (diff {})",
            (got - want).abs()
        );
    }

    #[test]
    fn sawtooth_descriptor_signatures_and_errors() {
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
        assert!(labels.contains(&"Y = sawtooth(t)"));
        assert!(labels.contains(&"Y = sawtooth(t, xmax)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.SAWTOOTH.XMAX_RANGE"));
    }

    #[test]
    fn sawtooth_default_is_rising_within_first_period() {
        assert_close(expect_num(call(Value::Num(0.0)).unwrap()), -1.0);
        assert_close(expect_num(call(Value::Num(PI / 2.0)).unwrap()), -0.5);
        assert_close(expect_num(call(Value::Num(PI)).unwrap()), 0.0);
        assert_close(expect_num(call(Value::Num(3.0 * PI / 2.0)).unwrap()), 0.5);
    }

    #[test]
    fn sawtooth_period_boundary_wraps_to_minus_one() {
        assert_close(expect_num(call(Value::Num(TWO_PI)).unwrap()), -1.0);
        assert_close(expect_num(call(Value::Num(4.0 * PI)).unwrap()), -1.0);
        assert_close(expect_num(call(Value::Num(-TWO_PI)).unwrap()), -1.0);
    }

    #[test]
    fn sawtooth_negative_time_wraps_into_period() {
        assert_close(expect_num(call(Value::Num(-PI)).unwrap()), 0.0);
        assert_close(expect_num(call(Value::Num(-PI / 2.0)).unwrap()), 0.5);
    }

    #[test]
    fn sawtooth_vector_two_periods_ranges_from_minus_one_to_just_below_one() {
        let n: usize = 100;
        let total = 4.0 * PI;
        let step = total / (n as f64 - 1.0);
        let data: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
        let tensor = Tensor::new(data.clone(), vec![1, n]).unwrap();
        let result = expect_tensor(call(Value::Tensor(tensor)).unwrap());
        assert_eq!(result.shape(), vec![1, n]);
        let min = result
            .materialize_f64()
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = result
            .materialize_f64()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (-1.0 - 1e-12..=-1.0 + 1e-12).contains(&min),
            "min should be -1, got {min}"
        );
        assert!(
            max <= 1.0 + 1e-12 && max > 0.95,
            "max should approach 1 from below, got {max}"
        );
        // First and last samples land exactly on period boundaries (t=0 and t=4*pi)
        // so the rising sawtooth resets to -1 at both ends.
        assert_close(result.materialize_f64()[0], -1.0);
        assert_close(*result.materialize_f64().last().unwrap(), -1.0);

        // Each sample must satisfy the elementwise formula.
        for (idx, (&t, &y)) in data.iter().zip(result.materialize_f64().iter()).enumerate() {
            assert_close(y, sawtooth_scalar(t, 1.0));
            if !y.is_finite() {
                panic!("non-finite sample at index {idx}");
            }
        }

        // There should be exactly two period boundaries in (0, 4*pi]: the inner
        // reset near index 50 (sample just after 2*pi) and the closing sample.
        let reset_count = result
            .materialize_f64()
            .iter()
            .enumerate()
            .filter(|(idx, &y)| *idx > 0 && y < result.materialize_f64()[idx - 1])
            .count();
        assert_eq!(
            reset_count, 2,
            "expected two period resets across two periods"
        );
    }

    #[test]
    fn sawtooth_xmax_half_is_triangle_wave_with_peak_at_pi() {
        let half = Value::Num(0.5);
        assert_close(
            expect_num(call_with_xmax(Value::Num(0.0), half.clone()).unwrap()),
            -1.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(PI / 2.0), half.clone()).unwrap()),
            0.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(PI), half.clone()).unwrap()),
            1.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(3.0 * PI / 2.0), half.clone()).unwrap()),
            0.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(TWO_PI), half).unwrap()),
            -1.0,
        );
    }

    #[test]
    fn sawtooth_xmax_zero_is_pure_falling() {
        let zero = Value::Num(0.0);
        assert_close(
            expect_num(call_with_xmax(Value::Num(0.0), zero.clone()).unwrap()),
            1.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(PI), zero.clone()).unwrap()),
            0.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(TWO_PI), zero).unwrap()),
            1.0,
        );
    }

    #[test]
    fn sawtooth_xmax_one_is_pure_rising() {
        let one = Value::Num(1.0);
        assert_close(
            expect_num(call_with_xmax(Value::Num(0.0), one.clone()).unwrap()),
            -1.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(PI), one.clone()).unwrap()),
            0.0,
        );
        // The default call (no xmax) must agree with the explicit xmax = 1 form.
        for &t in &[-3.7, -1.0, 0.0, 0.25, PI, 5.5, 9.0] {
            assert_close(
                expect_num(call(Value::Num(t)).unwrap()),
                expect_num(call_with_xmax(Value::Num(t), one.clone()).unwrap()),
            );
        }
    }

    #[test]
    fn sawtooth_xmax_rejects_out_of_range() {
        assert!(call_with_xmax(Value::Num(0.0), Value::Num(-0.1)).is_err());
        assert!(call_with_xmax(Value::Num(0.0), Value::Num(1.1)).is_err());
        assert!(call_with_xmax(Value::Num(0.0), Value::Num(f64::NAN)).is_err());
        assert!(call_with_xmax(Value::Num(0.0), Value::Num(f64::INFINITY)).is_err());
    }

    #[test]
    fn sawtooth_xmax_rejects_non_scalar() {
        let tensor = Tensor::new(vec![0.5, 1.0], vec![1, 2]).unwrap();
        assert!(call_with_xmax(Value::Num(0.0), Value::Tensor(tensor)).is_err());
    }

    #[test]
    fn sawtooth_int_and_logical_promote_to_double() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let int_result = expect_num(call(Value::Int(IntValue::I32(0))).unwrap());
        assert_close(int_result, -1.0);

        let bool_result = expect_num(call(Value::Bool(false)).unwrap());
        assert_close(bool_result, -1.0);

        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = expect_tensor(call(Value::LogicalArray(logical)).unwrap());
        assert_eq!(result.shape(), vec![1, 2]);
        assert_close(result.materialize_f64()[0], -1.0);
        assert_close(result.materialize_f64()[1], sawtooth_scalar(1.0, 1.0));
    }

    #[test]
    fn sawtooth_single_extension_promotes_to_documented_double_domain() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = Tensor::from_numeric_storage(
            runmat_value::NumericStorage::F32(vec![0.0, 1.0]),
            vec![1, 2],
        )
        .unwrap();
        let result = expect_tensor(call(Value::Tensor(input)).unwrap());
        assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_close(result.materialize_f64()[0], -1.0);
        assert_close(result.materialize_f64()[1], sawtooth_scalar(1.0, 1.0));
    }

    #[test]
    fn sawtooth_extensions_follow_compatibility_mode() {
        let single = || {
            Value::Tensor(
                Tensor::from_numeric_storage(
                    runmat_value::NumericStorage::F32(vec![0.0, 1.0]),
                    vec![1, 2],
                )
                .expect("single input"),
            )
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = call(single()).expect_err("MATLAB mode rejects single input");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:SawtoothNondoubleInputExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(call(single()).is_ok());
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
            Some("RunMat:compatibility:SawtoothGpuInputExtension")
        );
    }

    #[test]
    fn sawtooth_automatic_residency_can_materialize_in_matlab_mode() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, std::f64::consts::PI], vec![1, 2]).unwrap();
            let automatic = gpu_helpers::upload_tensor(provider, &input).unwrap();
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let result = call(Value::GpuTensor(automatic)).expect("automatic host materialization");
            assert_eq!(expect_tensor(result).materialize_f64(), vec![-1.0, 0.0]);
        });
    }

    #[test]
    fn sawtooth_nonfinite_inputs_return_nan() {
        assert!(expect_num(call(Value::Num(f64::NAN)).unwrap()).is_nan());
        assert!(expect_num(call(Value::Num(f64::INFINITY)).unwrap()).is_nan());
        assert!(expect_num(call(Value::Num(f64::NEG_INFINITY)).unwrap()).is_nan());
    }

    #[test]
    fn sawtooth_complex_input_errors() {
        let err = call(Value::Complex(1.0, 2.0)).expect_err("complex sawtooth should error");
        assert!(err
            .message()
            .contains("sawtooth: input must be real; complex values are not supported"));
    }

    #[test]
    fn sawtooth_text_input_errors() {
        let err = call(Value::String("0".into())).expect_err("text sawtooth should error");
        assert!(err.message().contains("sawtooth: expected numeric input"));
    }

    #[test]
    fn sawtooth_preserves_matrix_shape() {
        let tensor = Tensor::new(vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0], vec![2, 2]).unwrap();
        let result = expect_tensor(call(Value::Tensor(tensor)).unwrap());
        assert_eq!(result.shape(), vec![2, 2]);
        let expected = [-1.0, -0.5, 0.0, 0.5];
        for (got, want) in result.materialize_f64().iter().zip(expected) {
            assert_close(*got, want);
        }
    }
}
