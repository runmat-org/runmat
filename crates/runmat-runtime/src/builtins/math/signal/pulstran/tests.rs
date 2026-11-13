use super::evaluation::callback_samples;
use super::*;
use futures::executor::block_on;
use runmat_builtins::builtin_function_by_name;
use runmat_value::{CharArray, IntegerStorage, StringArray};

fn call(t: Value, d: Value, pulse: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    block_on(pulstran_builtin(None, t, d, pulse, rest))
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
fn pulstran_rectpuls_named_char_reproduces_pulse_train() {
    let t = Tensor::new(vec![-0.5, 0.0, 0.5, 1.0], vec![1, 4]).unwrap();
    let d = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
    let pulse = Value::CharArray(CharArray::new_row("rectpuls"));
    let out = expect_tensor(
        call(
            Value::Tensor(t),
            Value::Tensor(d),
            pulse,
            vec![Value::Num(0.25)],
        )
        .expect("pulstran"),
    );
    assert_eq!(out.shape(), [1, 4]);
    assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn pulstran_accepts_delay_amplitude_matrix() {
    let t = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
    let d = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
    let pulse =
        Value::StringArray(StringArray::new(vec!["rectpuls".to_string()], vec![1, 1]).unwrap());
    let out = expect_tensor(
        call(
            Value::Tensor(t),
            Value::Tensor(d),
            pulse,
            vec![Value::Num(0.25)],
        )
        .expect("pulstran"),
    );
    assert_eq!(out.materialize_f64(), vec![2.0, 3.0]);
}

#[test]
fn pulstran_reads_typed_integer_times_and_delay_amplitudes_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let t = integer_tensor(vec![0, 1], vec![1, 2]);
    let d = integer_tensor(vec![0, 1, 2, 3], vec![2, 2]);
    let pulse =
        Value::StringArray(StringArray::new(vec!["rectpuls".to_string()], vec![1, 1]).unwrap());
    let out = expect_tensor(
        call(
            Value::Tensor(t),
            Value::Tensor(d),
            pulse,
            vec![Value::Num(0.25)],
        )
        .expect("pulstran"),
    );
    assert_eq!(out.materialize_f64(), vec![2.0, 3.0]);
}

#[test]
fn pulstran_reads_native_single_times_and_delays_authoritatively() {
    let t = Tensor::from_f32(vec![0.0, 1.0], vec![1, 2]).expect("single times");
    let d =
        Tensor::from_f32(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).expect("single delay amplitudes");
    let pulse =
        Value::StringArray(StringArray::new(vec!["rectpuls".to_string()], vec![1, 1]).unwrap());
    let out = expect_tensor(
        call(
            Value::Tensor(t),
            Value::Tensor(d),
            pulse,
            vec![Value::Num(0.25)],
        )
        .expect("pulstran"),
    );
    assert_eq!(out.materialize_f64(), vec![2.0, 3.0]);
}

#[test]
fn pulstran_sampled_prototype_interpolates_at_sample_rate() {
    let t = Tensor::new(vec![0.0, 0.5, 1.0, 1.5, 2.0], vec![1, 5]).unwrap();
    let d = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
    let prototype = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
    let out = expect_tensor(
        call(
            Value::Tensor(t),
            Value::Tensor(d),
            Value::Tensor(prototype),
            vec![Value::Num(2.0)],
        )
        .expect("pulstran"),
    );
    assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn pulstran_reads_typed_integer_sampled_prototype_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let t = Tensor::new(vec![0.0, 0.5, 1.0, 1.5, 2.0], vec![1, 5]).unwrap();
    let d = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
    let prototype = integer_tensor(vec![0, 1, 0], vec![1, 3]);
    let out = expect_tensor(
        call(
            Value::Tensor(t),
            Value::Tensor(d),
            Value::Tensor(prototype),
            vec![Value::Num(2.0)],
        )
        .expect("pulstran"),
    );
    assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn pulstran_callback_samples_read_typed_integer_storage_exactly() {
    let samples = block_on(callback_samples(
        Value::Tensor(integer_tensor(vec![2, 4, 6], vec![1, 3])),
        3,
    ))
    .expect("callback samples");
    assert_eq!(samples, vec![2.0, 4.0, 6.0]);

    let repeated = block_on(callback_samples(
        Value::Tensor(integer_tensor(vec![7], vec![1, 1])),
        3,
    ))
    .expect("scalar callback sample");
    assert_eq!(repeated, vec![7.0, 7.0, 7.0]);
}

#[test]
fn pulstran_callback_samples_reject_inexact_wide_integer_without_rounding() {
    let wide = (1_u64 << 53) + 1;
    let samples = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
        .expect("wide callback sample");
    let error =
        block_on(callback_samples(Value::Tensor(samples), 3)).expect_err("inexact callback sample");
    assert_eq!(error.identifier(), PULSTRAN_ERROR_INVALID_PULSE.identifier);
    assert!(error.message().contains("exactly representable"));

    let exact = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54]), vec![1, 1])
        .expect("exact wide callback sample");
    assert_eq!(
        block_on(callback_samples(Value::Tensor(exact), 2)).expect("exact wide sample"),
        vec![(1_u64 << 54) as f64; 2]
    );
}

#[test]
fn pulstran_callback_samples_reject_inexact_resident_integer_with_source_live() {
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

    test_support::with_test_provider(|provider| {
        let wide = (1_u64 << 53) + 1;
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(std::slice::from_ref(&wide)),
                shape: &[1, 1],
            })
            .expect("resident callback result");
        let error = block_on(callback_samples(Value::GpuTensor(handle.clone()), 2))
            .expect_err("inexact resident callback result");
        assert_eq!(error.identifier(), PULSTRAN_ERROR_INVALID_PULSE.identifier);
        assert!(error.message().contains("exactly representable"));
        assert!(handle.allocation_lease().validate_live().is_ok());
        handle.release().expect("release callback result");
    });
}

#[test]
fn pulstran_scalar_control_keeps_operational_cause_and_retry() {
    let cause = crate::build_runtime_error("provider cancelled")
        .with_identifier("RunMat:Cancelled")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build();
    let owner = pulstran_error_with_source(
        &PULSTRAN_ERROR_INVALID_PARAMETER,
        "width: scalar gather failed",
        cause,
    );
    assert_eq!(
        owner.identifier(),
        PULSTRAN_ERROR_INVALID_PARAMETER.identifier
    );
    assert_eq!(owner.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    let source = std::error::Error::source(&owner).expect("typed cause");
    assert!(source.to_string().contains("provider cancelled"));
}

#[test]
fn resident_complex_operands_reject_before_download_with_owning_diagnostics() {
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView, HostNumericTensorView};

    test_support::with_test_provider(|provider| {
        let complex = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&[1.0, 0.5]),
                shape: &[1, 1],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .expect("complex resident operand");
        for (t, d, pulse, rest, expected) in [
            (
                Value::GpuTensor(complex.clone()),
                Value::Num(0.0),
                Value::from("rectpuls"),
                vec![],
                PULSTRAN_ERROR_INVALID_INPUT.identifier,
            ),
            (
                Value::Num(0.0),
                Value::GpuTensor(complex.clone()),
                Value::from("rectpuls"),
                vec![],
                PULSTRAN_ERROR_INVALID_DELAY.identifier,
            ),
            (
                Value::Num(0.0),
                Value::Num(0.0),
                Value::GpuTensor(complex.clone()),
                vec![],
                PULSTRAN_ERROR_INVALID_PULSE.identifier,
            ),
            (
                Value::Num(0.0),
                Value::Num(0.0),
                Value::from("rectpuls"),
                vec![Value::GpuTensor(complex.clone())],
                PULSTRAN_ERROR_INVALID_PARAMETER.identifier,
            ),
            (
                Value::Num(0.0),
                Value::Num(0.0),
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
                vec![Value::GpuTensor(complex.clone())],
                PULSTRAN_ERROR_INVALID_PARAMETER.identifier,
            ),
        ] {
            provider.reset_telemetry();
            let error = call(t, d, pulse, rest).expect_err("complex computation operand");
            assert_eq!(error.identifier(), expected);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        }
        provider.reset_telemetry();
        let error = block_on(callback_samples(Value::GpuTensor(complex.clone()), 1))
            .expect_err("complex callback waveform");
        assert_eq!(error.identifier(), PULSTRAN_ERROR_INVALID_PULSE.identifier);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(complex.allocation_lease().validate_live().is_ok());
        complex.release().expect("release complex operand");
    });
}

#[test]
fn invalid_computation_shapes_precede_earlier_resident_integer_exactness_readback() {
    use crate::builtins::common::{gpu_helpers, test_support};
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

    test_support::with_test_provider(|provider| {
        let wide = (1_u64 << 53) + 1;
        let times = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(std::slice::from_ref(&wide)),
                shape: &[1, 1],
            })
            .expect("resident wide times");
        let bad_matrix = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![1.0; 6], vec![2, 3]).expect("invalid matrix role"),
        )
        .expect("resident matrix");
        let bad_scalar = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("invalid scalar role"),
        )
        .expect("resident nonscalar");
        for (d, pulse, rest, expected) in [
            (
                Value::GpuTensor(bad_matrix.clone()),
                Value::from("rectpuls"),
                vec![],
                PULSTRAN_ERROR_INVALID_DELAY.identifier,
            ),
            (
                Value::Num(0.0),
                Value::GpuTensor(bad_matrix.clone()),
                vec![],
                PULSTRAN_ERROR_INVALID_PULSE.identifier,
            ),
            (
                Value::Num(0.0),
                Value::from("rectpuls"),
                vec![Value::GpuTensor(bad_scalar.clone())],
                PULSTRAN_ERROR_INVALID_PARAMETER.identifier,
            ),
            (
                Value::Num(0.0),
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
                vec![Value::GpuTensor(bad_scalar.clone())],
                PULSTRAN_ERROR_INVALID_PARAMETER.identifier,
            ),
        ] {
            provider.reset_telemetry();
            let error = call(Value::GpuTensor(times.clone()), d, pulse, rest)
                .expect_err("known-invalid role shape");
            assert_eq!(error.identifier(), expected);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        }
        for handle in [times, bad_matrix, bad_scalar] {
            assert!(handle.allocation_lease().validate_live().is_ok());
            handle.release().expect("release original operand");
        }
    });
}

#[test]
fn callback_complex_integer_samples_reject_before_floating_projection() {
    use runmat_value::IntegerComplexStorage;

    let wide = (1_u64 << 53) + 1;
    let storage = IntegerComplexStorage::new(
        IntegerStorage::U64(vec![wide]),
        IntegerStorage::U64(vec![0]),
    )
    .expect("complex integer storage");
    let value = Value::ComplexTensor(
        runmat_value::ComplexTensor::new_integer(storage, vec![1, 1])
            .expect("complex integer sample"),
    );
    let error = block_on(callback_samples(value, 1)).expect_err("unsupported callback class");
    assert_eq!(error.identifier(), PULSTRAN_ERROR_INVALID_PULSE.identifier);
    assert!(error.message().contains("complex storage"));
}

#[test]
fn restoration_rejects_distinct_view_of_other_borrowed_operand() {
    use crate::builtins::math::signal::common::waveform_restore_fixture::RestoreProbeProvider;

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = RestoreProbeProvider::new();
    let times = provider.input(&[0.0, 1.0]);
    let delays = provider.input(&[0.0, 1.0]);
    let alias = delays
        .with_view(runmat_accelerate_api::GpuTensorView::contiguous(vec![1, 2]))
        .expect("distinct borrowed delay view");
    assert!(alias.shares_allocation_with(&delays));
    provider.alias_on_restore(&alias);
    let borrowed_times = times.clone();
    let borrowed_delays = delays.clone();
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let error = call(
        Value::GpuTensor(times),
        Value::GpuTensor(delays),
        Value::from("rectpuls"),
        vec![],
    )
    .expect_err("other borrowed allocation cannot become output");
    assert!(error.message().contains("borrowed input"));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    assert!(borrowed_times.allocation_lease().validate_live().is_ok());
    assert!(borrowed_delays.allocation_lease().validate_live().is_ok());
}

#[test]
fn restoration_protects_borrowed_prototype_and_scalar_control_allocations() {
    use crate::builtins::math::signal::common::waveform_restore_fixture::RestoreProbeProvider;

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    for prototype_mode in [true, false] {
        let provider = RestoreProbeProvider::new();
        let (times, other_allocation) = if prototype_mode {
            (provider.input(&[0.0, 1.0]), provider.input(&[1.0, 0.0]))
        } else {
            (provider.input(&[0.0]), provider.input(&[1.0]))
        };
        let borrowed_other = other_allocation.clone();
        let output_alias = other_allocation
            .with_view(runmat_accelerate_api::GpuTensorView::contiguous(vec![
                if prototype_mode { 2 } else { 1 },
            ]))
            .expect("distinct output view");
        assert!(output_alias.shares_allocation_with(&borrowed_other));
        provider.alias_on_restore(&output_alias);
        let source_lifetime = times.clone();
        let other_lifetime = borrowed_other.clone();
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let (pulse, rest) = if prototype_mode {
            (Value::GpuTensor(borrowed_other), vec![Value::Num(1.0)])
        } else {
            (
                Value::from("rectpuls"),
                vec![Value::GpuTensor(borrowed_other)],
            )
        };
        let error = call(Value::GpuTensor(times), Value::Num(0.0), pulse, rest)
            .expect_err("borrowed prototype/control alias");
        assert!(error.message().contains("borrowed input"));
        assert!(source_lifetime.allocation_lease().validate_live().is_ok());
        assert!(other_lifetime.allocation_lease().validate_live().is_ok());
    }
}

#[test]
fn pulstran_rejects_invalid_delay_shape_and_bad_sample_rate() {
    let t = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
    let bad_d = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2, 1]).unwrap());
    let err = call(
        t.clone(),
        bad_d,
        Value::CharArray(CharArray::new_row("rectpuls")),
        Vec::new(),
    )
    .expect_err("delay shape");
    assert_eq!(err.identifier(), PULSTRAN_ERROR_INVALID_DELAY.identifier);

    let d = Value::Num(0.0);
    let prototype = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
    let err = call(t, d, prototype, vec![Value::Num(0.0)]).expect_err("sample rate");
    assert_eq!(
        err.identifier(),
        PULSTRAN_ERROR_INVALID_PARAMETER.identifier
    );
}

#[test]
fn pulstran_automatic_residency_is_transparent_and_explicit_residency_is_gated() {
    use crate::builtins::common::test_support;

    test_support::with_test_provider(|provider| {
        let times = Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("times");
        let automatic = gpu_helpers::upload_tensor(provider, &times).expect("upload");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let output = call(
            Value::GpuTensor(automatic),
            Value::Num(0.0),
            Value::from("rectpuls"),
            vec![Value::Num(0.25)],
        )
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

        let explicit = gpu_helpers::upload_tensor(provider, &times).expect("upload");
        let explicit = explicit
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let error = call(
            Value::GpuTensor(explicit),
            Value::Num(0.0),
            Value::from("rectpuls"),
            vec![Value::Num(0.25)],
        )
        .expect_err("strict explicit input");
        assert_eq!(
            error.identifier(),
            PULSTRAN_EXPLICIT_GPU_EXTENSION.error_identifier
        );
    });
}

#[test]
fn at_prefixed_builtin_uses_the_same_integer_parameter_gate() {
    use runmat_value::IntValue;
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = call(
        Value::Num(0.0),
        Value::Num(0.0),
        Value::FunctionHandle("@rectpuls".into()),
        vec![Value::Int(IntValue::I16(1))],
    )
    .expect_err("typed integer width must be gated");
    assert_eq!(
        error.identifier(),
        PULSTRAN_INTEGER_PARAMETER_EXTENSION.error_identifier
    );
}

#[test]
fn at_prefixed_builtin_resident_width_controls_output_ownership() {
    use crate::builtins::common::test_support;

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        for explicit in [false, true] {
            let width = Tensor::new(vec![1.0], vec![1, 1]).expect("width");
            let resident = gpu_helpers::upload_tensor(provider, &width).expect("resident width");
            let resident = if explicit {
                resident
                    .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                    .expect("explicit width")
            } else {
                resident
            };
            let output = call(
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("times")),
                Value::Num(0.0),
                Value::FunctionHandle("@rectpuls".into()),
                vec![Value::GpuTensor(resident.clone())],
            )
            .expect("resident width waveform");
            let Value::GpuTensor(result) = &output else {
                panic!("resident numeric control must retain owner");
            };
            assert_eq!(runmat_accelerate_api::handle_is_explicit(result), explicit);
            assert!(resident.allocation_lease().validate_live().is_ok());
            assert_eq!(
                test_support::gather(output)
                    .expect("gather output")
                    .materialize_f64(),
                vec![1.0, 0.0]
            );
        }
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn pulstran_wgpu_fallback_enforces_double_residency_for_all_integer_classes() {
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
        let times = Tensor::new_integer(storage, vec![1, 2]).expect("integer times");
        let handle = gpu_helpers::upload_tensor(provider, &times).expect("upload");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let result = call(
            Value::GpuTensor(handle),
            Value::Num(0.0),
            Value::from("rectpuls"),
            vec![Value::Num(0.25)],
        );
        let output = result.expect("resident integer pulstran");
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
fn pulstran_is_registered() {
    let entry = runmat_builtins::builtin_catalog_entry_by_name("pulstran").expect("catalog");
    assert!(builtin_function_by_name("pulstran").is_none());
    assert_eq!(entry.bindings.len(), 1);
    assert_eq!(entry.bindings[0].variant, "default");
    let bindings = crate::builtin::runtime_builtin_bindings_by_name("pulstran");
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        entry.binding_identity(&entry.bindings[0])
    );
}
