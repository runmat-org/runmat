use super::*;
use crate::builtins::common::random;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

struct MalformedRandomProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    alias: Option<GpuTensorHandle>,
}

impl runmat_accelerate_api::AccelProvider for MalformedRandomProvider {
    fn upload(
        &self,
        host: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn random_exponential(&self, _mu: f64, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        if let Some(alias) = &self.alias {
            return Ok(alias.clone());
        }
        self.inner.upload(&runmat_accelerate_api::HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
    }
}

fn reset() -> impl Drop {
    let guard = random::test_guard();
    runmat_accelerate_api::clear_provider();
    random::reset_rng();
    guard
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
    Tensor::new_integer(storage, shape).expect("integer tensor")
}

#[test]
fn exprnd_scalar_deterministic() {
    let _guard = random::test_guard();
    let _guard = reset();
    let result = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    ))
    .expect("exprnd");
    let expected = random::expected_exponential_sequence(2.0, 1)[0];
    match result {
        Value::Num(v) => {
            assert!(v > 0.0);
            assert!((v - expected).abs() < 1e-12);
        }
        other => panic!("expected scalar, got {other:?}"),
    }
}

#[test]
fn exprnd_matrix_dims() {
    let _guard = random::test_guard();
    let _guard = reset();
    let args = vec![Value::Num(1.0), Value::Num(3.0), Value::Num(4.0)];
    let result = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("exprnd");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[3, 4]);
            assert!(t.materialize_f64().iter().all(|&v| v > 0.0));
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn exprnd_size_vec() {
    let _guard = random::test_guard();
    let _guard = reset();
    let size = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
    let args = vec![Value::Num(1.0), Value::Tensor(size)];
    let result = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("exprnd");
    match result {
        Value::Tensor(t) => assert_eq!(t.shape(), &[3, 4]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn exprnd_reads_typed_integer_mu_and_size_exactly() {
    let _guard = random::test_guard();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    reset();
    let mu = poisoned_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]);
    let size = poisoned_int_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
    let result = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(mu),
            Value::Tensor(size),
        ]),
    ))
    .expect("exprnd");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 3]);
            assert!(t.materialize_f64().iter().all(|&v| v > 0.0));
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn exprnd_rejects_negative_mu() {
    let args = vec![Value::Num(-1.0)];
    let err = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect_err("negative mu should error");
    assert_eq!(
        err.identifier(),
        EXPRND_ERROR_MU_MUST_BE_POSITIVE.identifier
    );
}

#[test]
fn exprnd_rejects_zero_mu() {
    let args = vec![Value::Num(0.0)];
    let err = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect_err("zero mu should error");
    assert_eq!(
        err.identifier(),
        EXPRND_ERROR_MU_MUST_BE_POSITIVE.identifier
    );
}

#[test]
fn exprnd_distribution_mean() {
    let _guard = random::test_guard();
    let _guard = reset();
    let mu = 3.0_f64;
    let n = 50_000_usize;
    let args = vec![Value::Num(mu), Value::Num(n as f64), Value::Num(1.0)];
    let result = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("exprnd");
    let data = match result {
        Value::Tensor(t) => t.materialize_f64(),
        other => panic!("expected tensor, got {other:?}"),
    };
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    assert!(
        (mean - mu).abs() / mu < 0.05,
        "sample mean {mean:.4} not within 5% of mu={mu}"
    );
}

#[test]
fn exprnd_array_mu_preserves_shape_and_native_single() {
    let _guard = random::test_guard();
    reset();
    let mu = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(mu)]),
    ))
    .expect("array mu");
    let Value::Tensor(tensor) = result else {
        panic!("expected tensor output");
    };
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    let expected: Vec<f64> = random::expected_uniform_sequence(4)
        .into_iter()
        .zip([1.0, 2.0, 3.0, 4.0])
        .map(|(uniform, mean)| ((-mean * uniform.max(f64::MIN_POSITIVE).ln()) as f32) as f64)
        .collect();
    assert_eq!(tensor.materialize_f64(), expected);
    assert!(tensor.materialize_f64().iter().all(|value| *value > 0.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn exprnd_normalizes_rank_zero_and_rank_one_mu_shapes() {
    let _guard = random::test_guard();
    reset();

    let scalar = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
            Tensor::new(vec![1.0], Vec::new()).unwrap(),
        )]),
    ))
    .expect("rank-zero mu");
    assert!(matches!(scalar, Value::Num(value) if value >= 0.0));

    let vector = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0], vec![2]).unwrap(),
        )]),
    ))
    .expect("rank-one mu");
    let Value::Tensor(vector) = vector else {
        panic!("rank-one output");
    };
    assert_eq!(vector.shape(), &[2, 1]);
}

#[test]
fn exprnd_array_mu_requires_matching_explicit_size() {
    let mu = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let error = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(mu),
            Value::Num(3.0),
            Value::Num(1.0),
        ]),
    ))
    .expect_err("shape mismatch");
    assert_eq!(error.identifier(), EXPRND_ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn exprnd_normalizes_empty_and_trailing_singleton_sizes() {
    let _guard = random::test_guard();
    reset();
    let before = random::snapshot().unwrap();
    let empty = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(-2.0),
            Value::Num(3.0),
        ]),
    ))
    .expect("negative size produces empty");
    let Value::Tensor(empty) = empty else {
        panic!("expected empty tensor");
    };
    assert_eq!(empty.shape(), &[0, 3]);
    assert_eq!(before.state, random::snapshot().unwrap().state);

    let output = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(3.0),
            Value::Num(1.0),
            Value::Num(1.0),
        ]),
    ))
    .expect("trailing singleton dimensions");
    let Value::Tensor(output) = output else {
        panic!("expected tensor");
    };
    assert_eq!(output.shape(), &[3, 1]);
}

#[test]
fn exprnd_rejects_logical_complex_and_excess_outputs_before_rng() {
    let _guard = random::test_guard();
    reset();
    for value in [Value::Bool(true), Value::Complex(1.0, 0.0)] {
        let before = random::snapshot().unwrap();
        let error = block_on(exprnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![value]),
        ))
        .expect_err("invalid class");
        assert_eq!(error.identifier(), EXPRND_ERROR_INVALID_ARGUMENT.identifier);
        assert_eq!(before.state, random::snapshot().unwrap().state);
    }
    let before = random::snapshot().unwrap();
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .expect_err("excess outputs");
    assert_eq!(error.identifier(), EXPRND_ERROR_INVALID_ARGUMENT.identifier);
    assert_eq!(before.state, random::snapshot().unwrap().state);
}

#[test]
fn exprnd_zero_output_evaluation_still_advances_rng_once() {
    let _guard = random::test_guard();
    reset();
    let before = random::snapshot().unwrap().state;
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let _ = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .expect("zero-output call");
    assert_ne!(before, random::snapshot().unwrap().state);
}

#[test]
fn exprnd_uses_host_stream_even_when_provider_has_random_hook() {
    let _guard = random::test_guard();
    reset();
    let _accel_guard = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static dyn runmat_accelerate_api::AccelProvider =
        Box::leak(Box::new(MalformedRandomProvider {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            alias: None,
        }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let before = random::snapshot().expect("rng snapshot").state;
    let value = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        ]),
    ))
    .expect("host stream result");
    assert!(matches!(value, Value::Tensor(_)));
    assert_ne!(before, random::snapshot().expect("rng snapshot").state);
}

#[test]
fn resident_size_controls_are_protected_without_selecting_output_owner() {
    let _guard = reset();
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let dimensions = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &dimensions).unwrap();
        let parsed = block_on(parse_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::GpuTensor(handle.clone()),
            ]),
        ))
        .unwrap();
        assert!(parsed.gpu_source.is_none());
        assert_eq!(parsed.shape, vec![2, 3]);
        assert_eq!(parsed.borrowed_gpu.len(), 1);
        assert!(gpu_helpers::same_gpu_handle(
            &parsed.borrowed_gpu[0],
            &handle
        ));
    });
}

#[test]
fn borrowed_size_handle_does_not_select_provider_random_stream() {
    use runmat_accelerate_api::AccelProvider;
    let _guard = reset();
    let inner = runmat_accelerate::simple_provider::InProcessProvider::new();
    let handle = inner
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let provider = Box::leak(Box::new(MalformedRandomProvider {
        inner,
        alias: Some(handle.clone()),
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let before = random::snapshot().unwrap().state;
    let output = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::GpuTensor(handle.clone()),
        ]),
    ))
    .expect("host session random draw");
    assert!(matches!(output, Value::Tensor(_)));
    assert_ne!(before, random::snapshot().unwrap().state);
    let original = block_on(provider.download(&handle)).unwrap();
    assert_eq!(original.data, vec![1.0, 2.0]);
}

#[test]
fn exprnd_integer_extensions_follow_compatibility_mode() {
    let _guard = random::test_guard();
    reset();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
            runmat_value::IntValue::U16(2),
        )]),
    ))
    .expect_err("integer mu is gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:ExprndIntegerMeanExtension")
    );
    let error = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Int(runmat_value::IntValue::U16(2)),
        ]),
    ))
    .expect_err("integer size is gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:ExprndIntegerSizeExtension")
    );
}

#[test]
fn exprnd_runmat_extension_accepts_all_integer_mean_classes() {
    let _guard = random::test_guard();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = vec![
        IntegerStorage::I8(vec![2]),
        IntegerStorage::I16(vec![2]),
        IntegerStorage::I32(vec![2]),
        IntegerStorage::I64(vec![2]),
        IntegerStorage::U8(vec![2]),
        IntegerStorage::U16(vec![2]),
        IntegerStorage::U32(vec![2]),
        IntegerStorage::U64(vec![2]),
    ];
    for storage in storages {
        reset();
        let mu = poisoned_int_tensor(storage, vec![1, 1]);
        let sample = block_on(exprnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(mu)]),
        ))
        .expect("integer mean extension");
        assert!(matches!(sample, Value::Num(value) if value > 0.0));
    }
}

#[test]
fn exprnd_resident_single_mean_stays_host_when_owner_is_physically_f64() {
    let _guard = random::test_guard();
    reset();
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let mu = Tensor::from_f32(vec![2.0, 3.0], vec![2, 1]).expect("single mu");
        let input = gpu_helpers::upload_tensor(provider, &mu).expect("upload single mu");
        let output = block_on(exprnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(input)]),
        ))
        .expect("resident exprnd");
        let Value::Tensor(output) = output else {
            panic!("expected host single output");
        };
        assert_eq!(output.shape(), &[2, 1]);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert!(output.materialize_f64().iter().all(|value| *value > 0.0));
    });
}

#[test]
fn allocation_overflow_fails_before_sampling() {
    let _guard = reset();
    let before = random::snapshot().unwrap().state;
    let error = block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(4294967296.0),
            Value::Num(4294967296.0),
        ]),
    ))
    .expect_err("overflow");
    assert_eq!(error.identifier(), EXPRND_ERROR_INVALID_ARGUMENT.identifier);
    assert_eq!(before, random::snapshot().unwrap().state);
    assert_eq!(checked_element_count(&[usize::MAX, 0, 2]).unwrap(), 0);
}

#[test]
fn nonfinite_means_fail_before_sampling() {
    let _guard = reset();
    let before = random::snapshot().unwrap().state;
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            block_on(exprnd_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(value)])
            ))
            .unwrap_err()
            .identifier(),
            EXPRND_ERROR_MU_MUST_BE_POSITIVE.identifier
        );
    }
    assert_eq!(before, random::snapshot().unwrap().state);
}

#[test]
fn multidimensional_size_tensor_is_not_a_vector() {
    let _guard = reset();
    let size = Tensor::new(vec![2.0; 4], vec![1, 2, 2]).unwrap();
    let before = random::snapshot().unwrap().state;
    assert!(block_on(exprnd_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Tensor(size)
        ])
    ))
    .is_err());
    assert_eq!(before, random::snapshot().unwrap().state);
}

#[test]
fn explicit_single_owner_mismatch_fails_before_sampling() {
    let _guard = reset();
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let mu = Tensor::from_f32(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &mu)
            .unwrap()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated explicit view");
        let before = random::snapshot().unwrap().state;
        let error = block_on(exprnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle.clone(),
            )]),
        ))
        .unwrap_err();
        assert!(error.message().contains("explicit gpuArray"));
        assert_eq!(before, random::snapshot().unwrap().state);
        assert!(block_on(provider.download(&handle)).is_ok());
    });
}
