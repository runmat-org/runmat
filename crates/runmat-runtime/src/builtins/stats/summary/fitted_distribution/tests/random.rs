use super::*;

#[test]
fn canonical_random_binding_uses_invocation_cancellation() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::rc::Rc;

    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    runtime.set_error_namespace("Invocation");
    let cancelled = CancellationSource::new();
    cancelled.cancel(CancellationReason::User);
    let runtime = runtime.with_cancellation_token(cancelled.token());
    let error = block_on(random_builtin_registered(
        Some(runtime),
        Value::from("Poisson"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(3.0),
            Value::Num(4.0),
        ]),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("Invocation:Cancelled"));
}

#[test]
fn random_named_logical_parameter_follows_numeric_evaluator() {
    let _rng_guard = crate::builtins::common::random::test_guard();
    let samples = block_on(random_builtin(
        Value::from("Poisson"),
        vec![Value::Bool(false), Value::Num(2.0)],
    ))
    .expect("logical Poisson parameter is numeric zero");
    let Value::Tensor(samples) = samples else {
        panic!("expected two-by-two samples");
    };
    assert_eq!(samples.shape(), &[2, 2]);
    assert!(samples.materialize_f64().iter().all(|value| *value == 0.0));
}

#[test]
fn fitted_poisson_sampling_polls_between_draws_and_within_inverse_search() {
    let _rng_guard = crate::builtins::common::random::test_guard();
    use std::cell::Cell;
    let fit = FittedDistribution {
        kind: DistributionKind::Poisson,
        parameters: vec![1000.0],
        nlogl: f64::NAN,
        observations: f64::NAN,
    };
    let calls = Cell::new(0);
    let mut turn =
        crate::builtins::common::random::try_acquire_turn(None, RANDOM_NAME).expect("random turn");
    let error = random_samples(&mut turn, &fit, 4, || {
        calls.set(calls.get() + 1);
        if calls.get() > 2 {
            Err(invalid_for(RANDOM_NAME, "cancelled by fixture"))
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    assert!(calls.get() > 2);
    assert_eq!(error.message(), "cancelled by fixture");
}

#[test]
fn random_dimension_product_overflow_is_an_error_not_a_panic() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let fitted = block_on(fitdist_builtin(
        vec_tensor(&[1.0, 2.0, 3.0]),
        Value::String("Normal".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fitted object");
    let error = block_on(random_builtin(
        fitted,
        vec![
            Value::Int(runmat_value::IntValue::U64(1_u64 << 32)),
            Value::Int(runmat_value::IntValue::U64(1_u64 << 32)),
        ],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:random:InvalidArgument"));
}

#[test]
fn random_restores_parameter_residency_but_not_size_control_residency() {
    use crate::builtins::common::test_support;

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let parameter = Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1])
            .expect("integer parameter");
        let parameter = gpu_helpers::upload_tensor(provider, &parameter).expect("upload");
        let parameter = parameter
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let output = block_on(random_builtin(
            Value::String("Normal".into()),
            vec![
                Value::GpuTensor(parameter),
                Value::Num(1.0),
                Value::Num(2.0),
            ],
        ))
        .expect("resident random output");
        let Value::GpuTensor(output_handle) = &output else {
            panic!("expected resident random output");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
        let gathered = test_support::gather(output).expect("gather random output");
        assert_eq!(gathered.shape(), &[2, 2]);
        assert!(gathered
            .materialize_f64()
            .iter()
            .all(|value| value.is_finite()));

        let size =
            Tensor::new_integer(IntegerStorage::U8(vec![2]), vec![1, 1]).expect("integer size");
        let size = gpu_helpers::upload_tensor(provider, &size).expect("upload size");
        let size = size
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let output = block_on(random_builtin(
            Value::String("Normal".into()),
            vec![Value::Num(0.0), Value::Num(1.0), Value::GpuTensor(size)],
        ))
        .expect("resident size control");
        assert!(matches!(output, Value::Tensor(_)));
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn random_wgpu_parameter_residency_enforces_double_for_all_integer_classes() {
    use crate::builtins::common::test_support;

    let _accel_guard = test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_cdf_integer_storages(0) {
        let parameter = Tensor::new_integer(storage, vec![1, 1]).expect("integer parameter");
        let handle = gpu_helpers::upload_tensor(provider, &parameter).expect("upload");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let result = block_on(random_builtin(
            Value::String("Normal".into()),
            vec![Value::GpuTensor(handle), Value::Num(1.0), Value::Num(2.0)],
        ));
        let output = result.expect("resident random output");
        let Value::GpuTensor(output_handle) = &output else {
            panic!("expected resident random output");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
        assert_eq!(
            runmat_accelerate_api::handle_precision(output_handle),
            Some(ProviderPrecision::F64)
        );
        let gathered = test_support::gather(output).expect("gather random output");
        assert_eq!(gathered.shape(), &[2, 2]);
        assert!(gathered
            .materialize_f64()
            .iter()
            .all(|value| value.is_finite()));
    }
}
