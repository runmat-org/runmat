use super::*;
use futures::executor::block_on;
fn run(input: Tensor, options: options::Options, count: usize) -> Vec<Value> {
    let mut cooperation = Cooperation::new(None);
    block_on(async {
        let prepared = prepare::prepare(&input, &options, &mut cooperation).await?;
        engine::evaluate(prepared, &options, count, None, &mut cooperation).await
    })
    .unwrap()
}
fn array(value: &Value) -> &Tensor {
    let Value::Tensor(value) = value else {
        panic!("numeric array")
    };
    value
}
fn values(value: &Value) -> Vec<f64> {
    array(value).materialize_f64()
}
fn near(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
}
#[test]
fn orthogonal_observations_reconstruct_with_analytic_variance_and_full_space_statistics() {
    let input = Tensor::new(vec![-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0], vec![4, 2]).unwrap();
    let outputs = run(input.clone(), options::Options::default(), 6);
    let c = values(&outputs[0]);
    let score = values(&outputs[1]);
    for column in 0..2 {
        for row in 0..4 {
            near(
                score[row] * c[column] + score[4 + row] * c[2 + column],
                crate::builtins::common::tensor::tensor_value_f64(&input, column * 4 + row),
            );
        }
        for other in 0..2 {
            near(
                c[column * 2] * c[other * 2] + c[column * 2 + 1] * c[other * 2 + 1],
                if column == other { 1.0 } else { 0.0 },
            );
        }
        let pivot = if c[column * 2].abs() >= c[column * 2 + 1].abs() {
            c[column * 2]
        } else {
            c[column * 2 + 1]
        };
        assert!(pivot >= 0.0);
    }
    for value in values(&outputs[2]) {
        near(value, 4.0 / 3.0);
    }
    for value in values(&outputs[3]) {
        near(value, 1.5);
    }
    for value in values(&outputs[4]) {
        near(value, 50.0);
    }
    assert_eq!(values(&outputs[5]), [0.0, 0.0]);
    let reduced = run(
        input,
        options::Options {
            components: Some(1),
            ..Default::default()
        },
        6,
    );
    assert_eq!(array(&reduced[0]).shape(), [2, 1]);
    assert_eq!(array(&reduced[1]).shape(), [4, 1]);
    assert_eq!(array(&reduced[2]).shape(), [2, 1]);
    assert_eq!(array(&reduced[4]).shape(), [2, 1]);
    for value in values(&reduced[3]) {
        near(value, 1.5);
    }
}
#[test]
fn economy_full_basis_missing_rows_float_class_and_zero_spectrum_have_real_semantics() {
    let input = Tensor::new(vec![-1.0, 1.0, 0.0, 0.0, 1.0, -1.0], vec![2, 3]).unwrap();
    let compact = run(input.clone(), options::Options::default(), 6);
    assert_eq!(array(&compact[0]).shape(), [3, 1]);
    let full = run(
        input,
        options::Options {
            economy: false,
            ..Default::default()
        },
        6,
    );
    assert_eq!(array(&full[0]).shape(), [3, 3]);
    assert_eq!(array(&full[2]).shape(), [3, 1]);
    let latent = values(&full[2]);
    near(latent[0], 4.0);
    assert_eq!(&latent[1..], [0.0, 0.0]);
    assert_eq!(&values(&full[1])[2..], [0.0, 0.0, 0.0, 0.0]);
    let input = Tensor::from_f32(
        vec![-1.0, f32::NAN, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        vec![4, 2],
    )
    .unwrap();
    let outputs = run(input.clone(), options::Options::default(), 6);
    for output in &outputs {
        assert_eq!(array(output).numeric_dtype(), NumericDType::F32);
    }
    assert!(values(&outputs[1])[1].is_nan() && values(&outputs[3])[1].is_nan());
    let mut cooperation = Cooperation::new(None);
    assert!(block_on(prepare::prepare(
        &input,
        &options::Options {
            rows: runmat_builtins::PcaRows::All,
            ..Default::default()
        },
        &mut cooperation
    ))
    .is_err());
    let zero = run(
        Tensor::new(vec![0.0; 6], vec![3, 2]).unwrap(),
        options::Options::default(),
        6,
    );
    assert_eq!(values(&zero[2]), [0.0, 0.0]);
    assert_eq!(values(&zero[3]), [0.0, 0.0, 0.0]);
    assert!(values(&zero[4]).iter().all(|value| value.is_nan()));
    let input = Tensor::new(vec![0.0, 2.0, 100.0, 0.0, 2.0, f64::NAN], vec![3, 2]).unwrap();
    let outputs = run(input, options::Options::default(), 6);
    assert_eq!(values(&outputs[5]), [34.0, 1.0]);
    assert!(values(&outputs[1])[2].is_nan() && values(&outputs[3])[2].is_nan());
    let input = Tensor::new(vec![0.0, 2.0, 1e300, 0.0, 2.0, f64::NAN], vec![3, 2]).unwrap();
    let prepared = block_on(prepare::prepare(
        &input,
        &options::Options::default(),
        &mut Cooperation::new(None),
    ))
    .unwrap();
    assert!(prepared.scale >= prepared.means[0].abs());
    assert!(prepared
        .normalized
        .materialize_f64()
        .iter()
        .all(|value| value.is_finite()));
    let input = Tensor::new(
        vec![0.0, 2.0, f64::INFINITY, 0.0, 2.0, f64::NAN],
        vec![3, 2],
    )
    .unwrap();
    assert!(block_on(prepare::prepare(
        &input,
        &options::Options::default(),
        &mut Cooperation::new(None)
    ))
    .is_err());
}
#[test]
fn uncentered_scale_output_selection_and_unsupported_controls_are_not_ignored() {
    let input = Tensor::new(vec![1e200, 2e200, 3e200], vec![3, 1]).unwrap();
    let outputs = run(
        input,
        options::Options {
            centered: false,
            ..Default::default()
        },
        6,
    );
    assert_eq!(values(&outputs[5]), [0.0]);
    assert!(values(&outputs[2])[0].is_infinite());
    near(values(&outputs[4])[0], 100.0);
    for (actual, expected) in values(&outputs[1]).iter().zip([1.0, 2.0, 3.0]) {
        near(actual / 1e200, expected);
    }
    let outputs = run(
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap(),
        options::Options::default(),
        1,
    );
    assert_eq!(outputs.len(), 1);
    for rest in [
        vec![Value::from("Algorithm"), Value::from("eig")],
        vec![Value::from("Weights"), Value::Num(1.0)],
        vec![Value::from("Rows"), Value::from("pairwise")],
        vec![Value::from("NumComponents"), Value::Num(0.0)],
    ] {
        let mut args = vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap(),
        )];
        args.extend(rest);
        assert!(block_on(evaluate(
            None,
            crate::builtins::common::test_support::builtin_arguments(args)
        ))
        .is_err());
    }
    let resident = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![3, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap()
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    let error = block_on(evaluate(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(resident)]),
    ))
    .unwrap_err();
    assert!(error
        .message()
        .contains("host fallback requires automatic residency"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}

#[test]
fn public_entry_refuses_uncertified_finite_workspace_and_preserves_contextual_cancellation() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
        RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Finite(ManagedHostEffectLedger);
    impl RuntimeParallelService for Finite {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1024 * 1024);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
        );
    let args = crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap(),
    )]);
    let error = block_on(evaluate(Some(&runtime), args)).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(ledger.snapshot().peak_reserved_bytes, 0);
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(source.token());
    runtime.set_error_namespace("PcaInvocation");
    source.cancel(runmat_execution::CancellationReason::User);
    let args = crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]);
    let error = block_on(evaluate(Some(&runtime), args)).unwrap_err();
    assert_eq!(error.identifier(), Some("PcaInvocation:Cancelled"));
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
}
