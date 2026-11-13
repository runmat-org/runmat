use super::*;
use futures::executor::block_on;
use runmat_builtins::KsdensityFunction;
use runmat_value::NumericDType;
fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}
fn run(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
    let output = block_on(evaluate(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    ))?;
    let Value::OutputList(output) = output else {
        panic!("outputs")
    };
    Ok(output.iter().cloned().collect())
}
fn array(value: &Value) -> &Tensor {
    let Value::Tensor(value) = value else {
        panic!("array")
    };
    value
}
fn width(outputs: &[Value]) -> f64 {
    let Value::Num(value) = outputs[2] else {
        panic!("double bandwidth")
    };
    value
}
fn near(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}
#[test]
fn normal_mixture_pdf_cdf_and_survivor_have_independent_analytic_oracles() {
    let output = run(vec![
        tensor(vec![-1.0, 1.0], vec![2, 1]),
        tensor(vec![0.0, 1.0], vec![2, 1]),
        Value::from("Bandwidth"),
        Value::Num(1.0),
    ])
    .unwrap();
    let f = array(&output[0]).materialize_f64();
    near(f[0], (-0.5f64).exp() / (2.0 * std::f64::consts::PI).sqrt());
    near(
        f[1],
        (1.0 + (-2.0f64).exp()) / (2.0 * (2.0 * std::f64::consts::PI).sqrt()),
    );
    assert_eq!(array(&output[0]).shape(), [2, 1]);
    assert_eq!(array(&output[1]).materialize_f64(), [0.0, 1.0]);
    assert_eq!(width(&output), 1.0);
    for (function, expected) in [("cdf", [0.0, 0.5, 1.0]), ("survivor", [1.0, 0.5, 0.0])] {
        let output = run(vec![
            tensor(vec![-1.0, 1.0], vec![2, 1]),
            tensor(vec![f64::NEG_INFINITY, 0.0, f64::INFINITY], vec![1, 3]),
            Value::from("Bandwidth"),
            Value::Num(1.0),
            Value::from("Function"),
            Value::from(function),
        ])
        .unwrap();
        for (actual, expected) in array(&output[0])
            .materialize_f64()
            .into_iter()
            .zip(expected)
        {
            near(actual, expected);
        }
    }
}
#[test]
fn explicit_bandwidth_supports_degenerates_and_preserves_point_class_without_density_clamping() {
    for data in [vec![4.0], vec![4.0, 4.0, 4.0]] {
        let n = data.len();
        let points = Value::Tensor(Tensor::from_f32(vec![4.0, 5.0], vec![2, 1]).unwrap());
        let output = run(vec![
            tensor(data.clone(), vec![n, 1]),
            points,
            Value::from("Bandwidth"),
            Value::Num(0.25),
        ])
        .unwrap();
        assert_eq!(array(&output[0]).numeric_dtype(), NumericDType::F64);
        assert_eq!(array(&output[1]).numeric_dtype(), NumericDType::F32);
        assert_eq!(width(&output), 0.25);
        near(
            array(&output[0]).materialize_f64()[0],
            4.0 / (2.0 * std::f64::consts::PI).sqrt(),
        );
        assert!(array(&output[0]).materialize_f64()[0] > 1.0);
        let error = run(vec![tensor(data, vec![n, 1])]).unwrap_err();
        assert!(error
            .message()
            .contains("specify a finite positive Bandwidth"));
    }
    let data = vec![0.0; 100];
    let points = Tensor::new(vec![-38.4], vec![1, 1]).unwrap();
    let actual = block_on(engine::evaluate(
        &data,
        &points,
        1.0,
        KsdensityFunction::Cdf,
        None,
        &mut Cooperation::new(None),
    ))
    .unwrap()[0];
    let expected = super::super::distribution_math::standard_normal_cdf(-38.4);
    assert!(expected > 0.0);
    assert_eq!(actual, expected);
    let points = Tensor::new(vec![4e-299], vec![1, 1]).unwrap();
    let actual = block_on(engine::evaluate(
        &[0.0, 0.0],
        &points,
        1e-300,
        KsdensityFunction::Pdf,
        None,
        &mut Cooperation::new(None),
    ))
    .unwrap()[0];
    let expected = (-800.0 - 0.5 * (2.0 * std::f64::consts::PI).ln() - 1e-300f64.ln()).exp();
    assert!(actual > 0.0 && (actual / expected - 1.0).abs() < 1e-10);
    let points = Tensor::new(vec![f64::NEG_INFINITY, f64::INFINITY], vec![1, 2]).unwrap();
    let actual = block_on(engine::evaluate(
        &[1e300],
        &points,
        1e-300,
        KsdensityFunction::Cdf,
        None,
        &mut Cooperation::new(None),
    ))
    .unwrap();
    assert_eq!(actual, [0.0, 1.0]);
}
#[test]
fn normal_reference_mad_grid_and_extreme_scales_have_no_arbitrary_unit_fallback() {
    let output = run(vec![tensor(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5, 1])]).unwrap();
    let expected =
        (4.0f64 / 15.0).powf(0.2) / super::super::distribution_math::standard_normal_inv(0.75);
    near(width(&output), expected);
    assert_eq!(array(&output[0]).shape(), [1, 100]);
    let points = array(&output[1]).materialize_f64();
    near(points[0], -2.0 - 3.0 * expected);
    near(points[99], 2.0 + 3.0 * expected);
    for scale in [1e-200, 1e200] {
        let mut data = vec![-scale, 0.0, scale];
        let actual = block_on(bandwidth::normal_reference(
            &mut data,
            None,
            &mut Cooperation::new(None),
        ))
        .unwrap();
        assert!(actual > 0.0 && actual.is_finite());
        near(
            actual / scale,
            (4.0f64 / 9.0).powf(0.2) / super::super::distribution_math::standard_normal_inv(0.75),
        );
    }
    let mut data = vec![-1e-200, 0.0, 1e-200, 2e-200, 1e300];
    let actual = block_on(bandwidth::normal_reference(
        &mut data,
        None,
        &mut Cooperation::new(None),
    ))
    .unwrap();
    assert!(actual > 0.0 && actual < 1e-199);
    let mut data = vec![0.0, 0.0, 0.0, 0.0, 1.0];
    let actual = block_on(bandwidth::normal_reference(
        &mut data,
        None,
        &mut Cooperation::new(None),
    ))
    .unwrap();
    near(actual, (0.2f64).sqrt() * (4.0f64 / 15.0).powf(0.2));
}
#[test]
fn unsupported_forms_exact_count_geometry_and_explicit_intent_reject_without_ignored_controls() {
    for controls in [
        vec![Value::from("Kernel"), Value::from("triangle")],
        vec![Value::from("Support"), Value::from("positive")],
        vec![Value::from("Weights"), Value::Num(1.0)],
        vec![Value::from("Bandwidth"), Value::from("plug-in")],
        vec![Value::from("Function"), Value::from("icdf")],
        vec![Value::from("NumPoints"), Value::Num(0.0)],
    ] {
        let mut args = vec![tensor(vec![-1.0, 1.0], vec![2, 1])];
        args.extend(controls);
        assert!(run(args).is_err());
    }
    assert!(run(vec![
        tensor(vec![-1.0, 1.0], vec![2, 1]),
        Value::Num(0.0),
        Value::from("NumPoints"),
        Value::Num(10.0)
    ])
    .is_err());
    let output = run(vec![
        tensor(vec![-1.0, 1.0], vec![2, 1]),
        tensor(vec![], vec![0, 0]),
        Value::from("NumPoints"),
        Value::Num(7.0),
    ])
    .unwrap();
    assert_eq!(array(&output[0]).shape(), [1, 7]);
    let error = block_on(engine::grid(
        &[-1.0, 1.0],
        1.0,
        usize::MAX,
        None,
        &mut Cooperation::new(None),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:ksdensity:AllocationFailed")
    );
    for role in [0, 1, 2] {
        let resident = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let args = match role {
            0 => vec![Value::GpuTensor(resident)],
            1 => vec![
                tensor(vec![-1.0, 1.0], vec![2, 1]),
                Value::GpuTensor(resident),
            ],
            _ => vec![
                tensor(vec![-1.0, 1.0], vec![2, 1]),
                Value::from("Bandwidth"),
                Value::GpuTensor(resident),
            ],
        };
        let error = run(args).unwrap_err();
        assert!(error
            .message()
            .contains("host fallback requires automatic residency"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
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
    let args = crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]);
    let error = block_on(evaluate(Some(&runtime), args)).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(ledger.snapshot().peak_reserved_bytes, 0);
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(source.token());
    runtime.set_error_namespace("DensityInvocation");
    source.cancel(runmat_execution::CancellationReason::User);
    let args = crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]);
    let error = block_on(evaluate(Some(&runtime), args)).unwrap_err();
    assert_eq!(error.identifier(), Some("DensityInvocation:Cancelled"));
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
}
