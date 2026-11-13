use super::*;
use runmat_builtins::DatetimeBinaryOperation as Op;
const OPERATIONS: [Op; 8] = [
    Op::Eq,
    Op::Ne,
    Op::Lt,
    Op::Le,
    Op::Gt,
    Op::Ge,
    Op::Plus,
    Op::Minus,
];
async fn invoke_owned(
    operation: Op,
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    use super::super::*;
    match operation {
        Op::Eq => datetime_eq(runtime, args).await,
        Op::Ne => datetime_ne(runtime, args).await,
        Op::Lt => datetime_lt(runtime, args).await,
        Op::Le => datetime_le(runtime, args).await,
        Op::Gt => datetime_gt(runtime, args).await,
        Op::Ge => datetime_ge(runtime, args).await,
        Op::Plus => datetime_plus(runtime, args).await,
        Op::Minus => datetime_minus(runtime, args).await,
    }
}
fn invoke(operation: Op, values: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(invoke_owned(
        operation,
        None,
        crate::builtins::common::test_support::builtin_arguments(values),
    ))
}
fn date(values: Vec<f64>, shape: Vec<usize>) -> Value {
    datetime_object_from_serials(None, values, shape, "yyyy-MM-dd").unwrap()
}
fn scalar(value: f64) -> Value {
    date(vec![value], vec![1, 1])
}
fn serial_tensor(value: &Value) -> Tensor {
    serials_from_datetime_value(value).expect("datetime result")
}
#[test]
fn missing_and_infinite_equality_are_complementary_without_changing_double_or_finite_tolerance() {
    // Primary MathWorks relational rules: NaT is unequal even to NaT;
    // equal-sign infinities are equal. The existing finite tolerance remains.
    for (left, right, expected) in [
        (f64::NAN, 0., false),
        (0., f64::NAN, false),
        (f64::NAN, f64::NAN, false),
        (f64::INFINITY, f64::INFINITY, true),
        (f64::NEG_INFINITY, f64::NEG_INFINITY, true),
        (f64::INFINITY, f64::NEG_INFINITY, false),
        (0., 0., true),
        (0., 0.5e-12, true),
        (0., 2e-12, false),
    ] {
        let lhs = scalar(left);
        let rhs = scalar(right);
        assert_eq!(
            invoke(Op::Eq, vec![lhs.clone(), rhs.clone()]).unwrap(),
            Value::Num(if expected { 1. } else { 0. })
        );
        assert_eq!(
            invoke(Op::Ne, vec![lhs, rhs]).unwrap(),
            Value::Num(if expected { 0. } else { 1. })
        );
    }
    for op in [Op::Lt, Op::Le, Op::Gt, Op::Ge] {
        assert_eq!(
            invoke(op, vec![scalar(f64::NAN), scalar(0.)]).unwrap(),
            Value::Num(0.)
        );
    }
}
#[test]
fn actual_rhs_dispatch_and_format_keep_datetime_and_numeric_day_difference_distinct() {
    let lhs = scalar(739_252.);
    let duration = crate::builtins::duration::duration_object_from_days_tensor(
        None,
        Tensor::new(vec![1.], vec![1, 1]).unwrap(),
        crate::builtins::duration::DEFAULT_DURATION_FORMAT,
    )
    .unwrap();
    let calendar =
        calendar_duration_object_from_components(None, vec![0.], vec![1.], vec![1, 1]).unwrap();
    for rhs in [Value::Num(1.), duration.clone(), calendar.clone()] {
        for (op, expected) in [(Op::Plus, 739_253.), (Op::Minus, 739_251.)] {
            let output = invoke(op, vec![lhs.clone(), rhs.clone()]).unwrap();
            assert_eq!(
                tensor::tensor_value_f64(&serial_tensor(&output), 0),
                expected
            );
            assert_eq!(datetime_format_from_value(&output).unwrap(), "yyyy-MM-dd");
        }
    }
    assert_eq!(
        invoke(Op::Minus, vec![lhs.clone(), scalar(739_250.)]).unwrap(),
        Value::Num(2.)
    );
    for op in [Op::Eq, Op::Ne, Op::Lt, Op::Le, Op::Gt, Op::Ge] {
        for rhs in [duration.clone(), calendar.clone()] {
            let error = invoke(op, vec![lhs.clone(), rhs]).unwrap_err();
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        }
    }
    assert!(invoke(Op::Plus, vec![lhs, scalar(1.)]).is_err());
}
#[test]
fn full_empty_geometry_native_single_input_and_scalar_or_exact_only_are_retained() {
    for shape in [vec![0, 5], vec![2, 0, 3], vec![0], vec![3]] {
        let count: usize = shape.iter().product();
        let tensor = Tensor::from_f32(vec![1.; count], shape.clone()).unwrap();
        let lhs = datetime_object_from_serial_tensor(None, tensor, "yyyy-MM-dd").unwrap();
        let difference = invoke(Op::Minus, vec![lhs.clone(), lhs.clone()]).unwrap();
        let Value::Tensor(difference) = difference else {
            panic!("nonscalar datetime difference")
        };
        assert_eq!(difference.shape(), shape);
        assert_eq!(difference.as_f64_slice().unwrap(), vec![0.; count]);
        for op in OPERATIONS {
            let output = invoke(op, vec![lhs.clone(), Value::Num(1.)]).unwrap();
            let tensor = if op.is_comparison() {
                let Value::Tensor(tensor) = output else {
                    panic!("nonscalar double output")
                };
                tensor
            } else {
                serial_tensor(&output)
            };
            assert_eq!(tensor.shape(), shape);
            assert!(
                tensor.as_f64_slice().is_some(),
                "actual native publication is double"
            );
        }
    }
    let scalar_single = datetime_object_from_serial_tensor(
        None,
        Tensor::from_f32(vec![1.], vec![]).unwrap(),
        "yyyy-MM-dd",
    )
    .unwrap();
    assert_eq!(
        invoke(Op::Eq, vec![scalar_single, Value::Num(1.)]).unwrap(),
        Value::Num(1.)
    );
    let lhs = date(vec![1., 2.], vec![2, 1]);
    let rhs = Value::Tensor(Tensor::new(vec![1., 2., 3.], vec![1, 3]).unwrap());
    for op in OPERATIONS {
        let error = invoke(op, vec![lhs.clone(), rhs.clone()]).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:datetime:InvalidInput"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
}
#[test]
fn native_integer_rhs_range_logical_extension_and_no_resident_gather_are_truthful() {
    use runmat_value::IntValue;
    let lhs = scalar(0.);
    for integer in [
        IntValue::I8(1),
        IntValue::U8(1),
        IntValue::I16(1),
        IntValue::U16(1),
        IntValue::I32(1),
        IntValue::U32(1),
        IntValue::I64(1),
        IntValue::U64(1),
    ] {
        let output = invoke(Op::Plus, vec![lhs.clone(), Value::Int(integer)]).unwrap();
        assert_eq!(tensor::tensor_value_f64(&serial_tensor(&output), 0), 1.);
    }
    for integer in [IntValue::I64(-1_000_000_001), IntValue::U64(u64::MAX)] {
        assert!(invoke(Op::Plus, vec![lhs.clone(), Value::Int(integer)]).is_err());
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = invoke(Op::Plus, vec![lhs.clone(), Value::Bool(true)]).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DatetimeLogicalInputExtension")
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let resident = owner
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    for op in OPERATIONS {
        let error = invoke(
            op,
            vec![lhs.clone(), Value::GpuTensor(resident.handle.clone())],
        )
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(resident.probe.retirements(), 0);
        resident.handle.allocation_lease().validate_live().unwrap();
    }
}
#[test]
fn actual_selected_context_borrowed_arguments_output_preflight_and_cancellation_are_terminal() {
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(source.token());
    for op in OPERATIONS {
        let values = [scalar(1.), Value::Num(1.)];
        let args = crate::call::BuiltinArguments::from_borrowed(&values, Some(&runtime)).unwrap();
        let output =
            futures::executor::block_on(invoke_owned(op, Some(runtime.clone()), args)).unwrap();
        if op.is_comparison() {
            assert!(matches!(output, Value::Num(_)));
        } else {
            assert!(is_datetime_object(&output));
        }
        let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
        let error = invoke(op, vec![]).unwrap_err();
        assert_eq!(error.identifier(), op.output_error().identifier);
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
    let values = [scalar(1.), Value::Num(1.)];
    let args = crate::call::BuiltinArguments::from_borrowed(&values, Some(&runtime)).unwrap();
    source.cancel(runmat_execution::CancellationReason::User);
    let error =
        futures::executor::block_on(invoke_owned(Op::Plus, Some(runtime), args)).unwrap_err();
    assert!(error.identifier().unwrap().contains("Cancelled"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}
#[test]
fn finite_calendar_method_work_refuses_before_unowned_scratch_and_object_publication() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
        RuntimeParallelService, RuntimeServicePorts,
    };
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
    let ledger = ManagedHostEffectLedger::new(65_536);
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(std::rc::Rc::new(Finite(ledger.clone()))),
    );
    for op in OPERATIONS {
        let args = crate::builtins::common::test_support::builtin_arguments(vec![
            scalar(1.),
            Value::Num(1.),
        ]);
        let error =
            futures::executor::block_on(invoke_owned(op, Some(runtime.clone()), args)).unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
