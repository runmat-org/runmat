use super::*;
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use std::rc::Rc;

fn calendar_value(months: Vec<f64>, days: Vec<f64>, shape: Vec<usize>) -> Value {
    calendar_duration_object_from_components(None, months, days, shape).unwrap()
}
fn scalar(months: f64, days: f64) -> Value {
    calendar_value(vec![months], vec![days], vec![1, 1])
}
fn arguments(values: Vec<Value>) -> crate::call::BuiltinArguments {
    crate::builtins::common::test_support::builtin_arguments(values)
}
async fn invoke(
    op: Op,
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    use super::super::*;
    match op {
        Op::Plus => calendar_duration_plus(runtime, args).await,
        Op::Minus => calendar_duration_minus(runtime, args).await,
        Op::Eq => calendar_duration_eq(runtime, args).await,
        Op::Ne => calendar_duration_ne(runtime, args).await,
        Op::IsDatetime => {
            let value =
                super::super::super::component_builtins::isdatetime_builtin(runtime.clone(), args)?;
            RootedValue::in_optional_context(value, runtime.as_ref())
        }
        Op::DateTick => {
            let value = datetick_builtin(runtime.clone(), args).await?;
            RootedValue::in_optional_context(value, runtime.as_ref())
        }
    }
}
fn run(op: Op, values: Vec<Value>) -> RootedValue {
    futures::executor::block_on(invoke(op, None, arguments(values))).unwrap()
}
#[test]
fn calendar_components_shapes_and_double_comparison_handle_missing_and_infinite_values() {
    for (left, right, equal) in [
        (0., 0., true),
        (0., 0.5e-12, true),
        (0., 2e-12, false),
        (f64::NAN, 0., false),
        (f64::NAN, f64::NAN, false),
        (f64::INFINITY, f64::INFINITY, true),
        (f64::NEG_INFINITY, f64::NEG_INFINITY, true),
        (f64::INFINITY, f64::NEG_INFINITY, false),
    ] {
        let lhs = scalar(left, 0.);
        let rhs = scalar(right, 0.);
        assert_eq!(
            run(Op::Eq, vec![lhs.clone(), rhs.clone()]).value(),
            &Value::Num(if equal { 1. } else { 0. })
        );
        assert_eq!(
            run(Op::Ne, vec![lhs, rhs]).value(),
            &Value::Num(if equal { 0. } else { 1. })
        );
    }
    let lhs = scalar(1., 2.);
    let rhs = calendar_value(vec![2., 3.], vec![4., 5.], vec![1, 2]);
    for (op, expected_months, expected_days) in [
        (Op::Plus, vec![3., 4.], vec![6., 7.]),
        (Op::Minus, vec![-1., -2.], vec![-2., -3.]),
    ] {
        let output = run(op, vec![lhs.clone(), rhs.clone()]);
        let (months, days) = calendar_duration_tensors_from_value(output.value()).unwrap();
        assert_eq!(months.materialize_f64(), expected_months);
        assert_eq!(days.materialize_f64(), expected_days);
        assert_eq!(months.shape(), [1, 2]);
    }
    let empty = calendar_value(vec![], vec![], vec![0, 2]);
    let output = run(Op::Eq, vec![empty.clone(), empty]);
    let Value::Tensor(tensor) = output.value() else {
        panic!("empty comparison tensor")
    };
    assert_eq!(tensor.shape(), [0, 2]);
    let other_shape = calendar_value(vec![2., 3.], vec![4., 5.], vec![2, 1]);
    assert!(
        futures::executor::block_on(invoke(Op::Plus, None, arguments(vec![rhs, other_shape])))
            .is_err()
    );
    let date =
        datetime_object_from_serials(None, vec![739_282.], vec![1, 1], "yyyy-MM-dd").unwrap();
    let output = run(Op::Plus, vec![scalar(1., 0.), date.clone()]);
    assert!(is_datetime_object(output.value()));
    assert_eq!(
        datetime_format_from_value(output.value()).unwrap(),
        datetime_format_from_value(&date).unwrap()
    );
    for op in [Op::Minus, Op::Eq, Op::Ne] {
        let error = futures::executor::block_on(invoke(
            op,
            None,
            arguments(vec![scalar(1., 0.), date.clone()]),
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
}
#[test]
fn public_residual_calls_keep_class_status_outputs_and_output_preflight() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    for op in [
        Op::IsDatetime,
        Op::DateTick,
        Op::Plus,
        Op::Minus,
        Op::Eq,
        Op::Ne,
    ] {
        let values = match op {
            Op::IsDatetime => {
                vec![datetime_object_from_serials(None, vec![], vec![0, 2], "yyyy").unwrap()]
            }
            Op::DateTick => vec![
                Value::from("ignored"),
                Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            ],
            _ => vec![scalar(1., 0.), scalar(1., 0.)],
        };
        let output = futures::executor::block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                op.name(),
                &values,
                1,
            ),
        )
        .unwrap();
        match op {
            Op::IsDatetime => assert_eq!(output.value(), &Value::Bool(true)),
            Op::DateTick => assert_eq!(output.value(), &Value::Num(0.)),
            Op::Eq => assert_eq!(output.value(), &Value::Num(1.)),
            Op::Ne => assert_eq!(output.value(), &Value::Num(0.)),
            _ => assert!(is_calendar_duration_object(output.value())),
        }
        futures::executor::block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                op.name(),
                &values,
                0,
            ),
        )
        .expect("zero-output call still performs the existing operation");
        let error = futures::executor::block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                op.name(),
                &values,
                2,
            ),
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::CALENDAR_RESIDUAL_ERROR_OUTPUTS.identifier
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
}
#[test]
fn class_inspection_never_downloads_and_selected_context_does_not_fall_back_to_ambient() {
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let resident = owner
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    assert_eq!(
        run(
            Op::IsDatetime,
            vec![Value::GpuTensor(resident.handle.clone())]
        )
        .value(),
        &Value::Bool(false)
    );
    let source = runmat_execution::cancellation::CancellationSource::new();
    let ambient = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(source.token());
    source.cancel(runmat_execution::CancellationReason::User);
    let selected = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let values = arguments(vec![Value::Num(1.)]);
    let _ambient = ambient.enter();
    let result =
        futures::executor::block_on(invoke(Op::IsDatetime, Some(selected), values)).unwrap();
    assert_eq!(result.value(), &Value::Bool(false));
    assert_eq!(resident.probe.retirements(), 0);
    resident.handle.allocation_lease().validate_live().unwrap();
}
#[test]
fn finite_frontier_precedes_scratch_and_predicate_uses_selected_cancellation_without_workspace() {
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
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(source.token())
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
        );
    let value = datetime_object_from_serials(None, vec![1.], vec![1, 1], "yyyy").unwrap();
    assert_eq!(
        futures::executor::block_on(invoke(
            Op::IsDatetime,
            Some(runtime.clone()),
            arguments(vec![value.clone()])
        ))
        .unwrap()
        .value(),
        &Value::Bool(true)
    );
    for op in [Op::DateTick, Op::Plus, Op::Minus, Op::Eq, Op::Ne] {
        let values = if op == Op::DateTick {
            vec![]
        } else {
            vec![scalar(1., 0.), scalar(2., 0.)]
        };
        let error =
            futures::executor::block_on(invoke(op, Some(runtime.clone()), arguments(values)))
                .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::CALENDAR_RESIDUAL_ERROR_WORKSPACE.identifier
        );
        assert!(matches!(
            std::error::Error::source(&error)
                .and_then(|cause| cause.downcast_ref::<crate::execution::ExecutionServiceError>()),
            Some(crate::execution::ExecutionServiceError::Infrastructure(_))
        ));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
        let error =
            futures::executor::block_on(invoke(op, Some(runtime.clone()), arguments(vec![])))
                .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::CALENDAR_RESIDUAL_ERROR_OUTPUTS.identifier
        );
    }
    source.cancel(runmat_execution::CancellationReason::User);
    for op in [
        Op::IsDatetime,
        Op::DateTick,
        Op::Plus,
        Op::Minus,
        Op::Eq,
        Op::Ne,
    ] {
        let values = match op {
            Op::IsDatetime => vec![value.clone()],
            Op::DateTick => vec![],
            _ => vec![scalar(1., 0.), scalar(2., 0.)],
        };
        let error =
            futures::executor::block_on(invoke(op, Some(runtime.clone()), arguments(values)))
                .unwrap_err();
        assert!(error.identifier().unwrap().contains("Cancelled"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}
