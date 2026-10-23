use super::*;

struct MutablePolicy {
    ledger: ManagedHostEffectLedger,
    finite: Rc<std::cell::Cell<bool>>,
}
impl RuntimeParallelService for MutablePolicy {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.ledger.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: self.finite.get().then_some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.ledger.clone())
    }
}

#[test]
fn callback_policy_upgrade_and_reused_unbounded_factor_fail_before_publication() {
    use crate::builtins::math::linalg::ops::dense_factor::{DenseFactor, DenseFactorError};
    use crate::context::host_allocation::HostAllocationError;
    use std::cell::Cell;
    let finite = Rc::new(Cell::new(false));
    let ledger = ManagedHostEffectLedger::new(1 << 20);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(MutablePolicy {
                ledger: ledger.clone(),
                finite: finite.clone(),
            })),
        );
    let matrix = Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap());
    let mut factor =
        futures::executor::block_on(DenseFactor::<f64>::factor(&matrix, 1, &runtime)).unwrap();
    finite.set(true);
    let mut unpublished = [99.0];
    assert!(matches!(
        futures::executor::block_on(factor.solve_into(&[4.0], &mut unpublished, &runtime)),
        Err(DenseFactorError::Admission(
            HostAllocationError::StaleAuthority
        ))
    ));
    assert_eq!(unpublished, [99.0]);
    finite.set(false);
    let observed = finite.clone();
    let calls = Rc::new(Cell::new(0));
    let counted = calls.clone();
    runtime.set_bound_function_invoker(Some(Rc::new(move |bound_runtime, _, arguments, _, _| {
        counted.set(counted.get() + 1);
        observed.set(true);
        let input = arguments.supplied()[0].clone();
        crate::execution::admit_executable_sequence_with(&bound_runtime, move || async move {
            crate::user_functions::test_support::rooted_single_value_sequence(input)
        })
    })));
    let target =
        crate::user_functions::test_support::exact_local_handle(&runtime, "policy_upgrade", 9122);
    let parsed = parse(target, Value::Num(1.0), vec![]).unwrap();
    let error = futures::executor::block_on(execution::execute(parsed, Some(&runtime), Some(5)))
        .expect_err("changed policy must not publish unclaimed output");
    assert_eq!(
        error.identifier(),
        Some("RunMat:parallel:InfrastructureFailure")
    );
    assert_eq!(calls.get(), 1, "an effectful callback must not be replayed");
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn pending_or_cancelled_output_copy_retires_its_unpublished_claims() {
    use super::super::super::iterative::{copy, value::IterativeValue};
    use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::{
        future::Future,
        num::NonZeroU64,
        task::{Context, Poll},
    };
    for cancel in [false, true] {
        let cancellation = CancellationSource::new();
        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            cancellation.token(),
        )
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
        let receipt = HostAllocationReceipt::admit(
            Some(&runtime),
            HostAllocationPlan::new(0, <f64 as IterativeValue>::output_bytes(2048).unwrap())
                .unwrap(),
        )
        .unwrap()
        .unwrap();
        let before = ledger.snapshot().live_claimed_bytes;
        let mut budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            cancellation.token(),
            || Box::pin(std::future::pending()),
        );
        let source = vec![7.0; 2048];
        let mut copy = Box::pin(copy::admitted_with_budget::<f64>(
            &receipt,
            &runtime,
            &source,
            "gmres",
            &mut budget,
        ));
        let mut cx = Context::from_waker(futures::task::noop_waker_ref());
        assert!(copy.as_mut().poll(&mut cx).is_pending());
        assert!(
            ledger.snapshot().live_claimed_bytes > before,
            "the suspended constructor must own its exact backing"
        );
        if cancel {
            cancellation.cancel(CancellationReason::User);
            assert!(matches!(copy.as_mut().poll(&mut cx), Poll::Ready(Err(_))));
        }
        drop(copy);
        assert_eq!(ledger.snapshot().live_claimed_bytes, before);
        drop(receipt);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn default_bindings_admit_scalar_and_native_single_order_one_systems() {
    for name in ["pcg", "gmres"] {
        let answer = crate::call_builtin(name, &[Value::Num(2.0), Value::Num(4.0)]).unwrap();
        let Value::Tensor(answer) = answer.value() else {
            panic!("double column output")
        };
        assert_eq!(answer.shape(), &[1, 1]);
        assert_eq!(answer.as_f64_slice().unwrap(), &[2.0]);
        let answer =
            crate::call_builtin(name, &[Value::Complex(2.0, 0.0), Value::Complex(4.0, 2.0)])
                .unwrap();
        let Value::ComplexTensor(answer) = answer.value() else {
            panic!("complex column output")
        };
        assert_eq!(answer.shape(), &[1, 1]);
        let value = answer.as_f64_slice().unwrap()[0];
        assert!((value.0 - 2.0).abs() < 1e-12 && (value.1 - 1.0).abs() < 1e-12);
        let answer = crate::call_builtin(
            name,
            &[
                Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::from_f32(vec![4.0], vec![1, 1]).unwrap()),
            ],
        )
        .unwrap();
        let Value::Tensor(answer) = answer.value() else {
            panic!("single column output")
        };
        assert_eq!(answer.as_f32_slice().unwrap(), &[2.0]);
    }
    let empty = || Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
    let parsed = parse(
        Value::Complex(2.0, 1.0),
        Value::Complex(3.0, 4.0),
        vec![
            empty(),
            empty(),
            empty(),
            Value::Complex(2.0, 0.0),
            empty(),
            Value::Complex(0.1, -0.2),
        ],
    )
    .unwrap();
    let Value::OutputList(answer) =
        futures::executor::block_on(execution::execute(parsed, None, Some(5))).unwrap()
    else {
        panic!("five outputs")
    };
    assert_eq!(answer[1], Value::Num(0.0));
    let Value::ComplexTensor(x) = &answer[0] else {
        panic!("complex solution")
    };
    let x = x.as_f64_slice().unwrap()[0];
    assert!((x.0 - 2.0).abs() < 1e-12 && (x.1 - 1.0).abs() < 1e-12);
}

#[test]
fn status_text_obeys_flag_output_and_zero_output_discards_the_solution() {
    for count in [0, 1, 2] {
        let (a, b) = system();
        let capture = crate::console::begin_capture().unwrap();
        let parsed = parse(a, b, vec![]).unwrap();
        let result =
            futures::executor::block_on(execution::execute(parsed, None, Some(count))).unwrap();
        if count == 0 {
            let Value::OutputList(values) = result else {
                panic!("discarded output")
            };
            assert_eq!(values.len(), 0);
        }
        let text = capture.finish().unwrap();
        assert_eq!(text.is_empty(), count >= 2);
    }
}

#[test]
fn callback_language_error_is_terminal_and_admission_precedes_effects() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::cell::Cell;
    for budget in [0, 1 << 20] {
        for preconditioner in [false, true] {
            for cancel in [false, true] {
                let ledger = ManagedHostEffectLedger::new(budget);
                let cancellation = CancellationSource::new();
                let callback_cancellation = cancellation.clone();
                let runtime = RuntimeContext::with_cancellation(
                    Rc::new(crate::execution::RuntimeExecutionService::new()),
                    cancellation.token(),
                )
                .with_service_ports(
                    RuntimeServicePorts::default()
                        .with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
                );
                let calls = Rc::new(Cell::new(0));
                let observed = Rc::clone(&calls);
                runtime.set_bound_function_invoker(Some(Rc::new(move |bound_runtime, _, arguments, _, _| {
                    observed.set(observed.get() + 1);
                    let output = arguments.supplied()[0].clone();
                    if cancel {
                        callback_cancellation.cancel(CancellationReason::User);
                    }
                    crate::execution::admit_executable_sequence_with(&bound_runtime.clone(), move || async move {
                        if cancel {
                            crate::user_functions::test_support::rooted_single_value_sequence(
                                output,
                            )
                        } else {
                            Err(crate::build_runtime_error("callback deliberately failed")
                                .with_identifier("RunMat:test:CallbackFailure")
                                .with_source(std::io::Error::other("original GMRES callback cause"))
                                .with_gpu_gather_retry(crate::GpuGatherRetry::Requested)
                                .build())
                        }
                    })
                })));
                let target = crate::user_functions::test_support::exact_local_handle(
                    &runtime,
                    "gmres_failure",
                    9121,
                );
                let (a, b) = system();
                let parsed = if preconditioner {
                    let empty =
                        || Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap());
                    parse(a, b, vec![empty(), empty(), empty(), target])
                } else {
                    parse(target, b, vec![])
                }
                .unwrap();
                let error = futures::executor::block_on(execution::execute(
                    parsed,
                    Some(&runtime),
                    Some(5),
                ))
                .unwrap_err();
                assert_eq!(calls.get(), if budget == 0 { 0 } else { 1 });
                if budget != 0 {
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                    if cancel {
                        assert!(error
                            .identifier()
                            .is_some_and(|id| id.ends_with(":Cancelled")));
                        let mut source = std::error::Error::source(&error);
                        let mut typed_cancelled = false;
                        while let Some(cause) = source {
                            typed_cancelled |= matches!(
                                cause.downcast_ref::<crate::execution::ExecutionServiceError>(),
                                Some(crate::execution::ExecutionServiceError::Cancelled)
                            );
                            source = cause.source();
                        }
                        assert!(typed_cancelled, "retain original cancellation cause");
                    } else {
                        assert_eq!(error.identifier(), Some("RunMat:test:CallbackFailure"));
                        assert!(error.source.as_ref().is_some_and(
                            |cause| cause.to_string() == "original GMRES callback cause"
                        ));
                    }
                }
                assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
                assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
            }
        }
    }
}

#[test]
fn native_classes_defaults_and_preconditioner_status() {
    use runmat_value::{ComplexElement, ComplexStorage, ComplexTensor, NumericDType};
    for single in [false, true] {
        for complex in [false, true] {
            let (a, b) = if complex {
                let make = |values: Vec<f64>, shape: Vec<usize>| {
                    let storage = if single {
                        ComplexStorage::F32(
                            values
                                .into_iter()
                                .map(|v| ComplexElement(v as f32, 0.0))
                                .collect::<Vec<_>>()
                                .into(),
                        )
                    } else {
                        ComplexStorage::F64(
                            values
                                .into_iter()
                                .map(|v| ComplexElement(v, 0.0))
                                .collect::<Vec<_>>()
                                .into(),
                        )
                    };
                    Value::ComplexTensor(
                        ComplexTensor::from_complex_storage(storage, shape).unwrap(),
                    )
                };
                (
                    make(vec![4.0, 2.0, 1.0, 3.0], vec![2, 2]),
                    make(vec![1.0, 0.0], vec![2, 1]),
                )
            } else if single {
                (
                    Value::Tensor(Tensor::from_f32(vec![4.0, 2.0, 1.0, 3.0], vec![2, 2]).unwrap()),
                    Value::Tensor(Tensor::from_f32(vec![1.0, 0.0], vec![2, 1]).unwrap()),
                )
            } else {
                system()
            };
            let parsed = parse(a, b, vec![]).unwrap();
            assert_eq!(parsed.tolerance, if single { 1e-3 } else { 1e-6 });
            let Value::OutputList(result) =
                futures::executor::block_on(execution::execute(parsed, None, Some(5))).unwrap()
            else {
                panic!("five outputs")
            };
            assert_eq!(result[1], Value::Num(0.0));
            let dtype = if single {
                NumericDType::F32
            } else {
                NumericDType::F64
            };
            match &result[0] {
                Value::Tensor(x) => {
                    assert!(!complex);
                    assert_eq!(x.numeric_dtype(), dtype);
                }
                Value::ComplexTensor(x) => {
                    assert!(complex);
                    assert_eq!(x.numeric_dtype(), dtype);
                }
                _ => panic!("native solution"),
            }
            let Value::Tensor(iter) = &result[3] else {
                panic!("iteration pair")
            };
            assert_eq!(iter.as_f64_slice().unwrap()[0], 1.0);
        }
    }
    let (a, b) = system();
    let empty = || Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
    let initial = Value::Tensor(Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap());
    let singular = Value::Tensor(Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap());
    let parsed = parse(
        a,
        b,
        vec![
            empty(),
            empty(),
            empty(),
            singular,
            empty(),
            initial.clone(),
        ],
    )
    .unwrap();
    let Value::OutputList(result) =
        futures::executor::block_on(execution::execute(parsed, None, Some(5))).unwrap()
    else {
        panic!("five outputs")
    };
    assert_eq!(result[0], initial);
    assert_eq!(result[1], Value::Num(2.0));
    assert_eq!(result[2], Value::Num(f64::INFINITY));
    let Value::Tensor(iter) = &result[3] else {
        panic!("iteration pair")
    };
    assert_eq!(iter.as_f64_slice().unwrap(), &[0.0, 0.0]);
    let Value::Tensor(history) = &result[4] else {
        panic!("history")
    };
    assert_eq!(history.len(), 0);
}

#[test]
fn managed_callback_inputs_can_escape_without_aliasing_recurrence_scratch() {
    use std::cell::RefCell;
    let ledger = ManagedHostEffectLedger::new(1 << 20);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
    let retained = Rc::new(RefCell::new(Vec::<Value>::new()));
    let observed = Rc::clone(&retained);
    runtime.set_bound_function_invoker(Some(Rc::new(move |bound_runtime, _, arguments, _, _| {
        let [input] = arguments.supplied() else {
            panic!("one callback input")
        };
        let input = input.clone();
        observed.borrow_mut().push(input.clone());
        crate::execution::admit_executable_sequence_with(&bound_runtime, move || async move {
            crate::user_functions::test_support::rooted_single_value_sequence(input)
        })
    })));
    let target =
        crate::user_functions::test_support::exact_local_handle(&runtime, "gmres_identity", 9120);
    let rhs = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let parsed = parse(target, rhs, vec![]).unwrap();
    let result =
        futures::executor::block_on(execution::execute(parsed, Some(&runtime), Some(5))).unwrap();
    let Value::OutputList(outputs) = &result else {
        panic!("five outputs")
    };
    assert_eq!(outputs[1], Value::Num(0.0));
    let snapshots = retained.borrow().clone();
    assert!(snapshots.len() >= 3);
    let Value::Tensor(first) = &snapshots[0] else {
        panic!("native callback input")
    };
    assert_eq!(first.as_f64_slice().unwrap(), &[0.0, 0.0]);
    drop(result);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    retained.borrow_mut().clear();
    drop(snapshots);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
