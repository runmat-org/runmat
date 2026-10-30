use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use std::{cell::Cell, rc::Rc};

struct Managed(ManagedHostEffectLedger);
impl RuntimeParallelService for Managed {
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

fn runtime(budget: u64) -> (RuntimeContext, ManagedHostEffectLedger) {
    let (runtime, ledger, _) = cancellable_runtime(budget);
    (runtime, ledger)
}

fn cancellable_runtime(
    budget: u64,
) -> (
    RuntimeContext,
    ManagedHostEffectLedger,
    runmat_execution::cancellation::CancellationSource,
) {
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let ledger = ManagedHostEffectLedger::new_with_epoch(budget, 296);
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
    );
    (runtime, ledger, cancellation)
}

struct FundingProbe {
    ledger: ManagedHostEffectLedger,
    polls: Rc<Cell<usize>>,
    ready: bool,
    failure: bool,
}

impl Future for FundingProbe {
    type Output = Result<(), HostAllocationError>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.set(self.polls.get() + 1);
        if self.failure {
            Poll::Ready(Err(HostAllocationError::Overflow))
        } else if self.ready {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }
}

impl Drop for FundingProbe {
    fn drop(&mut self) {
        let snapshot = self.ledger.snapshot();
        assert!(snapshot.live_claimed_bytes + snapshot.live_reserved_bytes > 0);
    }
}

#[test]
fn command_supervisor_retains_funding_through_cancelled_completion_or_abandonment() {
    for finish in [false, true] {
        let (selected, ledger, cancellation) = cancellable_runtime(1_000_000);
        let polls = Rc::new(Cell::new(0));
        let mut future = admit_execution_supervisor_with(&selected, || {
            let supervised = selected.clone();
            let probe = FundingProbe {
                ledger: ledger.clone(),
                polls: Rc::clone(&polls),
                ready: false,
                failure: false,
            };
            async move {
                let _probe = probe;
                futures::future::poll_fn(|_| {
                    let previous = polls.replace(polls.get() + 1);
                    if previous == 0 {
                        Poll::Pending
                    } else {
                        assert!(supervised.is_cancelled());
                        // A supervisor publishes the child's terminal response,
                        // rather than having that response intercepted outside.
                        Poll::Ready(Ok::<_, RuntimeError>(()))
                    }
                })
                .await
            }
        })
        .unwrap();
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
        cancellation.cancel(runmat_execution::CancellationReason::User);
        if finish {
            assert!(matches!(
                Pin::new(&mut future).poll(&mut context),
                Poll::Ready(Ok(()))
            ));
        }
        drop(future);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn native_concrete_future_is_admitted_before_factory_and_refunded_on_abandonment() {
    optional::check();
    // Opaque embedding futures are not evidence of their concrete layout.
    // Managed rejection precedes factory effects; ordinary adoption retains
    // the exact original Box instead of allocating a Box around its pointer.
    let (managed, legacy_ledger) = runtime(1_000_000);
    let legacy_factory = Cell::new(false);
    let denied = adopt_unbounded_legacy_rooted_future(Some(&managed), || {
        legacy_factory.set(true);
        Box::pin(async { Ok::<(), RuntimeError>(()) })
    });
    assert!(denied.is_err());
    assert!(!legacy_factory.get());
    assert_eq!(legacy_ledger.snapshot().live_reserved_bytes, 0);
    let active = managed.enter();
    let concrete = crate::user_functions::admit_user_function_with(None, || {
        legacy_factory.set(true);
        std::future::pending::<Result<crate::execution::RootedValueSequence, RuntimeError>>()
    });
    assert!(
        concrete.is_err(),
        "None cannot admit an uncharged selected callable"
    );
    let lexical = crate::user_functions::admit_lexical_function_with(None, || {
        legacy_factory.set(true);
        std::future::pending::<Result<crate::call::lexical::RootedLexicalCallResult, RuntimeError>>(
        )
    });
    assert!(lexical.is_err());
    assert!(!legacy_factory.get());
    let denied = adopt_unbounded_legacy_rooted_future(None, || {
        legacy_factory.set(true);
        Box::pin(async { Ok::<(), RuntimeError>(()) })
    });
    assert!(
        denied.is_err(),
        "legacy None cannot bypass an active finite owner"
    );
    let borrowed = admit_optional_hostcall_with_contextual_error(None, || {
        legacy_factory.set(true);
        std::future::pending::<Result<(), RuntimeError>>()
    });
    assert!(
        borrowed.is_err(),
        "borrowed None cannot invent a finite owner"
    );
    assert!(!legacy_factory.get());
    drop(active);
    let ordinary = Cell::new(false);
    let borrowed = admit_optional_hostcall_with_contextual_error(None, || async {
        ordinary.set(true);
        Ok::<(), RuntimeError>(())
    })
    .expect("ordinary borrowed hostcall needs no manufactured runtime");
    assert!(!ordinary.get());
    futures::executor::block_on(borrowed).unwrap();
    assert!(ordinary.get());
    let concrete = crate::user_functions::admit_user_function_with(None, || {
        std::future::pending::<Result<crate::execution::RootedValueSequence, RuntimeError>>()
    })
    .expect("truly unscoped concrete callable remains ordinary");
    drop(concrete);
    let lexical = crate::user_functions::admit_lexical_function_with(None, || {
        std::future::pending::<Result<crate::call::lexical::RootedLexicalCallResult, RuntimeError>>(
        )
    })
    .expect("truly unscoped lexical callable remains ordinary");
    drop(lexical);
    // Context-free concrete factories have no execution grant, but retain
    // the original ordinary cancellation guard just like opaque embeddings.
    // Check before first poll, after a pending poll, and at final publication.
    for lexical in [false, true] {
        for mode in 0..3 {
            use futures::FutureExt;
            let source = runmat_execution::cancellation::CancellationSource::new();
            let selected =
                RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                    .with_cancellation_token(source.token());
            selected.set_error_namespace("OriginalNone398");
            let polls = Rc::new(Cell::new(0));
            let observed = Rc::clone(&polls);
            let inside_source = source.clone();
            let work = move || async move {
                observed.set(observed.get() + 1);
                if mode == 1 {
                    std::future::pending::<()>().await;
                }
                if mode == 2 {
                    assert!(inside_source.cancel(runmat_execution::CancellationReason::Deadline));
                }
            };
            let entry = selected.enter();
            let mut future = if lexical {
                futures::future::Either::Left(
                    crate::user_functions::admit_lexical_function_with(None, move || async move {
                        work().await;
                        crate::execution::RootedResult::new(
                            crate::call::lexical::LexicalCallResult {
                                outputs: runmat_value::ValueSequence::empty(),
                                captures: Vec::new(),
                            },
                        )
                        .map_err(|error| error.into_runtime_error())
                    })
                    .unwrap()
                    .map(|result| result.map(|_| ())),
                )
            } else {
                futures::future::Either::Right(
                    crate::user_functions::admit_user_function_with(None, move || async move {
                        work().await;
                        crate::execution::RootedValueSequence::new(
                            runmat_value::ValueSequence::empty(),
                        )
                        .map_err(|error| error.into_runtime_error())
                    })
                    .unwrap()
                    .map(|result| result.map(|_| ())),
                )
            };
            drop(entry);
            let (foreign, _) = runtime(1_000_000);
            foreign.set_error_namespace("ForeignNone398");
            let _foreign = foreign.enter();
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            if mode == 1 {
                assert!(Pin::new(&mut future).poll(&mut context).is_pending());
                assert_eq!(polls.get(), 1);
            }
            if mode != 2 {
                assert!(source.cancel(runmat_execution::CancellationReason::User));
            }
            let Poll::Ready(Err(error)) = Pin::new(&mut future).poll(&mut context) else {
                panic!("late cancellation must prevent ordinary callable publication");
            };
            assert_eq!(error.identifier(), Some("OriginalNone398:Cancelled"));
            assert!(matches!(
                std::error::Error::source(&error).and_then(|source| {
                    source.downcast_ref::<crate::execution::ExecutionServiceError>()
                }),
                Some(crate::execution::ExecutionServiceError::Cancelled)
            ));
            assert_eq!(polls.get(), usize::from(mode != 0));
            assert!(
                !foreign.is_cancelled(),
                "the ambient scope is not cancellation authority"
            );
            drop(future);
        }
    }
    let ordinary = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let original = Cell::new(std::ptr::null::<()>());
    let adopted = adopt_unbounded_legacy_rooted_future(Some(&ordinary), || {
        let future = Box::pin(async {
            std::future::pending::<()>().await;
            Ok::<(), RuntimeError>(())
        });
        original.set(future.as_ref().get_ref() as *const _ as *const ());
        future
    })
    .expect("ordinary embedding keeps its original owner");
    match &adopted.body {
        ContinuationBody::Running { future, claim, .. } => {
            assert_eq!(
                future.as_ref().get_ref() as *const _ as *const (),
                original.get()
            );
            assert!(
                claim.is_none(),
                "opaque adoption never fabricates a managed grant"
            );
        }
        ContinuationBody::Rejected(_) => panic!("ordinary factory was accepted"),
    }
    drop(adopted);
    let (cancelled, _, source) = cancellable_runtime(1_000_000);
    cancelled.set_error_namespace("LegacyOwner397");
    source.cancel(runmat_execution::CancellationReason::Deadline);
    let error = adopt_unbounded_legacy_rooted_future(Some(&cancelled), || {
        legacy_factory.set(true);
        Box::pin(async { Ok::<(), RuntimeError>(()) })
    })
    .err()
    .expect("captured cancellation precedes embedding factory");
    assert_eq!(error.identifier(), Some("LegacyOwner397:Cancelled"));
    assert!(!legacy_factory.get());
    // Rejection never carries a receipt. The inline carrier must not pay for
    // both the full RuntimeError and the mutually exclusive running receipt.
    const _: () = assert!(
        std::mem::size_of::<RuntimeContinuationFuture<'_, (), RuntimeError>>()
            < std::mem::size_of::<HostAllocationReceipt>() + std::mem::size_of::<RuntimeError>()
    );
    let (denied, ledger) = runtime(1);
    let called = Cell::new(false);
    let result = admit_native_hostcall_with(
        &denied,
        |cause| cause,
        || {
            called.set(true);
            async {
                let bytes = [0_u8; 64];
                std::future::pending::<()>().await;
                std::hint::black_box(bytes);
                Ok::<(), HostAllocationError>(())
            }
        },
    );
    assert!(result.is_err());
    assert!(!called.get());
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    // The actual concrete allocation, not its external carrier, is quoted.
    assert_eq!(
        rooted_future_layout_with_error::<FundingProbe, (), HostAllocationError>().unwrap(),
        std::mem::size_of::<FundingProbe>() as u64
    );
    for mode in 0..5 {
        let (admitted, ledger, cancellation) = cancellable_runtime(1_000_000);
        let polls = Rc::new(Cell::new(0));
        let mut future = admit_native_hostcall_with(
            &admitted,
            |cause| cause,
            || FundingProbe {
                ledger: ledger.clone(),
                polls: Rc::clone(&polls),
                ready: mode == 1,
                failure: mode == 4,
            },
        )
        .unwrap();
        assert!(ledger.snapshot().live_claimed_bytes + ledger.snapshot().live_reserved_bytes > 0);
        if mode == 2 {
            assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
            assert!(admitted.is_cancelled());
        }
        if mode == 3 {
            ledger.revoke();
        }
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        let result = future.as_mut().poll(&mut context);
        match mode {
            0 => assert!(result.is_pending()),
            1 => assert!(matches!(result, Poll::Ready(Ok(())))),
            4 => {
                assert!(matches!(
                    result,
                    Poll::Ready(Err(HostAllocationError::Overflow))
                ));
                assert_eq!(polls.get(), 1);
            }
            _ => {
                assert!(matches!(result, Poll::Ready(Err(_))));
                assert_eq!(polls.get(), 0);
            }
        }
        drop(future);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
    // The same physical hostcall owner may borrow a live caller frame and
    // return a borrowed non-Value domain result. No wrapper Box or cloned
    // payload is needed to manufacture a static lifetime.
    for ready in [false, true] {
        let (selected, ledger) = runtime(1_000_000);
        let borrowed = [11_u8; 64];
        let polls = Cell::new(0);
        let mut future = admit_native_hostcall_with_contextual_error(&selected, || async {
            polls.set(polls.get() + 1);
            if !ready {
                std::future::pending::<()>().await;
            }
            Ok::<_, RuntimeError>(&borrowed)
        })
        .unwrap();
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        match future.as_mut().poll(&mut context) {
            Poll::Ready(Ok(value)) if ready => assert!(std::ptr::eq(value, &borrowed)),
            Poll::Pending if !ready => {}
            other => panic!("borrowed hostcall completion: {other:?}"),
        }
        assert_eq!(polls.get(), 1);
        drop(future);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[path = "native_metadata_tests/optional.rs"]
mod optional;

#[test]
fn cancelled_native_admission_never_constructs_or_polls_child() {
    let (cancelled, ledger, cancellation) = cancellable_runtime(1_000_000);
    assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
    assert!(cancelled.is_cancelled());
    let called = Cell::new(false);
    let result = admit_native_hostcall_with(
        &cancelled,
        |cause| cause,
        || {
            called.set(true);
            std::future::ready(Ok::<(), HostAllocationError>(()))
        },
    );
    assert!(result.is_err());
    assert!(!called.get());
    // The infallible builtin boundary carries rejection inline; it does not
    // allocate an unfunded ready-error Box or construct the rejected child.
    let mut rejected = admit_rooted_future_with(
        Some(cancelled.clone()),
        "cancelled fixture",
        semantic_control_error,
        |_| {
            called.set(true);
            std::future::ready(Ok::<(), RuntimeError>(()))
        },
    );
    assert!(matches!(
        &rejected.body,
        ContinuationBody::Rejected(Some(_))
    ));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(
        rejected.as_mut().poll(&mut context),
        Poll::Ready(Err(_))
    ));
    assert!(!called.get());
    drop(rejected);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    // Both the initial admission and later poll must use the selected
    // cancellation authority, never a foreign ambient diagnostic namespace.
    for already_cancelled in [true, false] {
        let (selected, ledger, cancellation) = cancellable_runtime(1_000_000);
        selected.set_error_namespace("SelectedHostcall");
        if already_cancelled {
            assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
            assert!(selected.is_cancelled());
        }
        let constructed = Cell::new(false);
        let result = admit_native_hostcall_with_contextual_error(&selected, || {
            constructed.set(true);
            std::future::pending::<Result<(), RuntimeError>>()
        });
        let error = if already_cancelled {
            assert!(!constructed.get());
            result.err().expect("cancelled admission")
        } else {
            assert!(constructed.get());
            let mut future = result.expect("admitted hostcall");
            let (foreign, _) = runtime(1_000_000);
            foreign.set_error_namespace("ForeignHostcall");
            let _foreign_scope = foreign.enter();
            assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
            assert!(selected.is_cancelled());
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            let Poll::Ready(Err(error)) = future.as_mut().poll(&mut context) else {
                panic!("selected cancellation precedes child poll");
            };
            drop(future);
            error
        };
        assert_eq!(error.identifier(), Some("SelectedHostcall:Cancelled"));
        assert!(matches!(
            std::error::Error::source(&error).and_then(|source| {
                source.downcast_ref::<crate::execution::ExecutionServiceError>()
            }),
            Some(crate::execution::ExecutionServiceError::Cancelled)
        ));
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn dynamic_load_preserves_original_miss_owner_and_terminal_control_failures() {
    use crate::call::RootedFunctionArguments;
    use crate::execution::{RootedValueSequence, RootedValues};
    use crate::user_functions::await_dynamic_function_load;
    use std::ops::ControlFlow;

    let (denied, ledger) = runtime(0);
    let factories = Cell::new(0);
    let rejected = admit_dynamic_function_load_with(&denied, || {
        factories.set(factories.get() + 1);
        async {
            ControlFlow::Continue(RootedFunctionArguments::ordinary(
                RootedValues::new(Vec::new()).unwrap(),
            ))
        }
    });
    assert!(rejected.is_err());
    assert_eq!(
        factories.get(),
        0,
        "admission precedes source selection and future construction"
    );
    assert!(matches!(
        futures::executor::block_on(await_dynamic_function_load(rejected)),
        ControlFlow::Break(Err(_))
    ));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    for mode in 0..6 {
        let (selected, ledger, cancellation) = cancellable_runtime(1_000_000);
        selected.set_error_namespace("DynamicControl396");
        // Inputs already have their independent original root/backing owner;
        // this row qualifies continuation control, not their allocation budget.
        let arguments = RootedFunctionArguments::ordinary(
            RootedValues::new(vec![runmat_value::Value::String("original".into())]).unwrap(),
        );
        let runmat_value::Value::String(runmat_value::StringElement::Text(original)) =
            &arguments[0]
        else {
            unreachable!()
        };
        let pointer = original.as_ptr();
        let polls = Rc::new(Cell::new(0));
        let observed = Rc::clone(&polls);
        let mut future = admit_dynamic_function_load_with(&selected, move || async move {
            observed.set(observed.get() + 1);
            if mode == 1 {
                std::future::pending::<()>().await;
            }
            if mode == 4 {
                return ControlFlow::Break(Err(crate::build_runtime_error(
                    "selected source failed",
                )
                .with_identifier("RunMat:DynamicControl396:Selected")
                .build()));
            }
            if mode == 5 {
                return ControlFlow::Break(
                    RootedValueSequence::new(runmat_value::ValueSequence::empty())
                        .map_err(|error| error.into_runtime_error()),
                );
            }
            ControlFlow::Continue(arguments)
        })
        .unwrap();
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        if mode == 1 {
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            assert!(future.as_mut().poll(&mut context).is_pending());
            drop(future);
            assert_eq!(polls.get(), 1);
        } else {
            if mode == 2 {
                assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
            }
            if mode == 3 {
                ledger.revoke();
            }
            match futures::executor::block_on(await_dynamic_function_load(Ok(future))) {
                ControlFlow::Continue(arguments) if mode == 0 => {
                    let runmat_value::Value::String(value) = &arguments[0] else {
                        unreachable!()
                    };
                    assert_eq!(
                        value.as_str().expect("nonmissing text").as_ptr(),
                        pointer,
                        "a miss moves the original owner without cloning"
                    );
                }
                ControlFlow::Break(Err(error)) if mode == 2 => {
                    assert_eq!(error.identifier(), Some("DynamicControl396:Cancelled"));
                    assert_eq!(polls.get(), 0);
                }
                ControlFlow::Break(Err(_)) if mode == 3 => assert_eq!(polls.get(), 0),
                ControlFlow::Break(Err(error)) if mode == 4 => assert_eq!(
                    error.identifier(),
                    Some("RunMat:DynamicControl396:Selected")
                ),
                ControlFlow::Break(Ok(output)) if mode == 5 => assert!(output.value().is_empty()),
                _ => panic!("dynamic control changed its selected/miss/error outcome"),
            }
        }
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
