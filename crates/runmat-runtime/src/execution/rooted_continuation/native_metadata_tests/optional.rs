//! Optional owned and borrowed layouts share one read-only admission guard.
use super::*;
struct ReentrantPolicy {
    ledger: ManagedHostEffectLedger,
    reads: Rc<Cell<usize>>,
}
impl RuntimeParallelService for ReentrantPolicy {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        crate::context::with_original_scope(None, || self.reads.set(self.reads.get() + 1));
        let snapshot = self.ledger.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        crate::context::with_original_scope(None, || self.reads.set(self.reads.get() + 1));
        Some(self.ledger.clone())
    }
}
pub(super) fn check() {
    let (managed, ledger) = runtime(1_000_000);
    let reads = Rc::new(Cell::new(0));
    let managed = managed.with_service_ports(RuntimeServicePorts::default().with_parallel(
        Rc::new(ReentrantPolicy {
            ledger: ledger.clone(),
            reads: Rc::clone(&reads),
        }),
    ));
    let effects = Cell::new(false);
    let active = managed.enter();
    let rejected =
        admit_rooted_future_with(None, "owned absent", semantic_control_error, |context| {
            assert!(context.is_none());
            effects.set(true);
            std::future::ready(Ok::<(), RuntimeError>(()))
        });
    assert!(matches!(
        &rejected.body,
        ContinuationBody::Rejected(Some(_))
    ));
    drop(rejected);
    let rejected = admit_rooted_future(None, "existing absent", semantic_control_error, async {
        effects.set(true);
        Ok::<(), RuntimeError>(())
    });
    assert!(matches!(
        &rejected.body,
        ContinuationBody::Rejected(Some(_))
    ));
    drop(rejected);
    assert!(
        !effects.get(),
        "owned None rejects before factory or child polling"
    );
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(
        reads.get(),
        4,
        "external policy reads run outside the TLS borrow"
    );
    // A real unselected scope masks the managed ambient owner, rather than
    // borrowing its grant. The selected factory still receives exactly None.
    let masked = crate::context::with_original_scope(None, || {
        admit_rooted_future_with(None, "masked ordinary", semantic_control_error, |context| {
            assert!(context.is_none());
            assert!(crate::context::legacy::active().is_none());
            std::future::ready(Ok::<(), RuntimeError>(()))
        })
    });
    assert!(matches!(
        &masked.body,
        ContinuationBody::Running {
            claim: None,
            cancellation: None,
            ..
        }
    ));
    futures::executor::block_on(masked).unwrap();
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    drop(active);
    // Initial cancellation prevents construction; later cancellation retains
    // the original namespace before first poll, after Pending, and at Ready.
    for owned in [false, true] {
        for mode in 0..4 {
            let source = runmat_execution::cancellation::CancellationSource::new();
            let original =
                RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                    .with_cancellation_token(source.token());
            original.set_error_namespace("OptionalLayoutGuard");
            if mode == 0 {
                assert!(source.cancel(runmat_execution::CancellationReason::User));
            }
            let constructed = Rc::new(Cell::new(false));
            let observed = Rc::clone(&constructed);
            let polls = Rc::new(Cell::new(0));
            let observed_polls = Rc::clone(&polls);
            let inside = source.clone();
            let make = move || {
                observed.set(true);
                async move {
                    observed_polls.set(observed_polls.get() + 1);
                    if mode == 2 {
                        std::future::pending::<()>().await;
                    }
                    if mode == 3 {
                        assert!(inside.cancel(runmat_execution::CancellationReason::Deadline));
                    }
                    Ok::<(), RuntimeError>(())
                }
            };
            let entry = original.enter();
            let admitted = if owned {
                Ok(admit_rooted_future_with(
                    None,
                    "owned cancellation",
                    semantic_control_error,
                    |context| {
                        assert!(context.is_none());
                        make()
                    },
                ))
            } else {
                admit_optional_hostcall_with_contextual_error(None, make)
            };
            drop(entry);
            if mode == 0 {
                assert!(!constructed.get());
            } else {
                assert!(constructed.get());
            }
            let (foreign, _) = runtime(1_000_000);
            foreign.set_error_namespace("ForeignOptionalLayout");
            let _foreign = foreign.enter();
            let error = match admitted {
                Err(error) => {
                    assert_eq!(mode, 0);
                    error
                }
                Ok(mut future) => {
                    let mut context = Context::from_waker(futures::task::noop_waker_ref());
                    if mode == 2 {
                        assert!(future.as_mut().poll(&mut context).is_pending());
                    }
                    if mode == 1 || mode == 2 {
                        assert!(source.cancel(runmat_execution::CancellationReason::User));
                    }
                    let Poll::Ready(Err(error)) = future.as_mut().poll(&mut context) else {
                        panic!("original optional cancellation must prevent publication");
                    };
                    drop(future);
                    error
                }
            };
            assert_eq!(error.identifier(), Some("OptionalLayoutGuard:Cancelled"));
            assert!(matches!(
                std::error::Error::source(&error).and_then(|source| {
                    source.downcast_ref::<crate::execution::ExecutionServiceError>()
                }),
                Some(crate::execution::ExecutionServiceError::Cancelled)
            ));
            assert_eq!(polls.get(), usize::from(mode == 2 || mode == 3));
            assert!(!foreign.is_cancelled());
        }
    }
}
