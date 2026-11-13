use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_value::MException;
use std::cell::Cell;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

struct ManagedControl(runmat_runtime::context::host_effect::ManagedHostEffectLedger);

impl runmat_runtime::context::RuntimeParallelService for ManagedControl {
    fn supports(&self, _: runmat_runtime::context::ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> runmat_runtime::context::RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        runmat_runtime::context::RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(
        &self,
    ) -> Option<runmat_runtime::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

struct PendingCatchProbe {
    runtime: runmat_runtime::context::RuntimeContext,
    polls: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
}

impl Future for PendingCatchProbe {
    type Output = NativeExecutorResult<CallCompletion>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        assert_eq!(
            self.runtime.active_catch_exception().unwrap().identifier,
            "RunMat:native"
        );
        self.polls.set(self.polls.get() + 1);
        Poll::Pending
    }
}

impl Drop for PendingCatchProbe {
    fn drop(&mut self) {
        assert_eq!(
            self.runtime.active_catch_exception().unwrap().identifier,
            "RunMat:native"
        );
        self.drops.set(self.drops.get() + 1);
    }
}

#[test]
fn pending_native_call_scopes_each_poll_and_drop() {
    runmat_gc::gc_test_context(|| {
        for capacity in [0, 1 << 20] {
            let mut state = observed_state(&Rc::new(Probe::default()));
            state.function.blocks[0].catch_scopes = vec![runmat_mir::CatchScopeId(0)];
            let entry = state.function.entry;
            state
                .bind_catch_exception(
                    entry,
                    MException::new("RunMat:native".into(), "native".into()),
                )
                .unwrap();
            state.enter_site_block(entry);
            assert_eq!(
                state.active_catch_exception().unwrap().identifier,
                "RunMat:native"
            );
            let polls = Rc::new(Cell::new(0));
            let drops = Rc::new(Cell::new(0));
            let ledger =
                runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(capacity);
            state.runtime = state.runtime.clone().with_service_ports(
                state
                    .runtime
                    .service_ports()
                    .clone()
                    .with_parallel(Rc::new(ManagedControl(ledger.clone()))),
            );
            let factories = Cell::new(0);
            let future = admit_completion_with(&state, || {
                factories.set(factories.get() + 1);
                PendingCatchProbe {
                    runtime: state.runtime.clone(),
                    polls: polls.clone(),
                    drops: drops.clone(),
                }
            });
            if capacity == 0 {
                assert!(future.is_err());
                assert_eq!(
                    factories.get(),
                    0,
                    "refusal precedes concrete child construction"
                );
                assert_eq!(drops.get(), 0);
                assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
                assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
                continue;
            }
            let mut future = future.unwrap();
            assert_eq!(factories.get(), 1);
            assert!(ledger.snapshot().live_reserved_bytes > 0);
            let waker = futures::task::noop_waker();
            let mut context = Context::from_waker(&waker);
            assert!(future.as_mut().poll(&mut context).is_pending());
            assert!(state.runtime.active_catch_exception().is_none());
            drop(future);
            assert_eq!(polls.get(), 1);
            assert_eq!(drops.get(), 1);
            assert!(state.runtime.active_catch_exception().is_none());
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

            // The same original reservation funds two sequential physical
            // payload blocks. Its first claim must retire before Ready allocates.
            let reservation = terminator::prepare_payload(&state.runtime).unwrap();
            let reserved = ledger.snapshot();
            let completion = terminator::completion(
                &state.runtime,
                reservation,
                TerminatorCompletion::BranchTruth(true),
            )
            .unwrap();
            let CallCompletion::Terminator(completion) = completion else {
                panic!("typed terminator completion");
            };
            let first = ledger.snapshot();
            assert!(first.live_claimed_bytes > reserved.live_claimed_bytes);
            assert_eq!(first.live_reserved_bytes, reserved.live_reserved_bytes);
            let pending = completion.into_inner();
            assert_eq!(
                ledger.snapshot().live_claimed_bytes,
                reserved.live_claimed_bytes
            );
            let ready = pending.finish(&state.runtime).unwrap();
            assert!(ledger.snapshot().live_claimed_bytes > reserved.live_claimed_bytes);
            // Publication releases the receipt but not the Ready block's claim.
            assert!(matches!(
                ready.into_inner().into_parts(),
                (TerminatorReadyStage::BranchTruth(true), None)
            ));
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        }
    });
}

#[test]
fn denied_for_next_keeps_the_original_loop_owner_in_place() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let header = runmat_native_codegen::NativeBlockId(0);
        let iterator = futures::executor::block_on(
            runmat_runtime::iteration::ForColumnIterator::new(Value::Num(7.0)),
        )
        .unwrap();
        state.start_for_loop(header, header, iterator);
        state.current_request = Some(runmat_runtime::native::NativeSiteRequest {
            function: 0,
            block: 0,
            position: 0,
            phase: runmat_runtime::native::NativeSitePhase::TERMINATOR,
            ordinal: 0,
            reserved: 0,
        });
        let ledger = runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(0);
        state.runtime = state.runtime.clone().with_service_ports(
            state
                .runtime
                .service_ports()
                .clone()
                .with_parallel(Rc::new(ManagedControl(ledger.clone()))),
        );
        let error = match begin_for_next(&mut state, header) {
            Err(NativeExecutorError::Runtime(error)) => error,
            _ => panic!("zero budget must refuse the original continuation allocation"),
        };
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        loop {
            if let Some(allocation) = cause
                .downcast_ref::<runmat_runtime::context::host_allocation::HostAllocationError>(
            ) {
                assert!(allocation.is_capacity_denial());
                break;
            }
            cause = cause.source().expect("original typed allocation refusal");
        }
        assert!(state.pending_call.is_none());
        assert!(state.has_for_loop(header));
        let mut active = state.take_for_loop(header).unwrap();
        let value = futures::executor::block_on(active.iterator.next())
            .unwrap()
            .unwrap();
        assert_eq!(
            value.value(),
            &Value::Num(7.0),
            "refusal did not advance the original iterator"
        );
        drop((value, active));
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    });
}
