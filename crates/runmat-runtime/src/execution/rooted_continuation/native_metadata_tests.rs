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
    let ledger = ManagedHostEffectLedger::new_with_epoch(budget, 296);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
        );
    (runtime, ledger)
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
fn native_concrete_future_is_admitted_before_factory_and_refunded_on_abandonment() {
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
        let (admitted, ledger) = runtime(1_000_000);
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
            admitted.cancellation().cancel();
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
}

#[test]
fn cancelled_native_admission_never_constructs_or_polls_child() {
    let (cancelled, ledger) = runtime(1_000_000);
    cancelled.cancellation().cancel();
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
}
