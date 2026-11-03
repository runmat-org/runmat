use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use std::{cell::Cell, pin::Pin, rc::Rc, task::Poll};

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

struct Body {
    ledger: ManagedHostEffectLedger,
    dropped: Rc<Cell<bool>>,
    storage: [u8; 4096],
}
impl Future for Body {
    type Output = BuiltinResult<RootedValue>;
    fn poll(self: Pin<&mut Self>, _: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        assert_eq!(self.storage[0], 17);
        Poll::Pending
    }
}
impl Drop for Body {
    fn drop(&mut self) {
        let snapshot = self.ledger.snapshot();
        assert!(snapshot.live_claimed_bytes + snapshot.live_reserved_bytes > 0);
        self.dropped.set(true);
    }
}

#[test]
fn selected_body_is_small_outside_its_original_funded_owner() {
    for budget in [0, 1_000_000] {
        let ledger = ManagedHostEffectLedger::new_with_epoch(budget, 296);
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
                );
        let constructed = Cell::new(false);
        let dropped = Rc::new(Cell::new(false));
        let inputs =
            crate::execution::RootedValueLease::for_values(std::iter::empty::<&Value>()).unwrap();
        let future = admit_selected(Some(&runtime), inputs, || {
            assert!(!constructed.replace(true), "construct only the chosen body");
            assert!(ledger.snapshot().live_reserved_bytes > 0);
            Body {
                ledger: ledger.clone(),
                dropped: dropped.clone(),
                storage: [17; 4096],
            }
        });
        assert!(std::mem::size_of_val(&future) < std::mem::size_of::<Body>());
        assert_eq!(constructed.get(), budget != 0);
        drop(future);
        assert_eq!(dropped.get(), budget != 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
