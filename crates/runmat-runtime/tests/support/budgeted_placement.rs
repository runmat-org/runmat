//! Finite session setup and invocation capacity are separate actual owners.
//! No route, duration, observation, or capability is supplied by this fixture.
use runmat_accelerate::placement::{ManagedPlacementCapacity, PlacementSession};
use runmat_runtime::context::{
    host_allocation::HostAllocationError,
    host_effect::{HostEffectAllocationError, ManagedHostEffectLedger},
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use std::{error::Error, rc::Rc};

struct HostBudget(ManagedHostEffectLedger);
impl RuntimeParallelService for HostBudget {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

pub fn session() -> Rc<PlacementSession> {
    let setup = ManagedHostEffectLedger::new(1_048_576);
    Rc::new(
        PlacementSession::new_with_managed_observations(
            Default::default(),
            &setup,
            ManagedPlacementCapacity {
                in_flight_slots: 8,
                feedback_series: 16,
                signature_bytes_per_series: 16_384,
            },
        )
        .expect("bounded session observation/profile backing"),
    )
}

pub fn context(session: Rc<PlacementSession>, host_bytes: u64) -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(session)
            .with_parallel(Rc::new(HostBudget(ManagedHostEffectLedger::new(
                host_bytes,
            )))),
    )
}

pub fn assert_capacity_denied(error: &runmat_runtime::RuntimeError) {
    let mut cause: &(dyn Error + 'static) = error;
    loop {
        if let Some(allocation) = cause.downcast_ref::<HostAllocationError>() {
            assert_eq!(
                *allocation,
                HostAllocationError::Resource(HostEffectAllocationError::ExceedsCapacity)
            );
            return;
        }
        cause = cause
            .source()
            .unwrap_or_else(|| panic!("missing exact capacity cause: {error:?}"));
    }
}
