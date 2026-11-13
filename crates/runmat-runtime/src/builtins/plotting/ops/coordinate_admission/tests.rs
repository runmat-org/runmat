use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use std::rc::Rc;

struct Finite(ManagedHostEffectLedger);
impl RuntimeParallelService for Finite {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let state = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(state.free_bytes()),
            epoch: state.epoch,
            ..RuntimeParallelResources::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

#[test]
fn finite_state_has_no_unbounded_escape_and_cancelled_completion_cannot_publish() {
    let ledger = ManagedHostEffectLedger::new(4096);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
        );
    let error = CoordinateWork::admit(Some(&runtime), "scatter")
        .err()
        .unwrap();
    assert!(matches!(
        error.gpu_gather_retry,
        crate::GpuGatherRetry::Never
    ));
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let source = CancellationSource::new();
    let ordinary = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(source.token());
    let work = CoordinateWork::admit(Some(&ordinary), "line").unwrap();
    source.cancel(CancellationReason::User);
    let error = work.finish(Ok(17.0)).unwrap_err();
    assert!(matches!(
        error.gpu_gather_retry,
        crate::GpuGatherRetry::Never
    ));
    assert!(error.identifier().unwrap().contains("Cancelled"));
}
