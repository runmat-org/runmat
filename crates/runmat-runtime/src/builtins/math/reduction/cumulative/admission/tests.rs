use super::*;
use crate::{
    context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
        RuntimeParallelService, RuntimeServicePorts,
    },
    execution::RuntimeExecutionService,
};
use std::rc::Rc;
struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let s = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(s.free_bytes()),
            epoch: s.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
#[test]
fn actual_managed_context_refuses_unowned_host_scan_before_claims() {
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    );
    for kind in [
        CumulativeKind::Sum,
        CumulativeKind::Product,
        CumulativeKind::Minimum,
        CumulativeKind::Maximum,
    ] {
        assert_eq!(
            host(kind, Some(&runtime)).unwrap_err().gpu_gather_retry,
            crate::GpuGatherRetry::Never
        );
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        host(kind, None).unwrap();
    }
}
