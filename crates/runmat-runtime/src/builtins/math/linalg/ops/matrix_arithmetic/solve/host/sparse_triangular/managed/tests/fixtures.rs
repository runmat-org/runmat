//! Original managed-ledger context; factorization must remain unavailable.
use super::*;

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
struct NoLu;
impl RuntimeBlockingCpuService for NoLu {
    fn submit(
        &self,
        _: Option<&RuntimeContext>,
        _: OwnedCpuJob,
    ) -> crate::execution::OwnedCpuSubmission {
        panic!("a triangular solve must not factor through LU/COLAMD")
    }
}
pub(super) fn context(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default()
            .with_parallel(Rc::new(Parallel(ledger.clone())))
            .with_blocking_cpu(Rc::new(NoLu)),
    )
}
