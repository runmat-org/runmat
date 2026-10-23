//! Explicit frontier for legacy tabular metadata/serialization scratch.
//! This is not a finite allocation grant or a substitute output owner.
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
        HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::BuiltinResult;
pub(in crate::builtins::table) struct UnboundedTabularWork<'a> {
    runtime: Option<&'a RuntimeContext>,
    builtin: &'static str,
}
impl<'a> UnboundedTabularWork<'a> {
    pub fn admit(
        runtime: Option<&'a RuntimeContext>,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        let work = Self { runtime, builtin };
        work.check()?;
        Ok(work)
    }
    pub fn check(&self) -> BuiltinResult<()> {
        let source = match HostAllocationReceipt::policy(self.runtime) {
            Ok(HostAllocationPolicy::Unbounded) => return Ok(()),
            Ok(HostAllocationPolicy::Managed) => HostAllocationError::MissingManagedOwner,
            Err(source) => source,
        };
        let mut error=host_allocation_runtime_error(self.runtime,"RunMat:table:MetadataWorkspace",
            format!("{}: tabular metadata/serialization workspace requires unrestricted allocation; {source}",self.builtin),source);
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        Err(error)
    }
    pub fn finish<T>(&self, result: BuiltinResult<T>) -> BuiltinResult<T> {
        let output = result?;
        self.check()?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
        RuntimeParallelService, RuntimeServicePorts,
    };
    use crate::execution::RuntimeExecutionService;
    use std::{cell::Cell, rc::Rc};
    struct Policy {
        finite: Cell<bool>,
        ledger: ManagedHostEffectLedger,
    }
    impl RuntimeParallelService for Policy {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let state = self.ledger.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: self.finite.get().then_some(state.free_bytes()),
                epoch: state.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.ledger.clone())
        }
    }
    #[test]
    fn finite_and_rotated_tabular_work_refuses_before_allocating_or_publishing() {
        let ledger = ManagedHostEffectLedger::new(65536);
        let policy = Rc::new(Policy {
            finite: Cell::new(false),
            ledger: ledger.clone(),
        });
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_parallel(policy.clone()));
        let work = UnboundedTabularWork::admit(Some(&runtime), "writetable").unwrap();
        assert_eq!(work.finish(Ok(7)).unwrap(), 7);
        let free = ledger.snapshot().free_bytes();
        policy.finite.set(true);
        for error in [
            work.check().unwrap_err(),
            work.finish(Ok(7)).unwrap_err(),
            UnboundedTabularWork::admit(Some(&runtime), "categorical")
                .err()
                .expect("finite refuses"),
        ] {
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert!(error.message().contains(
                "tabular metadata/serialization workspace requires unrestricted allocation;"
            ));
        }
        assert_eq!(ledger.snapshot().free_bytes(), free);
    }
}
