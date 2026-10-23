use super::*;
use crate::{
    context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
        RuntimeParallelService, RuntimeServicePorts,
    },
    execution::RuntimeExecutionService,
};
use futures::executor::block_on;
use std::rc::Rc;
struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
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
#[test]
fn numeric_state_control_remains_legal_but_unowned_text_and_output_errors_never_mutate() {
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    );
    let old = current_digits();
    struct Reset(usize);
    impl Drop for Reset {
        fn drop(&mut self) {
            set_current_digits_for_test(self.0);
        }
    }
    let _reset = Reset(old);
    let args = BuiltinArguments::from_borrowed(&[Value::Num(50.)], Some(&runtime)).unwrap();
    let peak = ledger.snapshot().peak_claimed_bytes;
    assert_eq!(
        block_on(digits_builtin(Some(runtime.clone()), args)).unwrap(),
        Value::Num(old as f64)
    );
    assert_eq!(current_digits(), 50);
    assert_eq!(ledger.snapshot().peak_claimed_bytes, peak);
    let args = BuiltinArguments::from_borrowed(&[Value::from("default")], Some(&runtime)).unwrap();
    let error = block_on(digits_builtin(Some(runtime.clone()), args)).unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(current_digits(), 50);
    let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
    let args = BuiltinArguments::from_borrowed(&[Value::Num(12.)], Some(&runtime)).unwrap();
    assert_eq!(
        block_on(digits_builtin(Some(runtime.clone()), args))
            .unwrap_err()
            .identifier(),
        Some("RunMat:digits:TooManyOutputs")
    );
    assert_eq!(current_digits(), 50);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
