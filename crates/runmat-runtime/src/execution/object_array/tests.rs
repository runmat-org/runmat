use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            cpu_millicores_available: 1000,
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
fn runtime(capacity: u64) -> (RuntimeContext, ManagedHostEffectLedger, CancellationSource) {
    let ledger = ManagedHostEffectLedger::new_with_epoch(capacity, 1);
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    );
    (runtime, ledger, cancellation)
}
fn source() -> ObjectArray {
    ObjectArray::new(
        "ObjectArrayFixture",
        vec![Value::Object(runmat_value::ObjectInstance::new(
            "ObjectArrayFixture",
        ))],
        vec![1, 1],
    )
    .unwrap()
}
#[test]
fn admitted_slots_and_original_reshape_owners_have_exact_lifetimes() {
    let (runtime, ledger, _) = runtime(1 << 20);
    let source = source();
    let pointer = source.data().as_ptr();
    let output = copy_source(&source, Some(&runtime)).unwrap();
    let Value::ObjectArray(output_array) = output.value() else {
        panic!("array output")
    };
    assert_ne!(output_array.data().as_ptr(), pointer);
    assert_eq!(output_array.shape().as_ptr(), source.shape().as_ptr());
    assert_eq!(output_array.data(), source.data());
    let reframed = reshape(source.clone(), &[1, 1, 1], Some(&runtime)).unwrap();
    assert_eq!(reframed.data().as_ptr(), pointer);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop((output, reframed));
    assert!(
        ledger.snapshot().live_reserved_bytes > 0,
        "original alias retains promoted owners"
    );
    drop(source);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
#[test]
fn denied_and_cancelled_destinations_do_not_publish_or_leak() {
    let source = source();
    let (denied, ledger, _) = runtime(1);
    assert!(copy_source(&source, Some(&denied)).is_err());
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let (runtime, ledger, cancellation) = runtime(1 << 20);
    cancellation.cancel(CancellationReason::User);
    let error = copy_source(&source, Some(&runtime)).unwrap_err();
    assert!(error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
