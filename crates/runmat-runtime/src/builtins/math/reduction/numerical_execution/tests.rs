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
fn finite_array_paths_refuse_before_unowned_work_but_scalar_controls_remain_legal() {
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
        )
        .with_cancellation_token(cancellation.token());
    for kind in [
        NumericalReductionKind::Product,
        NumericalReductionKind::CountNonzero,
        NumericalReductionKind::Difference,
        NumericalReductionKind::Gradient,
        NumericalReductionKind::Trapezoid,
        NumericalReductionKind::CumulativeTrapezoid,
        NumericalReductionKind::NonNanMaximum,
    ] {
        let error = host(kind, Some(&runtime)).unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        host(kind, None).unwrap();
    }
    let result = futures::executor::block_on(crate::dispatcher::call_builtin_async_in_context(
        &runtime,
        "prod",
        &[Value::Num(3.0)],
    ))
    .unwrap();
    assert_eq!(result, Value::Num(3.0));
    let result = futures::executor::block_on(crate::dispatcher::call_builtin_async_in_context(
        &runtime,
        "nnz",
        &[Value::Int(runmat_value::IntValue::U64(u64::MAX))],
    ))
    .unwrap();
    assert_eq!(result, Value::Num(1.0));
    let call = NumericalCall::new(NumericalReductionKind::Product, Some(&runtime)).unwrap();
    cancellation.cancel(runmat_execution::CancellationReason::User);
    assert_eq!(
        call.check().unwrap_err().gpu_gather_retry,
        crate::GpuGatherRetry::Never
    );
}

#[test]
fn shared_optional_attempt_boundary_preserves_no_attempt_success_and_typed_failure() {
    let kind = NumericalReductionKind::Product;
    assert_eq!(completed::<usize>(kind, Ok(None)).unwrap(), None);
    assert_eq!(completed(kind, Ok(Some(7))).unwrap(), Some(7));
    let cause = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "provider execution witness");
    let error = completed::<usize>(
        kind,
        Err(runmat_accelerate_api::ProviderExecutionError::from(
            anyhow::Error::new(cause),
        )),
    )
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let mut source: &(dyn std::error::Error + 'static) = &error;
    while !source.is::<std::io::Error>() {
        source = source.source().expect("typed provider source retained");
    }
    assert_eq!(
        source.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::BrokenPipe
    );
}
