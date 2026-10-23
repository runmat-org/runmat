use super::*;
use crate::context::{
    host_allocation::HostTypedWorkspace, host_effect::ManagedHostEffectLedger, ParallelCapability,
    RuntimeContext, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use std::{
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

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
            ..RuntimeParallelResources::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
fn context(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
        )
}
struct Provider {
    owner: TestAllocationOwner,
    ledger: ManagedHostEffectLedger,
    cancellation: Option<CancellationSource>,
    calls: AtomicUsize,
    outputs: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "moving admission witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no unauthorized gather") })
    }
    fn moving_window<'a>(
        &'a self,
        request: &'a ProviderMovingWindowRequest<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(request.output_shape, &[3, 1]);
            assert_eq!((request.dim, request.before, request.after), (0, 1, 1));
            assert!(
                self.ledger.snapshot().live_claimed_bytes
                    >= HostTypedWorkspace::<usize>::quoted_bytes(2).unwrap()
            );
            let output = self.owner.allocate_zeroed(
                request.output_shape.to_vec(),
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            self.outputs.lock().unwrap().push(output.probe);
            if let Some(source) = &self.cancellation {
                source.cancel(CancellationReason::User);
            }
            Ok(output.handle)
        })
    }
}
fn provider(
    ledger: ManagedHostEffectLedger,
    cancellation: Option<CancellationSource>,
) -> &'static Provider {
    Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        ledger,
        cancellation,
        calls: AtomicUsize::new(0),
        outputs: Mutex::new(Vec::new()),
    }))
}
fn parsed() -> ParsedMoving {
    ParsedMoving {
        window: WindowSpec::Counts {
            before: 1,
            after: 1,
        },
        dim: Some(1),
        nan_mode: NanMode::Include,
        endpoints: Endpoints::Shrink,
        sample_points: None,
        normalization: VarianceNormalization::Sample,
    }
}

#[test]
fn finite_resident_request_funds_shape_and_retires_cancelled_output_exactly_once() {
    for cancel in [false, true] {
        let ledger = ManagedHostEffectLedger::new(65_536);
        let cancellation = CancellationSource::new();
        let context = context(&ledger).with_cancellation_token(cancellation.token());
        let provider = provider(ledger.clone(), cancel.then_some(cancellation));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let work = admission::MovingCall::new(Some(&context), "movsum").unwrap();
        let result = futures::executor::block_on(provider::moving_gpu(
            &work,
            "movsum",
            MovingOp::Sum,
            input.handle.clone(),
            &parsed(),
        ));
        if cancel {
            let error = result.unwrap_err();
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert!(error.identifier.unwrap().ends_with(":Cancelled"));
        } else {
            assert!(matches!(result.unwrap(), Value::GpuTensor(_)));
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}

#[test]
fn known_precision_restriction_and_explicit_intent_fail_before_provider_or_download() {
    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = context(&ledger);
    let provider = provider(ledger.clone(), None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![3, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    for runtime in [None, Some(&context)] {
        let work = admission::MovingCall::new(runtime, "movmean").unwrap();
        let handle = input
            .handle
            .clone()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let error = futures::executor::block_on(provider::moving_gpu(
            &work,
            "movmean",
            MovingOp::Mean,
            handle,
            &parsed(),
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(input.probe.retirements(), 0);
}

#[test]
fn finite_host_fence_precedes_argument_projection_and_retains_original_generation() {
    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = context(&ledger);
    let args = crate::builtins::common::test_support::builtin_arguments(vec![
        Value::Num(1.0),
        Value::Num(3.0),
    ]);
    let error = futures::executor::block_on(dispatch::moving_builtin(
        Some(&context),
        "movsum",
        MovingOp::Sum,
        args,
    ))
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    let old = CancellationSource::new();
    let ordinary = RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        old.token(),
    );
    let work = admission::MovingCall::new(Some(&ordinary), "movsum").unwrap();
    let successor = CancellationSource::new();
    let _scope = ordinary.scoped_cancellation_token(successor.token());
    old.cancel(CancellationReason::User);
    assert_eq!(
        work.check().unwrap_err().gpu_gather_retry,
        crate::GpuGatherRetry::Never
    );
}
