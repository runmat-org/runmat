//! Actual captured Runtime admission through source preparation and Metal.
use super::*;
use crate::fusion_exec::{execute_reduction_with_shape_placed_async, FusionExecutionOutcome};
use crate::graph::FusionCallOrigin;
use runmat_accelerate_api::{NumericElementType, ThreadProviderGuard};
use runmat_execution::cancellation::CancellationSource;
use runmat_runtime::context::host_effect::ManagedHostEffectLedger;
use runmat_runtime::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use std::rc::Rc;

pub(super) struct Managed(pub(super) ManagedHostEffectLedger);
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

fn plan() -> FusionGroupPlan {
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::Reduction,
            nodes: vec![],
            shape: ShapeInfo::Tensor(vec![Some(2), Some(3)]),
            span: InstrSpan { start: 0, end: 2 },
            pattern: None,
            stack_layout: None,
        },
        operations: vec![
            FusionOp::Builtin {
                name: "power".into(),
                origin: FusionCallOrigin::named("power"),
                inputs: vec![0, 3],
                output: Some(4),
            },
            FusionOp::Builtin {
                name: "sum".into(),
                origin: FusionCallOrigin::named("sum"),
                inputs: vec![4, 1],
                output: Some(2),
            },
        ],
        inputs: vec![0],
        stack_pattern: vec![0],
        constants: HashMap::new(),
        const_values: HashMap::from([(1, Value::Num(3.0)), (3, Value::Num(2.0))]),
        materialized_stores: vec![],
        output: Some(2),
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(4),
        reduction_dim: None,
        reduction_flavor: Some(runmat_accelerate_api::ReductionFlavor::Sum),
        reduction_axes: None,
        pattern: None,
    }
}

#[test]
#[ignore = "requires actual WGPU adapter; unavailable is failure"]
fn captured_runtime_budget_reaches_real_typed_reduction() {
    runmat_gc::gc_test_context(|| {
        let provider = owner();
        let _selected = ThreadProviderGuard::set(Some(provider));
        let plan = plan();
        // Public explicit-device intent, not an artificial timing override.
        // Upload precedes the managed call and is part of its input baseline.
        let source = provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[1., 2., 3., 4., 5., 6.]),
                shape: &[2, 3],
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let input = || Value::GpuTensor(source.clone());
        let execute = |ledger: &ManagedHostEffectLedger, cancelled: bool| {
            let cancellation = CancellationSource::new();
            if cancelled {
                cancellation.cancel(runmat_execution::CancellationReason::User);
            }
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
            )
            .with_cancellation_token(cancellation.token());
            block_on(execute_reduction_with_shape_placed_async(
                FusionExecutionRequest {
                    plan: &plan,
                    inputs: vec![input()],
                    placement: None,
                    runtime: Some(runtime),
                },
                1,
                6,
                &[2, 3],
                8,
            ))
        };
        for (budget, cancelled) in [(1, false), (16 * 1024 * 1024, true)] {
            let ledger = ManagedHostEffectLedger::new(budget);
            let allocations = provider.allocations.record_count().unwrap();
            let retained = provider.allocations.retained_bytes();
            let pipelines = provider.fused_pipeline_cache.lock().unwrap().len();
            let metrics = provider.metrics.counters();
            let error = execute(&ledger, cancelled)
                .err()
                .expect("original admission must refuse before effects");
            assert!(!error.is_after_effect());
            assert!(!error.is_unsupported());
            assert_eq!(provider.allocations.record_count().unwrap(), allocations);
            assert_eq!(provider.allocations.retained_bytes(), retained);
            source.allocation_lease().validate_live().unwrap();
            assert_eq!(
                provider.fused_pipeline_cache.lock().unwrap().len(),
                pipelines
            );
            assert_eq!(provider.metrics.counters(), metrics);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
        let ledger = ManagedHostEffectLedger::new(16 * 1024 * 1024);
        let output = match execute(&ledger, false).expect("finite source and real GPU execution") {
            FusionExecutionOutcome::Executed(Value::GpuTensor(output)) => output,
            _ => panic!("supported identity sum must execute through its typed GPU recipe"),
        };
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&output),
            Some(GpuHandleProvenance::Explicit)
        );
        assert_eq!(output.shape(), [2, 3]);
        let downloaded = block_on(provider.download_numeric(&output)).unwrap();
        assert_eq!(downloaded.shape, [2, 3]);
        assert_eq!(
            downloaded.data,
            HostNumericDataOwned::F32(vec![1., 4., 9., 16., 25., 36.])
        );
        let snapshot = ledger.snapshot();
        assert!(snapshot.peak_reserved_bytes > 0);
        assert!(snapshot.peak_reserved_bytes <= snapshot.capacity_bytes);
        drop(output);
        source.release().unwrap();
    });
}
