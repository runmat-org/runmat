use super::*;
use crate::fusion::{FusionGroup, FusionKernelSpec};
use crate::graph::InstrSpan;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, HostNumericTensorView, HostTensorView, ProviderFeasibility,
    ProviderFeasibilityQuery, ThreadProviderGuard,
};
use std::future::Future;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

struct FailingProvider {
    allocations: TestAllocationOwner,
    fail_upload_at: usize,
    uploads: AtomicUsize,
    dispatches: AtomicUsize,
    suspend_dispatch: AtomicBool,
    probes: Mutex<Vec<RetirementProbe>>,
}

impl FailingProvider {
    fn new(fail_upload_at: usize) -> &'static Self {
        Box::leak(Box::new(Self {
            allocations: TestAllocationOwner::new(),
            fail_upload_at,
            uploads: AtomicUsize::new(0),
            dispatches: AtomicUsize::new(0),
            suspend_dispatch: AtomicBool::new(false),
            probes: Mutex::new(Vec::new()),
        }))
    }
}

impl AccelProvider for FailingProvider {
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn device_info(&self) -> String {
        "elementwise-effect-fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("numeric upload must use the selected typed hook")
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let ordinal = self.uploads.fetch_add(1, Ordering::SeqCst) + 1;
        if ordinal == self.fail_upload_at {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "injected upload unsupported",
            )
            .into());
        }
        let allocated = self.allocations.allocate_zeroed(
            host.shape.to_vec(),
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )?;
        self.probes.lock().unwrap().push(allocated.probe);
        Ok(allocated.handle)
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("failure witness must not read back")
    }

    fn query_feasibility(&self, _: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        ProviderFeasibility::supported(Default::default())
    }

    fn fused_elementwise<'a>(
        &'a self,
        _: &'a str,
        _: runmat_accelerate_api::ProviderShaderRequirements,
        _: &'a [GpuTensorHandle],
        _: &'a [usize],
        _: usize,
        _: ProviderElementType,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.suspend_dispatch.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            self.dispatches.fetch_add(1, Ordering::SeqCst);
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "injected dispatch unsupported",
            )
            .into())
        })
    }
}

fn plan() -> FusionGroupPlan {
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: Vec::new(),
            shape: ShapeInfo::Tensor(vec![Some(2)]),
            span: InstrSpan { start: 0, end: 2 },
            pattern: None,
            stack_layout: None,
        },
        operations: vec![crate::fusion::FusionOp::Primitive {
            op: crate::graph::PrimitiveOp::Add,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        inputs: vec![0, 1],
        stack_pattern: vec![0, 1],
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: Vec::new(),
        output: Some(2),
        kernel: FusionKernelSpec {
            kind: FusionKind::ElementwiseChain,
            supported: true,
        },
        reduction_data: None,
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: None,
        pattern: None,
    }
}

fn invoke(provider: &'static FailingProvider) -> ProviderExecutionError {
    runmat_gc::gc_test_context(|| {
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = plan();
        let tensor = || Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap());
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![tensor(), tensor()],
            placement: None,
            runtime: None,
        };
        match futures::executor::block_on(execute_elementwise_placed(request)) {
            Err(error) => error,
            Ok(_) => panic!("injected provider hook must fail, not select a continuation"),
        }
    })
}

fn assert_original_unsupported(error: &ProviderExecutionError) {
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if source.is::<runmat_accelerate_api::UnsupportedProviderOperation>() {
            return;
        }
        source = source
            .source()
            .expect("original typed unsupported provider cause retained");
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn first_unsupported_upload_remains_pre_effect() {
    let provider = FailingProvider::new(1);
    let error = invoke(provider);
    assert!(error.is_unsupported());
    assert_original_unsupported(&error);
    assert!(error.to_string().contains("injected upload unsupported"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
    assert!(provider.probes.lock().unwrap().is_empty());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn late_unsupported_upload_cannot_replay_after_first_upload() {
    let provider = FailingProvider::new(2);
    let error = invoke(provider);
    assert!(!error.is_unsupported());
    assert_original_unsupported(&error);
    assert!(error.to_string().contains("injected upload unsupported"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
    let probes = provider.probes.lock().unwrap();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unsupported_dispatch_cannot_replay_after_uploads() {
    let provider = FailingProvider::new(usize::MAX);
    let error = invoke(provider);
    assert!(!error.is_unsupported());
    assert_original_unsupported(&error);
    assert!(error.to_string().contains("injected dispatch unsupported"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 1);
    let probes = provider.probes.lock().unwrap();
    assert_eq!(probes.len(), 2);
    assert!(probes.iter().all(|probe| probe.retirements() == 1));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn fused_hook_does_not_dispatch_before_poll_or_after_unpolled_drop() {
    let provider = FailingProvider::new(usize::MAX);
    let inputs = [];
    let shape = [1, 1];
    let future = provider.fused_elementwise(
        "",
        runmat_accelerate_api::ProviderShaderRequirements::Basic,
        &inputs,
        &shape,
        1,
        ProviderElementType::F64,
    );
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
    drop(future);
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
    let error = futures::executor::block_on(provider.fused_elementwise(
        "",
        runmat_accelerate_api::ProviderShaderRequirements::Basic,
        &inputs,
        &shape,
        1,
        ProviderElementType::F64,
    ))
    .unwrap_err();
    assert!(error.is_unsupported());
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pending_fused_hook_cancel_preserves_borrowed_inputs() {
    runmat_gc::gc_test_context(|| {
        let provider = FailingProvider::new(usize::MAX);
        provider.suspend_dispatch.store(true, Ordering::SeqCst);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let left = provider
            .allocations
            .allocate_zeroed(
                vec![2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let right = provider
            .allocations
            .allocate_zeroed(
                vec![2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let source = plan();
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![
                Value::GpuTensor(left.handle.clone()),
                Value::GpuTensor(right.handle.clone()),
            ],
            placement: None,
            runtime: None,
        };
        let mut future = Box::pin(execute_elementwise_placed(request));
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
        left.handle.allocation_lease().validate_live().unwrap();
        right.handle.allocation_lease().validate_live().unwrap();
        drop(future);
        left.handle.allocation_lease().validate_live().unwrap();
        right.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(left.probe.retirements(), 0);
        assert_eq!(right.probe.retirements(), 0);
    });
}
