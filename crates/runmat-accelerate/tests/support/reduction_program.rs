//! Genuine allocation owner with an observable, suspendable domain probe.
use runmat_accelerate::fusion::{FusionGroupPlan, FusionKind, FusionOp};
use runmat_accelerate::fusion_exec::FusionExecutionRequest;
use runmat_accelerate_api::test_support::{TestAllocation, TestAllocationOwner};
use runmat_accelerate_api::*;
use runmat_value::Value;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub fn new_owner() -> &'static ProbeOwner {
    Box::leak(Box::new(ProbeOwner {
        owner: TestAllocationOwner::new(),
        calls: AtomicUsize::new(0),
        ready: AtomicBool::new(true),
        fail: AtomicBool::new(false),
    }))
}
pub struct ProbeOwner {
    owner: TestAllocationOwner,
    pub calls: AtomicUsize,
    pub ready: AtomicBool,
    pub fail: AtomicBool,
}
#[derive(Debug, thiserror::Error)]
#[error("actual reduction producer probe failure")]
pub struct ProbeFailure;
impl ProbeOwner {
    pub fn reset(&self) {
        self.calls.store(0, Ordering::SeqCst);
        self.ready.store(true, Ordering::SeqCst);
        self.fail.store(false, Ordering::SeqCst);
    }
    pub fn input(&self) -> TestAllocation {
        self.owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap()
    }
}
impl AccelProvider for ProbeOwner {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "typed-reduction-probe-owner".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("receipt cannot upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("receipt cannot download")
    }
    fn classify_real_power<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, PowerResultDomain> {
        Box::pin(async move {
            assert_eq!(left.device_id(), self.device_id());
            assert_eq!(right.device_id(), self.device_id());
            self.calls.fetch_add(1, Ordering::SeqCst);
            std::future::poll_fn(|_| {
                if self.ready.load(Ordering::SeqCst) {
                    std::task::Poll::Ready(())
                } else {
                    std::task::Poll::Pending
                }
            })
            .await;
            if self.fail.load(Ordering::SeqCst) {
                return Err(ProbeFailure.into());
            }
            // Both authentic fixture allocations contain zero; 0^0 is real.
            Ok(PowerResultDomain::Real)
        })
    }
}
pub fn builtin(name: &str, inputs: Vec<u32>, output: u32) -> FusionOp {
    FusionOp::Builtin {
        name: name.into(),
        origin: Default::default(),
        inputs,
        output: Some(output),
    }
}
pub fn reduction(mut plan: FusionGroupPlan, input_ids: Vec<u32>, data: u32) -> FusionGroupPlan {
    plan.group.kind = FusionKind::Reduction;
    plan.kernel.kind = FusionKind::Reduction;
    plan.inputs = input_ids;
    plan.reduction_data = Some(data);
    plan
}
pub fn request(plan: &FusionGroupPlan, inputs: Vec<Value>) -> FusionExecutionRequest<'_> {
    FusionExecutionRequest {
        plan,
        inputs,
        placement: None,
        runtime: None,
    }
}
