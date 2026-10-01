use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostNumericTensorView, HostTensorView, ThreadProviderGuard,
};
use runmat_value::Tensor;
use std::sync::Mutex;

struct SiblingAliasProvider {
    owner: TestAllocationOwner,
    first: Mutex<Option<GpuTensorHandle>>,
    probe: Mutex<Option<RetirementProbe>>,
}

impl AccelProvider for SiblingAliasProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "normalization sibling alias fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("normalization must use typed upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("output restoration must not download") })
    }

    fn upload_numeric_raw(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        let mut first = self.first.lock().unwrap();
        if let Some(handle) = first.as_ref() {
            return Ok(handle.clone());
        }
        let HostNumericDataView::F64(data) = host.data else {
            panic!("fixture expects native double data")
        };
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(data.to_vec()),
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        *self.probe.lock().unwrap() = Some(allocation.probe);
        *first = Some(allocation.handle.clone());
        Ok(allocation.handle)
    }
}

#[test]
fn sibling_alias_is_rejected_before_any_output_is_published() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(SiblingAliasProvider {
        owner: TestAllocationOwner::new(),
        first: Mutex::new(None),
        probe: Mutex::new(None),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![4.0]),
            shape: vec![1, 1],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    let scalar = |value| Value::Tensor(Tensor::new(vec![value], vec![1, 1]).unwrap());
    let plan = NormalizeOutputPlan {
        single: false,
        source: Some(source.handle.clone()),
        requested: RequestedOutputs::Explicit(2),
    };
    let error = plan
        .finish(NormalizeEval {
            n: scalar(1.0),
            c: scalar(4.0),
            s: scalar(2.0),
        })
        .unwrap_err();
    assert!(error.message().contains("alias"));
    assert_eq!(
        error.identifier(),
        runmat_builtins::NORMALIZE_ERROR_INTERNAL.identifier
    );
    assert_eq!(error.context.builtin.as_deref(), Some("normalize"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    source.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(source.probe.retirements(), 0);
    let probe = provider.probe.lock().unwrap().clone().unwrap();
    // The fixture's retained reference remains valid; both invocation-owned
    // references must have been released, leaving it as the final owner.
    assert_eq!(probe.retirements(), 0);
    drop(provider.first.lock().unwrap().take());
    assert_eq!(probe.retirements(), 1);
    drop(source.handle);
    assert_eq!(source.probe.retirements(), 1);
}
