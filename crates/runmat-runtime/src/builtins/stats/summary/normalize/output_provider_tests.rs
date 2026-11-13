use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostNumericTensorView, HostTensorView, ThreadProviderGuard,
};
use runmat_value::{ComplexTensor, Tensor};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct SiblingAliasProvider {
    owner: TestAllocationOwner,
    retained: Mutex<Vec<GpuTensorHandle>>,
    probes: Mutex<Vec<RetirementProbe>>,
    uploads: AtomicUsize,
    alias_at: usize,
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

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let upload = self.uploads.fetch_add(1, Ordering::SeqCst);
        let mut retained = self.retained.lock().unwrap();
        if upload == self.alias_at {
            return Ok(retained.first().expect("prior output").clone());
        };
        let data = match host.data {
            HostNumericDataView::F64(data) => HostNumericDataOwned::F64(data.to_vec()),
            HostNumericDataView::F32(data) => HostNumericDataOwned::F32(data.to_vec()),
            other => panic!("fixture expects floating output, got {other:?}"),
        };
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data,
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.probes.lock().unwrap().push(allocation.probe);
        retained.push(allocation.handle.clone());
        Ok(allocation.handle)
    }
}

fn assert_sibling_alias_rejected(single: bool, complex: bool, requested: usize, alias_at: usize) {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(SiblingAliasProvider {
        owner: TestAllocationOwner::new(),
        retained: Mutex::new(Vec::new()),
        probes: Mutex::new(Vec::new()),
        uploads: AtomicUsize::new(0),
        alias_at,
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: if single {
                HostNumericDataOwned::F32(vec![4.0])
            } else {
                HostNumericDataOwned::F64(vec![4.0])
            },
            shape: vec![1, 1],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    let scalar = |value| {
        if complex {
            Value::ComplexTensor(ComplexTensor::new(vec![(value, 1.0)], vec![1, 1]).unwrap())
        } else {
            Value::Tensor(Tensor::new(vec![value], vec![1, 1]).unwrap())
        }
    };
    let plan = NormalizeOutputPlan {
        single,
        source: Some(source.handle.clone()),
        requested: RequestedOutputs::Explicit(requested),
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
    assert_eq!(provider.uploads.load(Ordering::SeqCst), requested);
    let probes = provider.probes.lock().unwrap().clone();
    assert_eq!(probes.len(), alias_at);
    for probe in &probes {
        assert_eq!(probe.retirements(), 0);
    }
    provider.retained.lock().unwrap().clear();
    for probe in &probes {
        assert_eq!(probe.retirements(), 1);
    }
    drop(source.handle);
    assert_eq!(source.probe.retirements(), 1);
}

#[test]
fn sibling_alias_is_rejected_before_any_output_is_published() {
    assert_sibling_alias_rejected(false, false, 2, 1);
    assert_sibling_alias_rejected(true, false, 2, 1);
    assert_sibling_alias_rejected(false, true, 2, 1);
    assert_sibling_alias_rejected(false, false, 3, 2);
}
