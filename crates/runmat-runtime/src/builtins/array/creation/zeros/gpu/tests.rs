use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned,
    HostNumericTensorView, HostTensorView,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct PreferredSingleWithDoubleUpload {
    owner: TestAllocationOwner,
    retained: Mutex<Vec<RetirementProbe>>,
    uploads: AtomicUsize,
}

impl AccelProvider for PreferredSingleWithDoubleUpload {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "preferred single, native double upload".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F32
    }
    fn upload(
        &self,
        _: &HostTensorView<'_>,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        anyhow::bail!("legacy untyped upload is not supported")
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("download is not needed by this constructor witness") })
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<runmat_accelerate_api::GpuTensorHandle>
    {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        host.validate()?;
        let data = match host.data {
            HostNumericDataView::F64(values) => HostNumericDataOwned::F64(values.to_vec()),
            HostNumericDataView::F32(values) => HostNumericDataOwned::F32(values.to_vec()),
            _ => return Err(anyhow::anyhow!("fixture accepts only floating storage").into()),
        };
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data,
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.retained.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
}

#[test]
fn default_gpu_zeros_uses_double_when_preferred_precision_is_single() {
    let _guard = test_support::accel_test_lock();
    let fixture = Box::leak(Box::new(PreferredSingleWithDoubleUpload {
        owner: TestAllocationOwner::new(),
        retained: Mutex::new(Vec::new()),
        uploads: AtomicUsize::new(0),
    }));
    let provider: &'static dyn AccelProvider = fixture;
    // SAFETY: the leaked fixture has a unique immutable device id and outlives its handles.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    assert!(block_on(super::super::zeros_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::from("single"),
            Value::from("double"),
            Value::from("gpuArray"),
        ])
    ))
    .is_err());
    assert_eq!(fixture.uploads.load(Ordering::SeqCst), 0);
    let result = block_on(super::super::zeros_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::from("gpuArray"),
        ]),
    ))
    .unwrap();
    let Value::GpuTensor(handle) = result else {
        panic!("expected explicit resident zeros");
    };
    assert_eq!(handle.shape(), vec![2, 3]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&handle),
        Some(ProviderPrecision::F64)
    );
    assert_eq!(fixture.uploads.load(Ordering::SeqCst), 1);
}
