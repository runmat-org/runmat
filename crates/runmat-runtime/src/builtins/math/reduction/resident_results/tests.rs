//! Fault-injected pair restoration with actual allocation retirement.
use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelProvider, GpuHandleProvenance, GpuTensorStorage, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostNumericTensorView, HostTensorView,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct Provider {
    owner: TestAllocationOwner,
    uploads: AtomicUsize,
    results: Mutex<Vec<RetirementProbe>>,
    first: Mutex<Option<GpuTensorHandle>>,
    alias_second: bool,
}
impl AccelProvider for Provider {
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(view.data.to_vec()),
                shape: view.shape.to_vec(),
                storage: GpuTensorStorage::Real,
            })?
            .handle)
    }
    fn upload_numeric_raw(
        &self,
        view: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.uploads.fetch_add(1, Ordering::SeqCst) == 1 {
            if self.alias_second {
                return Ok(self.first.lock().unwrap().take().unwrap());
            }
            return Err(anyhow::anyhow!("injected second-output upload failure").into());
        }
        let HostNumericDataView::F64(data) = view.data else {
            return Err(anyhow::anyhow!("fixture expects double scalar restoration").into());
        };
        let output = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(data.to_vec()),
            shape: view.shape.to_vec(),
            storage: view.storage,
        })?;
        self.results.lock().unwrap().push(output.probe);
        if self.alias_second {
            *self.first.lock().unwrap() = Some(output.handle.clone());
        }
        Ok(output.handle)
    }
    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("restoration may not gather") })
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "reduction output restoration test owner".into()
    }
}
fn provider(alias_second: bool) -> &'static Provider {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        uploads: AtomicUsize::new(0),
        results: Mutex::new(Vec::new()),
        first: Mutex::new(None),
        alias_second,
    }));
    unsafe {
        runmat_accelerate_api::register_device_provider(provider);
    }
    provider
}
#[test]
fn rejected_second_result_releases_first_without_freeing_source() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    for alias in [false, true] {
        let provider = provider(alias);
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let source = provider
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1., 2.]),
                shape: vec![2, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let handle = source
            .handle
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        drop(source.handle);
        let before = handle.wire_descriptor();
        assert!(restore_pair(
            &handle,
            Value::Num(2.),
            Value::Num(1.),
            2,
            "maxk",
            &[&handle]
        )
        .is_err());
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
        let results = provider.results.lock().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].retirements(), 1);
        assert_eq!(handle.wire_descriptor(), before);
        assert_eq!(source.probe.retirements(), 0);
        source
            .probe
            .with_data(&handle, |data| {
                let HostNumericDataView::F64(values) = data else {
                    panic!("source class")
                };
                assert_eq!(values, &[1., 2.]);
            })
            .unwrap();
        handle.release().unwrap();
        assert_eq!(source.probe.retirements(), 1);
    }
}
#[test]
fn discarded_and_unrequested_results_are_not_uploaded() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    for retained in [0, 1] {
        let provider = provider(false);
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let source = provider
            .upload(&HostTensorView {
                data: &[1., 2.],
                shape: &[2, 1],
            })
            .unwrap();
        let (first, second) = restore_pair(
            &source,
            Value::Num(1.),
            Value::Num(2.),
            retained,
            "bounds",
            &[&source],
        )
        .unwrap();
        assert_eq!(provider.uploads.load(Ordering::SeqCst), retained);
        assert!(!matches!(second, Value::GpuTensor(_)));
        assert!(provider
            .results
            .lock()
            .unwrap()
            .iter()
            .all(|probe| probe.retirements() == 0));
        drop(first);
        assert!(provider
            .results
            .lock()
            .unwrap()
            .iter()
            .all(|probe| probe.retirements() == 1));
        source.release().unwrap();
    }
}
