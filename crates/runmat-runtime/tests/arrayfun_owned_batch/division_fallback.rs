use super::*;
use runmat_accelerate_api::{
    allocation::WeakGpuTensorHandle,
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, GpuHandleProvenance, GpuTensorStorage,
    HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned, HostNumericTensorView,
    HostTensorView,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct Provider {
    owner: TestAllocationOwner,
    records: Mutex<Vec<(u64, RetirementProbe)>>,
    returned: Mutex<Option<WeakGpuTensorHandle>>,
    uploads: AtomicUsize,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "division other-original upload alias".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed upload required")
    }
    fn upload_numeric_raw(
        &self,
        _: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        self.returned
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .acquire()
            .map_err(Into::into)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("typed download required")
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            let probe = self
                .records
                .lock()
                .unwrap()
                .iter()
                .find(|(id, _)| *id == input.buffer_id())
                .unwrap()
                .1
                .clone();
            let data = probe.with_data(input, |data| match data {
                HostNumericDataView::F32(values) => HostNumericDataOwned::F32(values.to_vec()),
                _ => panic!("native single fixture"),
            })?;
            Ok(HostNumericTensorOwned {
                data,
                shape: input.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
}

#[test]
fn division_fallback_rejects_upload_alias_of_the_nonpreferred_original() {
    runmat_gc::gc_test_context(|| {
        for name in ["rdivide", "ldivide"] {
            let provider = Box::leak(Box::new(Provider {
                owner: TestAllocationOwner::new(),
                records: Mutex::new(Vec::new()),
                returned: Mutex::new(None),
                uploads: AtomicUsize::new(0),
            }));
            let _scope = ThreadProviderGuard::set(Some(provider));
            let mut handles = Vec::new();
            let mut probes = Vec::new();
            for values in [vec![2.0f32, 4.0], vec![1.0f32, 2.0]] {
                let allocation = provider
                    .owner
                    .allocate_numeric(HostNumericTensorOwned {
                        data: HostNumericDataOwned::F32(values),
                        shape: vec![1, 2],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap();
                provider
                    .records
                    .lock()
                    .unwrap()
                    .push((allocation.handle.buffer_id(), allocation.probe.clone()));
                probes.push(allocation.probe);
                handles.push(
                    allocation
                        .handle
                        .with_provenance(GpuHandleProvenance::Automatic)
                        .unwrap(),
                );
            }
            // With no explicit operand, rdivide prefers the left source and
            // ldivide's normalized numerator prefers the right source.
            let other = if name == "rdivide" { 1 } else { 0 };
            *provider.returned.lock().unwrap() = Some(handles[other].downgrade());
            let arguments = handles
                .iter()
                .cloned()
                .map(Value::GpuTensor)
                .collect::<Vec<_>>();
            let error = block_on(call_builtin_async_with_outputs(name, &arguments, 1)).unwrap_err();
            assert!(
                error.source.is_some(),
                "{name}: retained invalid upload cause"
            );
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
            for handle in &handles {
                handle.allocation_lease().validate_live().unwrap();
            }
            assert!(probes.iter().all(|probe| probe.retirements() == 0));
            drop(error);
            drop(arguments);
            drop(handles);
            assert!(probes.iter().all(|probe| probe.retirements() == 1));
        }
    });
}
