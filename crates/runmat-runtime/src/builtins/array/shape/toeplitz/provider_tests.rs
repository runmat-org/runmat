//! Fault-injected owned results exercise the complete builtin boundary.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    allocation::WeakGpuTensorHandle,
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostTensorOwned, HostTensorView,
};
use std::sync::Mutex;

enum Injection {
    None,
    Borrowed(WeakGpuTensorHandle),
    FreshMalformed,
}
struct Provider {
    owner: TestAllocationOwner,
    result: Mutex<Injection>,
    allocations: Mutex<Vec<RetirementProbe>>,
}
impl Provider {
    fn allocate(&self, data: Vec<f64>, shape: Vec<usize>) -> anyhow::Result<GpuTensorHandle> {
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(data),
            shape,
            storage: GpuTensorStorage::Real,
        })?;
        self.allocations.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
}
impl AccelProvider for Provider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.allocate(host.data.to_vec(), host.shape.to_vec())
    }
    fn upload_numeric_raw(
        &self,
        host: &runmat_accelerate_api::HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        match &*self.result.lock().unwrap() {
            Injection::Borrowed(handle) => {
                return Ok(handle.acquire().expect("live fixture input"))
            }
            Injection::FreshMalformed => return Ok(self.allocate(vec![0.0], vec![1, 1])?),
            Injection::None => {}
        }
        let HostNumericDataView::F64(data) = host.data else {
            return Err(anyhow::anyhow!("fixture accepts F64 payloads only").into());
        };
        Ok(self.allocate(data.to_vec(), host.shape.to_vec())?)
    }
    fn download<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async move {
            for probe in self.allocations.lock().unwrap().iter() {
                if let Ok(data) = probe.with_data(handle, |data| {
                    let HostNumericDataView::F64(values) = data else {
                        panic!("F64 fixture")
                    };
                    values.to_vec()
                }) {
                    return Ok(HostTensorOwned {
                        data,
                        shape: handle.shape().to_vec(),
                        storage: GpuTensorStorage::Real,
                    });
                }
            }
            anyhow::bail!("fixture allocation is not live");
        })
    }
    fn device_info(&self) -> String {
        "toeplitz owned fixture".into()
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
}

#[test]
fn rejected_uploads_preserve_both_operands_and_retire_only_fresh_results() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        result: Mutex::new(Injection::None),
        allocations: Mutex::new(Vec::new()),
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let c = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[2, 1],
        })
        .unwrap();
    let r = provider
        .upload(&HostTensorView {
            data: &[1.0, 3.0],
            shape: &[1, 2],
        })
        .unwrap();
    let column_view = c.view().clone();
    let row_view = r.view().clone();
    for injection in [
        Injection::Borrowed(c.downgrade()),
        Injection::Borrowed(r.downgrade()),
        Injection::FreshMalformed,
    ] {
        *provider.result.lock().unwrap() = injection;
        let error = block_on(toeplitz_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(c.clone()),
                Value::GpuTensor(r.clone()),
            ]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), ERROR_INTERNAL.identifier);
        assert_eq!(c.view(), &column_view);
        assert_eq!(r.view(), &row_view);
        assert_eq!(
            block_on(provider.download(&c)).unwrap().data,
            vec![1.0, 2.0]
        );
        assert_eq!(
            block_on(provider.download(&r)).unwrap().data,
            vec![1.0, 3.0]
        );
        let probes = provider.allocations.lock().unwrap();
        assert_eq!(probes[0].retirements(), 0);
        assert_eq!(probes[1].retirements(), 0);
        for probe in &probes[2..] {
            assert_eq!(probe.retirements(), 1);
        }
    }
    *provider.result.lock().unwrap() = Injection::None;
    drop(c);
    drop(r);
    let probes = provider.allocations.lock().unwrap();
    assert_eq!(probes.len(), 3);
    assert!(probes.iter().all(|probe| probe.retirements() == 1));
}
