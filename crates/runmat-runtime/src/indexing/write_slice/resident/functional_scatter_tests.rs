use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostNumericTensorView, HostTensorView,
    ProviderExecutionFuture,
};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct MutableScatterProvider {
    owner: TestAllocationOwner,
    values: Mutex<HashMap<u64, Vec<f64>>>,
    rhs_alias: Mutex<Option<GpuTensorHandle>>,
    downloads: AtomicUsize,
    inplace_calls: AtomicUsize,
}

impl MutableScatterProvider {
    fn new() -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            values: Mutex::new(HashMap::new()),
            rhs_alias: Mutex::new(None),
            downloads: AtomicUsize::new(0),
            inplace_calls: AtomicUsize::new(0),
        }
    }

    fn snapshot(&self, handle: &GpuTensorHandle) -> Vec<f64> {
        self.values.lock().unwrap()[&handle.buffer_id()].clone()
    }

    fn allocate(&self, data: Vec<f64>, shape: Vec<usize>) -> anyhow::Result<GpuTensorHandle> {
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(data.clone()),
            shape,
            storage: GpuTensorStorage::Real,
        })?;
        self.values
            .lock()
            .unwrap()
            .insert(allocation.handle.buffer_id(), data);
        Ok(allocation.handle)
    }
}

impl AccelProvider for MutableScatterProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "mutable functional scatter fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed upload required")
    }
    fn upload_numeric_raw(
        &self,
        view: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let HostNumericDataView::F64(data) = view.data else {
            panic!("fixture admits double only")
        };
        Ok(self.allocate(data.to_vec(), view.shape.to_vec())?)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        panic!("functional path must not download")
    }
    fn scatter_linear(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> anyhow::Result<()> {
        self.inplace_calls.fetch_add(1, Ordering::SeqCst);
        let rhs = self.snapshot(values);
        let mut all = self.values.lock().unwrap();
        let target = all.get_mut(&target.buffer_id()).expect("live target");
        for (&index, value) in indices.iter().zip(rhs) {
            target[index as usize] = value;
        }
        Ok(())
    }
    fn scatter_linear_functional<'a>(
        &'a self,
        target: &'a GpuTensorHandle,
        indices: &'a [u32],
        values: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            *self.rhs_alias.lock().unwrap() = Some(values.clone());
            let rhs = self.snapshot(values);
            let mut updated = self.snapshot(target);
            for (&index, value) in indices.iter().zip(rhs) {
                updated[index as usize] = value;
            }
            Ok(self.allocate(updated, target.shape().to_vec())?)
        })
    }
}

#[test]
fn functional_scatter_preserves_target_and_rhs_aliases_without_download() {
    let provider = Box::leak(Box::new(MutableScatterProvider::new()));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let target = provider.allocate(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let retained = target.clone();
    let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 2]);
    let Value::GpuTensor(output) =
        futures::executor::block_on(assign_gpu_slice_with_plan(&target, &plan, &Value::Num(9.0)))
            .unwrap()
    else {
        panic!("resident functional output")
    };
    assert!(!output.shares_allocation_with(&target));
    assert_eq!(provider.snapshot(&output), vec![1.0, 9.0]);
    assert_eq!(provider.snapshot(&target), vec![1.0, 2.0]);
    assert_eq!(provider.snapshot(&retained), vec![1.0, 2.0]);
    let rhs = provider
        .rhs_alias
        .lock()
        .unwrap()
        .clone()
        .expect("retained RHS");
    assert!(!output.shares_allocation_with(&rhs));
    assert_eq!(provider.snapshot(&rhs), vec![9.0]);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.inplace_calls.load(Ordering::SeqCst), 0);
    // A third-party provider implements only the original exactly-K contract.
    // Scalar-to-many must retain its expanded upload, not pass a one-element
    // RHS to that hook or force an unavailable host download.
    let both = IndexPlan::new(vec![0, 1], vec![1, 2], vec![2], 1, vec![1, 2]);
    let Value::GpuTensor(broadcast) = futures::executor::block_on(assign_gpu_slice_with_plan(
        &target,
        &both,
        &Value::Num(13.0),
    ))
    .unwrap() else {
        panic!("resident legacy broadcast")
    };
    assert_eq!(provider.snapshot(&broadcast), vec![13.0, 13.0]);
    assert_eq!(provider.snapshot(&target), vec![1.0, 2.0]);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.inplace_calls.load(Ordering::SeqCst), 0);
}
