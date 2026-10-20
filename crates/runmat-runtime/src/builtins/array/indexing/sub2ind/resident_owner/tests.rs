use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned,
    HostNumericTensorView, HostTensorOwned, HostTensorView, NumericElementType,
};
use runmat_value::Tensor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct Owner {
    allocations: TestAllocationOwner,
    precision: ProviderPrecision,
    uploads: AtomicUsize,
}

impl Owner {
    fn new(precision: ProviderPrecision) -> &'static Self {
        Box::leak(Box::new(Self {
            allocations: TestAllocationOwner::new(),
            precision,
            uploads: AtomicUsize::new(0),
        }))
    }
}

impl AccelProvider for Owner {
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
    }

    fn device_info(&self) -> String {
        "sub2ind exact-owner fixture".into()
    }

    fn precision(&self) -> ProviderPrecision {
        self.precision
    }

    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self
            .allocations
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(host.data.to_vec()),
                shape: host.shape.to_vec(),
                storage: GpuTensorStorage::Real,
            })?
            .handle)
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        let data = match host.data {
            runmat_accelerate_api::HostNumericDataView::F64(words) => {
                HostNumericDataOwned::F64(words.to_vec())
            }
            runmat_accelerate_api::HostNumericDataView::F32(words) => {
                HostNumericDataOwned::F32(words.to_vec())
            }
            _ => return Err(anyhow::anyhow!("unexpected test numeric class").into()),
        };
        Ok(self
            .allocations
            .allocate_numeric(HostNumericTensorOwned {
                data,
                shape: host.shape.to_vec(),
                storage: host.storage,
            })?
            .handle)
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("owner-admission tests never download") })
    }
}

struct DefaultOwner {
    allocations: TestAllocationOwner,
    records: Mutex<Vec<(GpuTensorHandle, RetirementProbe)>>,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
}

impl DefaultOwner {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            allocations: TestAllocationOwner::new(),
            records: Mutex::new(Vec::new()),
            uploads: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
        }))
    }

    fn retain(&self, data: Vec<f64>, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        let allocation = self.allocations.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(data),
            shape: shape.to_vec(),
            storage: GpuTensorStorage::Real,
        })?;
        self.records
            .lock()
            .unwrap()
            .push((allocation.handle.clone(), allocation.probe));
        self.uploads.fetch_add(1, Ordering::SeqCst);
        Ok(allocation.handle)
    }
}

impl AccelProvider for DefaultOwner {
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
    }

    fn device_info(&self) -> String {
        "sub2ind default-hook fallback owner".into()
    }

    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.retain(host.data.to_vec(), host.shape)
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let HostNumericDataView::F64(words) = host.data else {
            return Err(anyhow::anyhow!("default owner only admits Double").into());
        };
        self.retain(words.to_vec(), host.shape).map_err(Into::into)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let probe = self
                .records
                .lock()
                .unwrap()
                .iter()
                .find(|(owned, _)| owned.shares_allocation_with(handle))
                .map(|(_, probe)| probe.clone())
                .ok_or_else(|| anyhow::anyhow!("unknown test allocation"))?;
            let data = probe.with_data(handle, |payload| match payload {
                HostNumericDataView::F64(words) => words.to_vec(),
                _ => panic!("fixture owns only Double payloads"),
            })?;
            self.downloads.fetch_add(1, Ordering::SeqCst);
            Ok(HostTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn default_unsupported_hook_computes_host_indices_then_restores_exact_double_owner() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let owner = DefaultOwner::new();
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let rows = owner
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let columns = owner
        .upload(&HostTensorView {
            data: &[2.0, 3.0],
            shape: &[1, 2],
        })
        .unwrap();
    let unsupported = futures::executor::block_on(owner.sub2ind(
        &[2, 3],
        &[1, 2],
        &[&rows, &columns],
        &[false, false],
        2,
        &[1, 2],
    ))
    .unwrap_err();
    assert!(unsupported.is_unsupported());
    assert_eq!(owner.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
    rows.allocation_lease().validate_live().unwrap();
    columns.allocation_lease().validate_live().unwrap();

    let dims = Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap());
    let result = futures::executor::block_on(super::super::sub2ind_builtin(
        dims,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(rows.clone()),
            Value::GpuTensor(columns.clone()),
        ]),
    ))
    .expect("typed default Unsupported must admit host computation");
    let Value::GpuTensor(output) = result else {
        panic!("exact F64 input owner must restore Double result");
    };
    assert_eq!(output.shape(), [1, 2]);
    assert_eq!(output.device_id(), owner.device_id());
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(ProviderPrecision::F64)
    );
    assert!(!output.shares_allocation_with(&rows));
    assert!(!output.shares_allocation_with(&columns));
    let host = futures::executor::block_on(owner.download(&output)).unwrap();
    assert_eq!(host.data, [3.0, 6.0]);
    assert_eq!(host.shape, [1, 2]);
    assert_eq!(owner.uploads.load(Ordering::SeqCst), 3);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 3);
    rows.allocation_lease().validate_live().unwrap();
    columns.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn malformed_direct_results_retire_without_releasing_borrowed_input() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Owner::new(ProviderPrecision::F64);
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let source = provider
        .allocations
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let inputs = [&source.handle];
    assert!(direct_owner(&source.handle, &inputs).is_some());

    let wrong_class = provider
        .allocations
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let error = publish_direct(
        wrong_class.handle,
        &[1, 1],
        &source.handle,
        &inputs,
        provider,
    )
    .unwrap_err();
    assert_eq!(error.identifier(), SUB2IND_ERROR_PROVIDER.identifier);
    assert_eq!(wrong_class.probe.retirements(), 1);
    assert_eq!(source.probe.retirements(), 0);

    let wrong_shape = provider
        .allocations
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert!(publish_direct(
        wrong_shape.handle,
        &[1, 1],
        &source.handle,
        &inputs,
        provider,
    )
    .is_err());
    assert_eq!(wrong_shape.probe.retirements(), 1);

    let alien = Owner::new(ProviderPrecision::F64);
    let wrong_owner = alien
        .allocations
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert!(publish_direct(
        wrong_owner.handle,
        &[1, 1],
        &source.handle,
        &inputs,
        provider,
    )
    .is_err());
    assert_eq!(wrong_owner.probe.retirements(), 1);

    assert!(publish_direct(
        source.handle.clone(),
        &[1, 1],
        &source.handle,
        &inputs,
        provider,
    )
    .is_err());
    let other_subscript = provider
        .allocations
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert!(publish_direct(
        source.handle.clone(),
        &[1, 1],
        &source.handle,
        &[&other_subscript.handle],
        provider,
    )
    .is_err());
    assert_eq!(source.probe.retirements(), 0);
    source.handle.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn failed_rejection_keeps_cleanup_cause_and_host_fallback_keeps_double() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Owner::new(ProviderPrecision::F64);
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let source = provider
        .allocations
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let malformed = provider
        .allocations
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    malformed
        .probe
        .fail_retirement("injected rejection cleanup failure");
    let error = publish_direct(
        malformed.handle,
        &[1, 1],
        &source.handle,
        &[&source.handle],
        provider,
    )
    .unwrap_err();
    assert_eq!(error.identifier(), SUB2IND_ERROR_PROVIDER.identifier);
    assert!(error.message.contains("injected rejection cleanup failure"));
    assert_eq!(malformed.probe.retirements(), 1);
    assert_eq!(source.probe.retirements(), 0);

    let host = Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap());
    let restored = restore_host(Some(&source.handle), host, &[&source.handle]).unwrap();
    let Value::GpuTensor(output) = restored else {
        panic!("binary64 source should restore Double to its exact owner");
    };
    assert_eq!(output.device_id(), source.handle.device_id());
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(ProviderPrecision::F64)
    );
    assert!(!output.shares_allocation_with(&source.handle));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);

    let direct = provider
        .allocations
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let published = publish_direct(
        direct.handle,
        &[1, 1],
        &source.handle,
        &[&source.handle],
        provider,
    )
    .unwrap();
    let Value::GpuTensor(published) = published else {
        panic!("valid Double result must remain resident");
    };
    assert_eq!(published.device_id(), source.handle.device_id());
    assert_eq!(
        runmat_accelerate_api::handle_precision(&published),
        Some(ProviderPrecision::F64)
    );
    assert_eq!(direct.probe.retirements(), 0);

    let single_owner = Owner::new(ProviderPrecision::F32);
    let single = single_owner
        .allocations
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let host = Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap());
    let result = restore_host(Some(&single.handle), host, &[&single.handle]).unwrap();
    assert!(matches!(result, Value::Tensor(_)));
    assert_eq!(single_owner.uploads.load(Ordering::SeqCst), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn mixed_resident_owners_keep_host_computation_without_ambient_reupload() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let first = Owner::new(ProviderPrecision::F64);
    let second = Owner::new(ProviderPrecision::F64);
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(first));
    // SAFETY: fixture owners are leaked for process lifetime with distinct minted device ids.
    unsafe { runmat_accelerate_api::register_device_provider(second) };
    let left = first
        .allocations
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let right = second
        .allocations
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let selected = source(
        &Value::Num(2.0),
        crate::call::FunctionArgumentView::from_slice(&[
            Value::GpuTensor(left.handle.clone()),
            Value::GpuTensor(right.handle.clone()),
        ]),
    );
    assert!(
        selected.is_none(),
        "mixed sources have no restoration owner"
    );
    let host = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let result = restore_host(None, host, &[&left.handle, &right.handle]).unwrap();
    assert!(matches!(result, Value::Tensor(_)));
    assert_eq!(first.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(second.uploads.load(Ordering::SeqCst), 0);
    left.handle.allocation_lease().validate_live().unwrap();
    right.handle.allocation_lease().validate_live().unwrap();
}
