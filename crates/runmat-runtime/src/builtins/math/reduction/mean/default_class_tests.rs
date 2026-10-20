use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocation, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, GpuTensorStorage, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostNumericTensorView, HostTensorView,
    ThreadProviderGuard,
};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

struct Provider {
    owner: TestAllocationOwner,
    probes: Mutex<HashMap<u64, RetirementProbe>>,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
}
impl Provider {
    fn new() -> &'static Self {
        let provider = Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            probes: Mutex::new(HashMap::new()),
            uploads: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
        }));
        // SAFETY: this leaked fixture has a unique privately minted device id.
        unsafe { runmat_accelerate_api::register_device_provider(provider) };
        provider
    }
    fn remember(&self, allocation: TestAllocation) -> GpuTensorHandle {
        self.probes
            .lock()
            .unwrap()
            .insert(allocation.handle.buffer_id(), allocation.probe);
        allocation.handle
    }
}
fn copy_data(data: HostNumericDataView<'_>) -> anyhow::Result<HostNumericDataOwned> {
    Ok(match data {
        HostNumericDataView::F64(data) => HostNumericDataOwned::F64(data.to_vec()),
        HostNumericDataView::F32(data) => HostNumericDataOwned::F32(data.to_vec()),
        HostNumericDataView::I16(data) => HostNumericDataOwned::I16(data.to_vec()),
        HostNumericDataView::U8(data) => HostNumericDataOwned::U8(data.to_vec()),
        _ => anyhow::bail!("unsupported fixture class"),
    })
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "mean exact default class fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("untyped upload forbidden")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("untyped download forbidden") })
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        Ok(
            self.remember(self.owner.allocate_numeric(HostNumericTensorOwned {
                data: copy_data(host.data)?,
                shape: host.shape.to_vec(),
                storage: host.storage,
            })?),
        )
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            let probe = self
                .probes
                .lock()
                .unwrap()
                .get(&handle.buffer_id())
                .unwrap()
                .clone();
            let data = probe.with_data(handle, copy_data)??;
            Ok(HostNumericTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: handle.physical().storage(),
            })
        })
    }
}

#[test]
fn default_resident_integer_and_logical_means_are_double_on_exact_owner() {
    for logical in [false, true] {
        let owner = Provider::new();
        let unrelated = Provider::new();
        let _selected = ThreadProviderGuard::set(Some(unrelated));
        let allocation = if logical {
            owner
                .owner
                .allocate_logical(vec![1, 0], vec![2, 1])
                .unwrap()
        } else {
            owner
                .owner
                .allocate_numeric(HostNumericTensorOwned {
                    data: HostNumericDataOwned::I16(vec![1, 2]),
                    shape: vec![2, 1],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        };
        let input = owner.remember(allocation);
        let result = futures::executor::block_on(mean_builtin(
            Value::GpuTensor(input.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        let Value::GpuTensor(output) = result else {
            panic!("F64 fixture must restore resident output")
        };
        assert_eq!(output.device_id(), input.device_id());
        assert_eq!(output.shape(), &[1, 1]);
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(ProviderPrecision::F64)
        );
        assert!(!runmat_accelerate_api::handle_is_logical(&output));
        assert!(runmat_accelerate_api::handle_integer_type(&output).is_none());
        let downloaded = futures::executor::block_on(owner.download_numeric(&output)).unwrap();
        assert_eq!(
            downloaded.data,
            HostNumericDataOwned::F64(vec![if logical { 0.5 } else { 1.5 }])
        );
        assert_eq!(unrelated.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(unrelated.downloads.load(Ordering::SeqCst), 0);
        let output_probe = owner
            .probes
            .lock()
            .unwrap()
            .get(&output.buffer_id())
            .unwrap()
            .clone();
        let input_probe = owner
            .probes
            .lock()
            .unwrap()
            .get(&input.buffer_id())
            .unwrap()
            .clone();
        drop(output);
        assert_eq!(output_probe.retirements(), 1);
        assert_eq!(input_probe.retirements(), 0);
        input.allocation_lease().validate_live().unwrap();
        drop(input);
        assert_eq!(input_probe.retirements(), 1);
    }
}

#[test]
fn scalar_single_mean_keeps_supported_resident_identity_without_host_transfer() {
    let provider = Provider::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.remember(
        provider
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F32(vec![3.5]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap(),
    );
    let result = futures::executor::block_on(mean_builtin(
        Value::GpuTensor(input.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let Value::GpuTensor(output) = result else {
        panic!("Single identity must remain resident")
    };
    assert!(output.shares_allocation_with(&input));
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(ProviderPrecision::F32)
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
}

#[test]
fn host_default_integer_identity_dimensions_still_return_double() {
    let input = Tensor::new_integer(IntegerStorage::I16(vec![4, 5]), vec![2, 1]).unwrap();
    let result = futures::executor::block_on(mean_builtin(
        Value::Tensor(input),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)]),
    ))
    .unwrap();
    let Value::Tensor(output) = result else {
        panic!("expected non-scalar tensor")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F64);
    assert_eq!(output.shape(), vec![2, 1]);
}
