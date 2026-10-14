use futures::executor::block_on;
use runmat_accelerate_api::test_support::TestAllocationOwner;
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async_with_outputs, GpuGatherRetry};
use runmat_value::{Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Provider {
    owner: TestAllocationOwner,
    calculations: AtomicUsize,
    downloads: AtomicUsize,
    uploads: AtomicUsize,
}

impl Provider {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            calculations: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
        }))
    }
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "corrcoef discard admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("discarded output must not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("discarded output must not download") })
    }

    fn corrcoef<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a CorrcoefOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.calculations.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(ProviderExecutionError::from_source(std::io::Error::other(
                "discarded output must not calculate",
            )))
        })
    }
}

#[test]
fn zero_outputs_still_validate_numeric_carriers_rank_and_pair_size() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let invalid_type = block_on(call_builtin_async_with_outputs(
        "corrcoef",
        &[Value::from("nonnumeric")],
        0,
    ));
    assert!(invalid_type.is_err());

    let rank_three = Value::Tensor(Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap());
    let invalid_rank = block_on(call_builtin_async_with_outputs(
        "corrcoef",
        &[rank_three],
        0,
    ));
    assert!(invalid_rank.is_err());

    let left = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    let right = Value::Tensor(Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap());
    let invalid_pair = block_on(call_builtin_async_with_outputs(
        "corrcoef",
        &[left, right],
        0,
    ));
    assert_eq!(
        invalid_pair.unwrap_err().identifier(),
        Some("RunMat:corrcoef:SizeMismatch")
    );

    let valid_complex = block_on(call_builtin_async_with_outputs(
        "corrcoef",
        &[Value::Complex(1.0, 2.0)],
        0,
    ));
    assert!(
        matches!(valid_complex.as_ref().map(|root| root.value()), Ok(Value::OutputList(values)) if values.is_empty())
    );
    for carrier in [
        Value::Bool(true),
        Value::Int(runmat_value::IntValue::U64(3)),
    ] {
        let result = block_on(call_builtin_async_with_outputs("corrcoef", &[carrier], 0));
        assert!(
            matches!(result.as_ref().map(|root| root.value()), Ok(Value::OutputList(values)) if values.is_empty())
        );
    }
}

#[test]
fn zero_outputs_reject_invalid_resident_ownership_or_lease_without_provider_work() {
    let provider = Provider::new();
    let _active = ThreadProviderGuard::set(Some(provider));
    let valid = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let result = block_on(call_builtin_async_with_outputs(
        "corrcoef",
        &[Value::GpuTensor(valid.handle.clone())],
        0,
    ))
    .unwrap();
    assert!(matches!(result.value(), Value::OutputList(values) if values.is_empty()));
    assert!(valid.handle.allocation_lease().validate_live().is_ok());
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let logical = provider
        .owner
        .allocate_logical(vec![1, 0, 1, 1], vec![2, 2])
        .unwrap();
    let discarded_logical = block_on(call_builtin_async_with_outputs(
        "corrcoef",
        &[Value::GpuTensor(logical.handle.clone())],
        0,
    ));
    assert!(
        matches!(discarded_logical.as_ref().map(|root| root.value()), Ok(Value::OutputList(values)) if values.is_empty())
    );
    assert!(logical.handle.allocation_lease().validate_live().is_ok());

    let foreign_owner = TestAllocationOwner::new();
    let foreign = foreign_owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let unknown_owner = block_on(call_builtin_async_with_outputs(
        "corrcoef",
        &[Value::GpuTensor(foreign.handle.clone())],
        0,
    ))
    .unwrap_err();
    assert_eq!(unknown_owner.gpu_gather_retry, GpuGatherRetry::Never);
    assert_eq!(foreign.probe.retirements(), 0);

    let unavailable = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    unavailable.probe.mark_unavailable();
    let invalid_lease = block_on(call_builtin_async_with_outputs(
        "corrcoef",
        &[Value::GpuTensor(unavailable.handle.clone())],
        0,
    ))
    .unwrap_err();
    assert_eq!(invalid_lease.identifier(), Some("RunMat:corrcoef:Internal"));
    assert_eq!(provider.calculations.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(valid.probe.retirements(), 0);
    assert_eq!(foreign.probe.retirements(), 0);
}

#[test]
fn malformed_physical_views_cannot_become_operational_zero_output_inputs() {
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert!(input
        .handle
        .with_view(GpuTensorView::contiguous(vec![3, 2]))
        .is_err());
    let integer = owner
        .allocate_zeroed(vec![2, 2], NumericElementType::I32, GpuTensorStorage::Real)
        .unwrap();
    assert!(integer
        .handle
        .with_view(GpuTensorView::contiguous(vec![2, 2]).with_logical(true))
        .is_err());
    assert!(input.handle.allocation_lease().validate_live().is_ok());
    assert!(integer.handle.allocation_lease().validate_live().is_ok());
}
