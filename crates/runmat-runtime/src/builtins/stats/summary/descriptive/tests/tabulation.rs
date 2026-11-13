use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct RejectingDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

struct PendingDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for PendingDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "tabulate pending fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { futures::future::pending().await })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { futures::future::pending().await })
    }
}

#[tokio::test]
async fn tabulate_cancelled_pending_gather_leaves_borrowed_allocation_live() {
    let provider = Box::leak(Box::new(PendingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut pending = Box::pin(tabulate::tabulate_builtin(Value::GpuTensor(
        resident.handle.clone(),
    )));
    assert!(matches!(
        futures::poll!(pending.as_mut()),
        std::task::Poll::Pending
    ));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    drop(pending);
    assert_eq!(resident.probe.retirements(), 0);
    drop(resident.handle);
    assert_eq!(resident.probe.retirements(), 1);
}

impl AccelProvider for RejectingDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "tabulate gather fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("tabulate fixture download failure") })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("tabulate fixture native download failure") })
    }
}

#[tokio::test]
async fn tabulate_gather_failure_retains_cause_and_borrowed_lease() {
    let provider = Box::leak(Box::new(RejectingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let err = tabulate::tabulate_builtin(Value::GpuTensor(resident.handle.clone()))
        .await
        .expect_err("provider download must fail");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(err.context.builtin.as_deref(), Some("tabulate"));
    assert!(err.source.is_some());
    assert!(err.message().contains("tabulate fixture"));
    assert_eq!(resident.probe.retirements(), 0);
    drop(resident.handle);
    assert_eq!(resident.probe.retirements(), 1);
}

#[tokio::test]
async fn tabulate_rejects_nested_resident_cell_label_before_download() {
    let provider = Box::leak(Box::new(RejectingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let cell = runmat_value::CellArray::new(vec![Value::GpuTensor(resident.handle.clone())], 1, 1)
        .unwrap();
    let err = tabulate::tabulate_builtin(Value::Cell(cell))
        .await
        .expect_err("device tensor is not a scalar label");
    assert!(err
        .message()
        .contains("cell entries must be scalar strings or scalars"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
    drop(resident.handle);
    assert_eq!(resident.probe.retirements(), 1);
}

#[test]
fn tabulate_expands_positive_integer_levels() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 3.0, 3.0], vec![1, 3]).unwrap());
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    let (data, shape) = tensor_values(out);
    assert_eq!(shape, vec![3, 3]);
    assert_eq!(&data[0..3], &[1.0, 2.0, 3.0]);
    assert_eq!(&data[3..6], &[1.0, 0.0, 2.0]);
    assert_close(data[6], 100.0 / 3.0);
    assert_close(data[7], 0.0);
    assert_close(data[8], 200.0 / 3.0);
}

#[test]
fn tabulate_near_integer_floats_remain_distinct_exact_levels() {
    let adjacent = 1.0 + 5.0e-13;
    assert_ne!(adjacent, 1.0);
    let input = Value::Tensor(Tensor::new(vec![1.0, adjacent, adjacent], vec![1, 3]).unwrap());
    let (data, shape) = tensor_values(block_on(tabulate::tabulate_builtin(input)).unwrap());
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(&data[..2], &[1.0, adjacent]);
    assert_eq!(&data[2..4], &[1.0, 2.0]);
}

#[test]
fn tabulate_rejects_unbounded_positive_floating_levels_before_expansion() {
    assert_eq!(
        super::super::tabulation::expanded_level_count(1_000_000),
        Some(1_000_000)
    );
    assert_eq!(
        super::super::tabulation::expanded_level_count(1_000_001),
        None
    );
    assert_eq!(
        super::super::tabulation::expanded_level_count(u128::MAX),
        None
    );
    for level in [1.0e300, usize::MAX as f64, u64::MAX as f64] {
        let x = Value::Tensor(Tensor::new(vec![1.0, level], vec![1, 2]).unwrap());
        let error = block_on(tabulate::tabulate_builtin(x)).expect_err("huge gap rejected");
        assert!(error.message().contains("positive-level expansion"));
    }
}

#[test]
fn tabulate_empty_numeric_input_has_three_columns() {
    let input = Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).unwrap());
    let (data, shape) = tensor_values(block_on(tabulate::tabulate_builtin(input)).unwrap());
    assert!(data.is_empty());
    assert_eq!(shape, vec![0, 3]);
}

#[test]
fn tabulate_numeric_reads_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = int_tensor(IntegerStorage::U8(vec![1, 3, 3]), vec![1, 3]);
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    let Value::Cell(cell) = out else {
        panic!("expected exact integer cell table");
    };
    assert_eq!((cell.rows, cell.cols), (3, 3));
    assert_eq!(cell.get(0, 0).unwrap(), Value::Int(IntValue::U8(1)));
    assert_eq!(cell.get(1, 0).unwrap(), Value::Int(IntValue::U8(2)));
    assert_eq!(cell.get(2, 0).unwrap(), Value::Int(IntValue::U8(3)));
    assert_eq!(cell.get(0, 1).unwrap(), Value::Num(1.0));
    assert_eq!(cell.get(1, 1).unwrap(), Value::Num(0.0));
    assert_eq!(cell.get(2, 1).unwrap(), Value::Num(2.0));
}

#[test]
fn tabulate_integer_extension_preserves_wide_distinct_values() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let base = 1_u64 << 53;
    let x = int_tensor(IntegerStorage::U64(vec![base, base + 1, base]), vec![1, 3]);
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    let Value::Cell(cell) = out else {
        panic!("expected exact integer cell table");
    };
    assert_eq!((cell.rows, cell.cols), (2, 3));
    assert_eq!(cell.get(0, 0).unwrap(), Value::Int(IntValue::U64(base)));
    assert_eq!(cell.get(1, 0).unwrap(), Value::Int(IntValue::U64(base + 1)));
    assert_eq!(cell.get(0, 1).unwrap(), Value::Num(2.0));
    assert_eq!(cell.get(1, 1).unwrap(), Value::Num(1.0));
}

#[test]
fn tabulate_numeric_omits_nans_from_counts_and_percent() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 3.0, 3.0, f64::NAN], vec![1, 4]).unwrap());
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    let (data, shape) = tensor_values(out);
    assert_eq!(shape, vec![3, 3]);
    assert_eq!(&data[0..3], &[1.0, 2.0, 3.0]);
    assert_eq!(&data[3..6], &[1.0, 0.0, 2.0]);
    assert_close(data[6], 100.0 / 3.0);
    assert_close(data[7], 0.0);
    assert_close(data[8], 200.0 / 3.0);
}

#[test]
fn tabulate_logical_returns_cell_table() {
    let x =
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap());
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    match out {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 2);
            assert_eq!(cell.cols, 3);
            assert_eq!(cell.get(0, 0).unwrap(), Value::Bool(false));
            assert_eq!(cell.get(0, 1).unwrap(), Value::Num(1.0));
            assert_eq!(cell.get(1, 0).unwrap(), Value::Bool(true));
            assert_eq!(cell.get(1, 1).unwrap(), Value::Num(2.0));
        }
        other => panic!("expected cell table, got {other:?}"),
    }
}

#[test]
fn tabulate_string_array_returns_cell_table() {
    let x = Value::StringArray(
        StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![1, 3]).unwrap(),
    );
    let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
    match out {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 2);
            assert_eq!(cell.cols, 3);
            assert_eq!(cell.get(0, 1).unwrap(), Value::Num(1.0));
            assert_eq!(cell.get(1, 1).unwrap(), Value::Num(2.0));
        }
        other => panic!("expected cell table, got {other:?}"),
    }
}

#[test]
fn tabulate_rejects_multirow_characters_and_nested_cell_labels() {
    let chars =
        runmat_value::CharArray::from_unicode_scalars(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
    assert!(block_on(tabulate::tabulate_builtin(Value::CharArray(chars))).is_err());

    let inner = runmat_value::CellArray::new(vec![Value::from("a")], 1, 1).unwrap();
    let outer = runmat_value::CellArray::new(vec![Value::Cell(inner)], 1, 1).unwrap();
    let err = block_on(tabulate::tabulate_builtin(Value::Cell(outer)))
        .expect_err("nested cells are not scalar labels");
    assert!(err
        .message()
        .contains("cell entries must be scalar strings or scalars"));
}
