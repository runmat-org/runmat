use super::*;
use crate::execution::CpuCooperationPolicy;
use futures::task::noop_waker;
use runmat_accelerate_api::AccelProvider;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_value::{IntValue, Tensor};
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

struct CountedIntegerOwner {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
    numeric_downloads: AtomicUsize,
    fail_download: bool,
}

struct AdmissionProbeOwner {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    downloads: AtomicUsize,
    numeric_downloads: AtomicUsize,
}

impl AccelProvider for AdmissionProbeOwner {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "integer admission fixture".into()
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("admission should precede transfer") })
    }
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        anyhow::bail!("test allocation owner supplies input")
    }
    fn download_integer<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelIntegerDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("admission should precede transfer") })
    }
    fn download_numeric<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.numeric_downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("native complex download unsupported by fixture") })
    }
}

impl runmat_accelerate_api::AccelProvider for CountedIntegerOwner {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("integer preparation must use native readback") })
    }
    fn upload(
        &self,
        host: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &runmat_accelerate_api::HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<runmat_accelerate_api::GpuTensorHandle>
    {
        self.inner.upload_numeric_raw(host)
    }
    fn download_integer<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelIntegerDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self.fail_download {
            return Box::pin(async { anyhow::bail!("injected integer download failure") });
        }
        self.inner.download_integer(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.numeric_downloads.fetch_add(1, Ordering::SeqCst);
        if self.fail_download {
            return Box::pin(async { anyhow::bail!("injected integer download failure") });
        }
        self.inner.download_numeric(handle)
    }
}

fn host(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).expect("typed integer tensor"))
}

#[test]
fn all_classes_empty_nd_and_significant_bit_boundary() {
    let cases = [
        IntegerStorage::I8(vec![-1]),
        IntegerStorage::I16(vec![-1]),
        IntegerStorage::I32(vec![-1]),
        IntegerStorage::I64(vec![i64::MIN]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1_u64 << 54]),
    ];
    for storage in cases {
        let value = host(storage, vec![1, 1, 1]);
        let prepared =
            futures::executor::block_on(prepare_direct_integer_f64_boundary(&value, None))
                .expect("host boundary");
        assert!(prepared.is_exact());
        assert!(std::ptr::eq(prepared.source(), &value));
        assert!(prepared.resident_payload().is_none());
    }
    let empty = host(IntegerStorage::U64(vec![]), vec![0, 2, 3]);
    assert!(
        futures::executor::block_on(prepare_direct_integer_f64_boundary(&empty, None))
            .expect("empty boundary")
            .is_exact()
    );
    let inexact = host(
        IntegerStorage::U64(vec![1_u64 << 54, (1_u64 << 54) + 1]),
        vec![1, 2, 1],
    );
    assert!(
        !futures::executor::block_on(prepare_direct_integer_f64_boundary(&inexact, None))
            .expect("inexact classification")
            .is_exact()
    );
    assert!(integer_is_exact_f64(&IntValue::I64(i64::MIN)));
}

#[test]
fn wide_host_scan_uses_invocation_budget_and_preserves_typed_failure() {
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let observed = turns.clone();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4096).unwrap()),
        source.token(),
        move || {
            observed.set(observed.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let value = host(
        IntegerStorage::I64(vec![1; SCAN_CHUNK * 2 + 1]),
        vec![1, SCAN_CHUNK * 2 + 1],
    );
    assert!(
        futures::executor::block_on(prepare_direct_integer_f64_boundary(
            &value,
            Some(&mut budget)
        ))
        .expect("cooperative scan")
        .is_exact()
    );
    assert_eq!(turns.get(), 2);

    let mut failing = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        || Box::pin(async { Err(ExecutionServiceError::Infrastructure("turn failed".into())) }),
    );
    assert!(matches!(
        futures::executor::block_on(prepare_direct_integer_f64_boundary(&value, Some(&mut failing))),
        Err(IntegerBoundaryFailure::Execution(ExecutionServiceError::Infrastructure(message)))
        if message == "turn failed"
    ));
    source.cancel(CancellationReason::User);
    assert!(matches!(
        futures::executor::block_on(prepare_direct_integer_f64_boundary(
            &value,
            Some(&mut budget)
        )),
        Err(IntegerBoundaryFailure::Execution(
            ExecutionServiceError::Cancelled
        ))
    ));
}

#[test]
fn pending_turn_abandonment_drops_service_future() {
    struct PendingTurn(Rc<Cell<bool>>);
    impl Future for PendingTurn {
        type Output = Result<(), ExecutionServiceError>;
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }
    impl Drop for PendingTurn {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }
    let source = CancellationSource::new();
    let dropped = Rc::new(Cell::new(false));
    let witness = dropped.clone();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || Box::pin(PendingTurn(witness.clone())),
    );
    let value = host(IntegerStorage::U64(vec![1]), vec![1, 1]);
    let mut future = Box::pin(prepare_direct_integer_f64_boundary(
        &value,
        Some(&mut budget),
    ));
    let waker = noop_waker();
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    drop(future);
    assert!(dropped.get());
}

#[test]
fn aggregate_scope_is_explicit_not_a_synchronous_fallback() {
    let aggregate = Value::output_list(vec![Value::Int(IntValue::U64(1_u64 << 54))]);
    assert!(matches!(
        futures::executor::block_on(prepare_direct_integer_f64_boundary(&aggregate, None)),
        Err(IntegerBoundaryFailure::AggregateNeedsTraversal)
    ));
}

#[test]
fn resident_wide_payload_is_retained_after_one_download_and_source_stays_live() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static CountedIntegerOwner = Box::leak(Box::new(CountedIntegerOwner {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        downloads: AtomicUsize::new(0),
        numeric_downloads: AtomicUsize::new(0),
        fail_download: false,
    }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![1_u64 << 54, (1_u64 << 54) + 1]),
        vec![1, 2, 1],
    )
    .expect("native tensor");
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
        .expect("resident source");
    let source = Value::GpuTensor(handle.clone());
    let prepared = futures::executor::block_on(prepare_direct_integer_f64_boundary(&source, None))
        .expect("resident preparation");
    assert!(!prepared.is_exact());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.numeric_downloads.load(Ordering::SeqCst), 1);
    assert_eq!(
        prepared
            .resident_payload()
            .expect("retained payload")
            .real()
            .expect("real payload")
            .shape,
        vec![1, 2, 1]
    );
    assert!(matches!(
        &prepared.resident_payload().expect("retained payload").real().expect("real payload").data,
        HostIntegerDataOwned::U64(values) if values == &vec![1_u64 << 54, (1_u64 << 54) + 1]
    ));
    assert!(std::ptr::eq(prepared.source(), &source));
    drop(prepared);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.numeric_downloads.load(Ordering::SeqCst), 1);
    assert!(futures::executor::block_on(provider.download_integer(&handle)).is_ok());
    handle.release().expect("release source");
}

#[test]
fn resident_provider_failure_keeps_original_cause_and_source_lease() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static CountedIntegerOwner = Box::leak(Box::new(CountedIntegerOwner {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        downloads: AtomicUsize::new(0),
        numeric_downloads: AtomicUsize::new(0),
        fail_download: true,
    }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let tensor = Tensor::new_integer(IntegerStorage::I64(vec![1]), vec![1, 1]).unwrap();
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
    let value = Value::GpuTensor(handle.clone());
    let failure =
        match futures::executor::block_on(prepare_direct_integer_f64_boundary(&value, None)) {
            Err(IntegerBoundaryFailure::Operational(error)) => error,
            _ => panic!("provider failure must retain operational cause"),
        };
    assert!(failure.source.as_ref().is_some_and(|source| source
        .to_string()
        .contains("injected integer download failure")));
    assert_eq!(failure.gpu_gather_retry, GpuGatherRetry::Never);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.numeric_downloads.load(Ordering::SeqCst), 1);
    assert!(futures::executor::block_on(provider.inner.download_integer(&handle)).is_ok());
    handle.release().unwrap();
}

#[test]
fn native_complex_integer_components_are_scanned_once_without_floating_projection() {
    use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView, HostNumericTensorView};
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static CountedIntegerOwner = Box::leak(Box::new(CountedIntegerOwner {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        downloads: AtomicUsize::new(0),
        numeric_downloads: AtomicUsize::new(0),
        fail_download: false,
    }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let raw = [1_i64 << 54, i64::MIN, (1_i64 << 54) + 1, 0];
    let handle = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::I64(&raw),
            shape: &[1, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .expect("native complex upload");
    let value = Value::GpuTensor(handle.clone());
    let prepared = futures::executor::block_on(prepare_direct_integer_f64_boundary(&value, None))
        .expect("native complex exactness");
    assert!(!prepared.is_exact());
    let retained = prepared
        .resident_payload()
        .and_then(PreparedIntegerPayload::complex)
        .expect("one typed interleaved payload");
    assert_eq!(retained.storage, GpuTensorStorage::ComplexInterleaved);
    assert!(
        matches!(&retained.data, runmat_accelerate_api::HostNumericDataOwned::I64(values) if values.as_slice() == raw)
    );
    assert_eq!(provider.numeric_downloads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert!(futures::executor::block_on(provider.inner.download_numeric(&handle)).is_ok());
    handle.release().unwrap();
}

#[test]
fn unavailable_lease_fails_before_download_and_complex_provider_error_is_preserved() {
    use runmat_accelerate_api::{
        GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned, NumericElementType,
    };
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static AdmissionProbeOwner = Box::leak(Box::new(AdmissionProbeOwner {
        owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
        numeric_downloads: AtomicUsize::new(0),
    }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for data in [
        HostNumericDataOwned::I32(vec![1]),
        HostNumericDataOwned::I64(vec![1]),
    ] {
        let allocation = provider
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data,
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        allocation.probe.mark_unavailable();
        let value = Value::GpuTensor(allocation.handle.clone());
        let error =
            match futures::executor::block_on(prepare_direct_integer_f64_boundary(&value, None)) {
                Err(IntegerBoundaryFailure::Operational(error)) => error,
                _ => panic!("unavailable lease must fail"),
            };
        assert!(error.source.is_some());
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }

    let complex = provider
        .owner
        .allocate_zeroed(
            vec![1, 1],
            NumericElementType::I64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap();
    let complex_value = Value::GpuTensor(complex.handle.clone());
    let error = match futures::executor::block_on(prepare_direct_integer_f64_boundary(
        &complex_value,
        None,
    )) {
        Err(IntegerBoundaryFailure::Operational(error)) => error,
        _ => panic!("complex integer representation needs native provider readback"),
    };
    assert!(error.source.as_ref().is_some_and(|cause| cause
        .to_string()
        .contains("native complex download unsupported by fixture")));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.numeric_downloads.load(Ordering::SeqCst), 1);
}

#[test]
fn canonical_view_constructor_rejects_shape_overflow_before_any_transfer() {
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, GpuTensorView, NumericElementType,
    };
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::U64, GpuTensorStorage::Real)
        .unwrap();
    let malformed = allocation
        .handle
        .with_view(GpuTensorView::contiguous(vec![usize::MAX, 2]));
    assert!(
        malformed.is_err(),
        "owner must reject overflow before a handle exists"
    );
}
