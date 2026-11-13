//! A real in-process owner with controllable download suspension/failure.
use super::*;
use crate::builtins::common::test_support;
use runmat_accelerate_api::*;
use runmat_value::{CellArray, IntegerStorage, StructValue, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum DownloadMode {
    Ready,
    Pending,
    FailSecond,
}

struct DownloadOwner {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    mode: Mutex<DownloadMode>,
    downloads: AtomicUsize,
    abandoned: AtomicUsize,
}
struct PendingRead<'a>(&'a AtomicUsize);
impl Drop for PendingRead<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl AccelProvider for DownloadOwner {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }
    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(h)
    }
    fn download_numeric<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            let ordinal = self.downloads.fetch_add(1, Ordering::SeqCst);
            let mode = *self.mode.lock().unwrap();
            match mode {
                DownloadMode::Pending => {
                    let _read = PendingRead(&self.abandoned);
                    futures::future::pending::<()>().await;
                    unreachable!()
                }
                DownloadMode::FailSecond if ordinal == 1 => {
                    anyhow::bail!("injected download failure")
                }
                _ => self.inner.download_numeric(h).await,
            }
        })
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
}

fn cell(values: Vec<Value>) -> Value {
    let len = values.len();
    Value::Cell(CellArray::new(values, 1, len).unwrap())
}

#[test]
fn admission_precedes_access_and_nested_membership_preserves_exact_borrowed_values() {
    runmat_gc::gc_test_context(|| {
        use std::future::Future;
        let _lock = test_support::accel_test_lock();
        let owner = Box::leak(Box::new(DownloadOwner {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            mode: Mutex::new(DownloadMode::Ready),
            downloads: AtomicUsize::new(0),
            abandoned: AtomicUsize::new(0),
        }));
        let _provider = ThreadProviderGuard::set(Some(owner));
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(owner, &tensor).unwrap();
        let resident = Value::GpuTensor(handle.clone());
        let nested = cell(vec![cell(vec![resident.clone()])]);
        let invalid = cell(vec![Value::Struct(StructValue::new())]);
        let error = futures::executor::block_on(super::super::must_be_member::evaluate(
            crate::builtins::common::validation::tests::retained_arguments(vec![
                resident.clone(),
                invalid,
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:mustBeMember:InvalidArgument")
        );
        let invalid_shape = Value::CharArray(
            runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![2, 1])
                .unwrap(),
        );
        let error = futures::executor::block_on(value_is_member_async(&resident, &invalid_shape))
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:mustBeMember:InvalidArgument")
        );
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);

        let allowed = cell(vec![Value::Tensor(tensor)]);
        drop(Box::pin(value_is_member_async(&nested, &allowed)));
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        assert!(value_is_member(&allowed, &allowed).unwrap());
        assert!(futures::executor::block_on(value_is_member_async(&nested, &allowed)).unwrap());
        assert!(futures::executor::block_on(value_is_member_async(&allowed, &nested)).unwrap());
        assert!(!futures::executor::block_on(value_is_member_async(
            &nested,
            &Value::Num(u64::MAX as f64)
        ))
        .unwrap());
        assert!(futures::executor::block_on(value_is_member_atoms_async(
            &nested,
            &[ValidationAtom::Integer(runmat_value::IntValue::U64(
                u64::MAX
            ))]
        ))
        .unwrap());

        // The first materialized leaf is discarded on a later failed download;
        // neither borrowed provider allocation is released by membership.
        let pair = cell(vec![resident.clone(), nested.clone()]);
        owner.downloads.store(0, Ordering::SeqCst);
        *owner.mode.lock().unwrap() = DownloadMode::FailSecond;
        let error =
            futures::executor::block_on(value_is_member_async(&pair, &allowed)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:gpu:DownloadFailed"));
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 2);

        *owner.mode.lock().unwrap() = DownloadMode::Pending;
        let mut future = Box::pin(value_is_member_async(&nested, &allowed));
        let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
        assert!(future.as_mut().poll(&mut cx).is_pending());
        drop(future);
        assert_eq!(owner.abandoned.load(Ordering::SeqCst), 1);
        *owner.mode.lock().unwrap() = DownloadMode::Ready;
        assert!(futures::executor::block_on(owner.download_numeric(&handle)).is_ok());
        assert!(futures::executor::block_on(value_is_member_async(&nested, &allowed)).unwrap());
        assert_eq!(handle_integer_type(&handle), Some(IntegerElementType::U64));
        let physical = handle.allocation_lease().downgrade();
        drop((resident, nested, pair, allowed));
        runmat_gc::gc_collect_major().unwrap();
        assert!(futures::executor::block_on(owner.download_numeric(&handle)).is_ok());
        drop(handle);
        assert!(physical.acquire().is_none());
    });
}
