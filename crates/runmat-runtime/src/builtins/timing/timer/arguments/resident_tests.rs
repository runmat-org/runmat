//! Real owner witnesses for timer numeric-option admission and abandonment.

use super::*;
use runmat_accelerate_api::*;
use runmat_value::Tensor;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingOwner {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    mode: AtomicUsize,
    downloads: AtomicUsize,
    abandoned: AtomicUsize,
}

struct PendingRead<'a>(&'a AtomicUsize);

impl Drop for PendingRead<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl AccelProvider for CountingOwner {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn upload_numeric_raw(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            match self.mode.load(Ordering::SeqCst) {
                1 => {
                    let _pending = PendingRead(&self.abandoned);
                    futures::future::pending::<()>().await;
                    unreachable!()
                }
                2 => anyhow::bail!("timer injected download failure"),
                _ => self.inner.download_numeric(handle).await,
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

#[test]
fn late_property_rejection_never_reads_an_earlier_resident_value() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = Box::leak(Box::new(CountingOwner {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        mode: AtomicUsize::new(2),
        downloads: AtomicUsize::new(0),
        abandoned: AtomicUsize::new(0),
    }));
    let _provider = ThreadProviderGuard::set(Some(owner));
    let tensor = Tensor::new(vec![0.25], vec![1, 1]).expect("scalar tensor");
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(owner, &tensor)
        .expect("resident value")
        .with_provenance(GpuHandleProvenance::Automatic)
        .expect("automatic residency");
    let resident = Value::GpuTensor(handle.clone());

    for invalid in ["UnknownProperty", "Running"] {
        let error = futures::executor::block_on(prepare_timer_property_pairs(
            &[
                Value::String("Period".into()),
                resident.clone(),
                Value::String(invalid.into()),
                Value::Num(1.0),
            ],
            false,
        ))
        .err()
        .expect("late property rejection");
        assert_eq!(error.identifier(), Some("RunMat:timer:InvalidProperty"));
    }
    let error = futures::executor::block_on(prepare_timer_property_pairs(
        &[
            Value::String("Period".into()),
            resident.clone(),
            Value::String("BusyMode".into()),
            Value::String("invalid".into()),
        ],
        false,
    ))
    .err()
    .expect("late resident host validation");
    assert_eq!(error.identifier(), Some("RunMat:timer:InvalidProperty"));
    let mut running = default_timer_object("running".into());
    running
        .properties
        .insert("Running".into(), Value::String("on".into()));
    let error = futures::executor::block_on(super::super::setter::set_timer_property_object(
        Value::Object(running),
        TimerProperty::StartDelay,
        resident.clone(),
    ))
    .expect_err("frozen running property");
    assert_eq!(error.identifier(), Some("RunMat:timer:InvalidProperty"));
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);

    owner.mode.store(0, Ordering::SeqCst);
    let prepared = futures::executor::block_on(prepare_timer_property_pairs(
        &[Value::String("Period".into()), resident.clone()],
        false,
    ))
    .expect("automatic resident numeric option remains admitted");
    assert_eq!(prepared.len(), 1);
    assert!(matches!(&prepared[0].value, Value::Tensor(_)));
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 1);

    owner.mode.store(2, Ordering::SeqCst);
    let direct = futures::executor::block_on(gather_value_async(&resident))
        .expect_err("direct provider failure");
    let timer = futures::executor::block_on(prepare_timer_property_pairs(
        &[Value::String("Period".into()), resident.clone()],
        false,
    ))
    .err()
    .expect("timer provider failure");
    assert_eq!(timer.identifier(), direct.identifier());
    assert_eq!(timer.span, direct.span);
    assert_eq!(timer.gpu_gather_retry(), direct.gpu_gather_retry());
    assert_eq!(timer.failure_origin(), direct.failure_origin());
    assert_eq!(timer.identifier(), Some("RunMat:gather:DownloadFailed"));
    let source = std::error::Error::source(&timer).expect("owning provider source");
    assert!(format!("{source:?}").contains("timer injected download failure"));

    owner.mode.store(1, Ordering::SeqCst);
    let args = [Value::String("Period".into()), resident];
    let mut pending = Box::pin(prepare_timer_property_pairs(&args, false));
    let mut context = std::task::Context::from_waker(futures::task::noop_waker_ref());
    let before_pending = owner.downloads.load(Ordering::SeqCst);
    assert!(pending.as_mut().poll(&mut context).is_pending());
    assert_eq!(owner.downloads.load(Ordering::SeqCst), before_pending + 1);
    drop(pending);
    assert_eq!(owner.abandoned.load(Ordering::SeqCst), 1);
    handle
        .allocation_lease()
        .validate_live()
        .expect("abandoned gather cannot retire borrowed resident input");
}
