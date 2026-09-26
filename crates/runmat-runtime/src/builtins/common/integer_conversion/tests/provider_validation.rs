use super::super::cast_value;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle,
    HostIntegerTensorView, HostTensorView, IntegerElementType, ProviderPrecision,
    ThreadProviderGuard,
};
use runmat_types::IntegerClass;
use runmat_value::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex, OnceLock,
};

#[derive(Clone, Copy)]
enum ResultMode {
    MissingStorage,
    UnknownOwner,
    Alias,
    Failure,
    Unsupported,
    Pending,
}

struct Provider {
    inner: InProcessProvider,
    mode: Mutex<ResultMode>,
    calls: AtomicUsize,
    downloads: AtomicUsize,
    freed: Mutex<Vec<u64>>,
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn upload_integer(&self, view: &HostIntegerTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload_integer(view)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        self.inner.download(handle)
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.freed.lock().unwrap().push(handle.buffer_id);
        // This deliberately malformed provider still owns its returned buffer.
        let mut owned = handle.clone();
        owned.device_id = self.device_id();
        self.inner.free(&owned)
    }
    fn cast_to_integer<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        target: IntegerElementType,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mode = *self.mode.lock().unwrap();
        Box::pin(async move {
            match mode {
                ResultMode::Pending => {
                    runmat_accelerate_api::set_handle_logical(input, true);
                    runmat_accelerate_api::record_handle_transpose(input, 99, 88);
                    futures::future::pending().await
                }
                ResultMode::Alias => Ok(input.clone()),
                ResultMode::Failure => anyhow::bail!("injected cast execution failure"),
                ResultMode::Unsupported => Err(
                    runmat_accelerate_api::unsupported_provider_operation("cast unavailable"),
                ),
                ResultMode::MissingStorage | ResultMode::UnknownOwner => {
                    let mut output = self.inner.cast_to_integer(input, target).await?;
                    match mode {
                        ResultMode::MissingStorage => output.descriptor.storage = None,
                        ResultMode::UnknownOwner => output.device_id = u32::MAX,
                        _ => unreachable!(),
                    }
                    Ok(output)
                }
            }
        })
    }
}

#[test]
fn abandoning_pending_cast_restores_borrowed_metadata_and_allocation() {
    use futures::{task::noop_waker, FutureExt};
    use std::task::{Context, Poll};
    let _lock = test_support::accel_test_lock();
    let provider = provider(ResultMode::Pending);
    let _current = ThreadProviderGuard::set(Some(provider));
    let input = input(provider);
    let original_transpose = runmat_accelerate_api::handle_transpose_info(&input);
    let mut call = Box::pin(cast_value(
        Value::GpuTensor(input.clone()),
        IntegerClass::Int8,
    ));
    let waker = noop_waker();
    assert!(matches!(
        call.poll_unpin(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert!(runmat_accelerate_api::handle_is_logical(&input));
    drop(call);
    assert!(!runmat_accelerate_api::handle_is_logical(&input));
    assert_eq!(
        runmat_accelerate_api::handle_transpose_info(&input),
        original_transpose
    );
    assert!(provider.freed.lock().unwrap().is_empty());
    assert!(block_on(provider.inner.download(&input)).is_ok());
    provider.free(&input).unwrap();
}

fn provider(mode: ResultMode) -> &'static Provider {
    static PROVIDER: OnceLock<Provider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| Provider {
        inner: InProcessProvider::new(),
        mode: Mutex::new(mode),
        calls: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        freed: Mutex::new(Vec::new()),
    });
    *provider.mode.lock().unwrap() = mode;
    provider.calls.store(0, Ordering::Relaxed);
    provider.downloads.store(0, Ordering::Relaxed);
    provider.freed.lock().unwrap().clear();
    provider
}

fn input(provider: &Provider) -> GpuTensorHandle {
    provider
        .upload(&HostTensorView {
            data: &[3.5, -1.0],
            shape: &[1, 2],
        })
        .unwrap()
}

#[test]
fn malformed_fresh_outputs_are_freed_without_touching_borrowed_input() {
    let _lock = test_support::accel_test_lock();
    for mode in [
        ResultMode::MissingStorage,
        ResultMode::UnknownOwner,
        ResultMode::Alias,
    ] {
        let provider = provider(mode);
        let _current = ThreadProviderGuard::set(Some(provider));
        let input = input(provider);
        assert!(block_on(cast_value(
            Value::GpuTensor(input.clone()),
            IntegerClass::Int8
        ))
        .is_err());
        let freed = provider.freed.lock().unwrap().clone();
        assert!(!freed.contains(&input.buffer_id));
        assert_eq!(freed.len(), usize::from(!matches!(mode, ResultMode::Alias)));
        assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
        assert!(block_on(provider.inner.download(&input)).is_ok());
        provider.free(&input).unwrap();
    }
}

#[test]
fn missing_input_storage_is_rejected_before_hook_or_download() {
    let _lock = test_support::accel_test_lock();
    let provider = provider(ResultMode::Failure);
    let _current = ThreadProviderGuard::set(Some(provider));
    let input = input(provider);
    let mut malformed = input.clone();
    malformed.descriptor.storage = None;
    assert!(block_on(cast_value(Value::GpuTensor(malformed), IntegerClass::Int8)).is_err());
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    provider.free(&input).unwrap();
}

#[test]
fn only_typed_unsupported_permits_owner_download_fallback() {
    let _lock = test_support::accel_test_lock();
    for mode in [ResultMode::Failure, ResultMode::Unsupported] {
        let provider = provider(mode);
        let _current = ThreadProviderGuard::set(Some(provider));
        let input = input(provider);
        let result = block_on(cast_value(
            Value::GpuTensor(input.clone()),
            IntegerClass::Int8,
        ));
        match mode {
            ResultMode::Failure => {
                assert!(
                    matches!(result, Err(super::super::CastError::Internal(ref message)) if message.contains("injected cast execution failure"))
                );
                assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
            }
            ResultMode::Unsupported => {
                let Value::GpuTensor(output) = result.unwrap() else {
                    panic!("resident fallback")
                };
                assert_eq!(output.device_id, input.device_id);
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&output),
                    Some(IntegerElementType::I8)
                );
                assert_eq!(provider.downloads.load(Ordering::Relaxed), 1);
                provider.free(&output).unwrap();
            }
            _ => unreachable!(),
        }
        provider.free(&input).unwrap();
    }
}
