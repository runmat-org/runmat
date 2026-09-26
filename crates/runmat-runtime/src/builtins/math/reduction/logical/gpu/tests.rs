use super::*;
use crate::builtins::common::{spec::ReductionNaN, test_support};
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProviderFuture, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex, OnceLock,
    },
    task::{Context, Poll},
};

#[derive(Clone, Copy)]
enum Behavior {
    AliasInput,
    WrongShape,
    AliasIntermediate,
    PendingDownload,
    UnknownDevice,
    MissingStorage,
}

struct ReductionProvider {
    inner: InProcessProvider,
    behavior: Behavior,
    created: Mutex<Vec<GpuTensorHandle>>,
    calls: AtomicUsize,
}

impl ReductionProvider {
    fn new(behavior: Behavior) -> Self {
        Self {
            inner: InProcessProvider::new(),
            behavior,
            created: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
        }
    }
    fn result(
        &self,
        input: &GpuTensorHandle,
        axis: Option<usize>,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if matches!(self.behavior, Behavior::AliasInput)
            || (matches!(self.behavior, Behavior::AliasIntermediate) && axis == Some(1))
        {
            return Ok(input.clone());
        }
        let shape = if matches!(self.behavior, Behavior::WrongShape) {
            vec![1, 3]
        } else if let Some(axis) = axis {
            let mut shape = input.shape.clone();
            shape[axis] = 1;
            shape
        } else {
            vec![1, 1]
        };
        let data = vec![1.0; shape.iter().product()];
        let mut output = self.inner.upload(&HostTensorView {
            data: &data,
            shape: &shape,
        })?;
        self.created.lock().unwrap().push(output.clone());
        match self.behavior {
            Behavior::UnknownDevice => output.device_id = u32::MAX - 7,
            Behavior::MissingStorage => output.descriptor.storage = None,
            _ => {}
        }
        Ok(output)
    }
}

impl AccelProvider for ReductionProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        if matches!(self.behavior, Behavior::PendingDownload) {
            Box::pin(std::future::pending())
        } else {
            self.inner.download(handle)
        }
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        // This provider created the deliberately malformed handle, so it can
        // release its own allocation despite the bad returned device metadata.
        if matches!(self.behavior, Behavior::UnknownDevice) && handle.device_id != self.device_id()
        {
            let created = self.created.lock().unwrap();
            let original = created
                .iter()
                .find(|value| value.buffer_id == handle.buffer_id)
                .ok_or_else(|| anyhow::anyhow!("unknown malformed allocation"))?;
            return self.inner.free(original);
        }
        self.inner.free(handle)
    }
    fn reduce_all<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.result(input, None) })
    }
    fn reduce_any<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.result(input, None) })
    }
    fn reduce_all_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        axis: usize,
        _: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.result(input, Some(axis)) })
    }
    fn reduce_any_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        axis: usize,
        _: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.result(input, Some(axis)) })
    }
}

fn provider(behavior: Behavior) -> &'static ReductionProvider {
    static ALIAS: OnceLock<ReductionProvider> = OnceLock::new();
    static SHAPE: OnceLock<ReductionProvider> = OnceLock::new();
    static INTERMEDIATE: OnceLock<ReductionProvider> = OnceLock::new();
    static PENDING: OnceLock<ReductionProvider> = OnceLock::new();
    static UNKNOWN: OnceLock<ReductionProvider> = OnceLock::new();
    static MISSING: OnceLock<ReductionProvider> = OnceLock::new();
    match behavior {
        Behavior::AliasInput => ALIAS.get_or_init(|| ReductionProvider::new(behavior)),
        Behavior::WrongShape => SHAPE.get_or_init(|| ReductionProvider::new(behavior)),
        Behavior::AliasIntermediate => {
            INTERMEDIATE.get_or_init(|| ReductionProvider::new(behavior))
        }
        Behavior::PendingDownload => PENDING.get_or_init(|| ReductionProvider::new(behavior)),
        Behavior::UnknownDevice => UNKNOWN.get_or_init(|| ReductionProvider::new(behavior)),
        Behavior::MissingStorage => MISSING.get_or_init(|| ReductionProvider::new(behavior)),
    }
}

fn configs() -> [&'static LogicalReductionConfig; 2] {
    [
        &crate::builtins::math::reduction::all::CONFIG,
        &crate::builtins::math::reduction::any::CONFIG,
    ]
}

#[test]
fn malformed_owner_or_missing_storage_is_freed_before_next_dimension_hook() {
    let _state = test_support::accel_test_lock();
    for behavior in [Behavior::UnknownDevice, Behavior::MissingStorage] {
        let provider = provider(behavior);
        let _provider = ThreadProviderGuard::set(Some(provider));
        for config in configs() {
            provider.created.lock().unwrap().clear();
            provider.calls.store(0, Ordering::SeqCst);
            let input = provider
                .upload(&HostTensorView {
                    data: &[1.0; 4],
                    shape: &[2, 2],
                })
                .unwrap();
            let error = block_on(reduce(
                config,
                input.clone(),
                ReductionSpec::Dimensions(vec![1, 2]),
                ReductionNaN::Include,
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), config.internal.identifier);
            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                1,
                "malformed first result must not reach the next hook"
            );
            assert!(block_on(provider.inner.download(&input)).is_ok());
            let created = provider.created.lock().unwrap();
            assert_eq!(created.len(), 1);
            assert!(
                block_on(provider.inner.download(&created[0])).is_err(),
                "invoked provider must release output even with unknown device metadata"
            );
            provider.free(&input).unwrap();
        }
    }
}

#[test]
fn invalid_reduction_outputs_never_free_borrowed_inputs_and_free_fresh_results() {
    let _state = test_support::accel_test_lock();
    for behavior in [
        Behavior::AliasInput,
        Behavior::WrongShape,
        Behavior::AliasIntermediate,
    ] {
        let provider = provider(behavior);
        let _provider = ThreadProviderGuard::set(Some(provider));
        for config in configs() {
            provider.created.lock().unwrap().clear();
            let input = provider
                .upload(&HostTensorView {
                    data: &[1.0; 4],
                    shape: &[2, 2],
                })
                .unwrap();
            let spec = if matches!(behavior, Behavior::AliasIntermediate) {
                ReductionSpec::Dimensions(vec![1, 2])
            } else {
                ReductionSpec::All
            };
            let error =
                block_on(reduce(config, input.clone(), spec, ReductionNaN::Include)).unwrap_err();
            assert_eq!(error.identifier(), config.internal.identifier);
            assert!(block_on(provider.inner.download(&input)).is_ok());
            for output in provider.created.lock().unwrap().iter() {
                assert!(block_on(provider.inner.download(output)).is_err());
            }
            provider.free(&input).unwrap();
        }
    }
}

#[test]
fn cancelling_reduction_download_releases_fresh_outputs_but_not_input() {
    let _state = test_support::accel_test_lock();
    let provider = provider(Behavior::PendingDownload);
    let _provider = ThreadProviderGuard::set(Some(provider));
    for config in configs() {
        provider.created.lock().unwrap().clear();
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0; 4],
                shape: &[2, 2],
            })
            .unwrap();
        let mut future = Box::pin(reduce(
            config,
            input.clone(),
            ReductionSpec::All,
            ReductionNaN::Include,
        ));
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(&noop_waker())),
            Poll::Pending
        ));
        drop(future);
        assert!(block_on(provider.inner.download(&input)).is_ok());
        let created = provider.created.lock().unwrap();
        assert_eq!(created.len(), 1);
        assert!(block_on(provider.inner.download(&created[0])).is_err());
        provider.free(&input).unwrap();
    }
}
