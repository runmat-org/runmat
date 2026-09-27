use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle, HostTensorView,
    ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::Value;
use std::{
    future::Future,
    sync::{Mutex, OnceLock},
    task::{Context, Poll},
};

#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    PendingReduction,
    PendingDownload,
    FailedDownload,
    Alias,
    MissingStorage,
}

#[derive(Default)]
struct State {
    fault: Fault,
    temporary: Option<GpuTensorHandle>,
    freed: Vec<u64>,
    calls: usize,
}

struct Provider {
    inner: InProcessProvider,
    state: Mutex<State>,
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
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.state.lock().unwrap().freed.push(handle.buffer_id);
        self.inner.free(handle)
    }
    fn reduce_min<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let fault = {
                let mut state = self.state.lock().unwrap();
                state.calls += 1;
                state.fault
            };
            if matches!(fault, Fault::PendingReduction) {
                runmat_accelerate_api::set_handle_logical(input, true);
                return std::future::pending().await;
            }
            if matches!(fault, Fault::Alias) {
                return Ok(input.clone());
            }
            let mut output = self.inner.reduce_min(input).await?;
            self.state.lock().unwrap().temporary = Some(output.clone());
            if matches!(fault, Fault::MissingStorage) {
                output.descriptor.storage = None;
            }
            Ok(output)
        })
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let fault = self.state.lock().unwrap().fault;
            match fault {
                Fault::PendingDownload => std::future::pending().await,
                Fault::FailedDownload => anyhow::bail!("injected probe download failure"),
                _ => self.inner.download(input).await,
            }
        })
    }
}

fn fixture(fault: Fault) -> (&'static Provider, GpuTensorHandle) {
    static PROVIDER: OnceLock<Provider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| Provider {
        inner: InProcessProvider::new(),
        state: Mutex::default(),
    });
    *provider.state.lock().unwrap() = State {
        fault,
        ..State::default()
    };
    let input = provider
        .inner
        .upload(&HostTensorView {
            data: &[1.0, 4.0],
            shape: &[1, 2],
        })
        .unwrap();
    (provider, input)
}

// One runtime library supplies both dispatch inventory and implementation.
#[test]
fn logarithm_root_and_gammaln_probe_lifetimes_through_dispatch() {
    for name in [
        "log", "log10", "log1p", "log2", "sqrt", "realsqrt", "gammaln",
    ] {
        for fault in [
            Fault::PendingReduction,
            Fault::PendingDownload,
            Fault::FailedDownload,
            Fault::Alias,
            Fault::MissingStorage,
        ] {
            let (provider, input) = fixture(fault);
            let _provider = ThreadProviderGuard::set(Some(provider));
            let args = [Value::GpuTensor(input.clone())];
            drop(runmat_runtime::dispatcher::call_builtin_async_with_outputs(
                name, &args, 1,
            ));
            assert_eq!(provider.state.lock().unwrap().calls, 0);
            let mut call = Box::pin(runmat_runtime::dispatcher::call_builtin_async_with_outputs(
                name, &args, 1,
            ));
            let result = call.as_mut().poll(&mut Context::from_waker(&noop_waker()));
            if matches!(fault, Fault::PendingReduction | Fault::PendingDownload) {
                assert!(matches!(result, Poll::Pending), "{name}");
            } else {
                assert!(matches!(result, Poll::Ready(Err(_))), "{name}");
            }
            drop(call);
            assert!(!runmat_accelerate_api::handle_is_logical(&input), "{name}");
            let state = provider.state.lock().unwrap();
            assert_eq!(state.calls, 1, "{name}: no retry after provider failure");
            assert!(!state.freed.contains(&input.buffer_id), "{name}");
            if let Some(temporary) = &state.temporary {
                assert_eq!(
                    state
                        .freed
                        .iter()
                        .filter(|id| **id == temporary.buffer_id)
                        .count(),
                    1,
                    "{name}"
                );
            } else {
                assert!(state.freed.is_empty(), "{name}");
            }
            drop(state);
            assert_eq!(
                block_on(provider.inner.download(&input)).unwrap().data,
                [1.0, 4.0]
            );
            provider.inner.free(&input).unwrap();
        }
    }
}
