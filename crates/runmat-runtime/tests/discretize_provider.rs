use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, HostTensorView, ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::{Tensor, Value};
use std::{
    future::Future,
    sync::{Mutex, OnceLock},
    task::{Context, Poll},
};

#[derive(Default)]
struct State {
    downloads: usize,
    frees: usize,
    pending: bool,
}
struct Provider {
    inner: InProcessProvider,
    state: Mutex<State>,
}
impl AccelProvider for Provider {
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn download<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async move {
            let pending = {
                let mut state = self.state.lock().unwrap();
                state.downloads += 1;
                state.pending
            };
            runmat_accelerate_api::set_handle_logical(handle, true);
            if pending {
                std::future::pending().await
            } else {
                anyhow::bail!("injected download failure")
            }
        })
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.state.lock().unwrap().frees += 1;
        self.inner.free(handle)
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "discretize transfer fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
}

#[test]
fn failed_and_cancelled_downloads_restore_borrowed_metadata_without_free_or_retry() {
    static PROVIDER: OnceLock<Provider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| Provider {
        inner: InProcessProvider::new(),
        state: Mutex::new(State::default()),
    });
    let _provider = ThreadProviderGuard::set(Some(provider));
    let handle = provider
        .upload(&HostTensorView {
            data: &[0.5],
            shape: &[1, 1],
        })
        .unwrap();
    let args = [
        Value::GpuTensor(handle.clone()),
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
    ];
    let mut future = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
        "discretize",
        &args,
        1,
    ));
    drop(future);
    assert_eq!(provider.state.lock().unwrap().downloads, 0);
    provider.state.lock().unwrap().pending = true;
    future = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
        "discretize",
        &args,
        1,
    ));
    let waker = noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    drop(future);
    assert!(!runmat_accelerate_api::handle_is_logical(&handle));
    assert_eq!(provider.state.lock().unwrap().frees, 0);
    provider.state.lock().unwrap().pending = false;
    let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "discretize",
        &args,
        1,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:discretize:Internal"));
    assert_eq!(provider.state.lock().unwrap().downloads, 2);
    assert!(!runmat_accelerate_api::handle_is_logical(&handle));
    assert_eq!(provider.state.lock().unwrap().frees, 0);
    assert_eq!(
        block_on(provider.inner.download(&handle)).unwrap().data,
        vec![0.5]
    );
    provider.free(&handle).unwrap();
}
