use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, HostTensorView, ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::{Tensor, Value};
use std::{
    future::Future,
    sync::Mutex,
    task::{Context, Poll},
};

#[path = "support/retained_provider.rs"]
mod retained_provider;

#[derive(Default)]
struct State {
    downloads: usize,
    pending: bool,
    success: bool,
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
            let (pending, success) = {
                let mut state = self.state.lock().unwrap();
                state.downloads += 1;
                (state.pending, state.success)
            };
            runmat_accelerate_api::clear_residency(handle).unwrap();
            if pending {
                std::future::pending().await
            } else if success {
                self.inner.download(handle).await
            } else {
                anyhow::bail!("injected download failure")
            }
        })
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
fn failed_and_cancelled_downloads_restore_borrowed_residency_without_free_or_retry() {
    let provider = retained_provider::retained_test_provider!(PROVIDER: Provider = Provider {
        inner: InProcessProvider::new(),
        state: Mutex::new(State::default()),
    });
    let _provider = ThreadProviderGuard::set(Some(provider));
    runmat_accelerate::ensure_residency_hooks();
    let handle = provider
        .upload(&HostTensorView {
            data: &[0.5],
            shape: &[1, 1],
        })
        .unwrap();
    runmat_accelerate_api::mark_residency(&handle);
    let descriptor = handle.wire_descriptor();
    let physical = handle.allocation_lease().downgrade();
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
    assert!(runmat_accelerate::fusion_residency::is_resident(&handle));
    assert_eq!(handle.wire_descriptor(), descriptor);
    assert!(physical.acquire().is_some());
    provider.state.lock().unwrap().pending = false;
    let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "discretize",
        &args,
        1,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:discretize:Internal"));
    assert_eq!(provider.state.lock().unwrap().downloads, 2);
    assert!(runmat_accelerate::fusion_residency::is_resident(&handle));
    assert_eq!(handle.wire_descriptor(), descriptor);
    assert!(physical.acquire().is_some());
    assert_eq!(
        block_on(provider.inner.download(&handle)).unwrap().data,
        vec![0.5]
    );
    drop(args);
    drop(handle);
    assert!(physical.acquire().is_none());
    provider.state.lock().unwrap().success = true;
    let handle = provider
        .upload(&HostTensorView {
            data: &[0.0, 2.0, 4.0],
            shape: &[1, 3],
        })
        .unwrap();
    runmat_accelerate_api::mark_residency(&handle);
    let descriptor = handle.wire_descriptor();
    let physical = handle.allocation_lease().downgrade();
    let args = [
        Value::GpuTensor(handle.clone()),
        Value::Tensor(Tensor::new(vec![0.0, 1.0, 3.0, 5.0], vec![1, 4]).unwrap()),
    ];
    let owned_output = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "discretize",
        &args,
        1,
    ))
    .unwrap();
    let output = match owned_output.value() {
        Value::OutputList(values) => &values[0],
        other => other,
    };
    let Value::Tensor(output) = output else {
        panic!("numeric bins")
    };
    assert_eq!(output.materialize_f64(), vec![1.0, 2.0, 3.0]);
    assert!(runmat_accelerate::fusion_residency::is_resident(&handle));
    assert_eq!(handle.wire_descriptor(), descriptor);
    assert_eq!(provider.state.lock().unwrap().downloads, 3);
    assert!(physical.acquire().is_some());
    drop(args);
    drop(handle);
    assert!(physical.acquire().is_none());
}
