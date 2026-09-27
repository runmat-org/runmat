use runmat_accelerate_api::*;
use runmat_value::Value;
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[derive(Debug)]
struct Operational;
impl std::fmt::Display for Operational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dot not supported by provider")
    }
}
impl Error for Operational {}
struct Provider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    unsupported: bool,
    hooks: AtomicUsize,
    downloads: AtomicUsize,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "dot typed fallback fixture".into()
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(input)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(input)
    }
    fn dot<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        _: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.hooks.fetch_add(1, Ordering::SeqCst);
            if self.unsupported {
                Err(unsupported_provider_operation(
                    "dot not supported by provider",
                ))
            } else {
                Err(anyhow::Error::new(Operational))
            }
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn only_typed_unsupported_can_gather_dot_inputs() {
    for unsupported in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            unsupported,
            hooks: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let left = provider
            .inner
            .upload(&HostTensorView {
                data: &[1., 2.],
                shape: &[2, 1],
            })
            .unwrap();
        let right = provider
            .inner
            .upload(&HostTensorView {
                data: &[3., 4.],
                shape: &[2, 1],
            })
            .unwrap();
        let result = futures::executor::block_on(runmat_runtime::call_builtin_async(
            "dot",
            &[
                Value::GpuTensor(left.clone()),
                Value::GpuTensor(right.clone()),
            ],
        ));
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
        if unsupported {
            let output = result.unwrap();
            let value = match output {
                Value::Num(value) => value,
                Value::Tensor(tensor) => tensor.materialize_f64()[0],
                Value::GpuTensor(handle) => {
                    futures::executor::block_on(provider.inner.download(&handle))
                        .unwrap()
                        .data[0]
                }
                other => panic!("unexpected dot result {other:?}"),
            };
            assert_eq!(value, 11.);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
        } else {
            let error = result.unwrap_err();
            assert_eq!(
                error.gpu_gather_retry,
                runmat_runtime::GpuGatherRetry::Never
            );
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            let mut cursor: &(dyn Error + 'static) = &error;
            while !cursor.is::<Operational>() {
                cursor = cursor.source().expect("original typed error retained");
            }
        }
        left.allocation_lease().validate_live().unwrap();
        right.allocation_lease().validate_live().unwrap();
    }
}
