//! Public dispatch regression: histogram computation gathers once, and result
//! restoration never replays the builtin after an operational upload failure.
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::allocation::WeakAllocationLease;
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async_with_outputs, GpuGatherRetry};
use runmat_value::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Debug)]
struct RestoreFailure;
impl std::fmt::Display for RestoreFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("histogram result transfer failed")
    }
}
impl std::error::Error for RestoreFailure {}

struct Provider {
    inner: InProcessProvider,
    fail_at: usize,
    unsupported: bool,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
    outputs: Mutex<Vec<WeakAllocationLease>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "histogram native storage owner".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F32
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("histogram outputs require the native typed transfer")
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        assert_eq!(host.data.element_type(), NumericElementType::F64);
        let call = self.uploads.fetch_add(1, Ordering::SeqCst) + 1;
        if call == self.fail_at {
            return Err(if self.unsupported {
                unsupported_provider_operation("histogram native double unavailable")
            } else {
                anyhow::Error::new(RestoreFailure)
            }
            .into());
        }
        let output = self.inner.upload_numeric_raw(host)?;
        self.outputs
            .lock()
            .unwrap()
            .push(output.allocation_lease().downgrade());
        Ok(output)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }
}

fn contains_cause<T: std::error::Error + 'static>(
    error: &(dyn std::error::Error + 'static),
) -> bool {
    let mut cursor = Some(error);
    while let Some(error) = cursor {
        if error.is::<T>() {
            return true;
        }
        cursor = error.source();
    }
    false
}

#[test]
fn public_histogram_restoration_preserves_causes_classes_and_partial_ownership() {
    // Each failure position tests both an untouched output set and a genuinely
    // allocated counts output which must retire when locations cannot publish.
    for explicit in [false, true] {
        for unsupported in [false, true] {
            for fail_at in [0, 1, 2] {
                let provider = Box::leak(Box::new(Provider {
                    inner: InProcessProvider::new(),
                    fail_at,
                    unsupported,
                    uploads: AtomicUsize::new(0),
                    downloads: AtomicUsize::new(0),
                    outputs: Mutex::new(Vec::new()),
                }));
                let _selected = ThreadProviderGuard::set(Some(provider));
                let source = provider
                    .inner
                    .upload_numeric_raw(&HostNumericTensorView {
                        data: HostNumericDataView::F32(&[0.0, 0.0, 0.5, 1.0]),
                        shape: &[2, 2],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap();
                let input = source
                    .with_provenance(if explicit {
                        GpuHandleProvenance::Explicit
                    } else {
                        GpuHandleProvenance::Automatic
                    })
                    .unwrap();
                drop(source);
                let before = input.wire_descriptor();
                let input_weak = input.allocation_lease().downgrade();
                let args = [Value::GpuTensor(input.clone()), Value::Num(3.0)];
                let result = block_on(call_builtin_async_with_outputs("imhist", &args, 2));
                assert_eq!(
                    provider.downloads.load(Ordering::SeqCst),
                    1,
                    "no gather replay"
                );
                assert_eq!(
                    provider.uploads.load(Ordering::SeqCst),
                    if fail_at == 0 { 2 } else { fail_at }
                );
                if fail_at != 0 && (!unsupported || explicit) {
                    let error = result.expect_err("restoration failure must remain terminal");
                    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                    if unsupported {
                        assert!(contains_cause::<UnsupportedProviderOperation>(&error));
                    } else {
                        assert!(contains_cause::<RestoreFailure>(&error));
                    }
                } else {
                    result.expect("histogram outputs").with_owned(|value| {
                        let Value::OutputList(outputs) = value else {
                            panic!("two requested outputs must be retained");
                        };
                        assert_eq!(outputs.len(), 2);
                        let outputs = outputs.try_into_owned().expect("unique histogram outputs");
                        for (output, expected) in outputs
                            .into_iter()
                            .zip([vec![2., 1., 1.], vec![0., 0.5, 1.]])
                        {
                            match output {
                                Value::GpuTensor(handle) => {
                                    assert_eq!(fail_at, 0);
                                    assert_eq!(handle.device_id(), input.device_id());
                                    assert!(!handle.shares_allocation_with(&input));
                                    assert_eq!(
                                        handle.physical().element_type(),
                                        NumericElementType::F64
                                    );
                                    assert_eq!(handle.shape(), [3, 1]);
                                    assert_eq!(handle_is_explicit(&handle), explicit);
                                    let host =
                                        block_on(provider.inner.download_numeric(&handle)).unwrap();
                                    assert_eq!(host.data, HostNumericDataOwned::F64(expected));
                                    handle.release().unwrap();
                                }
                                Value::Tensor(tensor) => {
                                    assert!(unsupported && !explicit && fail_at != 0);
                                    assert_eq!(tensor.shape(), &[3, 1]);
                                    assert_eq!(
                                        tensor.numeric_dtype(),
                                        runmat_value::NumericDType::F64
                                    );
                                    assert_eq!(tensor.materialize_f64(), expected);
                                }
                                _ => panic!("unexpected histogram output"),
                            }
                        }
                    });
                }
                let outputs = provider.outputs.lock().unwrap();
                assert_eq!(outputs.len(), if fail_at == 0 { 2 } else { fail_at - 1 });
                assert!(
                    outputs.iter().all(|weak| weak.acquire().is_none()),
                    "partial or complete output owner leaked"
                );
                drop(outputs);
                assert_eq!(input.wire_descriptor(), before);
                assert!(input_weak.acquire().is_some());
                let unchanged = block_on(provider.inner.download_numeric(&input)).unwrap();
                assert_eq!(
                    unchanged.data,
                    HostNumericDataOwned::F32(vec![0., 0., 0.5, 1.])
                );
                drop(args);
                input.release().unwrap();
                assert!(input_weak.acquire().is_none());
            }
        }
    }
}
