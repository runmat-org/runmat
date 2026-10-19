//! Public dispatch witnesses for typed sinc capability and genuine owned fallback.
use futures::executor::block_on;
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async, GpuGatherRetry};
use runmat_value::Value;
use std::sync::Mutex;

#[path = "sinc_provider_contract/complex_single.rs"]
mod complex_single;

#[derive(Clone, Copy)]
enum Mode {
    Supported,
    Unsupported,
    PlainUnsupportedText,
    Failure,
    MalformedDownload,
    Alias,
}
struct Provider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    precision: ProviderPrecision,
    mode: Mode,
    trace: Mutex<Vec<&'static str>>,
}
impl AccelProvider for Provider {
    fn precision(&self) -> ProviderPrecision {
        self.precision
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "sinc capability and ownership fixture".into()
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.trace.lock().unwrap().push("upload");
        self.inner.upload(view)
    }
    fn upload_numeric_raw(
        &self,
        view: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.trace.lock().unwrap().push("upload");
        self.inner.upload_numeric_raw(view)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            self.trace.lock().unwrap().push("download");
            let mut result = self.inner.download(input).await?;
            if matches!(self.mode, Mode::MalformedDownload) {
                result.data.pop();
            }
            Ok(result)
        })
    }
    fn unary_sinc<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.trace.lock().unwrap().push("sinc");
            match self.mode {
                Mode::Supported => self.inner.unary_sinc(input).await,
                Mode::Unsupported | Mode::MalformedDownload => Err(unsupported_provider_operation(
                    "arbitrary typed capability detail",
                )),
                Mode::PlainUnsupportedText => {
                    Err(anyhow::anyhow!("unary_sinc not supported by provider"))
                }
                Mode::Failure => Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "sinc transport fault",
                )
                .into()),
                Mode::Alias => Ok(input.with_view(input.view().clone())?),
            }
        })
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            if input.physical().element_type() == NumericElementType::F32 {
                self.trace.lock().unwrap().push("download");
                self.inner.download_numeric(input).await
            } else {
                let host = self.download(input).await?;
                Ok(HostNumericTensorOwned {
                    data: HostNumericDataOwned::F64(host.data),
                    shape: host.shape,
                    storage: host.storage,
                })
            }
        })
    }
}

fn provider(mode: Mode) -> &'static Provider {
    Box::leak(Box::new(Provider {
        inner: Default::default(),
        precision: ProviderPrecision::F64,
        mode,
        trace: Mutex::new(Vec::new()),
    }))
}

fn input(provider: &Provider, complex: bool, explicit: bool) -> GpuTensorHandle {
    let (data, storage) = if complex {
        (
            vec![0.0, 0.0, 0.5, 0.25],
            GpuTensorStorage::ComplexInterleaved,
        )
    } else {
        (vec![0.0, 0.5], GpuTensorStorage::Real)
    };
    provider
        .inner
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F64(&data),
            shape: &[2, 1],
            storage,
        })
        .unwrap()
        .with_provenance(if explicit {
            GpuHandleProvenance::Explicit
        } else {
            GpuHandleProvenance::Automatic
        })
        .unwrap()
}

#[test]
fn supported_and_typed_unsupported_sinc_preserve_resident_shape_class_and_values() {
    for mode in [Mode::Supported, Mode::Unsupported] {
        for complex in [false, true] {
            for explicit in [false, true] {
                let provider = provider(mode);
                let _selected = ThreadProviderGuard::set(Some(provider));
                let input = input(provider, complex, explicit);
                let args = [Value::GpuTensor(input.clone())];
                let result = block_on(call_builtin_async("sinc", &args)).unwrap();
                let Value::GpuTensor(output) = result.value() else {
                    panic!("supported resident restoration required")
                };
                assert_eq!(output.shape(), input.shape());
                assert_eq!(handle_precision(output), Some(ProviderPrecision::F64));
                assert_eq!(handle_storage(output), handle_storage(&input));
                assert_eq!(handle_provenance(output), handle_provenance(&input));
                assert!(!output.shares_allocation_with(&input));
                let expected_trace = if matches!(mode, Mode::Supported) {
                    vec!["sinc"]
                } else {
                    vec!["sinc", "download", "upload"]
                };
                assert_eq!(*provider.trace.lock().unwrap(), expected_trace);
                let values = block_on(provider.inner.download(output)).unwrap().data;
                let expected = if complex {
                    let a = std::f64::consts::PI * 0.5;
                    let b = std::f64::consts::PI * 0.25;
                    let denominator = a * a + b * b;
                    vec![
                        1.0,
                        0.0,
                        (a.sin() * b.cosh() * a + a.cos() * b.sinh() * b) / denominator,
                        (a.cos() * b.sinh() * a - a.sin() * b.cosh() * b) / denominator,
                    ]
                } else {
                    vec![1.0, 2.0 / std::f64::consts::PI]
                };
                assert_eq!(values.len(), expected.len());
                for (actual, expected) in values.into_iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                drop(result);
                drop(args);
                input.allocation_lease().validate_live().unwrap();
                assert_eq!(
                    block_on(provider.inner.download(&input)).unwrap().shape,
                    vec![2, 1]
                );
                input.release().unwrap();
            }
        }
    }
}

#[test]
fn sinc_operational_text_and_alias_failures_never_download_or_retry() {
    for mode in [Mode::PlainUnsupportedText, Mode::Failure, Mode::Alias] {
        let provider = provider(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = input(provider, false, false);
        let args = [Value::GpuTensor(input.clone())];
        let error = block_on(call_builtin_async("sinc", &args)).unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never, "{error:?}");
        assert_eq!(*provider.trace.lock().unwrap(), vec!["sinc"]);
        if matches!(mode, Mode::Failure) {
            let carrier = error
                .source
                .as_ref()
                .unwrap()
                .downcast_ref::<ProviderExecutionError>()
                .unwrap();
            let cause = std::error::Error::source(carrier)
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap();
            assert_eq!(cause.kind(), std::io::ErrorKind::BrokenPipe);
            assert_eq!(cause.to_string(), "sinc transport fault");
        }
        drop(error);
        drop(args);
        input.allocation_lease().validate_live().unwrap();
        input.release().unwrap();
    }
}

#[test]
fn sinc_malformed_complex_download_rejects_before_restore_and_retains_owner() {
    let provider = provider(Mode::MalformedDownload);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = input(provider, true, false);
    let args = [Value::GpuTensor(input.clone())];
    let error = block_on(call_builtin_async("sinc", &args)).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:gather:DownloadFailed"));
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    assert!(
        error.source.is_some(),
        "gather rejection keeps its actual cause"
    );
    assert_eq!(*provider.trace.lock().unwrap(), vec!["sinc", "download"]);
    drop(error);
    drop(args);
    input.allocation_lease().validate_live().unwrap();
    input.release().unwrap();
}
