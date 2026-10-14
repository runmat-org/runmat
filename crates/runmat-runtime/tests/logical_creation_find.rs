//! Genuine native logical storage through single-library public dispatch.
use futures::executor::block_on;
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async_with_outputs, GpuGatherRetry};
use runmat_value::Value;
use std::sync::Mutex;

#[path = "../src/builtins/common/test_support/native_logical.rs"]
mod native_logical;

#[derive(Clone, Copy)]
enum Mode {
    Supported,
    Unsupported,
    Failure,
    Alias,
}

struct Provider {
    inner: native_logical::NativeLogicalProvider,
    mode: Mode,
    alias: Mutex<Option<GpuTensorHandle>>,
    trace: Mutex<Vec<&'static str>>,
}
impl AccelProvider for Provider {
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F32
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }
    fn upload_logical_raw(
        &self,
        host: &HostLogicalTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.trace.lock().unwrap().push("logical-upload");
        match self.mode {
            Mode::Supported => self.inner.upload_logical_raw(host),
            Mode::Unsupported => {
                Err(unsupported_provider_operation("native logical upload").into())
            }
            Mode::Failure => Err(runmat_accelerate_api::ProviderExecutionError::from(
                std::io::Error::new(std::io::ErrorKind::BrokenPipe, "logical transport failure"),
            )
            .with_unproven_effects()),
            Mode::Alias => Ok(self.alias.lock().unwrap().as_ref().unwrap().clone()),
        }
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.trace.lock().unwrap().push("download");
        self.inner.download_numeric(handle)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.trace.lock().unwrap().push("download");
        self.inner.download(handle)
    }
}
fn provider(mode: Mode) -> &'static Provider {
    Box::leak(Box::new(Provider {
        inner: Default::default(),
        mode,
        alias: Mutex::new(None),
        trace: Mutex::new(vec![]),
    }))
}

#[test]
fn logical_constructors_use_owned_u8_without_numeric_relabeling() {
    let provider = provider(Mode::Supported);
    let _active = ThreadProviderGuard::set(Some(provider));
    let prototype = provider
        .inner
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[7.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let borrowed = prototype.allocation_lease().downgrade();
    for (name, bit) in [("true", 1u8), ("false", 0u8)] {
        let result = block_on(call_builtin_async_with_outputs(
            name,
            &[
                Value::Num(2.0),
                Value::Num(3.0),
                Value::String("like".into()),
                Value::GpuTensor(prototype.clone()),
            ],
            1,
        ))
        .unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("resident logical output")
        };
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output.physical().element_type(), NumericElementType::U8);
        assert!(handle_is_logical(output));
        assert!(!output.shares_allocation_with(&prototype));
        assert_eq!(output.device_id(), provider.device_id());
        let retired = output.allocation_lease().downgrade();
        let host = block_on(provider.inner.download_numeric(output)).unwrap();
        assert!(matches!(host.data, HostNumericDataOwned::U8(data) if data == vec![bit;6]));
        drop(result);
        assert!(retired.acquire().is_none());
        assert!(borrowed.acquire().is_some());
    }
    assert!(!handle_is_logical(&prototype));
    drop(prototype);
    assert!(borrowed.acquire().is_none());
}

#[test]
fn find_logical_output_preserves_native_owner_and_rejects_operational_or_alias_uploads() {
    for mode in [
        Mode::Supported,
        Mode::Unsupported,
        Mode::Failure,
        Mode::Alias,
    ] {
        let owner = provider(mode);
        // This permanently retained provider must remain addressable while a
        // different provider is selected on this thread.
        unsafe { register_provider(owner) };
        let _owner_registration = ThreadProviderGuard::set(Some(owner));
        let input = owner
            .inner
            .upload_logical_raw(&HostLogicalTensorView {
                data: &[1, 1],
                shape: &[2, 1],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Automatic)
            .unwrap();
        let borrowed = input.allocation_lease().downgrade();
        if matches!(mode, Mode::Alias) {
            *owner.alias.lock().unwrap() = Some(input.clone());
        }
        let ambient = provider(Mode::Failure);
        let _ambient = ThreadProviderGuard::set(Some(ambient));
        let result = block_on(call_builtin_async_with_outputs(
            "find",
            &[Value::GpuTensor(input.clone())],
            3,
        ));
        assert_eq!(*owner.trace.lock().unwrap(), ["download", "logical-upload"]);
        assert!(
            ambient.trace.lock().unwrap().is_empty(),
            "ambient provider cannot steal output ownership"
        );
        match mode {
            Mode::Supported => {
                let result = result.unwrap();
                let Value::OutputList(outputs) = result.value() else {
                    panic!("three outputs")
                };
                assert_eq!(outputs.len(), 3);
                let Value::GpuTensor(output) = &outputs[2] else {
                    panic!("resident selected values")
                };
                assert_eq!(output.physical().element_type(), NumericElementType::U8);
                assert_eq!(output.shape(), &[2, 1]);
                assert!(handle_is_logical(output));
                assert_eq!(
                    handle_provenance(output),
                    Some(GpuHandleProvenance::Automatic)
                );
                assert_eq!(output.device_id(), owner.device_id());
                assert!(!output.shares_allocation_with(&input));
                let retired = output.allocation_lease().downgrade();
                let host = block_on(owner.inner.download_numeric(output)).unwrap();
                assert!(matches!(host.data, HostNumericDataOwned::U8(data) if data == vec![1,1]));
                drop(result);
                assert!(retired.acquire().is_none());
            }
            Mode::Unsupported => {
                assert!(matches!(
                    result.unwrap().value(),
                    Value::OutputList(outputs)
                        if outputs.len() == 3
                            && matches!(&outputs[0], Value::Tensor(t) if t.as_f64_slice() == Some(&[1.0, 2.0][..]) && t.shape() == [2, 1])
                            && matches!(&outputs[1], Value::Tensor(t) if t.as_f64_slice() == Some(&[1.0, 1.0][..]) && t.shape() == [2, 1])
                            && matches!(&outputs[2], Value::LogicalArray(data) if data.data == vec![1, 1] && data.shape == vec![2, 1])
                ));
            }
            Mode::Failure | Mode::Alias => {
                let error = result.unwrap_err();
                assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                let upload = error
                    .source
                    .as_ref()
                    .unwrap()
                    .downcast_ref::<NumericUploadError>()
                    .expect("typed upload cause");
                match mode {
                    Mode::Failure => {
                        let NumericUploadError::Provider(cause) = upload else {
                            panic!("provider failure")
                        };
                        assert_eq!(
                            std::error::Error::source(cause)
                                .and_then(|source| source.downcast_ref::<std::io::Error>())
                                .unwrap()
                                .kind(),
                            std::io::ErrorKind::BrokenPipe
                        );
                    }
                    Mode::Alias => assert!(matches!(upload, NumericUploadError::Invalid(_))),
                    _ => unreachable!(),
                }
            }
        }
        owner.alias.lock().unwrap().take();
        assert!(borrowed.acquire().is_some());
        assert!(
            matches!(block_on(owner.inner.download_numeric(&input)).unwrap().data, HostNumericDataOwned::U8(data) if data == vec![1,1])
        );
        drop(input);
        assert!(borrowed.acquire().is_none());
    }
}

#[test]
fn explicit_logical_find_rejects_unrepresentable_double_indices_before_value_upload() {
    let owner = provider(Mode::Supported);
    unsafe { register_provider(owner) };
    let _active = ThreadProviderGuard::set(Some(owner));
    let input = owner
        .inner
        .upload_logical_raw(&HostLogicalTensorView {
            data: &[1, 1],
            shape: &[2, 1],
        })
        .unwrap()
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let borrowed = input.allocation_lease().downgrade();
    let error = block_on(call_builtin_async_with_outputs(
        "find",
        &[Value::GpuTensor(input.clone())],
        3,
    ))
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    assert!(matches!(
        error
            .source
            .as_ref()
            .and_then(|source| source.downcast_ref::<NumericUploadError>()),
        Some(NumericUploadError::Provider(cause)) if cause.is_unsupported()
    ));
    assert_eq!(*owner.trace.lock().unwrap(), ["download"]);
    assert!(borrowed.acquire().is_some());
    assert!(
        matches!(block_on(owner.inner.download_numeric(&input)).unwrap().data, HostNumericDataOwned::U8(data) if data == vec![1, 1])
    );
    drop(input);
    assert!(borrowed.acquire().is_none());
}

#[test]
fn automatic_logical_find_retains_host_values_only_for_certified_unsupported() {
    let owner = provider(Mode::Unsupported);
    unsafe { register_provider(owner) };
    let _active = ThreadProviderGuard::set(Some(owner));
    let input = owner
        .inner
        .upload_logical_raw(&HostLogicalTensorView {
            data: &[1, 1],
            shape: &[2, 1],
        })
        .unwrap()
        .with_provenance(GpuHandleProvenance::Automatic)
        .unwrap();
    let output = block_on(call_builtin_async_with_outputs(
        "find",
        &[Value::GpuTensor(input.clone())],
        3,
    ))
    .unwrap();
    let Value::OutputList(outputs) = output.value() else {
        panic!("three outputs")
    };
    assert!(
        matches!(&outputs[2], Value::LogicalArray(data) if data.data == vec![1,1] && data.shape == vec![2,1])
    );
    assert_eq!(*owner.trace.lock().unwrap(), ["download", "logical-upload"]);
    assert!(input.allocation_lease().validate_live().is_ok());
}
