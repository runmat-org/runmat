use super::*;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, HostNumericTensorView, NumericElementType,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy)]
enum Mode {
    Malformed,
    Failure,
    Unsupported,
    Success,
}
struct Provider {
    device: u32,
    mode: Mode,
    freed: AtomicUsize,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.device
    }
    fn device_info(&self) -> String {
        "typed upload promotion fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> Result<GpuTensorHandle> {
        anyhow::bail!("unexpected legacy upload")
    }
    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused") })
    }
    fn free(&self, _: &GpuTensorHandle) -> Result<()> {
        self.freed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn upload_numeric_raw(&self, host: &HostNumericTensorView) -> Result<GpuTensorHandle> {
        match self.mode {
            Mode::Unsupported => Err(runmat_accelerate_api::unsupported_provider_operation(
                "integer transfer test",
            )),
            Mode::Failure => Err(anyhow!("injected integer allocation failure")),
            Mode::Malformed => Ok(GpuTensorHandle::new(host.shape.to_vec(), self.device, 1)),
            Mode::Success => Ok(GpuTensorHandle::new(host.shape.to_vec(), self.device, 1)
                .with_numeric_descriptor(host.element_type(), host.storage)),
        }
    }
}

#[test]
fn native_auto_integer_upload_fallback_is_only_typed_unsupported() {
    let value = Value::Tensor(
        Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![9_007_199_254_740_993, -3]),
            vec![1, 2],
        )
        .unwrap(),
    );
    for mode in [
        Mode::Malformed,
        Mode::Failure,
        Mode::Unsupported,
        Mode::Success,
    ] {
        let provider = Box::leak(Box::new(Provider {
            device: runmat_accelerate_api::next_device_id(),
            mode,
            freed: AtomicUsize::new(0),
        }));
        let auto = NativeAutoOffload::new(provider, ThresholdConfig::default());
        let result = auto.promote_tensor_protecting(&value, 1, None, &[]);
        match mode {
            Mode::Malformed => {
                assert!(runmat_accelerate_api::is_invalid_numeric_upload_result(
                    &result.unwrap_err()
                ));
                assert_eq!(provider.freed.load(Ordering::SeqCst), 1);
            }
            Mode::Failure => {
                assert!(result
                    .unwrap_err()
                    .to_string()
                    .contains("allocation failure"));
                assert_eq!(provider.freed.load(Ordering::SeqCst), 0);
            }
            Mode::Unsupported => {
                assert_eq!(result.unwrap(), value);
                assert_eq!(provider.freed.load(Ordering::SeqCst), 0);
            }
            Mode::Success => {
                let Value::GpuTensor(handle) = result.unwrap() else {
                    panic!("resident integer result")
                };
                assert_eq!(
                    handle.descriptor.element_type,
                    Some(NumericElementType::I64)
                );
                assert_eq!(handle.descriptor.storage, Some(GpuTensorStorage::Real));
                assert_eq!(handle.device_id, provider.device);
                provider.free(&handle).unwrap();
            }
        }
    }
}
