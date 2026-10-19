use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostNumericTensorView, NumericElementType,
};
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum Mode {
    Malformed,
    Failure,
    Unsupported,
    Success,
}
struct Provider {
    owner: TestAllocationOwner,
    mode: Mode,
    allocations: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
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
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        match self.mode {
            Mode::Unsupported => Err(runmat_accelerate_api::unsupported_provider_operation(
                "integer transfer test",
            )
            .into()),
            Mode::Failure => Err(anyhow!("injected integer allocation failure").into()),
            Mode::Malformed | Mode::Success => {
                let HostNumericDataView::I64(values) = host.data else {
                    panic!("exact integer upload")
                };
                let data = if matches!(self.mode, Mode::Malformed) {
                    HostNumericDataOwned::F64(vec![0.0; values.len()])
                } else {
                    HostNumericDataOwned::I64(values.to_vec())
                };
                let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
                    data,
                    shape: host.shape.to_vec(),
                    storage: host.storage,
                })?;
                self.allocations.lock().unwrap().push(allocation.probe);
                Ok(allocation.handle)
            }
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
            owner: TestAllocationOwner::new(),
            mode,
            allocations: Mutex::new(Vec::new()),
        }));
        let auto = NativeAutoOffload::new(provider, ThresholdConfig::default());
        let result = auto.promote_tensor_protecting(&value, 1, None, &[]);
        match mode {
            Mode::Malformed => {
                assert!(runmat_accelerate_api::is_invalid_numeric_upload_result(
                    &result.unwrap_err()
                ));
                assert_eq!(provider.allocations.lock().unwrap()[0].retirements(), 1);
            }
            Mode::Failure => {
                assert!(result
                    .unwrap_err()
                    .to_string()
                    .contains("allocation failure"));
                assert!(provider.allocations.lock().unwrap().is_empty());
            }
            Mode::Unsupported => {
                assert_eq!(result.unwrap(), value);
                assert!(provider.allocations.lock().unwrap().is_empty());
            }
            Mode::Success => {
                let Value::GpuTensor(handle) = result.unwrap() else {
                    panic!("resident integer result")
                };
                assert_eq!(handle.physical().element_type(), NumericElementType::I64);
                assert_eq!(handle.physical().storage(), GpuTensorStorage::Real);
                assert_eq!(handle.device_id(), provider.owner.device_id());
                let probe = provider.allocations.lock().unwrap()[0].clone();
                probe
                    .with_data(&handle, |data| {
                        let HostNumericDataView::I64(values) = data else {
                            panic!("native integer storage")
                        };
                        assert_eq!(values, &[9_007_199_254_740_993, -3]);
                    })
                    .unwrap();
                assert_eq!(probe.retirements(), 0);
                drop(handle);
                assert_eq!(probe.retirements(), 1);
            }
        }
    }
}
