use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocation, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned,
    HostTensorView, ThreadProviderGuard,
};
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};
use std::sync::Mutex;

// Actual native payloads belong to TestAllocationOwner. These probes only
// expose that same storage to the provider's typed download contract.
struct NativeProvider {
    owner: TestAllocationOwner,
    records: Mutex<Vec<(u64, RetirementProbe)>>,
}

impl NativeProvider {
    fn allocate(&self, data: HostNumericDataOwned, storage: GpuTensorStorage) -> TestAllocation {
        let allocation = self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data,
                storage,
                shape: vec![1, 1, 1],
            })
            .unwrap();
        self.records
            .lock()
            .unwrap()
            .push((allocation.handle.buffer_id(), allocation.probe.clone()));
        allocation
    }
}

impl AccelProvider for NativeProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "typed extrema capability fallback".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("extrema host fallback does not restore an allocation")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("native inputs must not pass through a double download") })
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            let probe = self
                .records
                .lock()
                .unwrap()
                .iter()
                .find(|(id, _)| *id == input.buffer_id())
                .unwrap()
                .1
                .clone();
            let data = probe.with_data(input, |data| match data {
                HostNumericDataView::F32(data) => HostNumericDataOwned::F32(data.to_vec()),
                HostNumericDataView::F64(data) => HostNumericDataOwned::F64(data.to_vec()),
                HostNumericDataView::U64(data) => HostNumericDataOwned::U64(data.to_vec()),
                _ => panic!("unexpected fixture class"),
            })?;
            Ok(HostNumericTensorOwned {
                data,
                shape: input.shape().to_vec(),
                storage: input.physical().storage(),
            })
        })
    }
}

#[test]
fn resident_extrema_fallback_preserves_single_complex_and_exact_integer_scalars() {
    let provider = Box::leak(Box::new(NativeProvider {
        owner: TestAllocationOwner::new(),
        records: Mutex::new(Vec::new()),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for name in ["min", "max"] {
        for (left_data, right_data, storage, expected) in [
            (
                HostNumericDataOwned::F64(vec![3.0, 0.0]),
                HostNumericDataOwned::F64(vec![4.0, 0.0]),
                GpuTensorStorage::ComplexInterleaved,
                Value::Num(if name == "min" { 3.0 } else { 4.0 }),
            ),
            (
                HostNumericDataOwned::F64(vec![3.0, 4.0]),
                HostNumericDataOwned::F64(vec![4.0, 0.0]),
                GpuTensorStorage::ComplexInterleaved,
                if name == "min" {
                    Value::Complex(4.0, 0.0)
                } else {
                    Value::Complex(3.0, 4.0)
                },
            ),
            (
                HostNumericDataOwned::F32(vec![3.5]),
                HostNumericDataOwned::F32(vec![1.25]),
                GpuTensorStorage::Real,
                Value::Tensor(
                    Tensor::from_f32(vec![if name == "min" { 1.25 } else { 3.5 }], vec![1, 1, 1])
                        .unwrap(),
                ),
            ),
            (
                HostNumericDataOwned::F32(vec![3.0, 4.0]),
                HostNumericDataOwned::F32(vec![4.0, 0.0]),
                GpuTensorStorage::ComplexInterleaved,
                Value::ComplexTensor(
                    ComplexTensor::from_f32(
                        vec![if name == "min" {
                            (4.0, 0.0)
                        } else {
                            (3.0, 4.0)
                        }],
                        vec![1, 1, 1],
                    )
                    .unwrap(),
                ),
            ),
            (
                HostNumericDataOwned::U64(vec![9_007_199_254_740_993]),
                HostNumericDataOwned::U64(vec![9_007_199_254_740_992]),
                GpuTensorStorage::Real,
                Value::Int(runmat_value::IntValue::U64(if name == "min" {
                    9_007_199_254_740_992
                } else {
                    9_007_199_254_740_993
                })),
            ),
        ] {
            let left = provider.allocate(left_data, storage);
            let right = provider.allocate(right_data, storage);
            for provenance in [
                runmat_accelerate_api::GpuHandleProvenance::Explicit,
                runmat_accelerate_api::GpuHandleProvenance::Automatic,
            ] {
                let result = futures::executor::block_on(crate::call_builtin_async(
                    name,
                    &[
                        Value::GpuTensor(left.handle.with_provenance(provenance).unwrap()),
                        Value::GpuTensor(right.handle.with_provenance(provenance).unwrap()),
                    ],
                ))
                .unwrap();
                assert_eq!(result, expected, "{name}");
                match result {
                    Value::Tensor(t) => assert!(matches!(
                        t.into_numeric_storage().unwrap(),
                        NumericStorage::F32(_)
                    )),
                    Value::ComplexTensor(t) => {
                        assert!(matches!(t.into_complex_storage(), ComplexStorage::F32(_)))
                    }
                    Value::Int(_) | Value::Num(_) | Value::Complex(..) => (),
                    other => panic!("unexpected result {other:?}"),
                }
            }
            for allocation in [left, right] {
                allocation
                    .handle
                    .allocation_lease()
                    .validate_live()
                    .unwrap();
                assert_eq!(allocation.probe.retirements(), 0);
                drop(allocation.handle);
                assert_eq!(allocation.probe.retirements(), 1);
            }
        }
    }
}

#[test]
fn resident_single_with_double_scalar_uses_resolved_double_class() {
    let provider = Box::leak(Box::new(NativeProvider {
        owner: TestAllocationOwner::new(),
        records: Mutex::new(Vec::new()),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    for (name, expected) in [("min", 1.25), ("max", 3.5)] {
        let input = provider.allocate(HostNumericDataOwned::F32(vec![3.5]), GpuTensorStorage::Real);
        for args in [
            [Value::GpuTensor(input.handle.clone()), Value::Num(1.25)],
            [Value::Num(1.25), Value::GpuTensor(input.handle.clone())],
        ] {
            let output =
                futures::executor::block_on(crate::call_builtin_async(name, &args)).unwrap();
            assert_eq!(output, Value::Num(expected));
        }
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}
