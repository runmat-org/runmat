use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorStorage, HostLogicalTensorView,
    HostNumericDataView, HostNumericTensorView, NumericElementType, ThreadProviderGuard,
};
use runmat_value::{ComplexTensor, Tensor, Value};

fn call(name: &str, input: Value) -> runmat_runtime::execution::RootedValue {
    block_on(runmat_runtime::call_builtin_async(name, &[input])).unwrap()
}

fn values(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(value) => vec![*value],
        Value::Complex(re, im) => vec![*re, *im],
        Value::Tensor(tensor) => tensor.materialize_f64(),
        Value::ComplexTensor(tensor) => tensor
            .materialize_f64()
            .into_iter()
            .flat_map(|(re, im)| [re, im])
            .collect(),
        other => panic!("unexpected host angle result: {other:?}"),
    }
}

#[test]
fn angle_fallback_preserves_full_rank_class_complexity_and_input_ownership() {
    let provider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    for name in ["tand", "cosd", "cospi"] {
        for (data, storage, element, host) in [
            (
                HostNumericDataView::F64(&[0.5]),
                GpuTensorStorage::Real,
                NumericElementType::F64,
                Value::Tensor(Tensor::new(vec![0.5], vec![1, 1, 1]).unwrap()),
            ),
            (
                HostNumericDataView::F32(&[0.5]),
                GpuTensorStorage::Real,
                NumericElementType::F32,
                Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1, 1]).unwrap()),
            ),
            (
                HostNumericDataView::F32(&[0.5, 0.25]),
                GpuTensorStorage::ComplexInterleaved,
                NumericElementType::F32,
                Value::ComplexTensor(
                    ComplexTensor::from_f32(vec![(0.5, 0.25)], vec![1, 1, 1]).unwrap(),
                ),
            ),
        ] {
            let input = runmat_accelerate_api::upload_numeric_protecting(
                provider,
                &HostNumericTensorView {
                    data,
                    storage,
                    shape: &[1, 1, 1],
                },
                &[],
            )
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
            let host_result = call(name, host);
            let expected = values(host_result.value());
            let resident_result = call(name, Value::GpuTensor(input.clone()));
            let Value::GpuTensor(output) = resident_result.value() else {
                panic!("{name} must preserve explicit resident output");
            };
            assert_eq!(output.shape(), &[1, 1, 1], "{name}");
            assert_eq!(output.physical().element_type(), element);
            assert_eq!(runmat_accelerate_api::handle_storage(output), storage);
            assert_eq!(output.device_id(), input.device_id());
            let downloaded = block_on(provider.download(output)).unwrap();
            assert_eq!(downloaded.data, expected, "{name}");
            resident_result.with_owned(|value| {
                let Value::GpuTensor(output) = value else {
                    unreachable!("resident output was checked above")
                };
                output.release().unwrap();
            });
            input.allocation_lease().validate_live().unwrap();
        }

        for shape in [vec![1, 1, 1], vec![1, 2, 1], vec![1, 0, 1]] {
            let logical = vec![1; shape.iter().product()];
            let input = runmat_accelerate_api::upload_logical_protecting(
                provider,
                &HostLogicalTensorView {
                    data: &logical,
                    shape: &shape,
                },
                &[],
            )
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
            for extensions in [false, true] {
                let _compat =
                    runmat_runtime::compatibility::push_runmat_extensions_enabled(extensions);
                let result = block_on(runmat_runtime::call_builtin_async(
                    name,
                    &[Value::GpuTensor(input.clone())],
                ));
                if !extensions {
                    assert!(result.is_err(), "strict {name} logical admission");
                    continue;
                }
                let host_result = call(name, Value::Bool(true));
                let expected = values(host_result.value());
                let resident_result = result.unwrap();
                let Value::GpuTensor(output) = resident_result.value() else {
                    panic!("resident logical {name} output required");
                };
                assert_eq!(output.shape(), shape);
                assert_eq!(output.physical().element_type(), NumericElementType::F64);
                assert!(!runmat_accelerate_api::handle_is_logical(output));
                assert_eq!(output.device_id(), input.device_id());
                let downloaded = block_on(provider.download(output)).unwrap();
                assert_eq!(downloaded.data, vec![expected[0]; logical.len()]);
                resident_result.with_owned(|value| {
                    let Value::GpuTensor(output) = value else {
                        unreachable!("resident output was checked above")
                    };
                    output.release().unwrap();
                });
                input.allocation_lease().validate_live().unwrap();
            }
        }
    }
}
