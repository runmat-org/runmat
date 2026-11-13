use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorStorage, HostLogicalTensorView,
    HostNumericDataOwned, HostNumericDataView, HostNumericTensorView, NumericElementType,
    ThreadProviderGuard,
};
use runmat_value::Value;

#[test]
fn unary_fallback_preserves_nd_singleton_native_classes_and_complexity() {
    let provider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for (name, data, storage, expected_data, expected_storage) in [
        (
            "sqrt",
            HostNumericDataView::F64(&[-4.0]),
            GpuTensorStorage::Real,
            HostNumericDataOwned::F64(vec![0.0, 2.0]),
            GpuTensorStorage::ComplexInterleaved,
        ),
        (
            "sqrt",
            HostNumericDataView::F32(&[-4.0]),
            GpuTensorStorage::Real,
            HostNumericDataOwned::F32(vec![0.0, 2.0]),
            GpuTensorStorage::ComplexInterleaved,
        ),
        (
            "sqrt",
            HostNumericDataView::F32(&[-4.0, 0.0]),
            GpuTensorStorage::ComplexInterleaved,
            HostNumericDataOwned::F32(vec![0.0, 2.0]),
            GpuTensorStorage::ComplexInterleaved,
        ),
        (
            "sqrt",
            HostNumericDataView::F32(&[4.0]),
            GpuTensorStorage::Real,
            HostNumericDataOwned::F32(vec![2.0]),
            GpuTensorStorage::Real,
        ),
        (
            "abs",
            HostNumericDataView::I16(&[-7]),
            GpuTensorStorage::Real,
            HostNumericDataOwned::I16(vec![7]),
            GpuTensorStorage::Real,
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
        let result = block_on(runmat_runtime::call_builtin_async(
            name,
            &[Value::GpuTensor(input.clone())],
        ))
        .unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("expected resident {name} result");
        };
        assert_eq!(output.shape(), &[1, 1, 1], "{name}");
        assert_eq!(output.device_id(), input.device_id());
        assert_eq!(
            runmat_accelerate_api::handle_storage(output),
            expected_storage
        );
        let host = block_on(provider.download_numeric(output)).unwrap();
        assert_eq!(host.shape, vec![1, 1, 1]);
        assert_eq!(host.data, expected_data);
        input.allocation_lease().validate_live().unwrap();
    }

    for name in ["sin", "cos", "tan"] {
        let input = runmat_accelerate_api::upload_logical_protecting(
            provider,
            &HostLogicalTensorView {
                data: &[1],
                shape: &[1, 1, 1],
            },
            &[],
        )
        .unwrap();
        let result = block_on(runmat_runtime::call_builtin_async(
            name,
            &[Value::GpuTensor(input.clone())],
        ))
        .unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("expected resident logical {name} result");
        };
        assert_eq!(output.shape(), &[1, 1, 1]);
        assert_eq!(output.physical().element_type(), NumericElementType::F64);
        assert!(!runmat_accelerate_api::handle_is_logical(output));
        assert_eq!(output.device_id(), input.device_id());
        let expected = match name {
            "sin" => 1.0f64.sin(),
            "cos" => 1.0f64.cos(),
            _ => 1.0f64.tan(),
        };
        assert_eq!(
            block_on(provider.download_numeric(output)).unwrap().data,
            HostNumericDataOwned::F64(vec![expected])
        );
    }
}
