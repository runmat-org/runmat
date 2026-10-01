use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorView, ThreadProviderGuard,
};
use runmat_value::Value;

fn upload(
    provider: &'static InProcessProvider,
    values: &[f64],
    shape: &[usize],
) -> GpuTensorHandle {
    runmat_accelerate_api::upload_numeric_protecting(
        provider,
        &HostNumericTensorView {
            data: HostNumericDataView::F64(values),
            storage: GpuTensorStorage::Real,
            shape,
        },
        &[],
    )
    .unwrap()
    .with_provenance(GpuHandleProvenance::Explicit)
    .unwrap()
}

fn resident_result(
    provider: &'static InProcessProvider,
    name: &str,
    inputs: &[GpuTensorHandle],
) -> GpuTensorHandle {
    let arguments = inputs
        .iter()
        .cloned()
        .map(Value::GpuTensor)
        .collect::<Vec<_>>();
    let Value::GpuTensor(result) = block_on(runmat_runtime::call_builtin_async(name, &arguments))
        .unwrap_or_else(|error| panic!("{name} failed: {error}"))
    else {
        panic!("{name} should preserve supported same-owner floating input residency");
    };
    assert_eq!(result.shape(), &[1, 1], "{name}");
    assert_eq!(result.device_id(), inputs[0].device_id(), "{name}");
    assert!(!result.shares_allocation_with(&inputs[0]), "{name}");
    assert!(inputs
        .iter()
        .all(|input| input.allocation_lease().validate_live().is_ok()));
    assert!(result.allocation_lease().validate_live().is_ok());
    let owner = runmat_accelerate_api::provider_for_handle(&result)
        .expect("result must retain its exact provider owner");
    assert!(runmat_accelerate_api::same_provider_instance(
        owner, provider
    ));
    result
}

#[test]
fn all_four_means_preserve_supported_same_owner_float_residency() {
    let provider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = upload(provider, &[1.0, 4.0], &[2, 1]);
    let other = upload(provider, &[2.0, 2.0], &[2, 1]);
    let expectations = [
        ("geomean", vec![input.clone()], 2.0),
        ("harmmean", vec![input.clone()], 1.6),
        ("rms", vec![input.clone()], 8.5_f64.sqrt()),
        ("rmse", vec![input.clone(), other.clone()], 2.5_f64.sqrt()),
    ];
    for (name, inputs, expected) in expectations {
        let result = resident_result(provider, name, &inputs);
        let host = block_on(provider.download_numeric(&result)).unwrap();
        let HostNumericDataOwned::F64(values) = host.data else {
            panic!("{name} changed floating class");
        };
        assert_eq!(values.len(), 1);
        assert!((values[0] - expected).abs() < 1e-12, "{name}");
    }
}
