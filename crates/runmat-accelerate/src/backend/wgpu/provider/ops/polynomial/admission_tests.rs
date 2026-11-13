use super::*;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataView, HostNumericTensorView, HostTensorView,
};

#[test]
#[ignore = "requires actual WGPU device; run explicitly for each native precision"]
fn complex_integration_is_supported_but_real_only_polynomial_hooks_are_not() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let data64 = [2., 4., 6., 8.];
    let data32 = [2f32, 4., 6., 8.];
    let data = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => HostNumericDataView::F64(&data64),
        runmat_accelerate_api::ProviderPrecision::F32 => HostNumericDataView::F32(&data32),
    };
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data,
            shape: &[1, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let output = provider.polyint_exec(&input, 3.).unwrap();
    assert_eq!(output.shape(), &[1, 3]);
    assert_eq!(
        output.physical().storage(),
        GpuTensorStorage::ComplexInterleaved
    );
    assert_eq!(
        futures::executor::block_on(provider.download(&output))
            .unwrap()
            .data,
        vec![1., 2., 6., 8., 3., 0.]
    );
    let points = provider
        .upload(&HostTensorView {
            data: &[1., 2.],
            shape: &[1, 2],
        })
        .unwrap();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &provider
            .polyval_exec(&input, &points, &ProviderPolyvalOptions { mu: None })
            .unwrap_err()
    ));
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &futures::executor::block_on(provider.polyfit_exec(
            &input,
            &points,
            1,
            None,
            runmat_accelerate_api::ProviderPolyfitOutputs::Coefficients,
        ))
        .unwrap_err()
    ));
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &futures::executor::block_on(provider.polyder_exec(&input)).unwrap_err()
    ));
    assert!(input.allocation_lease().validate_live().is_ok());
}
