use runmat_accelerate_api::{AccelProvider, FspecialFilter, FspecialRequest, HostNumericDataOwned};
use runmat_runtime::builtins::image::filters::fspecial::{
    kernel_error::FspecialKernelError, spec_from_request, FspecialFilterSpec,
};
use std::error::Error;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn fspecial_pure_generation_retains_values_shape_and_exact_error_contract() {
    fn native_error<T: Error + Send + Sync + 'static>() {}
    native_error::<FspecialKernelError>();
    let spec = spec_from_request(&FspecialFilter::Average { rows: 2, cols: 3 });
    let tensor = spec.generate_tensor().unwrap();
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.as_f64_slice().unwrap(), &[1. / 6.; 6]);
    for (spec, message) in [
        (
            FspecialFilterSpec::Average { rows: 0, cols: 3 },
            "fspecial: LENGTHS must be positive integers",
        ),
        (
            FspecialFilterSpec::Average {
                rows: usize::MAX,
                cols: 2,
            },
            "fspecial: LENGTHS are too large",
        ),
    ] {
        let error = spec.generate_tensor().unwrap_err();
        assert!(matches!(error, FspecialKernelError::InvalidInput(_)));
        assert_eq!(error.to_string(), message);
        let runtime: runmat_runtime::RuntimeError = error.into();
        assert_eq!(runtime.message(), message);
        assert_eq!(runtime.identifier(), Some("RunMat:fspecial:InvalidInput"));
        assert!(runtime.source().unwrap().is::<FspecialKernelError>());
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn fspecial_provider_keeps_original_pure_error_and_genuine_native_output() {
    let provider = runmat_accelerate::simple_provider::InProcessProvider::new();
    let error = provider
        .fspecial(&FspecialRequest {
            filter: FspecialFilter::Average { rows: 0, cols: 3 },
        })
        .unwrap_err();
    assert!(error.downcast_ref::<FspecialKernelError>().is_some());
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &error
    ));
    let output = provider
        .fspecial(&FspecialRequest {
            filter: FspecialFilter::Average { rows: 2, cols: 3 },
        })
        .unwrap();
    let downloaded = futures::executor::block_on(provider.download_numeric(&output)).unwrap();
    assert_eq!(downloaded.shape, [2, 3]);
    assert!(
        matches!(downloaded.data, HostNumericDataOwned::F64(ref values) if values == &[1. / 6.; 6])
    );
    output.allocation_lease().validate_live().unwrap();
    output.release().unwrap();
}
