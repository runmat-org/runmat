//! One runtime library, real native allocations, identical pure errors on WASM.
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
};
use runmat_runtime::builtins::math::linalg::structure::{
    bandwidth, ishermitian, issymmetric, kernel_error::StructureKernelError, symrcm,
};
use std::error::Error;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn rectangular_symrcm_public_host_and_provider_errors_match_without_retry() {
    let provider = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let tensor = runmat_value::Tensor::new(vec![1.; 6], vec![2, 3]).unwrap();
    let host = futures::executor::block_on(runmat_runtime::call_builtin_async(
        "symrcm",
        &[runmat_value::Value::Tensor(tensor)],
    ))
    .unwrap_err();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.; 6]),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let resident = futures::executor::block_on(runmat_runtime::call_builtin_async(
        "symrcm",
        &[runmat_value::Value::GpuTensor(input.clone())],
    ))
    .unwrap_err();
    assert_eq!(host.identifier(), Some("RunMat:symrcm:InvalidInput"));
    assert_eq!(resident.identifier(), host.identifier());
    assert_eq!(resident.message(), host.message());
    assert_eq!(resident.context.builtin, host.context.builtin);
    assert_eq!(
        resident.gpu_gather_retry,
        runmat_runtime::GpuGatherRetry::Never
    );
    let mut cursor: &(dyn Error + 'static) = &resident;
    let mut found_provider = false;
    while !cursor.is::<StructureKernelError>() {
        found_provider |= cursor.is::<runmat_accelerate_api::ProviderExecutionError>();
        cursor = cursor.source().expect("exact kernel cause retained");
    }
    assert!(found_provider);
    input.allocation_lease().validate_live().unwrap();
}

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn structure_kernels_preserve_host_results_and_language_error_metadata() {
    fn native_error<T: Error + Send + Sync + 'static>() {}
    native_error::<StructureKernelError>();
    let symmetric = [1., 2., 2., 3.];
    assert!(issymmetric::issymmetric_host_real_data(&[2, 2], &symmetric, false, 0.).unwrap());
    assert!(ishermitian::ishermitian_host_real_data(&[2, 2], &symmetric, false, 0.).unwrap());
    assert_eq!(
        bandwidth::bandwidth_host_real_data(&[2, 2], &symmetric).unwrap(),
        (1, 1)
    );
    assert_eq!(
        symrcm::symrcm_host_real_data(&[2, 2], &symmetric)
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        bandwidth::bandwidth_host_real_data(&[0, 0], &[]).unwrap(),
        (0, 0)
    );
    assert!(symrcm::symrcm_host_real_data(&[0, 0], &[])
        .unwrap()
        .is_empty());
    let failures = [
        issymmetric::issymmetric_host_real_data(&[2, 2, 2], &[1.; 8], false, 0.).unwrap_err(),
        ishermitian::ishermitian_host_real_data(&[2, 2, 2], &[1.; 8], false, 0.).unwrap_err(),
        bandwidth::bandwidth_host_real_data(&[2, 2, 2], &[1.; 8]).unwrap_err(),
        symrcm::symrcm_host_real_data(&[2, 3], &[1.; 6]).unwrap_err(),
        issymmetric::issymmetric_host_real_data(&[2, 2], &[1.], false, 0.).unwrap_err(),
        symrcm::symrcm_host_real_data(&[2, 2], &[1.]).unwrap_err(),
    ];
    let expected = [
        (
            "RunMat:issymmetric:InvalidInput",
            "issymmetric: invalid input: inputs must be 2-D matrices or vectors",
        ),
        (
            "RunMat:ishermitian:InvalidInput",
            "ishermitian: invalid input: inputs must be 2-D matrices or vectors",
        ),
        (
            "RunMat:bandwidth:InvalidInput",
            "bandwidth: invalid input: input must be a 2-D matrix",
        ),
        (
            "RunMat:symrcm:InvalidInput",
            "symrcm: invalid input: input matrix must be square",
        ),
    ];
    for (error, (identifier, message)) in failures.iter().zip(expected) {
        assert_eq!(error.identifier(), Some(identifier));
        assert_eq!(error.message(), message);
    }
    for error in failures {
        let message = error.message().to_owned();
        let identifier = error.identifier();
        let runtime: runmat_runtime::RuntimeError = error.into();
        assert_eq!(runtime.message(), message);
        assert_eq!(runtime.identifier(), identifier);
        let original = runtime
            .source()
            .unwrap()
            .downcast_ref::<StructureKernelError>()
            .unwrap();
        assert_eq!(original.message(), message);
        assert_eq!(original.identifier(), identifier);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn inprocess_structure_boundaries_retain_exact_kernel_errors_and_borrowed_owners() {
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{ProviderHermitianKind, ProviderSymmetryKind};
    let provider = InProcessProvider::new();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.; 8]),
            shape: &[2, 2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let errors = [
        provider
            .issymmetric(&input, ProviderSymmetryKind::Symmetric, 0.)
            .unwrap_err(),
        futures::executor::block_on(provider.ishermitian(
            &input,
            ProviderHermitianKind::Hermitian,
            0.,
        ))
        .unwrap_err(),
        provider.bandwidth(&input).unwrap_err(),
        futures::executor::block_on(provider.sym_rcm(&input)).unwrap_err(),
    ];
    for error in errors {
        assert!(
            error.downcast_ref::<StructureKernelError>().is_some(),
            "{error:?}"
        );
        assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
            &error
        ));
        input.allocation_lease().validate_live().unwrap();
    }
    let data = futures::executor::block_on(provider.download_numeric(&input)).unwrap();
    assert_eq!(data.shape, [2, 2, 2]);
    assert!(
        matches!(data.data, runmat_accelerate_api::HostNumericDataOwned::F64(ref values) if values == &[1.; 8])
    );
}
