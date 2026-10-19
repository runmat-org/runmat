use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
};
use runmat_runtime::builtins::math::linalg::ops::{dot_host_real_for_provider, DotKernelError};
use runmat_value::{Tensor, Value};
use std::error::Error;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pure_dot_and_actual_provider_keep_shape_diagnostic() {
    fn native_error<T: Error + Send + Sync + 'static>() {}
    native_error::<DotKernelError>();
    let a = Tensor::from_f32(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
    let b = Tensor::from_f32(vec![5., 6., 7., 8.], vec![2, 2]).unwrap();
    let result = dot_host_real_for_provider(&a, &b, Some(1)).unwrap();
    assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(result.shape(), &[1, 2]);
    assert_eq!(result.materialize_f64(), [17., 53.]);

    let provider = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let upload = |data, shape| {
        provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(data),
                shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
    };
    let left = upload(&[1., 2., 3., 4.], &[2, 2]);
    let right = upload(&[1., 2., 3.], &[3, 1]);
    let direct = futures::executor::block_on(provider.dot(&left, &right, Some(1))).unwrap_err();
    assert!(direct.downcast_ref::<DotKernelError>().is_some());
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &direct
    ));

    // Public shape admission correctly chooses its host path before calling
    // a provider on unlike shapes; preserve that behavior as well as its error.
    let host = futures::executor::block_on(runmat_runtime::call_builtin_async(
        "dot",
        &[
            Value::Tensor(Tensor::new(vec![1.; 4], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.; 3], vec![3, 1]).unwrap()),
            Value::Num(1.),
        ],
    ))
    .unwrap_err();
    let resident = futures::executor::block_on(runmat_runtime::call_builtin_async(
        "dot",
        &[
            Value::GpuTensor(left.clone()),
            Value::GpuTensor(right.clone()),
            Value::Num(1.),
        ],
    ))
    .unwrap_err();
    assert_eq!(resident.identifier(), host.identifier());
    assert_eq!(resident.message(), host.message());
    assert!(host.source().unwrap().is::<DotKernelError>());
    left.allocation_lease().validate_live().unwrap();
    right.allocation_lease().validate_live().unwrap();
}
