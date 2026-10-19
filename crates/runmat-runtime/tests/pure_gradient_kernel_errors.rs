use runmat_runtime::builtins::math::reduction::{
    gradient_complex_tensor_host, gradient_real_tensor_host,
    gradient_real_tensor_host_with_coordinates, GradientKernelError,
};
use runmat_value::{ComplexTensor, NumericDType, Tensor};
use std::error::Error;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pure_native_gradient_values_and_error_type_are_preserved() {
    fn native_error<T: Error + Send + Sync + 'static>() {}
    native_error::<GradientKernelError>();
    let input = Tensor::from_f32(vec![0., 2., 8.], vec![1, 3]).unwrap();
    let output = gradient_real_tensor_host(input.clone(), 2, 2.).unwrap();
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.shape(), &[1, 3]);
    assert_eq!(output.materialize_f64(), [1., 2., 3.]);
    let coordinates =
        gradient_real_tensor_host_with_coordinates(input, 2, vec![0., 1., 4.]).unwrap();
    assert_eq!(coordinates.materialize_f64(), [2., 2., 2.]);
    let complex = ComplexTensor::from_f64_values_with_dtype(
        vec![(0., 0.), (2., 4.), (8., 16.)],
        vec![1, 3],
        NumericDType::F32,
    )
    .unwrap();
    let output = gradient_complex_tensor_host(complex, 2, 2.).unwrap();
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.materialize_f64(), [(1., 2.), (2., 4.), (3., 6.)]);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn gradient_inputs_cannot_contain_mismatched_storage_and_shape() {
    assert!(Tensor::new(vec![1., 2., 3.], vec![1, 2]).is_err());
    assert!(Tensor::from_f32(vec![1., 2., 3.], vec![1, 2]).is_err());
    assert!(ComplexTensor::from_f32(vec![(1., 0.); 3], vec![1, 2]).is_err());
}
