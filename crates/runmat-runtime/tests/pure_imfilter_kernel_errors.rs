use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataView, HostNumericTensorView, ImfilterOptions,
};
use runmat_runtime::builtins::image::filters::kernel::{
    apply_imfilter_tensor, ImfilterKernelError,
};
use runmat_value::{Tensor, Value};
use std::error::Error;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pure_filter_and_provider_retain_exact_validation_cause() {
    fn native_error<T: Error + Send + Sync + 'static>() {}
    native_error::<ImfilterKernelError>();
    let options = ImfilterOptions::default();
    let image = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
    let identity = Tensor::new(vec![1.], vec![1, 1]).unwrap();
    let result = apply_imfilter_tensor(&image, &identity, &options, "imfilter").unwrap();
    assert_eq!(result.shape(), image.shape());
    assert_eq!(result.materialize_f64(), image.materialize_f64());
    let empty = Tensor::new(vec![], vec![0, 1]).unwrap();
    let error = apply_imfilter_tensor(&image, &empty, &options, "filter2").unwrap_err();
    let runtime: runmat_runtime::RuntimeError = error.into();
    assert_eq!(
        runtime.message(),
        "filter2: filter must be non-empty along every dimension"
    );
    assert_eq!(runtime.context.builtin.as_deref(), Some("filter2"));
    assert!(runtime.source().unwrap().is::<ImfilterKernelError>());

    let provider = runmat_accelerate::simple_provider::InProcessProvider::new();
    let upload = |data, shape| {
        provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(data),
                shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
    };
    let image = upload(&[1., 2., 3., 4.], &[2, 2]);
    let kernel = upload(&[], &[0, 1]);
    let error =
        futures::executor::block_on(provider.imfilter(&image, &kernel, &options)).unwrap_err();
    assert!(error.downcast_ref::<ImfilterKernelError>().is_some());
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &error
    ));
    image.allocation_lease().validate_live().unwrap();
    kernel.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn public_host_adapter_keeps_descriptor_and_original_plan_error() {
    let args = [
        Value::Tensor(Tensor::new(vec![], vec![0, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.], vec![1, 1]).unwrap()),
    ];
    let error = futures::executor::block_on(runmat_runtime::call_builtin_async("imfilter", &args))
        .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:imfilter:InvalidInput"));
    let mut cursor: &(dyn Error + 'static) = &error;
    while !cursor.is::<ImfilterKernelError>() {
        cursor = cursor
            .source()
            .expect("plan error survives language descriptor mapping");
    }
}
