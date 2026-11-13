use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{HostTensorView, ImfilterMode, ImfilterOptions, ImfilterShape};
use runmat_builtins::FILTER2_DESCRIPTOR;
use runmat_value::LogicalArray;

async fn filter2_builtin(
    kernel: Value,
    image: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    super::filter2_builtin(
        None,
        kernel,
        image,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

#[test]
fn conditional_host_workspace_is_typed_and_overflow_is_not_zero() {
    use runmat_execution::{ResourceBoundUnavailable, ResourceByteBound};
    let (scratch, output) = placement::conditional_host_bounds(
        FILTER2_BUILTIN,
        &[2, 2],
        &[1, 1],
        &ImfilterOptions::default(),
        true,
        false,
        true,
    );
    assert!(matches!(scratch, ResourceByteBound::Known(bytes) if bytes > 4 * 4));
    assert_eq!(output, ResourceByteBound::Known(16));
    assert_eq!(
        placement::host_scratch_excluding_result(scratch, output).unwrap(),
        ResourceByteBound::Known(match scratch {
            ResourceByteBound::Known(bytes) => bytes - 16,
            _ => unreachable!("first bound is known"),
        })
    );
    let (scratch, output) = placement::conditional_host_bounds(
        FILTER2_BUILTIN,
        &[usize::MAX, 2],
        &[1, 1],
        &ImfilterOptions::default(),
        false,
        false,
        false,
    );
    assert_eq!(
        scratch,
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound)
    );
    assert_eq!(
        output,
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound)
    );
}

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
    Tensor::new(data, vec![rows, cols]).expect("tensor construction")
}

fn error_message(err: &crate::RuntimeError) -> String {
    err.message().to_string()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn filter2_preserves_rank_zero_and_rank_one_single_tap_kernels() {
    for shape in [Vec::new(), vec![1]] {
        let kernel = Tensor::new(vec![2.0], shape).unwrap();
        let image = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let value = block_on(filter2_builtin(
            Value::Tensor(kernel),
            Value::Tensor(image),
            Vec::new(),
        ))
        .expect("filter2 single-tap kernel");
        let output = test_support::gather(value).expect("host output");
        assert_eq!(output.shape(), &[1, 2]);
        assert_eq!(output.materialize_f64(), vec![4.0, 6.0]);
    }
}

#[cfg(feature = "wgpu")]
struct WgpuFilterEnvGuard {
    skip_warmup: Option<std::ffi::OsString>,
    disable_imfilter: Option<std::ffi::OsString>,
}

#[cfg(feature = "wgpu")]
impl WgpuFilterEnvGuard {
    fn set() -> Self {
        let guard = Self {
            skip_warmup: std::env::var_os("RUNMAT_WGPU_SKIP_WARMUP"),
            disable_imfilter: std::env::var_os("RUNMAT_WGPU_DISABLE_IMFILTER"),
        };
        std::env::set_var("RUNMAT_WGPU_SKIP_WARMUP", "1");
        std::env::set_var("RUNMAT_WGPU_DISABLE_IMFILTER", "1");
        guard
    }
}

#[cfg(feature = "wgpu")]
impl Drop for WgpuFilterEnvGuard {
    fn drop(&mut self) {
        match self.skip_warmup.take() {
            Some(value) => std::env::set_var("RUNMAT_WGPU_SKIP_WARMUP", value),
            None => std::env::remove_var("RUNMAT_WGPU_SKIP_WARMUP"),
        }
        match self.disable_imfilter.take() {
            Some(value) => std::env::set_var("RUNMAT_WGPU_DISABLE_IMFILTER", value),
            None => std::env::remove_var("RUNMAT_WGPU_DISABLE_IMFILTER"),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn default_same_matches_imfilter_reference() {
    let kernel = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let image = tensor(vec![4.0, 1.0, 2.0, 0.0], 2, 2);
    let value = block_on(filter2_builtin(
        Value::Tensor(kernel.clone()),
        Value::Tensor(image.clone()),
        Vec::new(),
    ))
    .expect("filter2");
    let gathered = test_support::gather(value).expect("gather");
    let options = ImfilterOptions {
        shape: ImfilterShape::Same,
        mode: ImfilterMode::Correlation,
        ..Default::default()
    };
    let expected =
        apply_imfilter_tensor(&image, &kernel, &options, FILTER2_BUILTIN).expect("reference");
    assert_eq!(gathered.shape(), expected.shape());
    assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn full_shape_option_expands_output() {
    let kernel = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let value = block_on(filter2_builtin(
        Value::Tensor(kernel.clone()),
        Value::Tensor(image.clone()),
        vec![Value::from("full")],
    ))
    .expect("filter2");
    let gathered = test_support::gather(value).expect("gather");
    let options = ImfilterOptions {
        shape: ImfilterShape::Full,
        ..Default::default()
    };
    let expected =
        apply_imfilter_tensor(&image, &kernel, &options, FILTER2_BUILTIN).expect("reference");
    assert_eq!(gathered.shape(), expected.shape());
    assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn valid_shape_option_matches_reference() {
    let kernel = tensor(vec![1.0, 0.0, 0.0, -1.0], 2, 2);
    let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let value = block_on(filter2_builtin(
        Value::Tensor(kernel.clone()),
        Value::Tensor(image.clone()),
        vec![Value::from("valid")],
    ))
    .expect("filter2");
    let gathered = test_support::gather(value).expect("gather");
    let options = ImfilterOptions {
        shape: ImfilterShape::Valid,
        ..Default::default()
    };
    let expected =
        apply_imfilter_tensor(&image, &kernel, &options, FILTER2_BUILTIN).expect("reference");
    assert_eq!(gathered.shape(), expected.shape());
    assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn convolution_mode_rotates_kernel() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let kernel = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let value = block_on(filter2_builtin(
        Value::Tensor(kernel.clone()),
        Value::Tensor(image.clone()),
        vec![Value::from("conv")],
    ))
    .expect("filter2");
    let gathered = test_support::gather(value).expect("gather");
    let options = ImfilterOptions {
        mode: ImfilterMode::Convolution,
        ..Default::default()
    };
    let expected =
        apply_imfilter_tensor(&image, &kernel, &options, FILTER2_BUILTIN).expect("reference");
    assert_eq!(gathered.shape(), expected.shape());
    assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn explicit_corr_option_matches_default() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let kernel = tensor(vec![0.0, 1.0, 1.0, 0.0], 2, 2);
    let image = tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2);

    let default_value = block_on(filter2_builtin(
        Value::Tensor(kernel.clone()),
        Value::Tensor(image.clone()),
        Vec::new(),
    ))
    .expect("filter2 default");
    let corr_value = block_on(filter2_builtin(
        Value::Tensor(kernel.clone()),
        Value::Tensor(image.clone()),
        vec![Value::from("corr")],
    ))
    .expect("filter2 corr");

    let default_tensor = test_support::gather(default_value).expect("gather default");
    let corr_tensor = test_support::gather(corr_value).expect("gather corr");
    assert_eq!(default_tensor.shape(), corr_tensor.shape());
    assert_eq!(
        default_tensor.materialize_f64(),
        corr_tensor.materialize_f64()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conv_and_shape_order_independent() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let kernel = tensor(vec![1.0, -1.0, 2.0, -2.0], 2, 2);
    let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

    let conv_first = block_on(filter2_builtin(
        Value::Tensor(kernel.clone()),
        Value::Tensor(image.clone()),
        vec![Value::from("conv"), Value::from("full")],
    ))
    .expect("filter2 conv,full");
    let shape_first = block_on(filter2_builtin(
        Value::Tensor(kernel.clone()),
        Value::Tensor(image.clone()),
        vec![Value::from("full"), Value::from("conv")],
    ))
    .expect("filter2 full,conv");

    let conv_first_tensor = test_support::gather(conv_first).expect("gather conv first");
    let shape_first_tensor = test_support::gather(shape_first).expect("gather shape first");
    assert_eq!(conv_first_tensor.shape(), shape_first_tensor.shape());
    assert_eq!(
        conv_first_tensor.materialize_f64(),
        shape_first_tensor.materialize_f64()
    );
}

#[test]
fn multiple_trailing_options_require_runmat_extension_mode() {
    let kernel = tensor(vec![1.0], 1, 1);
    let image = tensor(vec![1.0], 1, 1);
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(filter2_builtin(
        Value::Tensor(kernel),
        Value::Tensor(image),
        vec![Value::from("same"), Value::from("full")],
    ))
    .expect_err("variadic options must be gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:Filter2VariadicOptionsExtension")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn non_string_option_is_rejected() {
    let kernel = tensor(vec![1.0], 1, 1);
    let image = tensor(vec![1.0], 1, 1);
    let err = block_on(filter2_builtin(
        Value::Tensor(kernel),
        Value::Tensor(image),
        vec![Value::Tensor(tensor(vec![2.0], 1, 1))],
    ))
    .expect_err("expected option parsing error");
    let message = error_message(&err);
    assert!(
        message.contains("expected string option"),
        "unexpected error message: {message}"
    );
    assert_eq!(err.identifier(), FILTER2_ERROR_INVALID_OPTION.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_input_promoted_to_double() {
    let kernel = tensor(vec![1.0, 1.0, 1.0, 1.0], 2, 2);
    let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).expect("logical array");

    let value = block_on(filter2_builtin(
        Value::Tensor(kernel.clone()),
        Value::LogicalArray(logical.clone()),
        Vec::new(),
    ))
    .expect("filter2 logical");

    let gathered = test_support::gather(value).expect("gather logical result");
    let image_tensor = Tensor::new(
        logical
            .data
            .iter()
            .map(|&b| if b != 0 { 1.0 } else { 0.0 })
            .collect(),
        logical.shape.clone(),
    )
    .expect("logical->tensor");
    let expected = apply_imfilter_tensor(
        &image_tensor,
        &kernel,
        &ImfilterOptions::default(),
        FILTER2_BUILTIN,
    )
    .expect("expected logical reference");

    assert_eq!(gathered.shape(), expected.shape());
    assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
}

#[test]
fn typed_integer_inputs_produce_double_and_single_input_is_preserved() {
    use runmat_value::IntegerStorage;

    let integer = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2]).unwrap();
    let scalar_kernel = Tensor::new_integer(IntegerStorage::U8(vec![2]), vec![1, 1]).unwrap();
    let integer_output = block_on(filter2_builtin(
        Value::Tensor(scalar_kernel),
        Value::Tensor(integer),
        Vec::new(),
    ))
    .expect("integer filter2");
    let Value::Tensor(integer_output) = integer_output else {
        panic!("expected tensor")
    };
    assert_eq!(integer_output.numeric_dtype(), NumericDType::F64);

    let single = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let single_output = block_on(filter2_builtin(
        Value::Num(1.0),
        Value::Tensor(single),
        Vec::new(),
    ))
    .expect("single filter2");
    let Value::Tensor(single_output) = single_output else {
        panic!("expected tensor")
    };
    assert_eq!(single_output.numeric_dtype(), NumericDType::F32);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rejects_unknown_option() {
    let kernel = tensor(vec![1.0], 1, 1);
    let image = tensor(vec![1.0], 1, 1);
    let err = block_on(filter2_builtin(
        Value::Tensor(kernel),
        Value::Tensor(image),
        vec![Value::from("replicate")],
    ))
    .expect_err("filter2 should error");
    let message = error_message(&err);
    assert!(
        message.contains("filter2"),
        "expected error mentioning builtin, got {message}"
    );
    assert_eq!(err.identifier(), FILTER2_ERROR_INVALID_OPTION.identifier);
}

#[test]
fn invalid_option_rejects_before_resident_download_or_upload() {
    use runmat_accelerate_api::AccelProvider;
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let image = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[2, 2],
        })
        .unwrap();
    provider.reset_telemetry();
    let error = block_on(filter2_builtin(
        Value::Num(1.0),
        Value::GpuTensor(image.clone()),
        vec![Value::from("invalid-padding")],
    ))
    .expect_err("invalid option");
    assert_eq!(error.identifier(), FILTER2_ERROR_INVALID_OPTION.identifier);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
    assert!(image.allocation_lease().validate_live().is_ok());
}

#[test]
fn invalid_kernel_geometry_rejects_before_resident_transfer() {
    use runmat_accelerate_api::AccelProvider;
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let image = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[2, 2],
        })
        .expect("resident image");
    let empty_kernel = Tensor::new(Vec::new(), vec![0, 2]).expect("empty kernel");
    provider.reset_telemetry();
    let error = block_on(filter2_builtin(
        Value::Tensor(empty_kernel),
        Value::GpuTensor(image.clone()),
        Vec::new(),
    ))
    .expect_err("empty kernel rejected");
    assert_eq!(error.identifier(), FILTER2_ERROR_INVALID_INPUT.identifier);
    let telemetry = provider.telemetry_snapshot();
    assert_eq!(telemetry.download_bytes, 0);
    assert_eq!(telemetry.upload_bytes, 0);
    assert!(image.allocation_lease().validate_live().is_ok());

    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let incompatible_kernel = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 1, 2],
        })
        .expect("resident higher-rank kernel");
    provider.reset_telemetry();
    let error = block_on(filter2_builtin(
        Value::GpuTensor(incompatible_kernel.clone()),
        Value::GpuTensor(image.clone()),
        Vec::new(),
    ))
    .expect_err("kernel axis without image axis rejected");
    assert_eq!(error.identifier(), FILTER2_ERROR_INVALID_INPUT.identifier);
    let telemetry = provider.telemetry_snapshot();
    assert_eq!(telemetry.download_bytes, 0);
    assert_eq!(telemetry.upload_bytes, 0);
    assert!(image.allocation_lease().validate_live().is_ok());
    assert!(incompatible_kernel
        .allocation_lease()
        .validate_live()
        .is_ok());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter2_gpu_matches_cpu() {
    test_support::with_test_provider(|provider| {
        let kernel = tensor(vec![1.0, 0.0, -1.0, 2.0], 2, 2);
        let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let image_view = HostTensorView {
            data: &image.materialize_f64(),
            shape: image.shape(),
        };
        let image_handle = provider.upload(&image_view).expect("upload image");
        let result = block_on(filter2_builtin(
            Value::Tensor(kernel.clone()),
            Value::GpuTensor(image_handle.clone()),
            Vec::new(),
        ))
        .expect("filter2 gpu");
        let gathered = test_support::gather(result).expect("gather");
        let expected = apply_imfilter_tensor(
            &image,
            &kernel,
            &ImfilterOptions::default(),
            FILTER2_BUILTIN,
        )
        .expect("ref");
        assert_eq!(gathered.shape(), expected.shape());
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    });
}

#[test]
fn explicit_resident_image_provenance_survives_provider_filter() {
    use runmat_accelerate_api::AccelProvider;
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let image = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[2, 2],
        })
        .expect("image")
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("explicit image view");
    let output = block_on(filter2_builtin(
        Value::Tensor(tensor(vec![1.0], 1, 1)),
        Value::GpuTensor(image.clone()),
        Vec::new(),
    ))
    .expect("resident filter2");
    assert!(runmat_accelerate_api::same_provider_instance(
        provider,
        runmat_accelerate_api::provider().expect("selected provider remains installed"),
    ));
    let Value::GpuTensor(output) = output else {
        panic!("provider result must remain resident");
    };
    assert!(runmat_accelerate_api::handle_is_explicit(&output));
    assert!(!output.shares_allocation_with(&image));
    assert!(image.allocation_lease().validate_live().is_ok());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter2_gpu_kernel_and_image() {
    test_support::with_test_provider(|provider| {
        let kernel = tensor(vec![0.0, 1.0, -1.0, 0.0], 2, 2);
        let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel_view = HostTensorView {
            data: &kernel.materialize_f64(),
            shape: kernel.shape(),
        };
        let image_view = HostTensorView {
            data: &image.materialize_f64(),
            shape: image.shape(),
        };
        let kernel_handle = provider.upload(&kernel_view).expect("upload kernel");
        let image_handle = provider.upload(&image_view).expect("upload image");
        let result = block_on(filter2_builtin(
            Value::GpuTensor(kernel_handle.clone()),
            Value::GpuTensor(image_handle.clone()),
            Vec::new(),
        ))
        .expect("filter2 gpu");
        let gathered = test_support::gather(result).expect("gather");
        let expected = apply_imfilter_tensor(
            &image,
            &kernel,
            &ImfilterOptions::default(),
            FILTER2_BUILTIN,
        )
        .expect("reference");
        assert_eq!(gathered.shape(), expected.shape());
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    });
}

#[test]
fn resident_double_image_with_single_kernel_returns_single() {
    test_support::with_test_provider(|provider| {
        let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let image_handle = provider
            .upload(&HostTensorView {
                data: &image.materialize_f64(),
                shape: image.shape(),
            })
            .expect("upload double image");
        let kernel = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single kernel");
        let result = block_on(filter2_builtin(
            Value::Tensor(kernel),
            Value::GpuTensor(image_handle),
            Vec::new(),
        ))
        .expect("filter2 mixed precision");
        let gathered = test_support::gather(result).expect("gather result");
        assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
        assert_eq!(gathered.materialize_f64(), image.materialize_f64());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn filter2_wgpu_same_matches_cpu() {
    let _lock = test_support::accel_test_lock();
    let _env = WgpuFilterEnvGuard::set();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");

    let kernel = tensor(vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], 3, 3);
    let image = tensor(
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ],
        4,
        4,
    );

    let kernel_view = HostTensorView {
        data: &kernel.materialize_f64(),
        shape: kernel.shape(),
    };
    let image_view = HostTensorView {
        data: &image.materialize_f64(),
        shape: image.shape(),
    };

    let kernel_handle = provider.upload(&kernel_view).expect("upload kernel");
    let image_handle = provider.upload(&image_view).expect("upload image");

    let gpu_value = block_on(filter2_builtin(
        Value::GpuTensor(kernel_handle.clone()),
        Value::GpuTensor(image_handle.clone()),
        Vec::new(),
    ))
    .expect("filter2 wgpu same");
    let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu result");

    let expected = apply_imfilter_tensor(
        &image,
        &kernel,
        &ImfilterOptions::default(),
        FILTER2_BUILTIN,
    )
    .expect("expected");

    assert_eq!(gpu_tensor.shape(), expected.shape());
    assert_eq!(
        gpu_tensor.materialize_f64().len(),
        expected.materialize_f64().len()
    );
    let tol = match runmat_accelerate_api::provider().unwrap().precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    for (gpu, cpu) in gpu_tensor
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64().iter())
    {
        assert!((gpu - cpu).abs() < tol, "{gpu} vs {cpu}");
    }

    let _ = kernel_handle.release();
    let _ = image_handle.release();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn filter2_wgpu_full_conv_matches_cpu() {
    let _lock = test_support::accel_test_lock();
    let _env = WgpuFilterEnvGuard::set();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");

    let kernel = tensor(vec![1.0, 2.0, 3.0, 0.0], 2, 2);
    let image = tensor(
        vec![
            3.0, 1.0, 4.0, //
            1.0, 5.0, 9.0, //
            2.0, 6.0, 5.0,
        ],
        3,
        3,
    );

    let kernel_view = HostTensorView {
        data: &kernel.materialize_f64(),
        shape: kernel.shape(),
    };
    let image_view = HostTensorView {
        data: &image.materialize_f64(),
        shape: image.shape(),
    };
    let kernel_handle = provider.upload(&kernel_view).expect("upload kernel");
    let image_handle = provider.upload(&image_view).expect("upload image");

    let gpu_value = block_on(filter2_builtin(
        Value::GpuTensor(kernel_handle.clone()),
        Value::GpuTensor(image_handle.clone()),
        vec![Value::from("full"), Value::from("conv")],
    ))
    .expect("filter2 wgpu full conv");
    let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu result");

    let options = ImfilterOptions {
        shape: ImfilterShape::Full,
        mode: ImfilterMode::Convolution,
        ..Default::default()
    };
    let expected = apply_imfilter_tensor(&image, &kernel, &options, FILTER2_BUILTIN)
        .expect("expected host result");

    assert_eq!(gpu_tensor.shape(), expected.shape());
    assert_eq!(
        gpu_tensor.materialize_f64().len(),
        expected.materialize_f64().len()
    );
    let tol = match runmat_accelerate_api::provider().unwrap().precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    for (gpu, cpu) in gpu_tensor
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64().iter())
    {
        assert!((gpu - cpu).abs() < tol, "{gpu} vs {cpu}");
    }

    let _ = kernel_handle.release();
    let _ = image_handle.release();
}

#[test]
fn filter2_descriptor_signatures_cover_surface() {
    let builtin = runmat_builtins::builtin_catalog_entry_by_name(FILTER2_BUILTIN)
        .expect("filter2 catalog entry");
    let labels: Vec<&str> = FILTER2_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert!(labels.contains(&"B = filter2(h, X)"));
    assert!(labels.contains(&"B = filter2(h, X, options...)"));
    assert_eq!(builtin.integer_capabilities.len(), 1);
    assert_eq!(builtin.extensions.len(), 5);
}

#[test]
fn filter2_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = FILTER2_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert!(codes.contains(&"RM.FILTER2.INVALID_INPUT"));
    assert!(codes.contains(&"RM.FILTER2.INVALID_OPTION"));
    assert!(codes.contains(&"RM.FILTER2.INTERNAL"));
}
