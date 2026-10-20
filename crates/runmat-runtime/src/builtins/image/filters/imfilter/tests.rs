use super::host_kernel::fallback_accumulation;
use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_builtins::{
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, IMFILTER_DESCRIPTOR,
};
use runmat_value::{IntegerStorage, Tensor, Value};

async fn imfilter_builtin(
    image: Value,
    kernel: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    super::imfilter_builtin(
        None,
        image,
        kernel,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

struct MutatingAliasingImfilterProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
}

impl MutatingAliasingImfilterProvider {
    fn new() -> Self {
        Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        }
    }
}

impl AccelProvider for MutatingAliasingImfilterProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn device_info(&self) -> String {
        "mutating-aliasing-imfilter-test-provider".to_string()
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn imfilter<'a>(
        &'a self,
        image: &'a GpuTensorHandle,
        _kernel: &'a GpuTensorHandle,
        _options: &'a ImfilterOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        assert!(image
            .with_view(image.view().clone().with_logical(true))
            .is_err());
        Box::pin(async move { Ok(image.clone()) })
    }
}

fn simple_tensor(data: &[f64], rows: usize, cols: usize) -> Tensor {
    Tensor::new(data.to_vec(), vec![rows, cols]).unwrap()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn imfilter_preserves_rank_zero_and_rank_one_single_tap_kernels() {
    for shape in [Vec::new(), vec![1]] {
        let image = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let kernel = Tensor::new(vec![2.0], shape).unwrap();
        let value = block_on(imfilter_builtin(
            Value::Tensor(image),
            Value::Tensor(kernel),
            Vec::new(),
        ))
        .expect("imfilter single-tap kernel");
        let output = test_support::gather(value).expect("host output");
        assert_eq!(output.shape(), &[1, 2]);
        assert_eq!(output.materialize_f64(), vec![4.0, 6.0]);
    }
}

#[test]
fn logical_image_keeps_logical_output_class() {
    let image =
        runmat_value::LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical image");
    let kernel = simple_tensor(&[1.0], 1, 1);
    let result = block_on(imfilter_builtin(
        Value::LogicalArray(image),
        Value::Tensor(kernel),
        Vec::new(),
    ))
    .expect("logical imfilter");
    let Value::LogicalArray(result) = result else {
        panic!("logical output expected");
    };
    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![0, 1, 1, 0]);
}

fn typed_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
    Tensor::new_integer(storage, shape).expect("integer tensor")
}

fn error_message(err: &crate::RuntimeError) -> String {
    err.message().to_string()
}

#[test]
fn resident_kernel_shape_rejects_before_download_or_upload() {
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
    let kernel = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 1, 2],
        })
        .expect("resident higher-rank kernel");
    provider.reset_telemetry();
    let error = block_on(imfilter_builtin(
        Value::GpuTensor(image.clone()),
        Value::GpuTensor(kernel.clone()),
        Vec::new(),
    ))
    .expect_err("unsupported public kernel shape");
    assert_eq!(error.identifier(), IMFILTER_ERROR_INVALID_INPUT.identifier);
    let telemetry = provider.telemetry_snapshot();
    assert_eq!(telemetry.download_bytes, 0);
    assert_eq!(telemetry.upload_bytes, 0);
    assert!(image.allocation_lease().validate_live().is_ok());
    assert!(kernel.allocation_lease().validate_live().is_ok());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn same_padding_default_zero() {
    let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    let kernel = simple_tensor(&[1.0; 9], 3, 3);
    let options = ImfilterOptions::default();
    let result =
        apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
    assert_eq!(result.shape(), &[2, 2]);
    assert!(result
        .materialize_f64()
        .iter()
        .all(|&v| (v - 10.0).abs() < 1e-12));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn replicate_padding() {
    let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    let kernel = simple_tensor(&[1.0; 9], 3, 3);
    let options = ImfilterOptions {
        padding: ImfilterPadding::Replicate,
        ..Default::default()
    };
    let result =
        apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
    assert_eq!(result.shape(), &[2, 2]);
    let expected = [18.0, 24.0, 21.0, 27.0];
    for (got, exp) in result.materialize_f64().iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-12);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn full_output_matches_expected_size() {
    let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    let kernel = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    let options = ImfilterOptions {
        shape: ImfilterShape::Full,
        ..Default::default()
    };
    let result =
        apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
    assert_eq!(result.shape(), &[3, 3]);
    let expected = [4.0, 14.0, 6.0, 11.0, 30.0, 11.0, 6.0, 14.0, 4.0];
    for (got, exp) in result.materialize_f64().iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-12);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn valid_output_respects_kernel_size() {
    let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let kernel = simple_tensor(&[1.0; 4], 2, 2);
    let options = ImfilterOptions {
        shape: ImfilterShape::Valid,
        ..Default::default()
    };
    let result =
        apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
    assert_eq!(result.shape(), &[1, 1]);
    assert!((result.materialize_f64()[0] - 10.0).abs() < 1e-12);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn convolution_matches_correlation_with_flipped_kernel() {
    let image = simple_tensor(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 3, 2);
    let kernel = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let corr_opts = ImfilterOptions::default();
    let corr = apply_imfilter_tensor(&image, &kernel, &corr_opts, IMFILTER_BUILTIN).expect("corr");
    let flipped_kernel = simple_tensor(&[4.0, 3.0, 2.0, 1.0], 2, 2);
    let corr_flipped = apply_imfilter_tensor(&image, &flipped_kernel, &corr_opts, IMFILTER_BUILTIN)
        .expect("corr flip");
    let conv_opts = ImfilterOptions {
        mode: ImfilterMode::Convolution,
        ..Default::default()
    };
    let conv = apply_imfilter_tensor(&image, &kernel, &conv_opts, IMFILTER_BUILTIN).expect("conv");
    assert_eq!(conv.shape(), corr_flipped.shape());
    for ((a, b), c) in conv
        .materialize_f64()
        .iter()
        .zip(corr_flipped.materialize_f64().iter())
        .zip(corr.materialize_f64().iter())
    {
        assert!((a - b).abs() < 1e-12 || (a - c).abs() < 1e-8);
    }
}

#[test]
fn raw_filter_plan_reads_typed_integer_image_storage_exactly() {
    let image = typed_tensor(IntegerStorage::I16(vec![1, 3, 2, 4]), vec![2, 2]);
    let kernel = simple_tensor(&[2.0], 1, 1);

    let result = apply_imfilter_tensor(
        &image,
        &kernel,
        &ImfilterOptions::default(),
        IMFILTER_BUILTIN,
    )
    .expect("imfilter");

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.materialize_f64(), vec![2.0, 6.0, 4.0, 8.0]);
}

#[test]
fn resident_integer_fallback_uses_documented_binary32_accumulation() {
    let image = simple_tensor(&[16_777_216.0, 1.0, -16_777_216.0], 3, 1);
    let kernel = simple_tensor(&[1.0, 1.0, 1.0], 3, 1);
    let options = ImfilterOptions {
        shape: ImfilterShape::Valid,
        ..Default::default()
    };
    let plan = build_imfilter_plan(image.shape(), &kernel, &options, IMFILTER_BUILTIN)
        .expect("filter plan");
    let values = image.materialize_f64();
    assert_eq!(
        plan.evaluate_with_accumulation(&values, &options, FilterAccumulation::HostDouble),
        vec![1.0]
    );
    assert_eq!(
        plan.evaluate_with_accumulation(&values, &options, FilterAccumulation::GpuSingle),
        vec![0.0]
    );
    assert_eq!(
        fallback_accumulation(ImageOutputClass::Integer(NumericDType::U8), false),
        FilterAccumulation::HostDouble,
        "automatic placement must remain numerically transparent"
    );
    assert_eq!(
        fallback_accumulation(ImageOutputClass::Integer(NumericDType::U8), true),
        FilterAccumulation::GpuSingle,
        "explicit gpuArray follows the documented GPU precision"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn circular_padding_wraps_indices() {
    let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let kernel = simple_tensor(&[0.0, 1.0, 1.0, 0.0], 2, 2);
    let options = ImfilterOptions {
        padding: ImfilterPadding::Circular,
        ..Default::default()
    };
    let result =
        apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
    let expected = [5.0, 5.0, 5.0, 5.0];
    for (got, exp) in result.materialize_f64().iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-12);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_fallback_uses_provider_upload() {
    test_support::with_test_provider(|provider| {
        let image = simple_tensor(&[1.0, 4.0, 2.0, 5.0], 2, 2);
        let kernel = simple_tensor(&[1.0, 1.0, 1.0, 1.0], 2, 2);
        let image_view = HostTensorView {
            data: &image.materialize_f64(),
            shape: image.shape(),
        };
        let kernel_view = HostTensorView {
            data: &kernel.materialize_f64(),
            shape: kernel.shape(),
        };
        let image_handle = provider.upload(&image_view).expect("upload image");
        let kernel_handle = provider.upload(&kernel_view).expect("upload kernel");
        let value = block_on(imfilter_builtin(
            Value::GpuTensor(image_handle),
            Value::GpuTensor(kernel_handle),
            Vec::new(),
        ))
        .expect("imfilter");
        let gathered = test_support::gather(value).expect("gather");
        assert_eq!(gathered.shape(), &[2, 2]);
        // A 2-by-2 kernel has its documented origin at the first element,
        // so same-size correlation samples forward from each output site.
        let expected = [12.0, 9.0, 7.0, 5.0];
        for (got, exp) in gathered.materialize_f64().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_gpu_form_rejects_non_double_kernel_before_upload() {
    test_support::with_test_provider(|provider| {
        let image = simple_tensor(&[1.0, 4.0, 2.0, 5.0], 2, 2);
        let image_view = HostTensorView {
            data: &image.materialize_f64(),
            shape: image.shape(),
        };
        let image_handle = provider.upload(&image_view).expect("upload image");
        let kernel = typed_tensor(IntegerStorage::I16(vec![2]), vec![1, 1]);

        let error = block_on(imfilter_builtin(
            Value::GpuTensor(image_handle),
            Value::Tensor(kernel),
            Vec::new(),
        ))
        .expect_err("typed H must reject");
        assert_eq!(error.identifier(), IMFILTER_ERROR_INVALID_INPUT.identifier);
        assert!(error.message().contains("H must be double"));
    });
}

#[test]
fn adversarial_provider_cannot_mutate_or_return_the_image_handle() {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(MutatingAliasingImfilterProvider::new()));
    unsafe {
        runmat_accelerate_api::register_provider(provider);
    }
    let image = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[2, 2],
        })
        .expect("image upload");
    let image = image
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("publish provenance view");

    let error = block_on(imfilter_builtin(
        Value::GpuTensor(image.clone()),
        Value::Tensor(simple_tensor(&[1.0], 1, 1)),
        Vec::new(),
    ))
    .expect_err("aliased provider output must reject");

    assert_eq!(error.identifier(), IMFILTER_ERROR_INTERNAL.identifier);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&image),
        Some(ProviderPrecision::F64)
    );
    assert_eq!(
        runmat_accelerate_api::handle_storage(&image),
        GpuTensorStorage::Real
    );
    assert_eq!(runmat_accelerate_api::handle_integer_type(&image), None);
    assert!(!runmat_accelerate_api::handle_is_logical(&image));
    assert_eq!(
        runmat_accelerate_api::handle_class_identity(&image),
        Some(runmat_types::standard::DOUBLE.owned())
    );
    assert_eq!(
        runmat_accelerate_api::handle_transpose_info(&image)
            .map(|info| { (info.base_rows, info.base_cols) }),
        None
    );
    assert!(runmat_accelerate_api::handle_is_explicit(&image));
    assert_eq!(
        block_on(provider.download(&image))
            .expect("protected input remains allocated")
            .data,
        vec![1.0, 2.0, 3.0, 4.0]
    );
    image.release().expect("release image");
}

#[test]
fn public_integer_images_preserve_every_documented_class() {
    let cases = [
        (
            IntegerStorage::I8(vec![-2, 63]),
            IntegerStorage::I8(vec![-4, 126]),
        ),
        (
            IntegerStorage::I16(vec![-2, 16_000]),
            IntegerStorage::I16(vec![-4, 32_000]),
        ),
        (
            IntegerStorage::I32(vec![-2, 1_000_000]),
            IntegerStorage::I32(vec![-4, 2_000_000]),
        ),
        (
            IntegerStorage::U8(vec![2, 127]),
            IntegerStorage::U8(vec![4, 254]),
        ),
        (
            IntegerStorage::U16(vec![2, 32_000]),
            IntegerStorage::U16(vec![4, 64_000]),
        ),
        (
            IntegerStorage::U32(vec![2, 1_000_000]),
            IntegerStorage::U32(vec![4, 2_000_000]),
        ),
    ];
    for (input, expected) in cases {
        let image = typed_tensor(input, vec![1, 2]);
        let result = block_on(imfilter_builtin(
            Value::Tensor(image),
            Value::Tensor(simple_tensor(&[2.0], 1, 1)),
            Vec::new(),
        ))
        .expect("integer imfilter");
        let Value::Tensor(output) = result else {
            panic!("expected integer tensor output");
        };
        assert_eq!(output.integer_storage(), Some(&expected));
    }
}

#[test]
fn public_kernel_must_be_nonempty_double_vector_or_matrix() {
    let image = simple_tensor(&[1.0], 1, 1);
    let rank_three = Tensor::new(vec![1.0], vec![1, 1, 1]).unwrap();
    let error = block_on(imfilter_builtin(
        Value::Tensor(image.clone()),
        Value::Tensor(rank_three),
        Vec::new(),
    ))
    .expect_err("rank-three H must reject");
    assert_eq!(error.identifier(), IMFILTER_ERROR_INVALID_INPUT.identifier);

    let empty = Tensor::new(Vec::new(), vec![0, 1]).unwrap();
    let error = block_on(imfilter_builtin(
        Value::Tensor(image),
        Value::Tensor(empty),
        Vec::new(),
    ))
    .expect_err("empty H must reject");
    assert_eq!(error.identifier(), IMFILTER_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn even_kernel_origin_and_symmetric_padding_match_public_rules() {
    let kernel = Tensor::new(vec![1.0, 0.0, 0.0, 0.0], vec![1, 4]).unwrap();
    let plan = build_imfilter_plan(
        &[1, 4],
        &kernel,
        &ImfilterOptions::default(),
        IMFILTER_BUILTIN,
    )
    .unwrap();
    let offsets: Vec<isize> = plan
        .kernel_points
        .iter()
        .map(|point| point.offsets[1])
        .collect();
    assert_eq!(offsets, vec![-1, 0, 1, 2]);
    assert_eq!(reflect_index(-1, 3), 0);
    assert_eq!(reflect_index(-2, 3), 1);
    assert_eq!(reflect_index(3, 3), 2);
    assert_eq!(reflect_index(4, 3), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn doc_example_average_filter_matches_expected() {
    let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    let kernel = Tensor::new(vec![1.0 / 9.0; 9], vec![3, 3]).unwrap();
    let result = apply_imfilter_tensor(
        &image,
        &kernel,
        &ImfilterOptions::default(),
        IMFILTER_BUILTIN,
    )
    .expect("imfilter");
    assert_eq!(result.shape(), &[2, 2]);
    for value in result.materialize_f64() {
        assert!((value - (10.0 / 9.0)).abs() < 1e-12);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn doc_example_convolution_same_matches_expected() {
    let image = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
    let kernel = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let options = ImfilterOptions {
        mode: ImfilterMode::Convolution,
        ..Default::default()
    };
    let result =
        apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
    assert_eq!(result.shape(), &[3, 2]);
    // The documented even-kernel origin is the first element for a 2-by-2
    // kernel. Convolution rotates H while retaining that origin.
    let expected = [25.0, 35.0, 30.0, 26.0, 32.0, 24.0];
    for (got, exp) in result.materialize_f64().iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-12, "got {got}, expected {exp}");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn specifying_numeric_pad_value_matches_manual_options() {
    let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let kernel = simple_tensor(&[1.0; 9], 3, 3);

    let manual = ImfilterOptions {
        padding: ImfilterPadding::Constant,
        constant_value: 5.0,
        ..Default::default()
    };
    let manual_res =
        apply_imfilter_tensor(&image, &kernel, &manual, IMFILTER_BUILTIN).expect("imfilter");

    let via_builtin = block_on(imfilter_builtin(
        Value::Tensor(image.clone()),
        Value::Tensor(kernel.clone()),
        vec![Value::Num(5.0)],
    ))
    .expect("imfilter builtin");
    let via_tensor = tensor::value_into_tensor_for("imfilter", via_builtin).expect("tensor");
    assert_eq!(manual_res.shape(), via_tensor.shape());
    for (a, b) in manual_res
        .materialize_f64()
        .iter()
        .zip(via_tensor.materialize_f64().iter())
    {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn parse_scalar_reads_typed_integer_tensor_exactly() {
    let scalar =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("scalar");
    assert_eq!(
        parse_scalar(IMFILTER_BUILTIN, &Value::Tensor(scalar)).unwrap(),
        u64::MAX as f64
    );

    let vector = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).expect("vector");
    assert!(parse_scalar(IMFILTER_BUILTIN, &Value::Tensor(vector)).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_option_string_raises_error() {
    let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let kernel = simple_tensor(&[1.0; 9], 3, 3);
    let err = block_on(imfilter_builtin(
        Value::Tensor(image),
        Value::Tensor(kernel),
        vec![Value::from("unsupported-mode")],
    ))
    .expect_err("imfilter should error");
    assert!(error_message(&err).contains("unknown option"));
    assert_eq!(err.identifier(), IMFILTER_ERROR_INVALID_OPTION.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fill_requires_scalar_value() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let kernel = simple_tensor(&[1.0; 4], 2, 2);
    let pad = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let err = block_on(imfilter_builtin(
        Value::Tensor(image),
        Value::Tensor(kernel),
        vec![Value::from("fill"), Value::Tensor(pad)],
    ))
    .expect_err("imfilter should error");
    assert!(error_message(&err).contains("scalar value"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn valid_with_larger_kernel_returns_empty_tensor() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let kernel = simple_tensor(&[1.0; 25], 5, 5);
    let result = block_on(imfilter_builtin(
        Value::Tensor(image),
        Value::Tensor(kernel),
        vec![Value::from("valid")],
    ))
    .expect("imfilter");
    match result {
        Value::Tensor(t) => {
            assert!(t.materialize_f64().is_empty());
            assert_eq!(t.shape(), &[0, 0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fill_without_value_defaults_to_zero_padding() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    let kernel = simple_tensor(&[1.0, 1.0, 1.0, 1.0], 2, 2);
    let default = block_on(imfilter_builtin(
        Value::Tensor(image.clone()),
        Value::Tensor(kernel.clone()),
        Vec::new(),
    ))
    .expect("imfilter default");
    let fill_only = block_on(imfilter_builtin(
        Value::Tensor(image),
        Value::Tensor(kernel),
        vec![Value::from("fill")],
    ))
    .expect("imfilter fill");
    assert_eq!(fill_only, default);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn imfilter_wgpu_matches_cpu_same_padding() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };

    let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let kernel = simple_tensor(&[1.0; 9], 3, 3);
    let cpu = apply_imfilter_tensor(
        &image,
        &kernel,
        &ImfilterOptions::default(),
        IMFILTER_BUILTIN,
    )
    .expect("cpu");

    let image_view = HostTensorView {
        data: &image.materialize_f64(),
        shape: image.shape(),
    };
    let image_handle = provider.upload(&image_view).expect("upload image");

    let gpu_value = block_on(imfilter_builtin(
        Value::GpuTensor(image_handle),
        Value::Tensor(kernel),
        Vec::new(),
    ))
    .expect("imfilter");
    let gathered = test_support::gather(gpu_value).expect("gather");

    assert_eq!(cpu.shape(), gathered.shape());
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    for (a, b) in cpu
        .materialize_f64()
        .iter()
        .zip(gathered.materialize_f64().iter())
    {
        assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn imfilter_wgpu_preserves_higher_rank_inputs_through_host_fallback() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");

    let mut shape = vec![1; runmat_accelerate::backend::wgpu::params::IMFILTER_MAX_RANK + 1];
    *shape.last_mut().expect("non-empty shape") = 2;
    let image = Tensor::new(vec![2.0, 4.0], shape.clone()).expect("higher-rank image");
    let kernel = Tensor::new(vec![3.0], vec![1, 1]).expect("scalar kernel");
    let image_handle = gpu_helpers::upload_tensor(provider, &image).expect("upload image");
    let kernel_handle = gpu_helpers::upload_tensor(provider, &kernel).expect("upload kernel");

    let filtered = block_on(imfilter_builtin(
        Value::GpuTensor(image_handle),
        Value::GpuTensor(kernel_handle),
        Vec::new(),
    ))
    .expect("higher-rank imfilter");
    let gathered = test_support::gather(filtered).expect("gather higher-rank output");

    assert_eq!(gathered.shape(), shape);
    assert_eq!(gathered.materialize_f64(), vec![6.0, 12.0]);
}

#[test]
fn imfilter_descriptor_signatures_cover_surface() {
    let labels: Vec<&str> = IMFILTER_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert!(labels.contains(&"B = imfilter(A, H)"));
    assert!(labels.contains(&"B = imfilter(A, H, options...)"));
}

#[test]
fn imfilter_extensions_are_gated_and_capability_covers_shape_changing_forms() {
    assert_eq!(IMFILTER_INTEGER_CAPABILITIES.len(), 1);
    assert_eq!(
        IMFILTER_INTEGER_CAPABILITIES[0].overload,
        BuiltinIntegerOverloadKind::Multiple
    );
    assert_eq!(
        IMFILTER_INTEGER_CAPABILITIES[0].overflow,
        BuiltinIntegerOverflowRule::Saturate
    );

    let image = Value::Tensor(simple_tensor(&[1.0], 1, 1));
    let kernel = Value::Tensor(simple_tensor(&[1.0], 1, 1));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for option in ["valid", "fill"] {
        let error = block_on(imfilter_builtin(
            image.clone(),
            kernel.clone(),
            vec![Value::from(option)],
        ))
        .expect_err("extension must reject in MATLAB-compatible mode");
        assert!(error
            .identifier()
            .is_some_and(|identifier| identifier.starts_with("RunMat:compatibility:")));
    }
}

#[test]
fn imfilter_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = IMFILTER_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert!(codes.contains(&"RM.IMFILTER.INVALID_INPUT"));
    assert!(codes.contains(&"RM.IMFILTER.INVALID_OPTION"));
    assert!(codes.contains(&"RM.IMFILTER.INTERNAL"));
}
