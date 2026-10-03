//! MATLAB-compatible `filter2` builtin implementing 2-D correlation and convolution.
//!
//! The implementation builds on the shared `imfilter` infrastructure so that host
//! and GPU behaviour stay perfectly aligned. By default the builtin performs a
//! correlation and returns an output the same size as the input image, matching
//! MathWorks MATLAB semantics.

use runmat_accelerate_api::{GpuTensorHandle, ImfilterMode, ImfilterOptions, ImfilterShape};
use runmat_builtins::{
    BuiltinErrorDescriptor, Filter2Option, FILTER2_DESCRIPTOR, FILTER2_ERROR_INTERNAL,
    FILTER2_ERROR_INVALID_INPUT, FILTER2_ERROR_INVALID_OPTION, FILTER2_INTEGER_GPU_EXTENSION,
    FILTER2_LOGICAL_GPU_EXTENSION, FILTER2_MODE_EXTENSION, FILTER2_ND_EXTENSION,
    FILTER2_VARIADIC_OPTIONS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

use super::imfilter::{apply_imfilter_tensor, filter_output_shape};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::filters::filter2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "filter2",
    op_kind: GpuOpKind::Custom("filter2"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("imfilter")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to the provider `imfilter` hook with zero padding and correlation/convolution options.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::filters::filter2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "filter2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not a fusion candidate; executes via the dedicated filtering pipeline.",
};

const FILTER2_BUILTIN: &str = "filter2";

fn filter2_descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    detail: Option<&str>,
) -> RuntimeError {
    let message = match detail {
        Some(detail) => format!("{}: {}", error.message, detail),
        None => error.message.to_string(),
    };
    let mut builder = build_runtime_error(message).with_builtin(FILTER2_BUILTIN);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn filter2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    filter2_descriptor_error(error, Some(detail.as_ref()))
}

fn filter2_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut mapped = filter2_error_with_detail(fallback, err.message());
        mapped.source = Some(Box::new(err));
        mapped
    }
}

#[runtime_builtin(
    name = "filter2",
    binding_variant = "default",
    builtin_path = "crate::builtins::image::filters::filter2"
)]
async fn filter2_builtin(
    kernel: Value,
    image: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    ensure_filter2_extensions(&kernel, &image, &rest)?;
    // Invalid controls must fail before a resident integer/logical operand is
    // gathered, or a floating kernel is uploaded to the image owner.
    let options = parse_filter2_options(&rest)?;
    preflight_filter2_geometry(&kernel, &image, options.shape)?;
    let output_single = value_is_single(&kernel) || value_is_single(&image);
    let image_precision_mismatch = matches!(&image, Value::GpuTensor(handle)
    if output_single
        && runmat_accelerate_api::provider_for_handle(handle).is_some_and(|provider| {
            provider.precision() != runmat_accelerate_api::ProviderPrecision::F32
        }));
    if is_nonfloating_gpu(&kernel) || is_nonfloating_gpu(&image) || image_precision_mismatch {
        let kernel = gather_filter2_value(kernel).await?;
        let image = gather_filter2_value(image).await?;
        return filter2_host(kernel, image, &options);
    }
    match (kernel, image) {
        (Value::GpuTensor(kernel_handle), Value::GpuTensor(image_handle)) => {
            filter2_gpu(Value::GpuTensor(kernel_handle), image_handle, &options).await
        }
        (Value::GpuTensor(kernel_handle), image_value) => {
            let kernel_tensor = gpu_helpers::gather_tensor_async(&kernel_handle)
                .await
                .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?;
            filter2_host(Value::Tensor(kernel_tensor), image_value, &options)
        }
        (kernel_value, Value::GpuTensor(image_handle)) => {
            filter2_gpu(kernel_value, image_handle, &options).await
        }
        (kernel_value, image_value) => filter2_host(kernel_value, image_value, &options),
    }
}

fn value_rank(value: &Value) -> Option<usize> {
    match value {
        Value::Tensor(tensor) => Some(tensor.shape.len()),
        Value::LogicalArray(array) => Some(array.shape.len()),
        Value::GpuTensor(handle) => Some(handle.shape().len()),
        _ => None,
    }
}

fn filter2_value_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Tensor(value) => Some(value.shape.clone()),
        Value::LogicalArray(value) => Some(value.shape.clone()),
        Value::GpuTensor(value) => Some(value.shape().to_vec()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(vec![1, 1]),
        _ => None,
    }
}

fn preflight_filter2_geometry(
    kernel: &Value,
    image: &Value,
    shape: ImfilterShape,
) -> BuiltinResult<()> {
    if let (Some(kernel_shape), Some(image_shape)) =
        (filter2_value_shape(kernel), filter2_value_shape(image))
    {
        filter_output_shape(&image_shape, &kernel_shape, shape, FILTER2_BUILTIN)
            .map_err(|error| filter2_map_error(error.into(), &FILTER2_ERROR_INVALID_INPUT))?;
    }
    Ok(())
}

fn is_nonfloating_gpu(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some() || runmat_accelerate_api::handle_is_logical(handle))
}

fn value_is_single(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_none()
            && !runmat_accelerate_api::handle_is_logical(handle)
            && runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn ensure_filter2_extensions(kernel: &Value, image: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if rest.len() > 1 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILTER2_VARIADIC_OPTIONS_EXTENSION,
            FILTER2_BUILTIN,
        )?;
    }
    if rest.iter().any(|value| {
        tensor::value_to_string(value).is_some_and(|text| {
            matches!(text.trim().to_ascii_lowercase().as_str(), "conv" | "corr")
        })
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILTER2_MODE_EXTENSION,
            FILTER2_BUILTIN,
        )?;
    }
    if value_rank(kernel).is_some_and(|rank| rank > 2)
        || value_rank(image).is_some_and(|rank| rank > 2)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILTER2_ND_EXTENSION,
            FILTER2_BUILTIN,
        )?;
    }
    for value in [kernel, image] {
        if let Value::GpuTensor(handle) = value {
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FILTER2_INTEGER_GPU_EXTENSION,
                    FILTER2_BUILTIN,
                )?;
            }
            if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FILTER2_LOGICAL_GPU_EXTENSION,
                    FILTER2_BUILTIN,
                )?;
            }
        }
    }
    Ok(())
}

async fn gather_filter2_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle)
            .await
            .map(Value::Tensor)
            .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT)),
        other => Ok(other),
    }
}

fn filter2_host(
    kernel_value: Value,
    image_value: Value,
    options: &ImfilterOptions,
) -> BuiltinResult<Value> {
    let kernel_tensor = tensor::value_into_tensor_for(FILTER2_BUILTIN, kernel_value)
        .map_err(|err| filter2_error_with_detail(&FILTER2_ERROR_INVALID_INPUT, err))?;
    let image_tensor = tensor::value_into_tensor_for(FILTER2_BUILTIN, image_value)
        .map_err(|err| filter2_error_with_detail(&FILTER2_ERROR_INVALID_INPUT, err))?;
    let output_single = kernel_tensor.numeric_dtype() == NumericDType::F32
        || image_tensor.numeric_dtype() == NumericDType::F32;
    let result = apply_imfilter_tensor(&image_tensor, &kernel_tensor, options, FILTER2_BUILTIN)
        .map_err(|err| filter2_map_error(err.into(), &FILTER2_ERROR_INVALID_INPUT))?;
    if output_single {
        let shape = result.shape.clone();
        let result = Tensor::from_f32(
            result
                .materialize_f64()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            shape,
        )
        .map_err(|err| filter2_error_with_detail(&FILTER2_ERROR_INTERNAL, err))?;
        return Ok(tensor::tensor_into_value(result));
    }
    Ok(tensor::tensor_into_value(result))
}

async fn filter2_gpu(
    kernel_value: Value,
    image_handle: GpuTensorHandle,
    options: &ImfilterOptions,
) -> BuiltinResult<Value> {
    let kernel_clone = kernel_value.clone();
    #[cfg(all(test, feature = "wgpu"))]
    {
        let kernel_is_wgpu = matches!(kernel_clone, Value::GpuTensor(ref h) if h.device_id() != 0);
        if kernel_is_wgpu || image_handle.device_id() != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    image_handle
        .allocation_lease()
        .validate_live()
        .map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                filter2_error_with_detail(
                    &FILTER2_ERROR_INVALID_INPUT,
                    "X allocation is unavailable",
                ),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
    let provider = match gpu_helpers::exact_provider_for_handle(&image_handle) {
        Some(p) => p,
        None => {
            let image_tensor = gpu_helpers::gather_tensor_async(&image_handle)
                .await
                .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?;
            return filter2_host(kernel_clone, Value::Tensor(image_tensor), options);
        }
    };

    let mut uploaded_kernel: Option<GpuTensorHandle> = None;
    let mut kernel_tensor_cache: Option<Tensor> = None;

    let kernel_handle = match kernel_value {
        Value::GpuTensor(handle) => {
            handle.allocation_lease().validate_live().map_err(|error| {
                crate::runtime_error::with_provider_execution_source(
                    filter2_error_with_detail(
                        &FILTER2_ERROR_INVALID_INPUT,
                        "H allocation is unavailable",
                    ),
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
                )
            })?;
            if gpu_helpers::exact_provider_for_handle(&handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
            {
                handle
            } else {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?;
                match upload_filter2_kernel(provider, &tensor, &image_handle, Some(&handle))? {
                    Some(uploaded) => {
                        kernel_tensor_cache = Some(tensor);
                        uploaded_kernel = Some(uploaded.clone());
                        uploaded
                    }
                    None => {
                        let image_tensor = gpu_helpers::gather_tensor_async(&image_handle)
                            .await
                            .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?;
                        return filter2_host(
                            Value::Tensor(tensor),
                            Value::Tensor(image_tensor),
                            options,
                        );
                    }
                }
            }
        }
        other => {
            let tensor = tensor::value_into_tensor_for(FILTER2_BUILTIN, other)
                .map_err(|err| filter2_error_with_detail(&FILTER2_ERROR_INVALID_INPUT, err))?;
            match upload_filter2_kernel(provider, &tensor, &image_handle, None)? {
                Some(uploaded) => {
                    kernel_tensor_cache = Some(tensor);
                    uploaded_kernel = Some(uploaded.clone());
                    uploaded
                }
                None => {
                    let image_tensor = gpu_helpers::gather_tensor_async(&image_handle)
                        .await
                        .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?;
                    return filter2_host(
                        Value::Tensor(tensor),
                        Value::Tensor(image_tensor),
                        options,
                    );
                }
            }
        }
    };

    let kernel_handle_clone = kernel_handle.clone();
    let expected_shape = filter_output_shape(
        image_handle.shape(),
        kernel_handle.shape(),
        options.shape,
        FILTER2_BUILTIN,
    )
    .map_err(|error| filter2_map_error(error.into(), &FILTER2_ERROR_INVALID_INPUT))?;

    match provider
        .imfilter(&image_handle, &kernel_handle, options)
        .await
    {
        Ok(output) => {
            drop(uploaded_kernel);
            let valid = output.shape() == expected_shape
                && output.device_id() == image_handle.device_id()
                && output.allocation_lease().validate_live().is_ok()
                && gpu_helpers::exact_provider_for_handle(&output).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
                && !output.shares_allocation_with(&image_handle)
                && !output.shares_allocation_with(&kernel_handle)
                && runmat_accelerate_api::handle_storage(&output)
                    == runmat_accelerate_api::GpuTensorStorage::Real
                && runmat_accelerate_api::handle_integer_type(&output).is_none()
                && !runmat_accelerate_api::handle_is_logical(&output)
                && runmat_accelerate_api::handle_precision(&output) == Some(provider.precision())
                && gpu_helpers::gpu_class_metadata_matches(
                    &output,
                    Some(provider.precision()),
                    None,
                    false,
                );
            if !valid {
                let mut error = filter2_error_with_detail(
                    &FILTER2_ERROR_INTERNAL,
                    "provider returned an invalid filter2 result",
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            Ok(Value::GpuTensor(output))
        }
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
            drop(uploaded_kernel);
            let image_tensor = gpu_helpers::gather_tensor_async(&image_handle)
                .await
                .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?;
            let kernel_tensor = if let Some(ref tensor) = kernel_tensor_cache {
                tensor.clone()
            } else {
                gpu_helpers::gather_tensor_async(&kernel_handle_clone)
                    .await
                    .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?
            };
            filter2_host(
                Value::Tensor(kernel_tensor),
                Value::Tensor(image_tensor),
                options,
            )
        }
        Err(error) => Err(crate::runtime_error::with_provider_execution_source(
            filter2_error_with_detail(
                &FILTER2_ERROR_INTERNAL,
                format!("provider execution failed: {error}"),
            ),
            error.into(),
        )),
    }
}

fn upload_filter2_kernel(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: &Tensor,
    image: &GpuTensorHandle,
    resident_kernel: Option<&GpuTensorHandle>,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    use runmat_accelerate_api::NumericUploadError;
    let protected = std::iter::once(image)
        .chain(resident_kernel)
        .collect::<Vec<_>>();
    match gpu_helpers::upload_tensor_protecting_typed(provider, tensor, &protected) {
        Ok(handle) => Ok(Some(handle)),
        Err(NumericUploadError::Provider(error))
            if gpu_helpers::provider_hook_is_unsupported(&error) =>
        {
            Ok(None)
        }
        Err(error) => Err(crate::runtime_error::with_provider_execution_source(
            filter2_error_with_detail(
                &FILTER2_ERROR_INTERNAL,
                format!("failed to upload H to X's provider: {error}"),
            ),
            runmat_accelerate_api::ProviderExecutionError::from_source(error),
        )),
    }
}

fn parse_filter2_options(args: &[Value]) -> BuiltinResult<ImfilterOptions> {
    let mut options = ImfilterOptions::default();
    for value in args {
        let Some(text) = tensor::value_to_string(value) else {
            return Err(filter2_error_with_detail(
                &FILTER2_ERROR_INVALID_OPTION,
                "expected string option ('same', 'full', 'valid', 'conv', 'corr')",
            ));
        };
        match Filter2Option::from_name(&text) {
            Some(Filter2Option::Same) => options.shape = ImfilterShape::Same,
            Some(Filter2Option::Full) => options.shape = ImfilterShape::Full,
            Some(Filter2Option::Valid) => options.shape = ImfilterShape::Valid,
            Some(Filter2Option::Convolution) => options.mode = ImfilterMode::Convolution,
            Some(Filter2Option::Correlation) => options.mode = ImfilterMode::Correlation,
            None => {
                return Err(filter2_error_with_detail(
                    &FILTER2_ERROR_INVALID_OPTION,
                    format!(
                        "unknown option '{}' (supported: 'same', 'full', 'valid', 'conv', 'corr')",
                        text.trim().to_ascii_lowercase()
                    ),
                ))
            }
        }
    }
    Ok(options)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostTensorView, ImfilterMode, ImfilterOptions, ImfilterShape};
    use runmat_value::LogicalArray;

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor::new(data, vec![rows, cols]).expect("tensor construction")
    }

    fn error_message(err: &crate::RuntimeError) -> String {
        err.message().to_string()
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
        assert_eq!(gathered.shape, expected.shape);
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
        assert_eq!(gathered.shape, expected.shape);
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
        assert_eq!(gathered.shape, expected.shape);
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
        assert_eq!(gathered.shape, expected.shape);
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
        assert_eq!(default_tensor.shape, corr_tensor.shape);
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
        assert_eq!(conv_first_tensor.shape, shape_first_tensor.shape);
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

        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    }

    #[test]
    fn typed_integer_inputs_produce_double_and_single_input_is_preserved() {
        use runmat_value::IntegerStorage;

        let integer =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2]).unwrap();
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
                shape: &image.shape,
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
            assert_eq!(gathered.shape, expected.shape);
            assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filter2_gpu_kernel_and_image() {
        test_support::with_test_provider(|provider| {
            let kernel = tensor(vec![0.0, 1.0, -1.0, 0.0], 2, 2);
            let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
            let kernel_view = HostTensorView {
                data: &kernel.materialize_f64(),
                shape: &kernel.shape,
            };
            let image_view = HostTensorView {
                data: &image.materialize_f64(),
                shape: &image.shape,
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
            assert_eq!(gathered.shape, expected.shape);
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
                    shape: &image.shape,
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
            shape: &kernel.shape,
        };
        let image_view = HostTensorView {
            data: &image.materialize_f64(),
            shape: &image.shape,
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

        assert_eq!(gpu_tensor.shape, expected.shape);
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
            shape: &kernel.shape,
        };
        let image_view = HostTensorView {
            data: &image.materialize_f64(),
            shape: &image.shape,
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

        assert_eq!(gpu_tensor.shape, expected.shape);
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
        let builtin =
            runmat_builtins::builtin_function_by_name(FILTER2_BUILTIN).expect("filter2 builtin");
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
}
