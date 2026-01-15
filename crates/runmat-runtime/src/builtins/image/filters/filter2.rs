//! MATLAB-compatible `filter2` builtin implementing 2-D correlation and convolution.
//!
//! The implementation builds on the shared `imfilter` infrastructure so that host
//! and GPU behaviour stay perfectly aligned. By default the builtin performs a
//! correlation and returns an output the same size as the input image, matching
//! MathWorks MATLAB semantics.

use runmat_accelerate_api::{
    GpuTensorHandle, HostTensorView, ImfilterMode, ImfilterOptions, ImfilterShape,
};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use super::imfilter::apply_imfilter_tensor;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "filter2",
        builtin_path = "crate::builtins::image::filters::filter2"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "filter2"
category: "image/filters"
keywords: ["filter2", "correlation", "convolution", "image filtering", "gpu"]
summary: "Apply 2-D correlation or convolution using MATLAB-compatible semantics."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Reuses the imfilter acceleration hook when available; otherwise RunMat gathers tensors to the host automatically."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::image::filters::filter2::tests"
  integration: "builtins::image::filters::filter2::tests::filter2_gpu_matches_cpu"
---

# What does the `filter2` function do in MATLAB / RunMat?
`filter2(H, X)` performs a 2-D correlation of the image (or matrix) `X` with the kernel `H`.
By default the operation returns an array the same size as `X`, mirroring MATLAB’s behaviour
and matching the identity `filter2(H, X) == conv2(X, rot90(H, 2), 'same')`.

## How does the `filter2` function behave in MATLAB / RunMat?
- The first argument is the kernel and the second is the image or matrix being filtered.
- The default configuration performs correlation with zero padding and `'same'` output sizing.
- Passing `'full'` or `'valid'` as the third argument switches the output size to the full
  convolution result or the strictly valid interior, respectively.
- Supplying `'conv'` rotates the kernel by 180° so the operation matches MATLAB convolution.
  Use `'corr'` for an explicit correlation request; it is the default.
- Inputs may be numeric or logical. Logical arrays are promoted to double precision prior to filtering.
- Both arguments may be gpuArray handles; RunMat keeps them on the device whenever the active
  acceleration provider exposes the `imfilter` hook.

## `filter2` Function GPU Execution Behaviour
The builtin delegates to the same acceleration hook that powers `imfilter`. When the provider
implements `imfilter`, `filter2` uploads host-resident kernels on demand, keeps GPU-resident images
on the device, and only downloads results when absolutely necessary. Providers that skip the hook
trigger a transparent fallback: RunMat gathers the operands to the host and executes the shared
reference implementation, guaranteeing MATLAB-compatible results without extra configuration.

## Examples of using the `filter2` function in MATLAB / RunMat

### Averaging a matrix with a 3x3 kernel
```matlab
H = ones(3) / 9;
X = [1 2 3; 4 5 6; 7 8 9];
Y = filter2(H, X);
```
Expected output:
```matlab
Y =
    1.3333    2.3333    1.7778
    3.0000    5.0000    3.6667
    2.6667    4.3333    3.1111
```

### Requesting the full convolution result
```matlab
H = [1 2; 3 4];
X = [4 1; 2 0];
Y = filter2(H, X, 'full');
```
Expected output:
```matlab
Y =
     4     9     2
    14    23     4
     6     8     0
```

### Restricting the result to the valid interior
```matlab
H = ones(3);
X = magic(5);
Y = filter2(H, X, 'valid');
```
Expected output:
```matlab
Y =
   100    98   116
    99   117   135
   118   136   134
```

### Switching to convolution mode
```matlab
H = [1 2; 3 4];
X = [1 2 3; 4 5 6; 7 8 9];
Y = filter2(H, X, 'conv');
```
Expected output:
```matlab
Y =
     4    11    18
    18    37    47
    36    67    77
```

### Filtering data already on the GPU
```matlab
H = gpuArray([1 0 -1; 1 0 -1; 1 0 -1]);
X = gpuArray([1 2 3; 4 5 6; 7 8 9]);
Y = filter2(H, X);
result = gather(Y);
```
Expected output:
```matlab
result =
     7     4    -7
    15     6   -15
    13     4   -13
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Usually not. When the acceleration provider exposes the `imfilter` hook, `filter2` keeps the
operands on the GPU and returns a GPU tensor. Host fallbacks are automatic; the builtin gathers
data only when the required hook is missing or reports an error. Explicit `gpuArray` calls remain
useful for deterministic residency or backwards compatibility with MATLAB scripts.

## FAQ

### Can I pass string padding modes like `imfilter`?
No. `filter2` mirrors MATLAB and only accepts `'same'`, `'full'`, `'valid'`, `'corr'`, and `'conv'`.
For advanced padding control use `imfilter`.

### Does the builtin normalise the kernel?
No. The kernel is used exactly as supplied. Use helpers such as `fspecial` or manual scaling when you
need a normalised filter.

### What happens when the kernel is larger than the image?
`'same'` still returns an output the same size as the input, padded with zeros where the kernel extends
beyond the image. `'valid'` returns an empty array whenever the kernel does not fully fit within the image.

### Can I combine `filter2` with gpuArray inputs?
Yes. If either argument is a gpuArray, RunMat forwards the computation to the active acceleration provider
and only gathers the data when the provider hook is unavailable.

### Does `filter2` support higher-dimensional arrays?
MathWorks MATLAB defines `filter2` for 2-D filtering. Use `imfilter` when you need explicit padding control
or want to extend the operation to higher-dimensional arrays.

### Does `filter2` preserve logical inputs?
Logical arrays are promoted to double precision before filtering, matching MATLAB behaviour.

### How do I perform separable filtering efficiently?
Apply successive 1-D `filter2` calls with thin kernels or use `imfilter` for more control over padding
and dimensionality.

## See Also
[imfilter](./imfilter), [fspecial](./fspecial), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/image/filters/filter2.rs`
- Report issues or differences at https://github.com/runmat-org/runmat/issues/new/choose
"#;

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

#[runtime_builtin(
    name = "filter2",
    category = "image/filters",
    summary = "Apply a 2-D correlation or convolution with MATLAB-compatible sizing.",
    keywords = "filter2,correlation,convolution,image filtering,gpu",
    accel = "custom-imfilter",
    builtin_path = "crate::builtins::image::filters::filter2"
)]
fn filter2_builtin(kernel: Value, image: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let options = parse_filter2_options(&rest)?;
    match (kernel, image) {
        (Value::GpuTensor(kernel_handle), Value::GpuTensor(image_handle)) => {
            Ok(filter2_gpu(Value::GpuTensor(kernel_handle), image_handle, &options)?)
        }
        (Value::GpuTensor(kernel_handle), image_value) => {
            let kernel_tensor = gpu_helpers::gather_tensor(&kernel_handle)?;
            Ok(filter2_host(Value::Tensor(kernel_tensor), image_value, &options)?)
        }
        (kernel_value, Value::GpuTensor(image_handle)) => {
            Ok(filter2_gpu(kernel_value, image_handle, &options)?)
        }
        (kernel_value, image_value) => (filter2_host(kernel_value, image_value, &options)).map_err(Into::into),
    }
}

fn filter2_host(
    kernel_value: Value,
    image_value: Value,
    options: &ImfilterOptions,
) -> Result<Value, String> {
    let kernel_tensor = tensor::value_into_tensor_for("filter2", kernel_value)?;
    let image_tensor = tensor::value_into_tensor_for("filter2", image_value)?;
    let result = apply_imfilter_tensor(&image_tensor, &kernel_tensor, options)?;
    Ok(tensor::tensor_into_value(result))
}

fn filter2_gpu(
    kernel_value: Value,
    image_handle: GpuTensorHandle,
    options: &ImfilterOptions,
) -> Result<Value, String> {
    let kernel_clone = kernel_value.clone();
    #[cfg(all(test, feature = "wgpu"))]
    {
        let kernel_is_wgpu = matches!(kernel_clone, Value::GpuTensor(ref h) if h.device_id != 0);
        if kernel_is_wgpu || image_handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => {
            let image_tensor = gpu_helpers::gather_tensor(&image_handle)?;
            let kernel_tensor = tensor::value_into_tensor_for("filter2", kernel_clone)?;
            let result = apply_imfilter_tensor(&image_tensor, &kernel_tensor, options)?;
            return Ok(tensor::tensor_into_value(result));
        }
    };

    let mut uploaded_kernel: Option<GpuTensorHandle> = None;
    let mut kernel_tensor_cache: Option<Tensor> = None;

    let kernel_handle = match kernel_value {
        Value::GpuTensor(handle) => handle,
        other => {
            let tensor = tensor::value_into_tensor_for("filter2", other)?;
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            match provider.upload(&view) {
                Ok(uploaded) => {
                    kernel_tensor_cache = Some(tensor);
                    uploaded_kernel = Some(uploaded.clone());
                    uploaded
                }
                Err(_) => {
                    let image_tensor = gpu_helpers::gather_tensor(&image_handle)?;
                    let result = apply_imfilter_tensor(&image_tensor, &tensor, options)?;
                    return Ok(tensor::tensor_into_value(result));
                }
            }
        }
    };

    let kernel_handle_clone = kernel_handle.clone();

    match provider.imfilter(&image_handle, &kernel_handle, options) {
        Ok(output) => {
            if let Some(uploaded) = uploaded_kernel {
                let _ = provider.free(&uploaded);
            }
            Ok(Value::GpuTensor(output))
        }
        Err(_) => {
            if let Some(uploaded) = uploaded_kernel {
                let _ = provider.free(&uploaded);
            }
            let image_tensor = gpu_helpers::gather_tensor(&image_handle)?;
            let kernel_tensor = if let Some(ref tensor) = kernel_tensor_cache {
                tensor.clone()
            } else {
                gpu_helpers::gather_tensor(&kernel_handle_clone)?
            };
            let result = apply_imfilter_tensor(&image_tensor, &kernel_tensor, options)?;
            Ok(tensor::tensor_into_value(result))
        }
    }
}

fn parse_filter2_options(args: &[Value]) -> Result<ImfilterOptions, String> {
    let mut options = ImfilterOptions::default();
    for value in args {
        let Some(text) = tensor::value_to_string(value) else {
            return Err(
                "filter2: expected string option ('same', 'full', 'valid', 'conv', 'corr')"
                    .to_string(),
            );
        };
        let lowered = text.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "same" => options.shape = ImfilterShape::Same,
            "full" => options.shape = ImfilterShape::Full,
            "valid" => options.shape = ImfilterShape::Valid,
            "conv" => options.mode = ImfilterMode::Convolution,
            "corr" => options.mode = ImfilterMode::Correlation,
            other => {
                return Err(format!(
                "filter2: unknown option '{}' (supported: 'same', 'full', 'valid', 'conv', 'corr')",
                other
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
    use runmat_accelerate_api::{HostTensorView, ImfilterMode, ImfilterOptions, ImfilterShape};
    use runmat_builtins::LogicalArray;

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor::new(data, vec![rows, cols]).expect("tensor construction")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_same_matches_imfilter_reference() {
        let kernel = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let image = tensor(vec![4.0, 1.0, 2.0, 0.0], 2, 2);
        let value = filter2_builtin(
            Value::Tensor(kernel.clone()),
            Value::Tensor(image.clone()),
            Vec::new(),
        )
        .expect("filter2");
        let gathered = test_support::gather(value).expect("gather");
        let options = ImfilterOptions {
            shape: ImfilterShape::Same,
            mode: ImfilterMode::Correlation,
            ..Default::default()
        };
        let expected = apply_imfilter_tensor(&image, &kernel, &options).expect("reference");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn full_shape_option_expands_output() {
        let kernel = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let value = filter2_builtin(
            Value::Tensor(kernel.clone()),
            Value::Tensor(image.clone()),
            vec![Value::from("full")],
        )
        .expect("filter2");
        let gathered = test_support::gather(value).expect("gather");
        let options = ImfilterOptions {
            shape: ImfilterShape::Full,
            ..Default::default()
        };
        let expected = apply_imfilter_tensor(&image, &kernel, &options).expect("reference");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn valid_shape_option_matches_reference() {
        let kernel = tensor(vec![1.0, 0.0, 0.0, -1.0], 2, 2);
        let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let value = filter2_builtin(
            Value::Tensor(kernel.clone()),
            Value::Tensor(image.clone()),
            vec![Value::from("valid")],
        )
        .expect("filter2");
        let gathered = test_support::gather(value).expect("gather");
        let options = ImfilterOptions {
            shape: ImfilterShape::Valid,
            ..Default::default()
        };
        let expected = apply_imfilter_tensor(&image, &kernel, &options).expect("reference");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn convolution_mode_rotates_kernel() {
        let kernel = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let value = filter2_builtin(
            Value::Tensor(kernel.clone()),
            Value::Tensor(image.clone()),
            vec![Value::from("conv")],
        )
        .expect("filter2");
        let gathered = test_support::gather(value).expect("gather");
        let options = ImfilterOptions {
            mode: ImfilterMode::Convolution,
            ..Default::default()
        };
        let expected = apply_imfilter_tensor(&image, &kernel, &options).expect("reference");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn explicit_corr_option_matches_default() {
        let kernel = tensor(vec![0.0, 1.0, 1.0, 0.0], 2, 2);
        let image = tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2);

        let default_value = filter2_builtin(
            Value::Tensor(kernel.clone()),
            Value::Tensor(image.clone()),
            Vec::new(),
        )
        .expect("filter2 default");
        let corr_value = filter2_builtin(
            Value::Tensor(kernel.clone()),
            Value::Tensor(image.clone()),
            vec![Value::from("corr")],
        )
        .expect("filter2 corr");

        let default_tensor = test_support::gather(default_value).expect("gather default");
        let corr_tensor = test_support::gather(corr_value).expect("gather corr");
        assert_eq!(default_tensor.shape, corr_tensor.shape);
        assert_eq!(default_tensor.data, corr_tensor.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_and_shape_order_independent() {
        let kernel = tensor(vec![1.0, -1.0, 2.0, -2.0], 2, 2);
        let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

        let conv_first = filter2_builtin(
            Value::Tensor(kernel.clone()),
            Value::Tensor(image.clone()),
            vec![Value::from("conv"), Value::from("full")],
        )
        .expect("filter2 conv,full");
        let shape_first = filter2_builtin(
            Value::Tensor(kernel.clone()),
            Value::Tensor(image.clone()),
            vec![Value::from("full"), Value::from("conv")],
        )
        .expect("filter2 full,conv");

        let conv_first_tensor = test_support::gather(conv_first).expect("gather conv first");
        let shape_first_tensor = test_support::gather(shape_first).expect("gather shape first");
        assert_eq!(conv_first_tensor.shape, shape_first_tensor.shape);
        assert_eq!(conv_first_tensor.data, shape_first_tensor.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_string_option_is_rejected() {
        let kernel = tensor(vec![1.0], 1, 1);
        let image = tensor(vec![1.0], 1, 1);
        let err = filter2_builtin(
            Value::Tensor(kernel),
            Value::Tensor(image),
            vec![Value::Tensor(tensor(vec![2.0], 1, 1))],
        )
        .expect_err("expected option parsing error");
        assert!(
            err.contains("expected string option"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_input_promoted_to_double() {
        let kernel = tensor(vec![1.0, 1.0, 1.0, 1.0], 2, 2);
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).expect("logical array");

        let value = filter2_builtin(
            Value::Tensor(kernel.clone()),
            Value::LogicalArray(logical.clone()),
            Vec::new(),
        )
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
        let expected = apply_imfilter_tensor(&image_tensor, &kernel, &ImfilterOptions::default())
            .expect("expected logical reference");

        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_unknown_option() {
        let kernel = tensor(vec![1.0], 1, 1);
        let image = tensor(vec![1.0], 1, 1);
        let err = filter2_builtin(
            Value::Tensor(kernel),
            Value::Tensor(image),
            vec![Value::from("replicate")],
        )
        .expect_err("filter2 should error");
        assert!(
            err.contains("filter2"),
            "expected error mentioning builtin, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filter2_gpu_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let kernel = tensor(vec![1.0, 0.0, -1.0, 2.0], 2, 2);
            let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
            let image_view = HostTensorView {
                data: &image.data,
                shape: &image.shape,
            };
            let image_handle = provider.upload(&image_view).expect("upload image");
            let result = filter2_builtin(
                Value::Tensor(kernel.clone()),
                Value::GpuTensor(image_handle.clone()),
                Vec::new(),
            )
            .expect("filter2 gpu");
            let gathered = test_support::gather(result).expect("gather");
            let expected =
                apply_imfilter_tensor(&image, &kernel, &ImfilterOptions::default()).expect("ref");
            assert_eq!(gathered.shape, expected.shape);
            assert_eq!(gathered.data, expected.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filter2_gpu_kernel_and_image() {
        test_support::with_test_provider(|provider| {
            let kernel = tensor(vec![0.0, 1.0, -1.0, 0.0], 2, 2);
            let image = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
            let kernel_view = HostTensorView {
                data: &kernel.data,
                shape: &kernel.shape,
            };
            let image_view = HostTensorView {
                data: &image.data,
                shape: &image.shape,
            };
            let kernel_handle = provider.upload(&kernel_view).expect("upload kernel");
            let image_handle = provider.upload(&image_view).expect("upload image");
            let result = filter2_builtin(
                Value::GpuTensor(kernel_handle.clone()),
                Value::GpuTensor(image_handle.clone()),
                Vec::new(),
            )
            .expect("filter2 gpu");
            let gathered = test_support::gather(result).expect("gather");
            let expected = apply_imfilter_tensor(&image, &kernel, &ImfilterOptions::default())
                .expect("reference");
            assert_eq!(gathered.shape, expected.shape);
            assert_eq!(gathered.data, expected.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn filter2_wgpu_same_matches_cpu() {
        std::env::set_var("RUNMAT_WGPU_SKIP_WARMUP", "1");
        std::env::set_var("RUNMAT_WGPU_DISABLE_IMFILTER", "1");
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
            data: &kernel.data,
            shape: &kernel.shape,
        };
        let image_view = HostTensorView {
            data: &image.data,
            shape: &image.shape,
        };

        let kernel_handle = provider.upload(&kernel_view).expect("upload kernel");
        let image_handle = provider.upload(&image_view).expect("upload image");

        let gpu_value = filter2_builtin(
            Value::GpuTensor(kernel_handle.clone()),
            Value::GpuTensor(image_handle.clone()),
            Vec::new(),
        )
        .expect("filter2 wgpu same");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu result");

        let expected =
            apply_imfilter_tensor(&image, &kernel, &ImfilterOptions::default()).expect("expected");

        assert_eq!(gpu_tensor.shape, expected.shape);
        assert_eq!(gpu_tensor.data.len(), expected.data.len());
        let tol = match runmat_accelerate_api::provider().unwrap().precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (gpu, cpu) in gpu_tensor.data.iter().zip(expected.data.iter()) {
            assert!((gpu - cpu).abs() < tol, "{gpu} vs {cpu}");
        }

        let _ = provider.free(&kernel_handle);
        let _ = provider.free(&image_handle);
        std::env::remove_var("RUNMAT_WGPU_SKIP_WARMUP");
        std::env::remove_var("RUNMAT_WGPU_DISABLE_IMFILTER");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn filter2_wgpu_full_conv_matches_cpu() {
        std::env::set_var("RUNMAT_WGPU_SKIP_WARMUP", "1");
        std::env::set_var("RUNMAT_WGPU_DISABLE_IMFILTER", "1");
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
            data: &kernel.data,
            shape: &kernel.shape,
        };
        let image_view = HostTensorView {
            data: &image.data,
            shape: &image.shape,
        };
        let kernel_handle = provider.upload(&kernel_view).expect("upload kernel");
        let image_handle = provider.upload(&image_view).expect("upload image");

        let gpu_value = filter2_builtin(
            Value::GpuTensor(kernel_handle.clone()),
            Value::GpuTensor(image_handle.clone()),
            vec![Value::from("full"), Value::from("conv")],
        )
        .expect("filter2 wgpu full conv");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu result");

        let options = ImfilterOptions {
            shape: ImfilterShape::Full,
            mode: ImfilterMode::Convolution,
            ..Default::default()
        };
        let expected =
            apply_imfilter_tensor(&image, &kernel, &options).expect("expected host result");

        assert_eq!(gpu_tensor.shape, expected.shape);
        assert_eq!(gpu_tensor.data.len(), expected.data.len());
        let tol = match runmat_accelerate_api::provider().unwrap().precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (gpu, cpu) in gpu_tensor.data.iter().zip(expected.data.iter()) {
            assert!((gpu - cpu).abs() < tol, "{gpu} vs {cpu}");
        }

        let _ = provider.free(&kernel_handle);
        let _ = provider.free(&image_handle);
        std::env::remove_var("RUNMAT_WGPU_SKIP_WARMUP");
        std::env::remove_var("RUNMAT_WGPU_DISABLE_IMFILTER");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
