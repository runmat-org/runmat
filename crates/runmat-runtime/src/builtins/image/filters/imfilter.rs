//! MATLAB-compatible `imfilter` builtin implementing multidimensional correlation and
//! convolution with configurable padding strategies.

use runmat_accelerate_api::{
    GpuTensorHandle, HostTensorView, ImfilterMode, ImfilterOptions, ImfilterPadding, ImfilterShape,
};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::filters::imfilter")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "imfilter",
    op_kind: GpuOpKind::Custom("imfilter"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("imfilter")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses provider-side filtering when available; otherwise gathers to host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::filters::imfilter")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "imfilter",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not a fusion candidate; emits standalone correlation kernels.",
};

#[runtime_builtin(
    name = "imfilter",
    category = "image/filters",
    summary = "Apply linear filters with MATLAB-compatible padding semantics.",
    keywords = "imfilter,image,filter,convolution,correlation,padding",
    accel = "custom-imfilter",
    builtin_path = "crate::builtins::image::filters::imfilter"
)]
fn imfilter_builtin(image: Value, kernel: Value, rest: Vec<Value>) -> Result<Value, String> {
    let options = parse_imfilter_options(&rest)?;
    match (image, kernel) {
        (Value::GpuTensor(image_handle), Value::GpuTensor(filter_handle)) => {
            imfilter_gpu(image_handle, Value::GpuTensor(filter_handle), options)
        }
        (Value::GpuTensor(image_handle), filter_value) => {
            imfilter_gpu(image_handle, filter_value, options)
        }
        (image_value, Value::GpuTensor(filter_handle)) => {
            let filter_tensor = gpu_helpers::gather_tensor(&filter_handle)?;
            imfilter_host_value(image_value, filter_tensor, options)
        }
        (image_value, filter_value) => {
            let filter_tensor = tensor::value_into_tensor_for("imfilter", filter_value)?;
            imfilter_host_value(image_value, filter_tensor, options)
        }
    }
}

fn imfilter_host_value(
    image_value: Value,
    kernel_tensor: Tensor,
    options: ImfilterOptions,
) -> Result<Value, String> {
    let image_tensor = tensor::value_into_tensor_for("imfilter", image_value)?;
    let result = apply_imfilter_tensor(&image_tensor, &kernel_tensor, &options)?;
    Ok(tensor::tensor_into_value(result))
}

fn imfilter_gpu(
    image_handle: GpuTensorHandle,
    kernel_value: Value,
    options: ImfilterOptions,
) -> Result<Value, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        let kernel_is_wgpu = matches!(kernel_value, Value::GpuTensor(ref h) if h.device_id != 0);
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
            return match kernel_value {
                Value::GpuTensor(handle) => {
                    let kernel_tensor = gpu_helpers::gather_tensor(&handle)?;
                    let result = apply_imfilter_tensor(&image_tensor, &kernel_tensor, &options)?;
                    Ok(tensor::tensor_into_value(result))
                }
                other => {
                    let kernel_tensor = tensor::value_into_tensor_for("imfilter", other)?;
                    let result = apply_imfilter_tensor(&image_tensor, &kernel_tensor, &options)?;
                    Ok(tensor::tensor_into_value(result))
                }
            };
        }
    };

    let (kernel_handle, uploaded_handle, kernel_tensor_for_fallback) = match kernel_value {
        Value::GpuTensor(handle) => (handle.clone(), None, None),
        other => {
            let tensor = tensor::value_into_tensor_for("imfilter", other)?;
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            match provider.upload(&view) {
                Ok(uploaded) => (uploaded.clone(), Some(uploaded), Some(tensor)),
                Err(_) => {
                    let image_tensor = gpu_helpers::gather_tensor(&image_handle)?;
                    let result = apply_imfilter_tensor(&image_tensor, &tensor, &options)?;
                    return Ok(tensor::tensor_into_value(result));
                }
            }
        }
    };

    match provider.imfilter(&image_handle, &kernel_handle, &options) {
        Ok(output) => {
            if let Some(uploaded) = uploaded_handle {
                let _ = provider.free(&uploaded);
            }
            Ok(Value::GpuTensor(output))
        }
        Err(_) => {
            if let Some(uploaded) = uploaded_handle {
                let _ = provider.free(&uploaded);
            }
            let image_tensor = gpu_helpers::gather_tensor(&image_handle)?;
            let kernel_tensor = if let Some(tensor) = kernel_tensor_for_fallback {
                tensor
            } else {
                gpu_helpers::gather_tensor(&kernel_handle)?
            };
            let result = apply_imfilter_tensor(&image_tensor, &kernel_tensor, &options)?;
            Ok(tensor::tensor_into_value(result))
        }
    }
}

fn parse_imfilter_options(args: &[Value]) -> Result<ImfilterOptions, String> {
    let mut options = ImfilterOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let mut consumed = 0usize;
        if matches_numeric_scalar(&args[idx]) {
            let scalar = parse_scalar("imfilter", &args[idx])?;
            options.padding = ImfilterPadding::Constant;
            options.constant_value = scalar;
        } else if let Some(text) = tensor::value_to_string(&args[idx]) {
            let lowered = text.trim().to_ascii_lowercase();
            match lowered.as_str() {
                "replicate" => options.padding = ImfilterPadding::Replicate,
                "symmetric" => options.padding = ImfilterPadding::Symmetric,
                "circular" => options.padding = ImfilterPadding::Circular,
                "fill" => {
                    options.padding = ImfilterPadding::Constant;
                    if let Some(next) = args.get(idx + 1) {
                        if matches_numeric_scalar(next) {
                            let scalar = parse_scalar("imfilter", next)?;
                            options.constant_value = scalar;
                            consumed = 1;
                        } else if tensor::value_to_string(next).is_some() {
                            options.constant_value = 0.0;
                        } else {
                            return Err(
                                "imfilter: expected numeric pad value after 'fill'".to_string()
                            );
                        }
                    } else {
                        options.constant_value = 0.0;
                    }
                }
                "same" => options.shape = ImfilterShape::Same,
                "full" => options.shape = ImfilterShape::Full,
                "valid" => options.shape = ImfilterShape::Valid,
                "conv" => options.mode = ImfilterMode::Convolution,
                "corr" => options.mode = ImfilterMode::Correlation,
                other => {
                    return Err(format!(
                        "imfilter: unknown option '{}' (supported: 'same', 'full', 'valid', 'replicate', 'symmetric', 'circular', 'fill', 'conv', 'corr')",
                        other
                    ))
                }
            }
        } else {
            return Err(format!(
                "imfilter: unsupported option {:?}; expected string flags or numeric pad values",
                args[idx]
            ));
        }
        idx += 1 + consumed;
    }
    Ok(options)
}

fn matches_numeric_scalar(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Tensor(_) | Value::LogicalArray(_)
    )
}

fn parse_scalar(name: &str, value: &Value) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) => {
            if t.data.len() == 1 {
                Ok(t.data[0])
            } else {
                Err(format!(
                    "{name}: expected scalar value, got tensor of size {}",
                    t.data.len()
                ))
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() == 1 {
                Ok(if la.data[0] != 0 { 1.0 } else { 0.0 })
            } else {
                Err(format!(
                    "{name}: expected scalar logical value, got array of size {}",
                    la.data.len()
                ))
            }
        }
        other => Err(format!("{name}: expected numeric scalar, got {:?}", other)),
    }
}

/// Core host implementation of `imfilter`, shared with the in-process acceleration provider.
#[derive(Clone, Debug)]
pub struct ImfilterKernelPoint {
    pub offsets: Vec<isize>,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct ImfilterPlan {
    pub rank: usize,
    pub output_shape_ext: Vec<usize>,
    pub final_shape: Vec<usize>,
    pub image_shape_ext: Vec<usize>,
    pub image_strides: Vec<usize>,
    pub base_offset: Vec<isize>,
    pub kernel_points: Vec<ImfilterKernelPoint>,
}

impl ImfilterPlan {
    #[inline]
    pub fn evaluate(&self, image_data: &[f64], options: &ImfilterOptions) -> Vec<f64> {
        evaluate_filter(
            &self.output_shape_ext,
            &self.base_offset,
            &self.image_shape_ext,
            &self.image_strides,
            &self.kernel_points,
            options,
            image_data,
        )
    }
}

pub fn build_imfilter_plan(
    image_shape: &[usize],
    kernel: &Tensor,
    options: &ImfilterOptions,
) -> Result<ImfilterPlan, String> {
    if kernel.data.is_empty() || kernel.shape.contains(&0) {
        return Err("imfilter: filter must be non-empty along every dimension".to_string());
    }

    let image_shape_norm = normalize_shape(image_shape);
    let kernel_shape_norm = normalize_shape(&kernel.shape);
    let rank = image_shape_norm.len().max(kernel_shape_norm.len());
    let image_ext = extend_shape(&image_shape_norm, rank);
    let kernel_ext = extend_shape(&kernel_shape_norm, rank);

    validate_kernel_shape(&image_shape_norm, &kernel_ext)?;

    let origin: Vec<usize> = kernel_ext.iter().map(|&dim| dim / 2).collect();
    let full_shape: Vec<usize> = image_ext
        .iter()
        .zip(kernel_ext.iter())
        .map(|(&img, &ker)| img + ker - 1)
        .collect();

    let kernel_points = build_kernel_points(kernel, &kernel_ext, &origin, options.mode);
    let zero_offset = vec![0isize; rank];
    let origin_signed: Vec<isize> = origin.iter().map(|&o| o as isize).collect();
    let neg_origin: Vec<isize> = origin.iter().map(|&o| -(o as isize)).collect();

    let (target_shape_ext, base_offset) = match options.shape {
        ImfilterShape::Full => (full_shape.clone(), neg_origin),
        ImfilterShape::Same => (image_ext.clone(), zero_offset),
        ImfilterShape::Valid => {
            let target: Vec<usize> = image_ext
                .iter()
                .zip(kernel_ext.iter())
                .map(|(&img, &ker)| if img >= ker { img - ker + 1 } else { 0 })
                .collect();
            (target, origin_signed)
        }
    };

    let mut final_shape = target_shape_ext.clone();
    while final_shape.len() > image_shape_norm.len() && final_shape.last() == Some(&1) {
        final_shape.pop();
    }
    if final_shape.is_empty() {
        final_shape.push(1);
    }

    let image_strides = compute_strides(&image_ext);

    Ok(ImfilterPlan {
        rank,
        output_shape_ext: target_shape_ext,
        final_shape,
        image_shape_ext: image_ext,
        image_strides,
        base_offset,
        kernel_points,
    })
}

pub fn apply_imfilter_tensor(
    image: &Tensor,
    kernel: &Tensor,
    options: &ImfilterOptions,
) -> Result<Tensor, String> {
    let plan = build_imfilter_plan(&image.shape, kernel, options)?;
    let data = plan.evaluate(&image.data, options);
    Tensor::new(data, plan.final_shape.clone()).map_err(|e| format!("imfilter: {e}"))
}

fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

fn validate_kernel_shape(image_shape: &[usize], kernel_ext: &[usize]) -> Result<(), String> {
    for (dim_idx, &ker_dim) in kernel_ext.iter().enumerate() {
        let img_dim = image_shape.get(dim_idx).copied().unwrap_or(1);
        if dim_idx >= image_shape.len() && ker_dim > 1 {
            return Err(format!(
                "imfilter: filter dimension {} is {}, but the image has no corresponding axis",
                dim_idx + 1,
                ker_dim
            ));
        }
        if img_dim == 0 {
            return Err("imfilter: image must not have zero-length dimensions".to_string());
        }
    }
    Ok(())
}

fn extend_shape(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut out = shape.to_vec();
    while out.len() < rank {
        out.push(1);
    }
    out
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides
}

fn advance_index(index: &mut [usize], shape: &[usize]) {
    for (idx, &dimension) in index.iter_mut().zip(shape.iter()) {
        *idx += 1;
        if *idx < dimension {
            return;
        }
        *idx = 0;
    }
}

fn build_kernel_points(
    kernel: &Tensor,
    kernel_shape: &[usize],
    origin: &[usize],
    mode: ImfilterMode,
) -> Vec<ImfilterKernelPoint> {
    let rank = kernel_shape.len();
    let strides = compute_strides(kernel_shape);
    let total = kernel.data.len();
    let mut points = Vec::with_capacity(total);
    if total == 0 {
        return points;
    }

    let mut index = vec![0usize; rank];
    for _ in 0..total {
        let linear = index_to_linear(&index, &strides);
        let value = match mode {
            ImfilterMode::Correlation => kernel.data[linear],
            ImfilterMode::Convolution => {
                kernel.data[flipped_linear_index(&index, kernel_shape, &strides)]
            }
        };
        let offsets = index
            .iter()
            .zip(origin.iter())
            .map(|(&idx, &orig)| idx as isize - orig as isize)
            .collect();
        points.push(ImfilterKernelPoint { offsets, value });
        advance_index(&mut index, kernel_shape);
    }

    points
}

fn flipped_linear_index(index: &[usize], shape: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .enumerate()
        .map(|(dim, &coord)| (shape[dim] - 1 - coord) * strides[dim])
        .sum()
}

fn index_to_linear(index: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .zip(strides.iter())
        .map(|(&coord, &stride)| coord * stride)
        .sum()
}

fn sample_with_padding(
    image: &[f64],
    image_shape: &[usize],
    image_strides: &[usize],
    base_index: &[usize],
    base_offset: &[isize],
    offsets: &[isize],
    options: &ImfilterOptions,
) -> f64 {
    if image.is_empty() {
        return options.constant_value;
    }

    let mut final_indices = Vec::with_capacity(image_shape.len());
    for (dim, (&base, &offset)) in base_index.iter().zip(offsets.iter()).enumerate() {
        let coord = base as isize + base_offset[dim] + offset;
        let len = image_shape[dim] as isize;
        if coord >= 0 && coord < len {
            final_indices.push(coord as usize);
            continue;
        }
        match options.padding {
            ImfilterPadding::Constant => return options.constant_value,
            ImfilterPadding::Replicate => final_indices.push(clamp_index(coord, len)),
            ImfilterPadding::Circular => final_indices.push(wrap_index(coord, len)),
            ImfilterPadding::Symmetric => final_indices.push(reflect_index(coord, len)),
        }
    }

    let linear: usize = final_indices
        .iter()
        .zip(image_strides.iter())
        .map(|(&coord, &stride)| coord * stride)
        .sum();
    image.get(linear).copied().unwrap_or(options.constant_value)
}

fn evaluate_filter(
    output_shape: &[usize],
    base_offset: &[isize],
    image_shape: &[usize],
    image_strides: &[usize],
    kernel_points: &[ImfilterKernelPoint],
    options: &ImfilterOptions,
    image_data: &[f64],
) -> Vec<f64> {
    let total = tensor::element_count(output_shape);
    let mut out = vec![0.0; total];
    if total == 0 || kernel_points.is_empty() {
        return out;
    }

    let mut out_index = vec![0usize; output_shape.len()];
    for out_value in out.iter_mut() {
        let mut sum = 0.0;
        for point in kernel_points {
            let value = sample_with_padding(
                image_data,
                image_shape,
                image_strides,
                &out_index,
                base_offset,
                &point.offsets,
                options,
            );
            sum += point.value * value;
        }
        *out_value = sum;
        advance_index(&mut out_index, output_shape);
    }

    out
}

fn clamp_index(coord: isize, len: isize) -> usize {
    if len <= 0 || coord <= 0 {
        0
    } else if coord >= len {
        (len - 1) as usize
    } else {
        coord as usize
    }
}

fn wrap_index(mut coord: isize, len: isize) -> usize {
    if len <= 0 {
        return 0;
    }
    coord %= len;
    if coord < 0 {
        coord += len;
    }
    coord as usize
}

fn reflect_index(coord: isize, len: isize) -> usize {
    if len <= 0 {
        return 0;
    }
    if len == 1 {
        return 0;
    }
    let period = 2 * len - 2;
    let mut value = coord % period;
    if value < 0 {
        value += period;
    }
    if value >= len {
        value = period - value;
    }
    value as usize
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{Tensor, Value};

    fn simple_tensor(data: &[f64], rows: usize, cols: usize) -> Tensor {
        Tensor::new(data.to_vec(), vec![rows, cols]).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn same_padding_default_zero() {
        let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 9], 3, 3);
        let options = ImfilterOptions::default();
        let result = apply_imfilter_tensor(&image, &kernel, &options).expect("imfilter");
        assert_eq!(result.shape, vec![2, 2]);
        assert!(result.data.iter().all(|&v| (v - 10.0).abs() < 1e-12));
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
        let result = apply_imfilter_tensor(&image, &kernel, &options).expect("imfilter");
        assert_eq!(result.shape, vec![2, 2]);
        let expected = [18.0, 24.0, 21.0, 27.0];
        for (got, exp) in result.data.iter().zip(expected.iter()) {
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
        let result = apply_imfilter_tensor(&image, &kernel, &options).expect("imfilter");
        assert_eq!(result.shape, vec![3, 3]);
        let expected = [0.0, 0.0, 0.0, 0.0, 4.0, 14.0, 0.0, 11.0, 30.0];
        for (got, exp) in result.data.iter().zip(expected.iter()) {
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
        let result = apply_imfilter_tensor(&image, &kernel, &options).expect("imfilter");
        assert_eq!(result.shape, vec![1, 1]);
        assert!((result.data[0] - 10.0).abs() < 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn convolution_matches_correlation_with_flipped_kernel() {
        let image = simple_tensor(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 3, 2);
        let kernel = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let corr_opts = ImfilterOptions::default();
        let corr = apply_imfilter_tensor(&image, &kernel, &corr_opts).expect("corr");
        let flipped_kernel = simple_tensor(&[4.0, 3.0, 2.0, 1.0], 2, 2);
        let corr_flipped =
            apply_imfilter_tensor(&image, &flipped_kernel, &corr_opts).expect("corr flip");
        let conv_opts = ImfilterOptions {
            mode: ImfilterMode::Convolution,
            ..Default::default()
        };
        let conv = apply_imfilter_tensor(&image, &kernel, &conv_opts).expect("conv");
        assert_eq!(conv.shape, corr_flipped.shape);
        for ((a, b), c) in conv
            .data
            .iter()
            .zip(corr_flipped.data.iter())
            .zip(corr.data.iter())
        {
            assert!((a - b).abs() < 1e-12 || (a - c).abs() < 1e-8);
        }
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
        let result = apply_imfilter_tensor(&image, &kernel, &options).expect("imfilter");
        let expected = [5.0, 5.0, 5.0, 5.0];
        for (got, exp) in result.data.iter().zip(expected.iter()) {
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
                data: &image.data,
                shape: &image.shape,
            };
            let kernel_view = HostTensorView {
                data: &kernel.data,
                shape: &kernel.shape,
            };
            let image_handle = provider.upload(&image_view).expect("upload image");
            let kernel_handle = provider.upload(&kernel_view).expect("upload kernel");
            let value = imfilter_builtin(
                Value::GpuTensor(image_handle),
                Value::GpuTensor(kernel_handle),
                Vec::new(),
            )
            .expect("imfilter");
            let gathered = test_support::gather(value).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            let expected = [1.0, 5.0, 3.0, 12.0];
            for (got, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((got - exp).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_example_average_filter_matches_expected() {
        let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
        let kernel = Tensor::new(vec![1.0 / 9.0; 9], vec![3, 3]).unwrap();
        let result =
            apply_imfilter_tensor(&image, &kernel, &ImfilterOptions::default()).expect("imfilter");
        assert_eq!(result.shape, vec![2, 2]);
        for value in result.data {
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
        let result = apply_imfilter_tensor(&image, &kernel, &options).expect("imfilter");
        assert_eq!(result.shape, vec![3, 2]);
        let expected = [1.0, 5.0, 9.0, 6.0, 25.0, 35.0];
        for (got, exp) in result.data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12);
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
        let manual_res = apply_imfilter_tensor(&image, &kernel, &manual).expect("imfilter");

        let via_builtin = imfilter_builtin(
            Value::Tensor(image.clone()),
            Value::Tensor(kernel.clone()),
            vec![Value::Num(5.0)],
        )
        .expect("imfilter builtin");
        let via_tensor = tensor::value_into_tensor_for("imfilter", via_builtin).expect("tensor");
        assert_eq!(manual_res.shape, via_tensor.shape);
        for (a, b) in manual_res.data.iter().zip(via_tensor.data.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_option_string_raises_error() {
        let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 9], 3, 3);
        let err = imfilter_builtin(
            Value::Tensor(image),
            Value::Tensor(kernel),
            vec![Value::from("unsupported-mode")],
        )
        .unwrap_err();
        assert!(err.contains("unknown option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_requires_scalar_value() {
        let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 4], 2, 2);
        let pad = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = imfilter_builtin(
            Value::Tensor(image),
            Value::Tensor(kernel),
            vec![Value::from("fill"), Value::Tensor(pad)],
        )
        .unwrap_err();
        assert!(err.contains("scalar value"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn valid_with_larger_kernel_returns_empty_tensor() {
        let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 25], 5, 5);
        let result = imfilter_builtin(
            Value::Tensor(image),
            Value::Tensor(kernel),
            vec![Value::from("valid")],
        )
        .expect("imfilter");
        match result {
            Value::Tensor(t) => {
                assert!(t.data.is_empty());
                assert_eq!(t.shape, vec![0, 0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_without_value_defaults_to_zero_padding() {
        let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0, 1.0, 1.0, 1.0], 2, 2);
        let default = imfilter_builtin(
            Value::Tensor(image.clone()),
            Value::Tensor(kernel.clone()),
            Vec::new(),
        )
        .expect("imfilter default");
        let fill_only = imfilter_builtin(
            Value::Tensor(image),
            Value::Tensor(kernel),
            vec![Value::from("fill")],
        )
        .expect("imfilter fill");
        assert_eq!(fill_only, default);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn imfilter_wgpu_matches_cpu_same_padding() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");

        let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 9], 3, 3);
        let cpu = apply_imfilter_tensor(&image, &kernel, &ImfilterOptions::default()).expect("cpu");

        let image_view = HostTensorView {
            data: &image.data,
            shape: &image.shape,
        };
        let kernel_view = HostTensorView {
            data: &kernel.data,
            shape: &kernel.shape,
        };
        let image_handle = provider.upload(&image_view).expect("upload image");
        let kernel_handle = provider.upload(&kernel_view).expect("upload kernel");

        let gpu_value = imfilter_builtin(
            Value::GpuTensor(image_handle),
            Value::GpuTensor(kernel_handle),
            Vec::new(),
        )
        .expect("imfilter");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_eq!(cpu.shape, gathered.shape);
        let tol = match runmat_accelerate_api::provider().unwrap().precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in cpu.data.iter().zip(gathered.data.iter()) {
            assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
        }
    }
}
