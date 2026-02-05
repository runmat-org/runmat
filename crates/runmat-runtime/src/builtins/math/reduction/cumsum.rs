//! MATLAB-compatible `cumsum` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, ProviderNanMode, ProviderScanDirection};
use runmat_builtins::{ComplexTensor, ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cumsum";

fn cumsum_type(args: &[Type], ctx: &ResolveContext) -> Type {
    cumulative_numeric_type(args, ctx)
}

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::reduction::type_resolvers::cumulative_numeric_type;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::cumsum")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cumsum",
    op_kind: GpuOpKind::Custom("scan"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("cumsum_scan")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may expose device prefix-sum kernels; the runtime gathers to host when hooks are absent or options are unsupported.",
};

fn cumsum_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::cumsum")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cumsum",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently lowers cumsum to the runtime implementation; providers can substitute specialised scan kernels.",
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CumsumDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CumsumNanMode {
    Include,
    Omit,
}

#[runtime_builtin(
    name = "cumsum",
    category = "math/reduction",
    summary = "Cumulative sum of scalars, vectors, matrices, or N-D tensors.",
    keywords = "cumsum,cumulative sum,running total,reverse,omitnan,gpu",
    accel = "reduction",
    type_resolver(cumsum_type),
    builtin_path = "crate::builtins::math::reduction::cumsum"
)]
async fn cumsum_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let (dim, direction, nan_mode) = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => cumsum_gpu(handle, dim, direction, nan_mode).await,
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cumsum_error(format!("cumsum: {e}")))?;
            let target_dim = dim.unwrap_or(1);
            let result = cumsum_complex_tensor(&tensor, target_dim, direction, nan_mode)?;
            Ok(complex_tensor_into_value(result))
        }
        Value::ComplexTensor(ct) => {
            let target_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&ct.shape));
            let result = cumsum_complex_tensor(&ct, target_dim, direction, nan_mode)?;
            Ok(complex_tensor_into_value(result))
        }
        other => cumsum_host(other, dim, direction, nan_mode),
    }
}

fn parse_arguments(
    args: &[Value],
) -> BuiltinResult<(Option<usize>, CumsumDirection, CumsumNanMode)> {
    if args.len() > 3 {
        return Err(cumsum_error("cumsum: unsupported arguments"));
    }

    let mut dim: Option<usize> = None;
    let mut direction = CumsumDirection::Forward;
    let mut direction_set = false;
    let mut nan_mode = CumsumNanMode::Include;
    let mut nan_set = false;

    for value in args {
        match value {
            Value::Int(_) | Value::Num(_) => {
                if dim.is_some() {
                    return Err(cumsum_error("cumsum: dimension specified more than once"));
                }
                dim = Some(
                    tensor::parse_dimension(value, "cumsum").map_err(|err| cumsum_error(err))?,
                );
            }
            Value::Tensor(t) if t.data.is_empty() => {
                // MATLAB allows [] as a placeholder for the default dimension; ignore it.
            }
            Value::LogicalArray(la) if la.data.is_empty() => {
                // Treat logical [] the same way.
            }
            _ => {
                if let Some(text) = tensor::value_to_string(value) {
                    let keyword = text.trim().to_ascii_lowercase();
                    match keyword.as_str() {
                        "forward" => {
                            if direction_set {
                                return Err(cumsum_error(
                                    "cumsum: direction specified more than once",
                                ));
                            }
                            direction = CumsumDirection::Forward;
                            direction_set = true;
                        }
                        "reverse" => {
                            if direction_set {
                                return Err(cumsum_error(
                                    "cumsum: direction specified more than once",
                                ));
                            }
                            direction = CumsumDirection::Reverse;
                            direction_set = true;
                        }
                        "omitnan" | "omitmissing" => {
                            if nan_set {
                                return Err(cumsum_error(
                                    "cumsum: missing-value handling specified more than once",
                                ));
                            }
                            nan_mode = CumsumNanMode::Omit;
                            nan_set = true;
                        }
                        "includenan" | "includemissing" => {
                            if nan_set {
                                return Err(cumsum_error(
                                    "cumsum: missing-value handling specified more than once",
                                ));
                            }
                            nan_mode = CumsumNanMode::Include;
                            nan_set = true;
                        }
                        "" => {
                            return Err(cumsum_error(
                                "cumsum: empty string option is not supported",
                            ));
                        }
                        other => {
                            return Err(cumsum_error(format!(
                                "cumsum: unrecognised option '{other}'"
                            )));
                        }
                    }
                } else {
                    return Err(cumsum_error(format!(
                        "cumsum: unsupported argument type {value:?}"
                    )));
                }
            }
        }
    }

    Ok((dim, direction, nan_mode))
}

fn cumsum_host(
    value: Value,
    dim: Option<usize>,
    direction: CumsumDirection,
    nan_mode: CumsumNanMode,
) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("cumsum", value).map_err(|err| cumsum_error(err))?;
    let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
    let result = cumsum_tensor(&tensor, target_dim, direction, nan_mode)?;
    Ok(tensor::tensor_into_value(result))
}

async fn cumsum_gpu(
    handle: GpuTensorHandle,
    dim: Option<usize>,
    direction: CumsumDirection,
    nan_mode: CumsumNanMode,
) -> BuiltinResult<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if matches!(direction, CumsumDirection::Reverse) && matches!(nan_mode, CumsumNanMode::Omit) {
        let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        let fallback_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&tensor.shape));
        let result = cumsum_tensor(&tensor, fallback_dim, direction, nan_mode)?;
        return Ok(tensor::tensor_into_value(result));
    }

    if let Some(target) = dim {
        if target == 0 {
            return Err(cumsum_error("cumsum: dimension must be >= 1"));
        }
        if target > handle.shape.len() {
            return Ok(Value::GpuTensor(handle));
        }
    }

    let fallback_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&handle.shape));
    if fallback_dim == 0 {
        return Err(cumsum_error("cumsum: dimension must be >= 1"));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let zero_based_dim = fallback_dim.saturating_sub(1);
        if zero_based_dim < handle.shape.len() {
            let provider_direction = match direction {
                CumsumDirection::Forward => ProviderScanDirection::Forward,
                CumsumDirection::Reverse => ProviderScanDirection::Reverse,
            };
            let provider_nan_mode = match nan_mode {
                CumsumNanMode::Include => ProviderNanMode::Include,
                CumsumNanMode::Omit => ProviderNanMode::Omit,
            };
            if let Ok(device_result) = provider.cumsum_scan(
                &handle,
                zero_based_dim,
                provider_direction,
                provider_nan_mode,
            ) {
                return Ok(Value::GpuTensor(device_result));
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let result = cumsum_tensor(&tensor, fallback_dim, direction, nan_mode)?;
    Ok(tensor::tensor_into_value(result))
}

fn cumsum_tensor(
    tensor: &Tensor,
    dim: usize,
    direction: CumsumDirection,
    nan_mode: CumsumNanMode,
) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(cumsum_error("cumsum: dimension must be >= 1"));
    }
    if tensor.data.is_empty() || dim > tensor.shape.len() {
        return Ok(tensor.clone());
    }

    let dim_index = dim - 1;
    let segment_len = tensor.shape[dim_index];
    if segment_len == 0 {
        return Ok(tensor.clone());
    }

    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let block = stride_before * segment_len;
    let mut output = vec![0.0f64; tensor.data.len()];

    for after in 0..stride_after {
        let base = after * block;
        for before in 0..stride_before {
            match direction {
                CumsumDirection::Forward => {
                    let mut sum = 0.0f64;
                    let mut sum_is_nan = false;
                    for k in 0..segment_len {
                        let idx = base + before + k * stride_before;
                        let value = tensor.data[idx];
                        match nan_mode {
                            CumsumNanMode::Include => {
                                if sum_is_nan {
                                    output[idx] = f64::NAN;
                                    continue;
                                }
                                if value.is_nan() {
                                    sum_is_nan = true;
                                    output[idx] = f64::NAN;
                                } else {
                                    sum += value;
                                    output[idx] = sum;
                                }
                            }
                            CumsumNanMode::Omit => {
                                if !value.is_nan() {
                                    sum += value;
                                }
                                output[idx] = sum;
                            }
                        }
                    }
                }
                CumsumDirection::Reverse => {
                    let mut sum = 0.0f64;
                    let mut sum_is_nan = false;
                    for offset in (0..segment_len).rev() {
                        let idx = base + before + offset * stride_before;
                        let value = tensor.data[idx];
                        match nan_mode {
                            CumsumNanMode::Include => {
                                if sum_is_nan {
                                    output[idx] = f64::NAN;
                                    continue;
                                }
                                if value.is_nan() {
                                    sum_is_nan = true;
                                    output[idx] = f64::NAN;
                                } else {
                                    sum += value;
                                    output[idx] = sum;
                                }
                            }
                            CumsumNanMode::Omit => {
                                if !value.is_nan() {
                                    sum += value;
                                }
                                output[idx] = sum;
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::new(output, tensor.shape.clone()).map_err(|e| cumsum_error(format!("cumsum: {e}")))
}

fn cumsum_complex_tensor(
    tensor: &ComplexTensor,
    dim: usize,
    direction: CumsumDirection,
    nan_mode: CumsumNanMode,
) -> BuiltinResult<ComplexTensor> {
    if dim == 0 {
        return Err(cumsum_error("cumsum: dimension must be >= 1"));
    }
    if tensor.data.is_empty() || dim > tensor.shape.len() {
        return Ok(tensor.clone());
    }

    let dim_index = dim - 1;
    let segment_len = tensor.shape[dim_index];
    if segment_len == 0 {
        return Ok(tensor.clone());
    }

    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let block = stride_before * segment_len;
    let mut output = vec![(0.0f64, 0.0f64); tensor.data.len()];

    for after in 0..stride_after {
        let base = after * block;
        for before in 0..stride_before {
            match direction {
                CumsumDirection::Forward => {
                    let mut sum = (0.0f64, 0.0f64);
                    let mut sum_is_nan = false;
                    for k in 0..segment_len {
                        let idx = base + before + k * stride_before;
                        let value = tensor.data[idx];
                        let value_is_nan = value.0.is_nan() || value.1.is_nan();
                        match nan_mode {
                            CumsumNanMode::Include => {
                                if sum_is_nan {
                                    output[idx] = (f64::NAN, f64::NAN);
                                    continue;
                                }
                                if value_is_nan {
                                    sum_is_nan = true;
                                    output[idx] = (f64::NAN, f64::NAN);
                                } else {
                                    sum.0 += value.0;
                                    sum.1 += value.1;
                                    output[idx] = sum;
                                }
                            }
                            CumsumNanMode::Omit => {
                                if !value_is_nan {
                                    sum.0 += value.0;
                                    sum.1 += value.1;
                                }
                                output[idx] = sum;
                            }
                        }
                    }
                }
                CumsumDirection::Reverse => {
                    let mut sum = (0.0f64, 0.0f64);
                    let mut sum_is_nan = false;
                    for offset in (0..segment_len).rev() {
                        let idx = base + before + offset * stride_before;
                        let value = tensor.data[idx];
                        let value_is_nan = value.0.is_nan() || value.1.is_nan();
                        match nan_mode {
                            CumsumNanMode::Include => {
                                if sum_is_nan {
                                    output[idx] = (f64::NAN, f64::NAN);
                                    continue;
                                }
                                if value_is_nan {
                                    sum_is_nan = true;
                                    output[idx] = (f64::NAN, f64::NAN);
                                } else {
                                    sum.0 += value.0;
                                    sum.1 += value.1;
                                    output[idx] = sum;
                                }
                            }
                            CumsumNanMode::Omit => {
                                if !value_is_nan {
                                    sum.0 += value.0;
                                    sum.1 += value.1;
                                }
                                output[idx] = sum;
                            }
                        }
                    }
                }
            }
        }
    }

    ComplexTensor::new(output, tensor.shape.clone())
        .map_err(|e| cumsum_error(format!("cumsum: {e}")))
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn default_dimension(tensor: &Tensor) -> usize {
    default_dimension_from_shape(&tensor.shape)
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn dim_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, value| acc.saturating_mul(value))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, Tensor as BuiltinsTensor};

    #[test]
    fn cumsum_type_keeps_shape() {
        let out = cumsum_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    fn cumsum_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::cumsum_builtin(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_scalar_num() {
        let result = cumsum_builtin(Value::Num(7.0), Vec::new()).expect("cumsum scalar");
        assert_eq!(result, Value::Num(7.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_matrix_default_dimension() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = cumsum_builtin(Value::Tensor(tensor), Vec::new()).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![1.0, 5.0, 2.0, 7.0, 3.0, 9.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_matrix_dimension_two() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = cumsum_builtin(Value::Tensor(tensor), args).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![1.0, 4.0, 3.0, 9.0, 6.0, 15.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_reverse_direction() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let result =
            cumsum_builtin(Value::Tensor(tensor), vec![Value::from("reverse")]).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![10.0, 9.0, 7.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_omit_nan_forward() {
        let tensor =
            BuiltinsTensor::new(vec![f64::NAN, 1.0, f64::NAN, 3.0], vec![4, 1]).expect("tensor");
        let result =
            cumsum_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![0.0, 1.0, 1.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_include_nan_propagates() {
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = cumsum_builtin(Value::Tensor(tensor), Vec::new()).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert!(out.data[1].is_nan());
                assert!(out.data[2].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_dim_greater_than_ndims_returns_input() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(5))];
        let result = cumsum_builtin(Value::Tensor(tensor.clone()), args).expect("cumsum");
        match result {
            Value::Tensor(out) => assert_eq!(out, tensor),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_complex_tensor() {
        let data = vec![(1.0, 2.0), (3.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 1]).unwrap();
        let result = cumsum_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("cumsum");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert!((out.data[0].0 - 1.0).abs() < 1e-12);
                assert!((out.data[0].1 - 2.0).abs() < 1e-12);
                assert!((out.data[1].0 - 4.0).abs() < 1e-12);
                assert!((out.data[1].1 - 1.0).abs() < 1e-12);
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cumsum_builtin(Value::GpuTensor(handle), Vec::new()).expect("cumsum");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![1.0, 3.0, 6.0, 10.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_reverse_with_omit_nan() {
        let tensor =
            BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0, 2.0], vec![4, 1]).expect("tensor");
        let result = cumsum_builtin(
            Value::Tensor(tensor),
            vec![Value::from("omitnan"), Value::from("reverse")],
        )
        .expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![7.0, 6.0, 6.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_accepts_omitmissing_synonym() {
        let tensor =
            BuiltinsTensor::new(vec![f64::NAN, 2.0, 3.0, f64::NAN], vec![4, 1]).expect("tensor");
        let result = cumsum_builtin(Value::Tensor(tensor), vec![Value::from("omitmissing")])
            .expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![0.0, 2.0, 5.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_accepts_includemissing_synonym() {
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0], vec![3, 1]).unwrap();
        let result = cumsum_builtin(Value::Tensor(tensor), vec![Value::from("includemissing")])
            .expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert!((out.data[0] - 1.0).abs() < 1e-12);
                assert!(out.data[1].is_nan());
                assert!(out.data[2].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_dimension_placeholder_is_ignored() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let placeholder = BuiltinsTensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let args = vec![Value::Tensor(placeholder), Value::from("reverse")];
        let result = cumsum_builtin(Value::Tensor(tensor), args).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![5.0, 4.0, 7.0, 5.0, 9.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_dimension_zero_errors() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = cumsum_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))]);
        match result {
            Err(err) => {
                assert!(
                    err.message().contains("dimension must be >= 1"),
                    "unexpected result: {err}"
                );
            }
            Ok(_) => panic!("expected error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_duplicate_direction_errors() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = cumsum_builtin(
            Value::Tensor(tensor),
            vec![Value::from("forward"), Value::from("reverse")],
        );
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_unknown_option_errors() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = cumsum_builtin(Value::Tensor(tensor), vec![Value::from("bogus")]);
        match result {
            Err(err) => {
                assert!(
                    err.message().contains("unrecognised option"),
                    "unexpected result: {err}"
                );
            }
            Ok(_) => panic!("expected error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_complex_omit_nan() {
        let data = vec![(1.0, 0.5), (f64::NAN, 2.0), (3.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![3, 1]).unwrap();
        let result = cumsum_builtin(Value::ComplexTensor(tensor), vec![Value::from("omitnan")])
            .expect("cumsum");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert!((out.data[0].0 - 1.0).abs() < 1e-12);
                assert!((out.data[0].1 - 0.5).abs() < 1e-12);
                assert!((out.data[1].0 - 1.0).abs() < 1e-12);
                assert!((out.data[1].1 - 0.5).abs() < 1e-12);
                assert!((out.data[2].0 - 4.0).abs() < 1e-12);
                assert!((out.data[2].1 - (-0.5)).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_gpu_omit_nan_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = BuiltinsTensor::new(vec![f64::NAN, 2.0, 3.0, f64::NAN], vec![4, 1])
                .expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cumsum_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")])
                .expect("cumsum");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![0.0, 2.0, 5.0, 5.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_gpu_dimension_exceeds_rank_returns_handle() {
        test_support::with_test_provider(|provider| {
            let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let original_id = handle.buffer_id;
            let result = cumsum_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Int(IntValue::I32(3))],
            )
            .expect("cumsum");
            match result {
                Value::GpuTensor(out) => {
                    assert_eq!(out.shape, tensor.shape);
                    assert_eq!(out.buffer_id, original_id);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cumsum_wgpu_matches_cpu_forward_dim1() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let cpu = cumsum_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu_result = cumsum_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
        let gathered = test_support::gather(gpu_result).expect("gather gpu");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(ct.shape, gathered.shape);
                for (a, b) in ct.data.iter().zip(gathered.data.iter()) {
                    assert!((a - b).abs() < 1e-9);
                }
            }
            other => panic!("unexpected cpu result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cumsum_wgpu_reverse_omitnan_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0, 2.0], vec![4, 1]).unwrap();
        let cpu = cumsum_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("reverse"), Value::from("omitnan")],
        )
        .unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = provider.upload(&view).unwrap();
        let gpu = cumsum_builtin(
            Value::GpuTensor(handle),
            vec![Value::from("reverse"), Value::from("omitnan")],
        )
        .unwrap();
        let gathered = test_support::gather(gpu).expect("gather gpu");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(ct.shape, gathered.shape);
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-9,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in ct.data.iter().zip(gathered.data.iter()) {
                    assert!((a - b).abs() < tol);
                }
            }
            other => panic!("unexpected cpu result {other:?}"),
        }
    }
}
