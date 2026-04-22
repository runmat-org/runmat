//! MATLAB-compatible `gradient` builtin with scalar-spacing GPU residency.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{ComplexTensor, ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "gradient";

fn gradient_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn gradient_type(args: &[Type], ctx: &ResolveContext) -> Type {
    numeric_unary_type(args, ctx)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::gradient")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gradient",
    op_kind: GpuOpKind::Custom("numerical-gradient"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("gradient_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may keep scalar-spacing gradients on device via `gradient_dim`; coordinate-vector spacing falls back to the host in this implementation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::gradient")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gradient",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Gradient preserves input shape and uses edge-aware finite differences, so providers expose it through a custom sink hook.",
};

#[runtime_builtin(
    name = "gradient",
    category = "math/reduction",
    summary = "Numerical gradients using central differences with MATLAB-compatible output ordering.",
    keywords = "gradient,numerical gradient,finite difference,vector field,gpu",
    accel = "gradient",
    type_resolver(gradient_type),
    builtin_path = "crate::builtins::math::reduction::gradient"
)]
async fn gradient_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);
    if requested_outputs == 0 {
        return Ok(Value::OutputList(Vec::new()));
    }

    let available_outputs = gradient_output_dims(value_shape(&value), value_len(&value));
    if requested_outputs > available_outputs.len() {
        return Err(gradient_error(format!(
            "gradient: requested {requested_outputs} outputs, but input supports at most {}",
            available_outputs.len()
        )));
    }

    let spacings = parse_spacings(&rest, available_outputs.len()).await?;
    let outputs =
        evaluate_gradient_outputs(value, &available_outputs[..requested_outputs], &spacings)
            .await?;

    if crate::output_count::current_output_count().is_some() {
        return Ok(Value::OutputList(outputs));
    }

    Ok(outputs
        .into_iter()
        .next()
        .expect("single-output gradient result"))
}

async fn evaluate_gradient_outputs(
    value: Value,
    requested_dims: &[usize],
    all_spacings: &[f64],
) -> BuiltinResult<Vec<Value>> {
    if let Value::GpuTensor(handle) = value {
        return gradient_gpu_outputs(handle, requested_dims, all_spacings).await;
    }

    evaluate_host_gradient_outputs(value, requested_dims, all_spacings)
}

fn evaluate_host_gradient_outputs(
    value: Value,
    requested_dims: &[usize],
    all_spacings: &[f64],
) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Tensor(tensor) => {
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(tensor::tensor_into_value(gradient_real_tensor_host(
                    tensor.clone(),
                    dim,
                    spacing,
                )?));
            }
            Ok(outputs)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(gradient_error)?;
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(tensor::tensor_into_value(gradient_real_tensor_host(
                    tensor.clone(),
                    dim,
                    spacing,
                )?));
            }
            Ok(outputs)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for(NAME, value).map_err(gradient_error)?;
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(tensor::tensor_into_value(gradient_real_tensor_host(
                    tensor.clone(),
                    dim,
                    spacing,
                )?));
            }
            Ok(outputs)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor {
                data: vec![(re, im)],
                shape: vec![1, 1],
                rows: 1,
                cols: 1,
            };
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(complex_tensor_into_value(gradient_complex_tensor_host(
                    tensor.clone(),
                    dim,
                    spacing,
                )?));
            }
            Ok(outputs)
        }
        Value::ComplexTensor(tensor) => {
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(complex_tensor_into_value(gradient_complex_tensor_host(
                    tensor.clone(),
                    dim,
                    spacing,
                )?));
            }
            Ok(outputs)
        }
        other => Err(gradient_error(format!(
            "gradient: unsupported input type {:?}; expected numeric or logical data",
            other
        ))),
    }
}

async fn gradient_gpu_outputs(
    handle: GpuTensorHandle,
    requested_dims: &[usize],
    all_spacings: &[f64],
) -> BuiltinResult<Vec<Value>> {
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
        return evaluate_host_gradient_outputs(gathered, requested_dims, all_spacings);
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let mut outputs = Vec::with_capacity(requested_dims.len());
        for &dim in requested_dims {
            let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
            match provider.gradient_dim(&handle, dim.saturating_sub(1), spacing) {
                Ok(device_result) => outputs.push(gpu_helpers::resident_gpu_value(device_result)),
                Err(_) => {
                    let gathered =
                        gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
                    return evaluate_host_gradient_outputs(gathered, requested_dims, all_spacings);
                }
            }
        }
        return Ok(outputs);
    }

    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
    evaluate_host_gradient_outputs(gathered, requested_dims, all_spacings)
}

fn spacing_for_dim(dim: usize, available_dims: &[usize], spacings: &[f64]) -> f64 {
    if spacings.len() == 1 {
        return spacings[0];
    }

    let index = available_dims
        .iter()
        .position(|candidate| *candidate == dim)
        .expect("spacing lookup requires matching dimension");
    spacings[index]
}

async fn parse_spacings(args: &[Value], available_dims: usize) -> BuiltinResult<Vec<f64>> {
    match args.len() {
        0 => Ok(vec![1.0; available_dims]),
        1 => {
            let spacing = parse_scalar_spacing(&args[0]).await?;
            Ok(vec![spacing; available_dims])
        }
        count if count == available_dims => {
            let mut spacings = Vec::with_capacity(args.len());
            for value in args {
                spacings.push(parse_scalar_spacing(value).await?);
            }
            Ok(spacings)
        }
        _ => Err(gradient_error(format!(
            "gradient: expected 0, 1, or {available_dims} scalar spacing arguments"
        ))),
    }
}

async fn parse_scalar_spacing(value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Tensor(tensor) if tensor.data.is_empty() => {
            return Err(gradient_error(
                "gradient: empty spacing arguments are not supported",
            ))
        }
        _ => {}
    }

    let Some(spacing) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(gradient_error)?
    else {
        return Err(gradient_error(
            "gradient: only scalar spacings are supported in this implementation",
        ));
    };

    if !spacing.is_finite() {
        return Err(gradient_error("gradient: spacing must be finite"));
    }
    if spacing == 0.0 {
        return Err(gradient_error("gradient: spacing must be nonzero"));
    }
    Ok(spacing)
}

fn value_shape(value: &Value) -> &[usize] {
    match value {
        Value::Tensor(tensor) => &tensor.shape,
        Value::LogicalArray(logical) => &logical.shape,
        Value::ComplexTensor(tensor) => &tensor.shape,
        Value::GpuTensor(handle) => &handle.shape,
        _ => &[],
    }
}

fn value_len(value: &Value) -> usize {
    match value {
        Value::Tensor(tensor) => tensor.data.len(),
        Value::LogicalArray(logical) => logical.data.len(),
        Value::ComplexTensor(tensor) => tensor.data.len(),
        Value::GpuTensor(handle) => product(&handle.shape),
        _ => 1,
    }
}

pub fn matlab_gradient_shape(shape: &[usize], len: usize) -> Vec<usize> {
    if shape.is_empty() {
        if len == 0 {
            Vec::new()
        } else {
            vec![1, 1]
        }
    } else if shape.len() == 1 {
        if shape[0] == 1 {
            vec![1, 1]
        } else {
            vec![1, shape[0]]
        }
    } else {
        shape.to_vec()
    }
}

fn gradient_output_dims(shape: &[usize], len: usize) -> Vec<usize> {
    let normalized_shape = matlab_gradient_shape(shape, len);
    let mut ext_shape = if normalized_shape.is_empty() {
        if len == 0 {
            vec![0, 0]
        } else {
            vec![1, 1]
        }
    } else {
        normalized_shape
    };
    if ext_shape.len() == 1 {
        ext_shape.push(1);
    }

    if ext_shape.len() <= 2 {
        let rows = ext_shape.first().copied().unwrap_or(1);
        let cols = ext_shape.get(1).copied().unwrap_or(1);
        if rows == 1 && cols == 1 {
            vec![1]
        } else if rows == 1 {
            vec![2]
        } else if cols == 1 {
            vec![1]
        } else {
            vec![2, 1]
        }
    } else {
        let mut dims = vec![2, 1];
        for dim in 3..=ext_shape.len() {
            dims.push(dim);
        }
        dims
    }
}

pub fn gradient_real_tensor_host(
    tensor: Tensor,
    dim: usize,
    spacing: f64,
) -> BuiltinResult<Tensor> {
    let Tensor {
        data, shape, dtype, ..
    } = tensor;
    let dim_index = dim.saturating_sub(1);
    let mut shape = matlab_gradient_shape(&shape, data.len());
    while shape.len() <= dim_index {
        shape.push(1);
    }

    if data.is_empty() {
        return Tensor::new_with_dtype(Vec::new(), shape, dtype)
            .map_err(|e| gradient_error(format!("gradient: {e}")));
    }

    let mut ext_shape = shape.clone();
    while ext_shape.len() <= dim_index {
        ext_shape.push(1);
    }
    let len_dim = ext_shape[dim_index];
    let stride_before = if dim_index == 0 {
        1usize
    } else {
        product(&ext_shape[..dim_index]).max(1)
    };
    let stride_after = if dim_index + 1 >= ext_shape.len() {
        1usize
    } else {
        product(&ext_shape[dim_index + 1..]).max(1)
    };

    let mut out = vec![0.0; data.len()];
    if len_dim > 1 {
        let block = stride_before
            .checked_mul(len_dim)
            .ok_or_else(|| gradient_error("gradient: block size overflow"))?;
        for after in 0..stride_after {
            let base = after
                .checked_mul(block)
                .ok_or_else(|| gradient_error("gradient: indexing overflow"))?;
            for before in 0..stride_before {
                for k in 0..len_dim {
                    let idx = base + before + k * stride_before;
                    out[idx] = if k == 0 {
                        (data[idx + stride_before] - data[idx]) / spacing
                    } else if k + 1 == len_dim {
                        (data[idx] - data[idx - stride_before]) / spacing
                    } else {
                        (data[idx + stride_before] - data[idx - stride_before]) / (2.0 * spacing)
                    };
                }
            }
        }
    }

    Tensor::new_with_dtype(out, shape, dtype).map_err(|e| gradient_error(format!("gradient: {e}")))
}

pub fn gradient_complex_tensor_host(
    tensor: ComplexTensor,
    dim: usize,
    spacing: f64,
) -> BuiltinResult<ComplexTensor> {
    let ComplexTensor { data, shape, .. } = tensor;
    let dim_index = dim.saturating_sub(1);
    let mut shape = matlab_gradient_shape(&shape, data.len());
    while shape.len() <= dim_index {
        shape.push(1);
    }

    if data.is_empty() {
        return ComplexTensor::new(Vec::new(), shape)
            .map_err(|e| gradient_error(format!("gradient: {e}")));
    }

    let mut ext_shape = shape.clone();
    while ext_shape.len() <= dim_index {
        ext_shape.push(1);
    }
    let len_dim = ext_shape[dim_index];
    let stride_before = if dim_index == 0 {
        1usize
    } else {
        product(&ext_shape[..dim_index]).max(1)
    };
    let stride_after = if dim_index + 1 >= ext_shape.len() {
        1usize
    } else {
        product(&ext_shape[dim_index + 1..]).max(1)
    };

    let mut out = vec![(0.0, 0.0); data.len()];
    if len_dim > 1 {
        let block = stride_before
            .checked_mul(len_dim)
            .ok_or_else(|| gradient_error("gradient: block size overflow"))?;
        for after in 0..stride_after {
            let base = after
                .checked_mul(block)
                .ok_or_else(|| gradient_error("gradient: indexing overflow"))?;
            for before in 0..stride_before {
                for k in 0..len_dim {
                    let idx = base + before + k * stride_before;
                    out[idx] = if k == 0 {
                        scale_complex(
                            sub_complex(data[idx + stride_before], data[idx]),
                            1.0 / spacing,
                        )
                    } else if k + 1 == len_dim {
                        scale_complex(
                            sub_complex(data[idx], data[idx - stride_before]),
                            1.0 / spacing,
                        )
                    } else {
                        scale_complex(
                            sub_complex(data[idx + stride_before], data[idx - stride_before]),
                            0.5 / spacing,
                        )
                    };
                }
            }
        }
    }

    ComplexTensor::new(out, shape).map_err(|e| gradient_error(format!("gradient: {e}")))
}

fn sub_complex(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (lhs.0 - rhs.0, lhs.1 - rhs.1)
}

fn scale_complex(value: (f64, f64), scale: f64) -> (f64, f64) {
    (value.0 * scale, value.1 * scale)
}

fn product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, value| acc.saturating_mul(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{NumericDType, Tensor};

    fn gradient_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::gradient_builtin(value, rest))
    }

    #[test]
    fn gradient_row_vector_returns_horizontal_derivative() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let result = gradient_builtin(Value::Tensor(tensor), Vec::new()).expect("gradient");
        assert_eq!(
            result,
            Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap())
        );
    }

    #[test]
    fn gradient_one_dimensional_tensor_is_treated_as_row_vector() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3]).unwrap();
        let result =
            gradient_builtin(Value::Tensor(tensor), vec![Value::Num(2.0)]).expect("gradient");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![1.5, 2.0, 2.5]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_matrix_outputs_follow_matlab_order() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = gradient_builtin(Value::Tensor(tensor), Vec::new()).expect("gradient");
        match result {
            Value::OutputList(outputs) => {
                let fx = test_support::gather(outputs[0].clone()).expect("fx");
                let fy = test_support::gather(outputs[1].clone()).expect("fy");
                assert_eq!(fx.data, vec![1.0, 1.0, 1.0, 1.0]);
                assert_eq!(fy.data, vec![2.0, 2.0, 2.0, 2.0]);
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn gradient_scalar_spacing_scales_output() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let result =
            gradient_builtin(Value::Tensor(tensor), vec![Value::Num(2.0)]).expect("gradient");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, vec![1.5, 2.0, 2.5]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_preserves_single_precision_host_tensor() {
        let tensor =
            Tensor::new_with_dtype(vec![1.0, 4.0, 9.0], vec![1, 3], NumericDType::F32).unwrap();
        let result = gradient_builtin(Value::Tensor(tensor), Vec::new()).expect("gradient");
        match result {
            Value::Tensor(out) => assert_eq!(out.dtype, NumericDType::F32),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_complex_host_supported() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 1.0), (4.0, 3.0), (9.0, 6.0)], vec![1, 3]).unwrap();
        let result = gradient_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("gradient");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.data, vec![(3.0, 2.0), (4.0, 2.5), (5.0, 3.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_rejects_coordinate_vector_spacing_in_v1() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let spacing = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let err =
            gradient_builtin(Value::Tensor(tensor), vec![Value::Tensor(spacing)]).unwrap_err();
        assert!(err.message().contains("scalar"));
    }

    #[test]
    fn gradient_rejects_too_many_outputs() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(2));
        let err = gradient_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err();
        assert!(err.message().contains("requested 2 outputs"));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn gradient_gpu_scalar_spacing_matches_cpu_and_stays_resident() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let host =
            Tensor::new_with_dtype(vec![1.0, 4.0, 9.0], vec![1, 3], NumericDType::F32).unwrap();
        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result =
            gradient_builtin(Value::GpuTensor(handle), vec![Value::Num(2.0)]).expect("gradient");
        match result {
            Value::GpuTensor(out) => {
                let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
                assert_eq!(gathered.data, vec![1.5, 2.0, 2.5]);
                assert_eq!(gathered.dtype, NumericDType::F32);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn gradient_gpu_one_dimensional_shape_matches_matlab_row_vector_semantics() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let data = [1.0, 4.0, 9.0];
        let shape = [3usize];
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result =
            gradient_builtin(Value::GpuTensor(handle), vec![Value::Num(2.0)]).expect("gradient");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 3]);
        assert_eq!(gathered.data, vec![1.5, 2.0, 2.5]);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn gradient_gpu_multi_output_uses_output_list() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let host = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let _out_guard = crate::output_count::push_output_count(Some(2));
        let result = gradient_builtin(Value::GpuTensor(handle), Vec::new()).expect("gradient");
        match result {
            Value::OutputList(outputs) => {
                assert!(matches!(outputs[0], Value::GpuTensor(_)));
                assert!(matches!(outputs[1], Value::GpuTensor(_)));
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }
}
