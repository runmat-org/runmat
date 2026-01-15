//! MATLAB-compatible `permute` builtin with GPU-aware semantics for RunMat.
//!
//! This module implements the full MATLAB semantics for reordering array
//! dimensions and integrates with the acceleration layer to keep gpuArray
//! outputs resident on the device whenever possible.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::permute")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "permute",
    op_kind: GpuOpKind::Custom("permute"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("permute")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers should implement a custom permute hook; the runtime falls back to gather→permute→upload when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::permute")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "permute",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Permute only changes metadata/data layout; fusion plans treat it as a boundary between kernels.",
};

#[runtime_builtin(
    name = "permute",
    category = "array/shape",
    summary = "Reorder the dimensions of arrays, tensors, logical masks, and gpuArray values.",
    keywords = "permute,dimension reorder,swap axes,gpu",
    accel = "custom",
    builtin_path = "crate::builtins::array::shape::permute"
)]
fn permute_builtin(value: Value, order: Value) -> Result<Value, String> {
    let order_vec = parse_order_argument(order)?;
    match value {
        Value::Tensor(t) => {
            validate_rank(&order_vec, t.shape.len())?;
            permute_tensor(t, &order_vec).map(tensor::tensor_into_value)
        }
        Value::LogicalArray(la) => {
            validate_rank(&order_vec, la.shape.len())?;
            permute_logical_array(la, &order_vec).map(Value::LogicalArray)
        }
        Value::ComplexTensor(ct) => {
            validate_rank(&order_vec, ct.shape.len())?;
            permute_complex_tensor(ct, &order_vec).map(Value::ComplexTensor)
        }
        Value::StringArray(sa) => {
            validate_rank(&order_vec, sa.shape.len())?;
            permute_string_array(sa, &order_vec).map(Value::StringArray)
        }
        Value::CharArray(ca) => {
            validate_rank(&order_vec, 2)?;
            permute_char_array(ca, &order_vec).map(Value::CharArray)
        }
        Value::GpuTensor(handle) => {
            validate_rank(&order_vec, handle.shape.len())?;
            permute_gpu(handle, &order_vec)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for("permute", value)?;
            validate_rank(&order_vec, tensor.shape.len())?;
            permute_tensor(tensor, &order_vec).map(tensor::tensor_into_value)
        }
        other => Err(format!(
            "permute: unsupported input type {:?}; expected numeric, logical, complex, string, or gpuArray values",
            other
        )),
    }
}

pub(crate) fn parse_order_argument(order: Value) -> Result<Vec<usize>, String> {
    let tensor = match order {
        Value::Tensor(t) => t,
        Value::LogicalArray(la) => tensor::logical_to_tensor(&la)
            .map_err(|e| format!("permute: unable to parse order vector: {e}"))?,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            tensor::value_into_tensor_for("permute", order)?
        }
        Value::GpuTensor(_) => {
            return Err(
                "permute: order vector must be specified on the host (numeric or logical array)"
                    .to_string(),
            )
        }
        Value::StringArray(_) | Value::CharArray(_) | Value::ComplexTensor(_) | Value::Cell(_) => {
            return Err("permute: order vector must be numeric".to_string())
        }
        other => {
            return Err(format!(
                "permute: order vector must be numeric, got {:?}",
                other
            ))
        }
    };
    parse_order_tensor(&tensor)
}

fn parse_order_tensor(tensor: &Tensor) -> Result<Vec<usize>, String> {
    if !is_vector(tensor) {
        return Err("permute: order must be a row or column vector".to_string());
    }
    if tensor.data.is_empty() {
        return Err("permute: order must contain at least one dimension".to_string());
    }
    let mut order = Vec::with_capacity(tensor.data.len());
    for &entry in &tensor.data {
        if !entry.is_finite() {
            return Err("permute: order indices must be finite".to_string());
        }
        let rounded = entry.round();
        if (rounded - entry).abs() > f64::EPSILON {
            return Err("permute: order indices must be integers".to_string());
        }
        if rounded < 1.0 {
            return Err("permute: order indices must be >= 1".to_string());
        }
        order.push(rounded as usize);
    }
    validate_permutation(&order)?;
    Ok(order)
}

fn validate_permutation(order: &[usize]) -> Result<(), String> {
    let rank = order.len();
    let mut seen = vec![false; rank];
    for &idx in order {
        if idx == 0 || idx > rank {
            return Err(format!(
                "permute: order indices must lie between 1 and {}, got {}",
                rank, idx
            ));
        }
        if seen[idx - 1] {
            return Err(format!("permute: duplicate dimension index {}", idx));
        }
        seen[idx - 1] = true;
    }
    Ok(())
}

pub(crate) fn validate_rank(order: &[usize], rank: usize) -> Result<(), String> {
    if rank > order.len() {
        Err(format!(
            "permute: order length ({}) must be at least ndims(A) ({})",
            order.len(),
            rank
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn permute_tensor(tensor: Tensor, order: &[usize]) -> Result<Tensor, String> {
    let Tensor { data, shape, .. } = tensor;
    let (out, new_shape) = permute_generic(&data, &shape, order)?;
    Tensor::new(out, new_shape).map_err(|e| format!("permute: {e}"))
}

pub(crate) fn permute_complex_tensor(
    ct: ComplexTensor,
    order: &[usize],
) -> Result<ComplexTensor, String> {
    let ComplexTensor { data, shape, .. } = ct;
    let (out, new_shape) = permute_generic(&data, &shape, order)?;
    ComplexTensor::new(out, new_shape).map_err(|e| format!("permute: {e}"))
}

pub(crate) fn permute_logical_array(
    la: LogicalArray,
    order: &[usize],
) -> Result<LogicalArray, String> {
    let LogicalArray { data, shape } = la;
    let (out, new_shape) = permute_generic(&data, &shape, order)?;
    LogicalArray::new(out, new_shape).map_err(|e| format!("permute: {e}"))
}

pub(crate) fn permute_string_array(
    sa: StringArray,
    order: &[usize],
) -> Result<StringArray, String> {
    let StringArray { data, shape, .. } = sa;
    let (out, new_shape) = permute_generic(&data, &shape, order)?;
    StringArray::new(out, new_shape).map_err(|e| format!("permute: {e}"))
}

pub(crate) fn permute_char_array(ca: CharArray, order: &[usize]) -> Result<CharArray, String> {
    match order.len() {
        0 => Err("permute: order must contain at least one dimension".to_string()),
        1 => {
            if order[0] == 1 {
                Ok(ca)
            } else {
                Err("permute: character arrays are 2-D; invalid dimension index".to_string())
            }
        }
        2 => {
            if order.iter().copied().any(|idx| idx == 0 || idx > 2) {
                return Err("permute: character arrays only support dimensions 1 and 2".to_string());
            }
            if order[0] == 1 && order[1] == 2 {
                return Ok(ca);
            }
            if order[0] == 2 && order[1] == 1 {
                let shape = vec![ca.rows, ca.cols];
                let (data, new_shape) = permute_generic(&ca.data, &shape, order)?;
                if new_shape.len() != 2 {
                    return Err("permute: character arrays must remain 2-D".to_string());
                }
                let rows = new_shape[0];
                let cols = new_shape[1];
                CharArray::new(data, rows, cols).map_err(|e| format!("permute: {e}"))
            } else {
                Err("permute: character arrays require order [1 2] or [2 1]".to_string())
            }
        }
        _ => Err("permute: character arrays only support 2-D permutations".to_string()),
    }
}

pub(crate) fn permute_gpu(handle: GpuTensorHandle, order: &[usize]) -> Result<Value, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        let zero_based: Vec<usize> = order.iter().map(|&idx| idx - 1).collect();
        if let Ok(out) = provider.permute(&handle, &zero_based) {
            return Ok(Value::GpuTensor(out));
        }
        let host_tensor = gpu_helpers::gather_tensor(&handle)?;
        let permuted = permute_tensor(host_tensor, order)?;
        let view = HostTensorView {
            data: &permuted.data,
            shape: &permuted.shape,
        };
        provider
            .upload(&view)
            .map(Value::GpuTensor)
            .map_err(|e| format!("permute: {e}"))
    } else {
        let host_tensor = gpu_helpers::gather_tensor(&handle)?;
        permute_tensor(host_tensor, order).map(tensor::tensor_into_value)
    }
}

fn permute_generic<T: Clone>(
    data: &[T],
    shape: &[usize],
    order: &[usize],
) -> Result<(Vec<T>, Vec<usize>), String> {
    let rank = order.len();
    if shape.len() > rank {
        return Err(format!(
            "permute: order length ({}) must be at least ndims(A) ({})",
            rank,
            shape.len()
        ));
    }
    let mut src_shape = shape.to_vec();
    if src_shape.len() < rank {
        src_shape.extend(std::iter::repeat_n(1, rank - src_shape.len()));
    }
    let total: usize = src_shape.iter().product();
    if total != data.len() {
        return Err("permute: input data length does not match shape product".to_string());
    }

    let mut zero_based = Vec::with_capacity(rank);
    for &idx in order {
        zero_based.push(idx - 1);
    }

    let mut dst_shape = vec![0usize; rank];
    for (dst_dim, &src_dim) in zero_based.iter().enumerate() {
        dst_shape[dst_dim] = src_shape[src_dim];
    }

    let src_strides = compute_strides(&src_shape);
    let dst_total: usize = dst_shape.iter().product();
    let mut dst_coords = vec![0usize; rank];
    let mut src_coords = vec![0usize; rank];
    let mut out = Vec::with_capacity(dst_total);

    for dst_index in 0..dst_total {
        let mut rem = dst_index;
        for (dim, &size) in dst_shape.iter().enumerate() {
            if size == 0 {
                dst_coords[dim] = 0;
            } else {
                dst_coords[dim] = rem % size;
                rem /= size;
            }
        }
        for (dst_dim, &src_dim) in zero_based.iter().enumerate() {
            src_coords[src_dim] = dst_coords[dst_dim];
        }
        let mut src_index = 0usize;
        for (dim, &coord) in src_coords.iter().enumerate() {
            src_index += coord * src_strides[dim];
        }
        out.push(data[src_index].clone());
    }

    Ok((out, dst_shape))
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

fn is_vector(tensor: &Tensor) -> bool {
    match tensor.shape.as_slice() {
        [] => true,
        [_] => true,
        [rows, cols] => *rows == 1 || *cols == 1,
        _ => false,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    fn tensor(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_swaps_dims() {
        let data: Vec<f64> = (1..=24).map(|n| n as f64).collect();
        let t = tensor(&data, &[2, 3, 4]);
        let order = tensor(&[2.0, 1.0, 3.0], &[1, 3]);
        let value = permute_builtin(Value::Tensor(t), Value::Tensor(order)).expect("permute");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2, 4]);
                assert_eq!(out.data.len(), 24);
            }
            _ => panic!("expected tensor result"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_adds_trailing_dimension() {
        let row = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
        let order = tensor(&[2.0, 1.0, 3.0], &[1, 3]);
        let value = permute_builtin(Value::Tensor(row), Value::Tensor(order)).expect("permute");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1, 1]);
            }
            _ => panic!("expected tensor result"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_rejects_duplicates() {
        let data: Vec<f64> = (1..=6).map(|n| n as f64).collect();
        let t = tensor(&data, &[2, 3]);
        let order = tensor(&[2.0, 2.0], &[1, 2]);
        let err = permute_builtin(Value::Tensor(t), Value::Tensor(order)).expect_err("should fail");
        assert!(err.contains("duplicate dimension index"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_requires_vector_order() {
        let t = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let order = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let err = permute_builtin(Value::Tensor(t), Value::Tensor(order)).expect_err("should fail");
        assert!(err.contains("order must be a row or column vector"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_rejects_zero_index() {
        let t = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let order = tensor(&[0.0, 1.0], &[1, 2]);
        let err = permute_builtin(Value::Tensor(t), Value::Tensor(order)).expect_err("should fail");
        assert!(err.contains("indices must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_rejects_non_integer_order() {
        let t = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let order = tensor(&[1.2, 2.0], &[1, 2]);
        let err = permute_builtin(Value::Tensor(t), Value::Tensor(order)).expect_err("should fail");
        assert!(err.contains("indices must be integers"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_order_length_must_cover_rank() {
        let data: Vec<f64> = (1..=8).map(|n| n as f64).collect();
        let t = tensor(&data, &[2, 2, 2]);
        let order = tensor(&[2.0, 1.0], &[1, 2]);
        let err = permute_builtin(Value::Tensor(t), Value::Tensor(order)).expect_err("should fail");
        assert!(err.contains("order length (2) must be at least ndims(A) (3)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_logical_preserves_type() {
        let la = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let order = tensor(&[2.0, 1.0], &[1, 2]);
        let value =
            permute_builtin(Value::LogicalArray(la), Value::Tensor(order)).expect("permute");
        match value {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![1, 1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_complex_tensor() {
        let ct = ComplexTensor::new(
            vec![(1.0, 0.0), (2.0, 0.5), (3.0, -1.0), (4.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let order = tensor(&[2.0, 1.0], &[1, 2]);
        let value =
            permute_builtin(Value::ComplexTensor(ct), Value::Tensor(order)).expect("permute");
        match value {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data[0], (1.0, 0.0));
            }
            _ => panic!("expected complex tensor"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_string_array() {
        let sa = StringArray::new(
            vec!["run".into(), "mat".into(), "gpu".into(), "array".into()],
            vec![2, 2],
        )
        .unwrap();
        let order = tensor(&[2.0, 1.0], &[1, 2]);
        let value = permute_builtin(Value::StringArray(sa), Value::Tensor(order)).expect("permute");
        match value {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data[0], "run");
                assert_eq!(out.data[1], "gpu");
            }
            _ => panic!("expected string array"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_char_array_transpose() {
        let ca = CharArray::new("abcd".chars().collect(), 2, 2).unwrap();
        let order = tensor(&[2.0, 1.0], &[1, 2]);
        let value = permute_builtin(Value::CharArray(ca), Value::Tensor(order)).expect("permute");
        match value {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                assert_eq!(out.data.iter().collect::<String>(), "acbd");
            }
            _ => panic!("expected char array"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_char_array_requires_two_dims() {
        let ca = CharArray::new("abcd".chars().collect(), 2, 2).unwrap();
        let order = tensor(&[1.0], &[1, 1]);
        let err =
            permute_builtin(Value::CharArray(ca), Value::Tensor(order)).expect_err("should fail");
        assert!(err.contains("order length (1) must be at least ndims(A) (2)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let host_data: Vec<f64> = (1..=12).map(|n| n as f64).collect();
            let host = tensor(&host_data, &[2, 2, 3]);
            let view = HostTensorView {
                data: &host.data,
                shape: &host.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let order = tensor(&[3.0, 1.0, 2.0], &[1, 3]);
            let value = permute_builtin(Value::GpuTensor(handle.clone()), Value::Tensor(order))
                .expect("permute");
            match value {
                Value::GpuTensor(result) => {
                    let gathered = gpu_helpers::gather_tensor(&result).expect("gather");
                    assert_eq!(gathered.shape, vec![3, 2, 2]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn permute_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");

        let data: Vec<f64> = (0..24).map(|n| n as f64).collect();
        let host = tensor(&data, &[2, 3, 4]);
        let order = tensor(&[3.0, 1.0, 2.0], &[1, 3]);

        let cpu_value = permute_builtin(Value::Tensor(host.clone()), Value::Tensor(order.clone()))
            .expect("cpu permute");

        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        let handle = provider.upload(&view).expect("upload to GPU");
        let gpu_value =
            permute_builtin(Value::GpuTensor(handle), Value::Tensor(order)).expect("gpu permute");
        let gathered = test_support::gather(gpu_value).expect("gather gpu result");

        match cpu_value {
            Value::Tensor(ct) => {
                assert_eq!(ct.shape, gathered.shape);
                assert_eq!(ct.data, gathered.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }
}
