//! MATLAB-compatible `diff` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::reduction::type_resolvers::diff_numeric_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "diff";

fn diff_type(args: &[Type], ctx: &ResolveContext) -> Type {
    diff_numeric_type(args, ctx)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::diff")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "diff",
    op_kind: GpuOpKind::Custom("finite-difference"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("diff_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers surface finite-difference kernels through `diff_dim`; the WGPU backend keeps tensors on the device.",
};

fn diff_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::diff")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "diff",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently delegates to the runtime implementation; providers can override with custom kernels.",
};

#[runtime_builtin(
    name = "diff",
    category = "math/reduction",
    summary = "Forward finite differences of scalars, vectors, matrices, or N-D tensors.",
    keywords = "diff,difference,finite difference,nth difference,gpu",
    accel = "diff",
    type_resolver(diff_type),
    builtin_path = "crate::builtins::math::reduction::diff"
)]
async fn diff_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (order, dim) = parse_arguments(&rest)?;
    if order == 0 {
        return Ok(value);
    }

    match value {
        Value::Tensor(tensor) => {
            diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(diff_error)?;
            diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for("diff", value).map_err(diff_error)?;
            diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor {
                data: vec![(re, im)],
                shape: vec![1, 1],
                rows: 1,
                cols: 1,
            };
            diff_complex_tensor(tensor, order, dim).map(complex_tensor_into_value)
        }
        Value::ComplexTensor(tensor) => {
            diff_complex_tensor(tensor, order, dim).map(complex_tensor_into_value)
        }
        Value::CharArray(chars) => diff_char_array(chars, order, dim),
        Value::GpuTensor(handle) => diff_gpu(handle, order, dim).await,
        other => Err(diff_error(format!(
            "diff: unsupported input type {:?}; expected numeric, logical, or character data",
            other
        ))),
    }
}

fn parse_arguments(args: &[Value]) -> BuiltinResult<(usize, Option<usize>)> {
    match args.len() {
        0 => Ok((1, None)),
        1 => {
            let order = parse_order(&args[0])?;
            Ok((order.unwrap_or(1), None))
        }
        2 => {
            let order = parse_order(&args[0])?.unwrap_or(1);
            let dim = parse_dimension_arg(&args[1])?;
            Ok((order, dim))
        }
        _ => Err(diff_error("diff: unsupported arguments")),
    }
}

fn parse_order(value: &Value) -> BuiltinResult<Option<usize>> {
    if is_empty_array(value) {
        return Ok(None);
    }
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(diff_error(
                    "diff: order must be a non-negative integer scalar",
                ));
            }
            Ok(Some(raw as usize))
        }
        Value::Num(n) => parse_numeric_order(*n).map(Some),
        Value::Tensor(t) if t.data.len() == 1 => parse_numeric_order(t.data[0]).map(Some),
        Value::Bool(b) => Ok(Some(if *b { 1 } else { 0 })),
        other => Err(diff_error(format!(
            "diff: order must be a non-negative integer scalar, got {:?}",
            other
        ))),
    }
}

fn parse_numeric_order(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(diff_error("diff: order must be finite"));
    }
    if value < 0.0 {
        return Err(diff_error(
            "diff: order must be a non-negative integer scalar",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(diff_error(
            "diff: order must be a non-negative integer scalar",
        ));
    }
    Ok(rounded as usize)
}

fn parse_dimension_arg(value: &Value) -> BuiltinResult<Option<usize>> {
    if is_empty_array(value) {
        return Ok(None);
    }
    match value {
        Value::Int(_) | Value::Num(_) => tensor::parse_dimension(value, "diff")
            .map(Some)
            .map_err(diff_error),
        Value::Tensor(t) if t.data.len() == 1 => {
            tensor::parse_dimension(&Value::Num(t.data[0]), "diff")
                .map(Some)
                .map_err(diff_error)
        }
        other => Err(diff_error(format!(
            "diff: dimension must be a positive integer scalar, got {:?}",
            other
        ))),
    }
}

fn is_empty_array(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if t.data.is_empty())
}

async fn diff_gpu(
    handle: GpuTensorHandle,
    order: usize,
    dim: Option<usize>,
) -> BuiltinResult<Value> {
    let working_dim = dim.unwrap_or_else(|| default_dimension(&handle.shape));
    if working_dim == 0 {
        return Err(diff_error("diff: dimension must be >= 1"));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(device_result) = provider.diff_dim(&handle, order, working_dim.saturating_sub(1))
        {
            return Ok(Value::GpuTensor(device_result));
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    diff_tensor_host(tensor, order, Some(working_dim)).map(tensor::tensor_into_value)
}

fn diff_char_array(chars: CharArray, order: usize, dim: Option<usize>) -> BuiltinResult<Value> {
    if order == 0 {
        return Ok(Value::CharArray(chars));
    }
    let shape = vec![chars.rows, chars.cols];
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, shape).map_err(|e| diff_error(format!("diff: {e}")))?;
    diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
}

pub fn diff_tensor_host(tensor: Tensor, order: usize, dim: Option<usize>) -> BuiltinResult<Tensor> {
    let mut current = tensor;
    let mut working_dim = dim.unwrap_or_else(|| default_dimension(&current.shape));
    for _ in 0..order {
        current = diff_tensor_once(current, working_dim)?;
        if current.data.is_empty() {
            break;
        }
        // Preserve explicit dimension if the caller provided one; update when defaulting and shape shrinks.
        if dim.is_none() && dimension_length(&current.shape, working_dim) == 0 {
            working_dim = default_dimension(&current.shape);
        }
    }
    Ok(current)
}

fn diff_complex_tensor(
    tensor: ComplexTensor,
    order: usize,
    dim: Option<usize>,
) -> BuiltinResult<ComplexTensor> {
    let mut current = tensor;
    let mut working_dim = dim.unwrap_or_else(|| default_dimension(&current.shape));
    for _ in 0..order {
        current = diff_complex_tensor_once(current, working_dim)?;
        if current.data.is_empty() {
            break;
        }
        if dim.is_none() && dimension_length(&current.shape, working_dim) == 0 {
            working_dim = default_dimension(&current.shape);
        }
    }
    Ok(current)
}

fn diff_tensor_once(tensor: Tensor, dim: usize) -> BuiltinResult<Tensor> {
    let Tensor {
        data, mut shape, ..
    } = tensor;
    let dim_index = dim.saturating_sub(1);
    while shape.len() <= dim_index {
        shape.push(1);
    }
    let len_dim = shape[dim_index];
    let mut output_shape = shape.clone();
    if len_dim <= 1 || data.is_empty() {
        output_shape[dim_index] = output_shape[dim_index].saturating_sub(1);
        return Tensor::new(Vec::new(), output_shape).map_err(|e| diff_error(format!("diff: {e}")));
    }
    output_shape[dim_index] = len_dim - 1;
    let stride_before = product(&shape[..dim_index]);
    let stride_after = product(&shape[dim_index + 1..]);
    let output_len = stride_before * (len_dim - 1) * stride_after;
    let mut out = Vec::with_capacity(output_len);

    for after in 0..stride_after {
        let after_base = after * stride_before * len_dim;
        for before in 0..stride_before {
            for k in 0..(len_dim - 1) {
                let idx0 = before + after_base + k * stride_before;
                let idx1 = idx0 + stride_before;
                out.push(data[idx1] - data[idx0]);
            }
        }
    }

    Tensor::new(out, output_shape).map_err(|e| diff_error(format!("diff: {e}")))
}

fn diff_complex_tensor_once(tensor: ComplexTensor, dim: usize) -> BuiltinResult<ComplexTensor> {
    let ComplexTensor {
        data, mut shape, ..
    } = tensor;
    let dim_index = dim.saturating_sub(1);
    while shape.len() <= dim_index {
        shape.push(1);
    }
    let len_dim = shape[dim_index];
    let mut output_shape = shape.clone();
    if len_dim <= 1 || data.is_empty() {
        output_shape[dim_index] = output_shape[dim_index].saturating_sub(1);
        return ComplexTensor::new(Vec::new(), output_shape)
            .map_err(|e| diff_error(format!("diff: {e}")));
    }
    output_shape[dim_index] = len_dim - 1;
    let stride_before = product(&shape[..dim_index]);
    let stride_after = product(&shape[dim_index + 1..]);
    let mut out = Vec::with_capacity(stride_before * (len_dim - 1) * stride_after);

    for after in 0..stride_after {
        let after_base = after * stride_before * len_dim;
        for before in 0..stride_before {
            for k in 0..(len_dim - 1) {
                let idx0 = before + after_base + k * stride_before;
                let idx1 = idx0 + stride_before;
                let (re0, im0) = data[idx0];
                let (re1, im1) = data[idx1];
                out.push((re1 - re0, im1 - im0));
            }
        }
    }

    ComplexTensor::new(out, output_shape).map_err(|e| diff_error(format!("diff: {e}")))
}

fn default_dimension(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|&dim| dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn dimension_length(shape: &[usize], dim: usize) -> usize {
    let dim_index = dim.saturating_sub(1);
    if dim_index < shape.len() {
        shape[dim_index]
    } else {
        1
    }
}

fn product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, val| acc.saturating_mul(val))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, Tensor};

    fn diff_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::diff_builtin(value, rest))
    }

    #[test]
    fn diff_type_defaults_tensor() {
        let out = diff_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Tensor { shape: Some(vec![None, None]) });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_row_vector_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let result = diff_builtin(Value::Tensor(tensor), Vec::new()).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_column_vector_second_order() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = diff_builtin(Value::Tensor(tensor), args).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![2.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_matrix_along_columns() {
        let tensor = Tensor::new(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], vec![3, 2]).unwrap();
        let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(2))];
        let result = diff_builtin(Value::Tensor(tensor), args).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(out.data, vec![1.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_handles_empty_when_order_exceeds_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(5))];
        let result = diff_builtin(Value::Tensor(tensor), args).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape[0], 0);
                assert!(out.data.is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_char_array_promotes_to_double() {
        let chars = CharArray::new("ACEG".chars().collect(), 1, 4).unwrap();
        let result = diff_builtin(Value::CharArray(chars), Vec::new()).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![2.0, 2.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_complex_tensor_preserves_type() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 1.0), (3.0, 2.0), (6.0, 5.0)], vec![1, 3]).unwrap();
        let result = diff_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("diff");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![(2.0, 1.0), (3.0, 3.0)]);
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_zero_order_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(0))];
        let result = diff_builtin(Value::Tensor(tensor.clone()), args).expect("diff");
        assert_eq!(result, Value::Tensor(tensor));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_accepts_empty_order_argument() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3, 1]).unwrap();
        let baseline = diff_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("diff");
        let empty = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = diff_builtin(Value::Tensor(tensor), vec![Value::Tensor(empty)]).expect("diff");
        assert_eq!(result, baseline);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_accepts_empty_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![1, 4]).unwrap();
        let baseline = diff_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("diff");
        let empty = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = diff_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::Tensor(empty)],
        )
        .expect("diff");
        assert_eq!(result, baseline);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_rejects_negative_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(-1))];
        let err = diff_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert!(err.message().contains("non-negative"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_rejects_non_integer_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Num(1.5)];
        let err = diff_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert!(err.message().contains("non-negative integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_rejects_invalid_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(0))];
        let err = diff_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert!(err.message().contains("dimension must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = diff_builtin(Value::GpuTensor(handle), Vec::new()).expect("diff");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert_eq!(gathered.data, vec![3.0, 5.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn diff_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];

        let cpu_result = diff_builtin(Value::Tensor(tensor.clone()), args.clone()).expect("diff");
        let expected = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = diff_builtin(Value::GpuTensor(handle), args).expect("diff");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gathered.shape, expected.shape);
        let tol = if matches!(
            provider.precision(),
            runmat_accelerate_api::ProviderPrecision::F32
        ) {
            1e-5
        } else {
            1e-12
        };
        for (a, b) in gathered.data.iter().zip(expected.data.iter()) {
            assert!((a - b).abs() < tol, "|{a} - {b}| >= {tol}");
        }
    }
}
