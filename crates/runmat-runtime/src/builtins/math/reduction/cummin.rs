//! MATLAB-compatible `cummin` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;

use runmat_accelerate_api::{
    GpuTensorHandle, ProviderCumminResult, ProviderNanMode, ProviderScanDirection,
};
use runmat_builtins::{ComplexTensor, ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cummin";

fn cummin_type(args: &[Type], ctx: &ResolveContext) -> Type {
    cumulative_numeric_type(args, ctx)
}

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::reduction::type_resolvers::cumulative_numeric_type;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::cummin")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cummin",
    op_kind: GpuOpKind::Custom("scan"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("cummin_scan")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes:
        "Providers may expose prefix-min kernels that return running values and indices; the runtime gathers to host when hooks or options are unsupported.",
};

fn cummin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::cummin")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cummin",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently lowers cummin to the runtime implementation; providers can substitute specialised scan kernels when available.",
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CumminDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CumminNanMode {
    Include,
    Omit,
}

/// Evaluation artifact returned by `cummin` that carries both values and indices.
#[derive(Debug, Clone)]
pub struct CumminEvaluation {
    values: Value,
    indices: Value,
}

impl CumminEvaluation {
    /// Consume the evaluation and return only the running minima (single-output call).
    pub fn into_value(self) -> Value {
        self.values
    }

    /// Consume the evaluation and return both minima and indices.
    pub fn into_pair(self) -> (Value, Value) {
        (self.values, self.indices)
    }

    /// Peek at the indices without consuming the evaluation.
    pub fn indices_value(&self) -> Value {
        self.indices.clone()
    }
}

#[runtime_builtin(
    name = "cummin",
    category = "math/reduction",
    summary = "Cumulative minimum and index tracking for scalars, vectors, matrices, or N-D tensors.",
    keywords = "cummin,cumulative minimum,running minimum,reverse,omitnan,indices,gpu",
    accel = "reduction",
    type_resolver(cummin_type),
    builtin_path = "crate::builtins::math::reduction::cummin"
)]
async fn cummin_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(value, &rest).await.map(|eval| eval.into_value())
}

/// Evaluate the builtin once and expose both outputs (value + indices).
pub async fn evaluate(value: Value, rest: &[Value]) -> BuiltinResult<CumminEvaluation> {
    let (dim, direction, nan_mode) = parse_arguments(rest)?;
    match value {
        Value::GpuTensor(handle) => cummin_gpu(handle, dim, direction, nan_mode).await,
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cummin_error(format!("cummin: {e}")))?;
            let target_dim = dim.unwrap_or(1);
            let (values, indices) =
                cummin_complex_tensor(&tensor, target_dim, direction, nan_mode)?;
            Ok(CumminEvaluation {
                values: complex_tensor_into_value(values),
                indices: tensor::tensor_into_value(indices),
            })
        }
        Value::ComplexTensor(ct) => {
            let target_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&ct.shape));
            let (values, indices) = cummin_complex_tensor(&ct, target_dim, direction, nan_mode)?;
            Ok(CumminEvaluation {
                values: complex_tensor_into_value(values),
                indices: tensor::tensor_into_value(indices),
            })
        }
        other => cummin_host(other, dim, direction, nan_mode),
    }
}

fn parse_arguments(
    args: &[Value],
) -> BuiltinResult<(Option<usize>, CumminDirection, CumminNanMode)> {
    if args.len() > 3 {
        return Err(cummin_error("cummin: unsupported arguments"));
    }

    let mut dim: Option<usize> = None;
    let mut direction = CumminDirection::Forward;
    let mut direction_set = false;
    let mut nan_mode = CumminNanMode::Include;
    let mut nan_set = false;

    for value in args {
        match value {
            Value::Int(_) | Value::Num(_) => {
                if dim.is_some() {
                    return Err(cummin_error("cummin: dimension specified more than once"));
                }
                dim = Some(
                    tensor::parse_dimension(value, "cummin").map_err(|err| cummin_error(err))?,
                );
            }
            Value::Tensor(t) if t.data.is_empty() => {
                // MATLAB allows [] placeholders; ignore them.
            }
            Value::LogicalArray(l) if l.data.is_empty() => {}
            _ => {
                if let Some(text) = tensor::value_to_string(value) {
                    let keyword = text.trim().to_ascii_lowercase();
                    match keyword.as_str() {
                        "forward" => {
                            if direction_set {
                                return Err(cummin_error(
                                    "cummin: direction specified more than once",
                                ));
                            }
                            direction = CumminDirection::Forward;
                            direction_set = true;
                        }
                        "reverse" => {
                            if direction_set {
                                return Err(cummin_error(
                                    "cummin: direction specified more than once",
                                ));
                            }
                            direction = CumminDirection::Reverse;
                            direction_set = true;
                        }
                        "omitnan" | "omitmissing" => {
                            if nan_set {
                                return Err(cummin_error(
                                    "cummin: missing-value handling specified more than once",
                                ));
                            }
                            nan_mode = CumminNanMode::Omit;
                            nan_set = true;
                        }
                        "includenan" | "includemissing" => {
                            if nan_set {
                                return Err(cummin_error(
                                    "cummin: missing-value handling specified more than once",
                                ));
                            }
                            nan_mode = CumminNanMode::Include;
                            nan_set = true;
                        }
                        "" => {
                            return Err(cummin_error(
                                "cummin: empty string option is not supported",
                            ));
                        }
                        other => {
                            return Err(cummin_error(format!(
                                "cummin: unrecognised option '{other}'"
                            )));
                        }
                    }
                } else {
                    return Err(cummin_error(format!(
                        "cummin: unsupported argument type {value:?}"
                    )));
                }
            }
        }
    }

    Ok((dim, direction, nan_mode))
}

fn cummin_host(
    value: Value,
    dim: Option<usize>,
    direction: CumminDirection,
    nan_mode: CumminNanMode,
) -> BuiltinResult<CumminEvaluation> {
    let tensor = tensor::value_into_tensor_for("cummin", value).map_err(|err| cummin_error(err))?;
    let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
    let (values, indices) = cummin_tensor(&tensor, target_dim, direction, nan_mode)?;
    Ok(CumminEvaluation {
        values: tensor::tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}

async fn cummin_gpu(
    handle: GpuTensorHandle,
    dim: Option<usize>,
    direction: CumminDirection,
    nan_mode: CumminNanMode,
) -> BuiltinResult<CumminEvaluation> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(target) = dim {
        if target == 0 {
            return Err(cummin_error("cummin: dimension must be >= 1"));
        }
    }

    let target_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&handle.shape));
    if target_dim == 0 {
        return Err(cummin_error("cummin: dimension must be >= 1"));
    }

    if target_dim > handle.shape.len() {
        let indices = ones_indices(&handle.shape)?;
        return Ok(CumminEvaluation {
            values: Value::GpuTensor(handle),
            indices: tensor::tensor_into_value(indices),
        });
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let zero_based_dim = target_dim.saturating_sub(1);
        if zero_based_dim < handle.shape.len() {
            let provider_direction = match direction {
                CumminDirection::Forward => ProviderScanDirection::Forward,
                CumminDirection::Reverse => ProviderScanDirection::Reverse,
            };
            let provider_nan_mode = match nan_mode {
                CumminNanMode::Include => ProviderNanMode::Include,
                CumminNanMode::Omit => ProviderNanMode::Omit,
            };
            if let Ok(ProviderCumminResult { values, indices }) = provider.cummin_scan(
                &handle,
                zero_based_dim,
                provider_direction,
                provider_nan_mode,
            ) {
                return Ok(CumminEvaluation {
                    values: Value::GpuTensor(values),
                    indices: Value::GpuTensor(indices),
                });
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let (values, indices) = cummin_tensor(&tensor, target_dim, direction, nan_mode)?;
    Ok(CumminEvaluation {
        values: tensor::tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}

fn cummin_tensor(
    tensor: &Tensor,
    dim: usize,
    direction: CumminDirection,
    nan_mode: CumminNanMode,
) -> BuiltinResult<(Tensor, Tensor)> {
    if dim == 0 {
        return Err(cummin_error("cummin: dimension must be >= 1"));
    }
    if tensor.data.is_empty() {
        let indices = Tensor::new(Vec::new(), tensor.shape.clone())
            .map_err(|e| cummin_error(format!("cummin: {e}")))?;
        return Ok((tensor.clone(), indices));
    }
    if dim > tensor.shape.len() {
        let indices = ones_indices(&tensor.shape)?;
        return Ok((tensor.clone(), indices));
    }

    let dim_index = dim - 1;
    let segment_len = tensor.shape[dim_index];
    if segment_len == 0 {
        let indices = Tensor::new(Vec::new(), tensor.shape.clone())
            .map_err(|e| cummin_error(format!("cummin: {e}")))?;
        return Ok((tensor.clone(), indices));
    }

    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let block = stride_before * segment_len;
    let mut values_out = vec![0.0f64; tensor.data.len()];
    let mut indices_out = vec![0.0f64; tensor.data.len()];

    for after in 0..stride_after {
        let base = after * block;
        for before in 0..stride_before {
            match direction {
                CumminDirection::Forward => {
                    let mut current = 0.0f64;
                    let mut current_index = 0usize;
                    let mut has_value = false;
                    let mut nan_fixed = false;
                    let mut nan_index = 0usize;
                    for k in 0..segment_len {
                        let idx = base + before + k * stride_before;
                        let value = tensor.data[idx];
                        let position = k + 1;
                        match nan_mode {
                            CumminNanMode::Include => {
                                if nan_fixed {
                                    values_out[idx] = f64::NAN;
                                    indices_out[idx] = nan_index as f64;
                                    continue;
                                }
                                if value.is_nan() {
                                    nan_fixed = true;
                                    nan_index = position;
                                    values_out[idx] = f64::NAN;
                                    indices_out[idx] = position as f64;
                                    continue;
                                }
                                if !has_value || value < current {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                            CumminNanMode::Omit => {
                                if value.is_nan() {
                                    if has_value {
                                        values_out[idx] = current;
                                        indices_out[idx] = current_index as f64;
                                    } else {
                                        values_out[idx] = f64::NAN;
                                        indices_out[idx] = f64::NAN;
                                    }
                                    continue;
                                }
                                if !has_value || value < current {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                        }
                    }
                }
                CumminDirection::Reverse => {
                    let mut current = 0.0f64;
                    let mut current_index = 0usize;
                    let mut has_value = false;
                    let mut nan_fixed = false;
                    let mut nan_index = 0usize;
                    for offset in (0..segment_len).rev() {
                        let idx = base + before + offset * stride_before;
                        let value = tensor.data[idx];
                        let position = offset + 1;
                        match nan_mode {
                            CumminNanMode::Include => {
                                if nan_fixed {
                                    values_out[idx] = f64::NAN;
                                    indices_out[idx] = nan_index as f64;
                                    continue;
                                }
                                if value.is_nan() {
                                    nan_fixed = true;
                                    nan_index = position;
                                    values_out[idx] = f64::NAN;
                                    indices_out[idx] = position as f64;
                                    continue;
                                }
                                if !has_value || value < current {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                            CumminNanMode::Omit => {
                                if value.is_nan() {
                                    if has_value {
                                        values_out[idx] = current;
                                        indices_out[idx] = current_index as f64;
                                    } else {
                                        values_out[idx] = f64::NAN;
                                        indices_out[idx] = f64::NAN;
                                    }
                                    continue;
                                }
                                if !has_value || value < current {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                        }
                    }
                }
            }
        }
    }

    let values_tensor = Tensor::new(values_out, tensor.shape.clone())
        .map_err(|e| cummin_error(format!("cummin: {e}")))?;
    let indices_tensor = Tensor::new(indices_out, tensor.shape.clone())
        .map_err(|e| cummin_error(format!("cummin: {e}")))?;
    Ok((values_tensor, indices_tensor))
}

fn cummin_complex_tensor(
    tensor: &ComplexTensor,
    dim: usize,
    direction: CumminDirection,
    nan_mode: CumminNanMode,
) -> BuiltinResult<(ComplexTensor, Tensor)> {
    if dim == 0 {
        return Err(cummin_error("cummin: dimension must be >= 1"));
    }
    if tensor.data.is_empty() {
        let indices = Tensor::new(Vec::new(), tensor.shape.clone())
            .map_err(|e| cummin_error(format!("cummin: {e}")))?;
        return Ok((tensor.clone(), indices));
    }
    if dim > tensor.shape.len() {
        let indices = ones_indices(&tensor.shape)?;
        return Ok((tensor.clone(), indices));
    }

    let dim_index = dim - 1;
    let segment_len = tensor.shape[dim_index];
    if segment_len == 0 {
        let indices = Tensor::new(Vec::new(), tensor.shape.clone())
            .map_err(|e| cummin_error(format!("cummin: {e}")))?;
        return Ok((tensor.clone(), indices));
    }

    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let block = stride_before * segment_len;
    let mut values_out = vec![(0.0f64, 0.0f64); tensor.data.len()];
    let mut indices_out = vec![0.0f64; tensor.data.len()];

    for after in 0..stride_after {
        let base = after * block;
        for before in 0..stride_before {
            match direction {
                CumminDirection::Forward => {
                    let mut current = (0.0f64, 0.0f64);
                    let mut current_index = 0usize;
                    let mut has_value = false;
                    let mut nan_fixed = false;
                    let mut nan_index = 0usize;
                    for k in 0..segment_len {
                        let idx = base + before + k * stride_before;
                        let value = tensor.data[idx];
                        let position = k + 1;
                        let value_is_nan = complex_is_nan(value);
                        match nan_mode {
                            CumminNanMode::Include => {
                                if nan_fixed {
                                    values_out[idx] = complex_nan();
                                    indices_out[idx] = nan_index as f64;
                                    continue;
                                }
                                if value_is_nan {
                                    nan_fixed = true;
                                    nan_index = position;
                                    values_out[idx] = complex_nan();
                                    indices_out[idx] = position as f64;
                                    continue;
                                }
                                if !has_value || complex_less(value, current) {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                            CumminNanMode::Omit => {
                                if value_is_nan {
                                    if has_value {
                                        values_out[idx] = current;
                                        indices_out[idx] = current_index as f64;
                                    } else {
                                        values_out[idx] = complex_nan();
                                        indices_out[idx] = f64::NAN;
                                    }
                                    continue;
                                }
                                if !has_value || complex_less(value, current) {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                        }
                    }
                }
                CumminDirection::Reverse => {
                    let mut current = (0.0f64, 0.0f64);
                    let mut current_index = 0usize;
                    let mut has_value = false;
                    let mut nan_fixed = false;
                    let mut nan_index = 0usize;
                    for offset in (0..segment_len).rev() {
                        let idx = base + before + offset * stride_before;
                        let value = tensor.data[idx];
                        let position = offset + 1;
                        let value_is_nan = complex_is_nan(value);
                        match nan_mode {
                            CumminNanMode::Include => {
                                if nan_fixed {
                                    values_out[idx] = complex_nan();
                                    indices_out[idx] = nan_index as f64;
                                    continue;
                                }
                                if value_is_nan {
                                    nan_fixed = true;
                                    nan_index = position;
                                    values_out[idx] = complex_nan();
                                    indices_out[idx] = position as f64;
                                    continue;
                                }
                                if !has_value || complex_less(value, current) {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                            CumminNanMode::Omit => {
                                if value_is_nan {
                                    if has_value {
                                        values_out[idx] = current;
                                        indices_out[idx] = current_index as f64;
                                    } else {
                                        values_out[idx] = complex_nan();
                                        indices_out[idx] = f64::NAN;
                                    }
                                    continue;
                                }
                                if !has_value || complex_less(value, current) {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                        }
                    }
                }
            }
        }
    }

    let values_tensor = ComplexTensor::new(values_out, tensor.shape.clone())
        .map_err(|e| cummin_error(format!("cummin: {e}")))?;
    let indices_tensor = Tensor::new(indices_out, tensor.shape.clone())
        .map_err(|e| cummin_error(format!("cummin: {e}")))?;
    Ok((values_tensor, indices_tensor))
}

fn complex_less(candidate: (f64, f64), current: (f64, f64)) -> bool {
    compare_complex_auto(candidate, current) == Ordering::Less
}

fn complex_is_nan(value: (f64, f64)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn complex_nan() -> (f64, f64) {
    (f64::NAN, f64::NAN)
}

fn compare_complex_auto(a: (f64, f64), b: (f64, f64)) -> Ordering {
    let a_mag = magnitude_squared(a);
    let b_mag = magnitude_squared(b);
    if a_mag < b_mag {
        return Ordering::Less;
    }
    if a_mag > b_mag {
        return Ordering::Greater;
    }
    let a_angle = a.1.atan2(a.0);
    let b_angle = b.1.atan2(b.0);
    if a_angle < b_angle {
        Ordering::Less
    } else if a_angle > b_angle {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn magnitude_squared(z: (f64, f64)) -> f64 {
    z.0.mul_add(z.0, z.1 * z.1)
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn ones_indices(shape: &[usize]) -> BuiltinResult<Tensor> {
    let len = tensor::element_count(shape);
    let data = if len == 0 {
        Vec::new()
    } else {
        vec![1.0f64; len]
    };
    Tensor::new(data, shape.to_vec()).map_err(|e| cummin_error(format!("cummin: {e}")))
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
    use runmat_builtins::IntValue;

    #[test]
    fn cummin_type_keeps_shape() {
        let out = cummin_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }

    fn evaluate(value: Value, rest: &[Value]) -> BuiltinResult<CumminEvaluation> {
        block_on(super::evaluate(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_scalar_returns_value_and_index() {
        let eval = evaluate(Value::Num(7.0), &[]).expect("cummin");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(7.0));
        assert_eq!(indices, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_matrix_default_dimension() {
        let tensor = Tensor::new(vec![4.0, 3.0, 2.0, 5.0, 7.0, 1.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![4.0, 3.0, 2.0, 2.0, 7.0, 1.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.shape, vec![2, 3]);
                assert_eq!(idx.data, vec![1.0, 2.0, 1.0, 1.0, 1.0, 2.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_dimension_two_tracks_rows() {
        let tensor = Tensor::new(vec![4.0, 3.0, 2.0, 5.0, 7.0, 1.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![4.0, 3.0, 2.0, 3.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.data, vec![1.0, 1.0, 2.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_reverse_direction() {
        let tensor = Tensor::new(vec![8.0, 3.0, 6.0, 2.0], vec![4, 1]).unwrap();
        let args = vec![Value::from("reverse")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.data, vec![2.0, 2.0, 2.0, 2.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.data, vec![4.0, 4.0, 4.0, 4.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_omit_nan_behaviour() {
        let tensor = Tensor::new(vec![f64::NAN, 5.0, f64::NAN, 3.0], vec![4, 1]).expect("tensor");
        let args = vec![Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert!(out.data[0].is_nan());
                assert_eq!(out.data[1], 5.0);
                assert_eq!(out.data[2], 5.0);
                assert_eq!(out.data[3], 3.0);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert!(idx.data[0].is_nan());
                assert_eq!(idx.data[1], 2.0);
                assert_eq!(idx.data[2], 2.0);
                assert_eq!(idx.data[3], 4.0);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.data[0], 1.0);
                assert!(out.data[1].is_nan());
                assert!(out.data[2].is_nan());
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.data[0], 1.0);
                assert_eq!(idx.data[1], 2.0);
                assert_eq!(idx.data[2], 2.0);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_dimension_greater_than_rank() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(5))];
        let eval = evaluate(Value::Tensor(tensor.clone()), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.data, tensor.data),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert!(idx.data.iter().all(|v| *v == 1.0)),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_allows_empty_dimension_placeholder() {
        let tensor = Tensor::new(vec![3.0, 1.0], vec![2, 1]).unwrap();
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let args = [Value::Tensor(placeholder), Value::from("reverse")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.data, vec![1.0, 1.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.data, vec![2.0, 2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_dimension_zero_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = [Value::Int(IntValue::I32(0))];
        match evaluate(Value::Tensor(tensor), &args) {
            Ok(_) => panic!("expected dimension error"),
            Err(err) => {
                assert!(err.message().contains("dimension must be >= 1"));
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_duplicate_direction_errors() {
        let tensor = Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap();
        let args = [Value::from("reverse"), Value::from("forward")];
        match evaluate(Value::Tensor(tensor), &args) {
            Ok(_) => panic!("expected duplicate direction error"),
            Err(err) => {
                assert!(err.message().contains("direction specified more than once"));
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_reverse_omitnan_combination() {
        let tensor =
            Tensor::new(vec![f64::NAN, 4.0, 2.0, f64::NAN, 3.0], vec![5, 1]).expect("tensor");
        let args = [Value::from("reverse"), Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.data, vec![2.0, 2.0, 2.0, 3.0, 3.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.data, vec![3.0, 3.0, 3.0, 5.0, 5.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_complex_vector() {
        let tensor =
            ComplexTensor::new(vec![(3.0, 0.0), (2.0, 0.0), (2.0, 1.0)], vec![3, 1]).unwrap();
        let eval = evaluate(Value::ComplexTensor(tensor), &[]).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::ComplexTensor(out) => {
                assert_eq!(out.data[0], (3.0, 0.0));
                assert_eq!(out.data[1], (2.0, 0.0));
                assert_eq!(out.data[2], (2.0, 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.data, vec![1.0, 2.0, 2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![4.0, 2.0, 7.0, 1.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("cummin");
            let (values, indices) = eval.into_pair();
            let gathered_values = test_support::gather(values).expect("gather values");
            let gathered_indices = test_support::gather(indices).expect("gather indices");
            assert_eq!(gathered_values.data, vec![4.0, 2.0, 2.0, 1.0]);
            assert_eq!(gathered_indices.data, vec![1.0, 2.0, 2.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_gpu_dimension_exceeds_rank_returns_indices() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![Value::Int(IntValue::I32(5))];
            let eval = evaluate(Value::GpuTensor(handle), &args).expect("cummin");
            let (values, indices) = eval.into_pair();
            let gathered_values = test_support::gather(values).expect("gather values");
            let gathered_indices = test_support::gather(indices).expect("gather indices");
            assert_eq!(gathered_values.data, tensor.data);
            assert!(gathered_indices.data.iter().all(|v| *v == 1.0));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cummin_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![4.0, 2.0, 7.0, 1.0, 5.0, 0.0], vec![3, 2]).unwrap();
        let cpu_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cummin cpu");
        let (cpu_vals, cpu_idx) = cpu_eval.into_pair();
        let expected_vals = match cpu_vals {
            Value::Tensor(t) => t,
            other => panic!("expected tensor values from cpu eval, got {other:?}"),
        };
        let expected_idx = match cpu_idx {
            Value::Tensor(t) => t,
            other => panic!("expected tensor indices from cpu eval, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("cummin gpu");
        let (gpu_vals, gpu_idx) = gpu_eval.into_pair();

        match (&gpu_vals, &gpu_idx) {
            (Value::GpuTensor(_), Value::GpuTensor(_)) => {}
            other => panic!("expected GPU tensors, got {other:?}"),
        }

        let gathered_vals = test_support::gather(gpu_vals).expect("gather values");
        let gathered_idx = test_support::gather(gpu_idx).expect("gather indices");

        assert_eq!(gathered_vals.shape, expected_vals.shape);
        assert_eq!(gathered_vals.data, expected_vals.data);
        assert_eq!(gathered_idx.shape, expected_idx.shape);
        assert_eq!(gathered_idx.data, expected_idx.data);
    }
}
