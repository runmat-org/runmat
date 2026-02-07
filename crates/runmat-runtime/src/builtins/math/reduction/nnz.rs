//! MATLAB-compatible `nnz` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    gpu_helpers,
    shape::{canonical_scalar_shape, is_scalar_shape, normalize_scalar_shape},
    tensor,
};
use crate::builtins::math::reduction::type_resolvers::count_nonzero_type;
use crate::dispatcher::download_handle_async;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    CharArray, ComplexTensor, LogicalArray, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

const NAME: &str = "nnz";

fn nnz_type(args: &[Type], ctx: &ResolveContext) -> Type {
    count_nonzero_type(args, ctx)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::nnz")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "nnz",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_nnz_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_nnz",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(512),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes: "Providers that implement reduce_nnz[_dim] keep counting on-device; the builtin downloads the MATLAB-compatible double result afterwards.",
};

fn nnz_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::nnz")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "nnz",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let zero = match ctx.scalar_ty {
                ScalarType::F64 => "f64(0.0)",
                _ => "0.0",
            };
            let one = match ctx.scalar_ty {
                ScalarType::F64 => "f64(1.0)",
                _ => "1.0",
            };
            Ok(format!(
                "if (isNan({input}) || {input} != {zero}) {{ accumulator += {one}; }}"
            ))
        },
    }),
    emits_nan: false,
    notes: "Fusion reductions treat NaN values as nonzero, mirroring MATLAB `nnz` semantics.",
};

#[runtime_builtin(
    name = "nnz",
    category = "math/reduction",
    summary = "Count the number of nonzero elements in an array with MATLAB-compatible semantics.",
    keywords = "nnz,nonzero,count,sparsity,gpu",
    accel = "reduction",
    type_resolver(nnz_type),
    builtin_path = "crate::builtins::math::reduction::nnz"
)]
async fn nnz_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let dim = parse_dimension_arg(&rest).await?;
    match value {
        Value::GpuTensor(handle) => nnz_gpu(handle, dim).await,
        other => nnz_host_value(other, dim),
    }
}

async fn parse_dimension_arg(args: &[Value]) -> BuiltinResult<Option<usize>> {
    match args.len() {
        0 => Ok(None),
        1 => {
            let dim = tensor::dimension_from_value_async(&args[0], "nnz", false)
                .await
                .map_err(nnz_error)?;
            match dim {
                Some(dim) => Ok(Some(dim)),
                None => Err(nnz_error(format!(
                    "nnz: dimension must be numeric, got {:?}",
                    args[0]
                ))),
            }
        }
        _ => Err(nnz_error("nnz: too many input arguments")),
    }
}

async fn nnz_gpu(handle: GpuTensorHandle, dim: Option<usize>) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider();
    match dim {
        None => {
            if let Some(p) = provider {
                if let Ok(result) = p.reduce_nnz(&handle).await {
                    let host = download_handle_async(p, &result)
                        .await
                        .map_err(|e| nnz_error(format!("nnz: {e}")))?;
                    let _ = p.free(&result);
                    let count = host.data.into_iter().next().unwrap_or(0.0);
                    return Ok(Value::Num(count));
                }
            }
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            nnz_host_value(Value::Tensor(tensor), None)
        }
        Some(dim) => {
            if let Some(p) = provider {
                let zero_based = dim.saturating_sub(1);
                if zero_based < handle.shape.len() {
                    if let Ok(result) = p.reduce_nnz_dim(&handle, zero_based).await {
                        let host = download_handle_async(p, &result)
                            .await
                            .map_err(|e| nnz_error(format!("nnz: {e}")))?;
                        let _ = p.free(&result);
                        let tensor = Tensor::new(host.data, host.shape)
                            .map_err(|e| nnz_error(format!("nnz: {e}")))?;
                        return Ok(tensor::tensor_into_value(tensor));
                    }
                }
            }
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            nnz_host_value(Value::Tensor(tensor), Some(dim))
        }
    }
}

fn nnz_host_value(value: Value, dim: Option<usize>) -> BuiltinResult<Value> {
    match dim {
        None => {
            let count = count_nonzero_value(&value)?;
            Ok(Value::Num(count as f64))
        }
        Some(dim) => {
            let mask = mask_from_value(&value)?;
            let tensor = reduce_mask_dim(&mask, dim)?;
            Ok(tensor::tensor_into_value(tensor))
        }
    }
}

fn count_nonzero_value(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) => Ok(count_nonzero_tensor(tensor)),
        Value::ComplexTensor(ct) => Ok(count_nonzero_complex_tensor(ct)),
        Value::LogicalArray(logical) => Ok(count_nonzero_logical(logical)),
        Value::CharArray(chars) => Ok(count_nonzero_char(chars)),
        Value::Num(n) => Ok(if is_nonzero_scalar(*n) { 1 } else { 0 }),
        Value::Int(i) => Ok(if i.to_i64() != 0 { 1 } else { 0 }),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Complex(re, im) => Ok(if is_nonzero_complex(*re, *im) { 1 } else { 0 }),
        Value::GpuTensor(_) => Err(nnz_error(
            "nnz: GPU inputs are handled before host evaluation",
        )),
        other => Err(nnz_error(format!(
            "nnz: expected numeric, logical, complex, or char array, got {}",
            describe_value_kind(other)
        ))),
    }
}

fn count_nonzero_tensor(tensor: &Tensor) -> usize {
    tensor
        .data
        .iter()
        .copied()
        .filter(|value| is_nonzero_scalar(*value))
        .count()
}

fn count_nonzero_complex_tensor(tensor: &ComplexTensor) -> usize {
    tensor
        .data
        .iter()
        .copied()
        .filter(|(re, im)| is_nonzero_complex(*re, *im))
        .count()
}

fn count_nonzero_logical(logical: &LogicalArray) -> usize {
    logical.data.iter().filter(|&&b| b != 0).count()
}

fn count_nonzero_char(chars: &CharArray) -> usize {
    chars.data.iter().filter(|&&ch| ch != '\0').count()
}

#[inline]
fn is_nonzero_scalar(value: f64) -> bool {
    value.is_nan() || value != 0.0
}

#[inline]
fn is_nonzero_complex(re: f64, im: f64) -> bool {
    re.is_nan() || im.is_nan() || re != 0.0 || im != 0.0
}

struct Mask {
    bits: Vec<u8>,
    shape: Vec<usize>,
}

fn mask_from_value(value: &Value) -> BuiltinResult<Mask> {
    match value {
        Value::Tensor(tensor) => {
            let shape = canonical_shape(&tensor.shape, tensor.data.len());
            let bits = tensor
                .data
                .iter()
                .map(|&v| if is_nonzero_scalar(v) { 1u8 } else { 0u8 })
                .collect();
            Ok(Mask { bits, shape })
        }
        Value::ComplexTensor(tensor) => {
            let shape = canonical_shape(&tensor.shape, tensor.data.len());
            let bits = tensor
                .data
                .iter()
                .map(|&(re, im)| if is_nonzero_complex(re, im) { 1u8 } else { 0u8 })
                .collect();
            Ok(Mask { bits, shape })
        }
        Value::LogicalArray(logical) => Ok(Mask {
            bits: logical
                .data
                .iter()
                .map(|&b| if b != 0 { 1 } else { 0 })
                .collect(),
            shape: canonical_shape(&logical.shape, logical.data.len()),
        }),
        Value::CharArray(chars) => {
            let bits = chars
                .data
                .iter()
                .map(|&ch| if ch != '\0' { 1u8 } else { 0u8 })
                .collect();
            Ok(Mask {
                bits,
                shape: vec![chars.rows, chars.cols],
            })
        }
        Value::Num(n) => Ok(Mask {
            bits: vec![if is_nonzero_scalar(*n) { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::Int(i) => Ok(Mask {
            bits: vec![if i.to_i64() != 0 { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::Bool(b) => Ok(Mask {
            bits: vec![if *b { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::Complex(re, im) => Ok(Mask {
            bits: vec![if is_nonzero_complex(*re, *im) { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::GpuTensor(_) => Err(nnz_error(
            "nnz: GPU inputs are handled before host evaluation",
        )),
        other => Err(nnz_error(format!(
            "nnz: expected numeric, logical, complex, or char array, got {}",
            describe_value_kind(other)
        ))),
    }
}

fn reduce_mask_dim(mask: &Mask, dim: usize) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(nnz_error("nnz: dimension must be >= 1"));
    }
    if mask.bits.is_empty() {
        let mut out_shape = canonical_shape(&mask.shape, 0);
        if dim <= out_shape.len() && !is_scalar_shape(&out_shape) {
            out_shape[dim - 1] = 1;
        }
        let out_len = out_shape.iter().copied().product::<usize>();
        let zeros = vec![0.0; out_len];
        return Tensor::new(zeros, out_shape).map_err(|e| nnz_error(format!("nnz: {e}")));
    }
    if is_scalar_shape(&mask.shape) {
        let data = vec![mask.bits[0] as f64];
        return Tensor::new(data, canonical_scalar_shape())
            .map_err(|e| nnz_error(format!("nnz: {e}")));
    }
    if dim > mask.shape.len() {
        return mask_to_tensor(mask);
    }
    let dim_index = dim - 1;
    let reduce_len = mask.shape[dim_index];
    let stride_before = if dim_index == 0 {
        1
    } else {
        mask.shape[..dim_index].iter().copied().product::<usize>()
    };
    let stride_after = if dim_index + 1 >= mask.shape.len() {
        1
    } else {
        mask.shape[dim_index + 1..]
            .iter()
            .copied()
            .product::<usize>()
    };
    let out_len = stride_before
        .checked_mul(stride_after)
        .ok_or_else(|| "nnz: dimension too large".to_string())?;
    let mut out_shape = mask.shape.clone();
    out_shape[dim_index] = 1;
    let mut output = vec![0f64; out_len];
    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut count = 0usize;
            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                if idx < mask.bits.len() && mask.bits[idx] != 0 {
                    count += 1;
                }
            }
            let out_idx = after * stride_before + before;
            if out_idx < output.len() {
                output[out_idx] = count as f64;
            }
        }
    }
    Tensor::new(output, out_shape).map_err(|e| nnz_error(format!("nnz: {e}")))
}

fn mask_to_tensor(mask: &Mask) -> BuiltinResult<Tensor> {
    let data = mask.bits.iter().map(|&b| b as f64).collect::<Vec<_>>();
    Tensor::new(data, canonical_shape(&mask.shape, mask.bits.len()))
        .map_err(|e| nnz_error(format!("nnz: {e}")))
}

fn canonical_shape(shape: &[usize], len: usize) -> Vec<usize> {
    if is_scalar_shape(shape) {
        if len <= 1 {
            return canonical_scalar_shape();
        }
        return vec![len, 1];
    }
    if tensor::element_count(shape) == len {
        return normalize_scalar_shape(shape);
    }
    shape.to_vec()
}

fn describe_value_kind(value: &Value) -> String {
    match value {
        Value::Int(_) => "integer scalar".to_string(),
        Value::Num(_) => "numeric scalar".to_string(),
        Value::Complex(_, _) => "complex scalar".to_string(),
        Value::Bool(_) => "logical scalar".to_string(),
        Value::LogicalArray(_) => "logical array".to_string(),
        Value::String(_) => "string scalar".to_string(),
        Value::StringArray(_) => "string array".to_string(),
        Value::CharArray(_) => "char array".to_string(),
        Value::Tensor(_) => "numeric tensor".to_string(),
        Value::ComplexTensor(_) => "complex tensor".to_string(),
        Value::Cell(_) => "cell array".to_string(),
        Value::Struct(_) => "struct".to_string(),
        Value::GpuTensor(_) => "GPU tensor".to_string(),
        Value::Object(obj) => format!("{} object", obj.class_name),
        Value::HandleObject(h) => format!("handle object ({})", h.class_name),
        Value::Listener(l) => format!("listener for {}", l.event_name),
        Value::FunctionHandle(_) | Value::Closure(_) => "function handle".to_string(),
        Value::ClassRef(_) => "class reference".to_string(),
        Value::MException(_) => "exception".to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray};

    #[test]
    fn nnz_type_returns_num() {
        assert_eq!(
            nnz_type(
                &[Type::Tensor { shape: None }],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Num
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_scalar_zero() {
        let result = nnz_host_value(Value::Num(0.0), None).expect("nnz");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_scalar_negative_zero_is_zero() {
        let result = nnz_host_value(Value::Num(-0.0), None).expect("nnz");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_scalar_nonzero() {
        let result = nnz_host_value(Value::Int(IntValue::I32(-5)), None).expect("nnz");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_tensor_counts_entries() {
        let tensor = Tensor::new(vec![1.0, 0.0, -3.0, f64::NAN], vec![2, 2]).unwrap();
        let result = nnz_host_value(Value::Tensor(tensor), None).expect("nnz");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_matrix_dimension_one() {
        let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let result = nnz_host_value(Value::Tensor(tensor), Some(1)).expect("nnz");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![1.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_matrix_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let result = nnz_host_value(Value::Tensor(tensor), Some(2)).expect("nnz");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![2.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_empty_matrix_dimension_returns_zero_counts() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = nnz_host_value(Value::Tensor(tensor), Some(1)).expect("nnz");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![0.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_scalar_with_dimension_returns_scalar() {
        let tensor = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let result = nnz_host_value(Value::Tensor(tensor), Some(1)).expect("nnz");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_three_dimensional_reduction_counts_per_slice() {
        let data = vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2, 2]).unwrap();
        let result = nnz_host_value(Value::Tensor(tensor), Some(3)).expect("nnz");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2, 1]);
                assert_eq!(out.data, vec![2.0, 0.0, 0.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_dimension_greater_than_ndims_returns_mask() {
        let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![2, 2]).unwrap();
        let result = nnz_host_value(Value::Tensor(tensor), Some(3)).expect("nnz");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![1.0, 0.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_empty_tensor_is_zero() {
        let tensor = Tensor::new(vec![], vec![0, 3]).unwrap();
        let result = nnz_host_value(Value::Tensor(tensor), None).expect("nnz");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_complex_tensor_counts() {
        let data = vec![(0.0, 0.0), (2.0, 0.0), (0.0, -4.0), (0.0, 0.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = nnz_host_value(Value::ComplexTensor(tensor), None).expect("nnz");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_logical_array_counts_true() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![4]).unwrap();
        let result = nnz_host_value(Value::LogicalArray(logical), None).expect("nnz");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_char_array_counts_nonzero_codepoints() {
        let chars = CharArray::new(vec!['a', '\0', 'c'], 1, 3).unwrap();
        let result = nnz_host_value(Value::CharArray(chars), None).expect("nnz");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_complex_scalar_nan_counts() {
        let result = nnz_host_value(Value::Complex(f64::NAN, 0.0), None).expect("nnz");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 0.0, 3.0], vec![5, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = futures::executor::block_on(nnz_gpu(handle, None)).expect("nnz");
            assert_eq!(result, Value::Num(3.0));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn nnz_wgpu_dim_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let cpu = nnz_host_value(Value::Tensor(tensor.clone()), Some(1)).expect("nnz");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu = futures::executor::block_on(nnz_gpu(handle, Some(1))).expect("nnz");
        match (cpu, gpu) {
            (Value::Tensor(ct), Value::Tensor(gt)) => {
                assert_eq!(gt.shape, ct.shape);
                assert_eq!(gt.data, ct.data);
            }
            other => panic!("unexpected comparison result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nnz_rejects_strings() {
        let err = nnz_host_value(Value::from("hello"), None).unwrap_err();
        assert!(
            err.message()
                .contains("expected numeric, logical, complex, or char array"),
            "unexpected error: {err}"
        );
        assert!(
            err.message().contains("string scalar"),
            "unexpected error: {err}"
        );
    }
}
