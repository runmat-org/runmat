//! MATLAB-compatible `zeros` builtin with GPU-aware semantics.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderPrecision};
use runmat_builtins::{ComplexTensor, LogicalArray, Value};
use runmat_macros::runtime_builtin;
use std::sync::OnceLock;

use crate::build_runtime_error;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionExprContext,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{shape::normalize_scalar_shape, tensor};
use runmat_builtins::NumericDType;
use runmat_builtins::Type;

use crate::builtins::array::type_resolvers::tensor_type_from_rank;
use runmat_builtins::ResolveContext;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::zeros")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "zeros",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("zeros"),
        ProviderHook::Custom("zeros_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Allocates device zeros when providers expose dedicated hooks; otherwise falls back to host upload.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("zeros").build()
}

fn zeros_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Num;
    }
    if args.iter().any(|arg| matches!(arg, Type::String)) {
        return Type::Unknown;
    }
    tensor_type_from_rank(args, ctx)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::zeros")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "zeros",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                ScalarType::I32 => "0".to_string(),
                ScalarType::Bool => "false".to_string(),
            };
            Ok(zero)
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner materialises zeros as literal constants; providers may substitute inexpensive fill kernels.",
};

#[runtime_builtin(
    name = "zeros",
    category = "array/creation",
    summary = "Create arrays filled with zeros.",
    keywords = "zeros,array,logical,gpu,like",
    accel = "array_construct",
    type_resolver(zeros_type),
    type_resolver_context = true,
    builtin_path = "crate::builtins::array::creation::zeros"
)]
async fn zeros_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedZeros::parse(rest).await?;
    build_output(parsed).await
}

struct ParsedZeros {
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    /// Single-precision request. Host tensors are stored as f64 today; we
    /// treat 'single' as a request for a numeric zeros tensor and honour
    /// single precision when allocating on GPU via 'like' or provider hooks.
    Single,
    Logical,
    Like(Value),
}

impl ParsedZeros {
    async fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<OutputTemplate> = None;
        let mut implicit_proto: Option<Value> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err(builtin_error(
                                "zeros: multiple 'like' specifications are not supported",
                            ));
                        }
                        if class_override.is_some() {
                            return Err(builtin_error(
                                "zeros: cannot combine 'like' with other class specifiers",
                            ));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error("zeros: expected prototype after 'like'"));
                        };
                        like_proto = Some(proto.clone());
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source = Some(shape_from_value(&proto)?);
                        }
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err(builtin_error(
                                "zeros: cannot combine 'like' with 'logical'",
                            ));
                        }
                        class_override = Some(OutputTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err(builtin_error(
                                "zeros: cannot combine 'like' with 'double'",
                            ));
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        if like_proto.is_some() {
                            return Err(builtin_error(
                                "zeros: cannot combine 'like' with 'single'",
                            ));
                        }
                        class_override = Some(OutputTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(builtin_error(format!(
                            "zeros: unrecognised option '{other}'"
                        )));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg).await? {
                tracing::trace!("zeros: parsed dimension arguments {:?}", parsed_dims);
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            tracing::debug!(
                arg_type = value_tag(&arg),
                "zeros: argument did not parse as dimensions"
            );

            if shape_source.is_none() {
                shape_source = Some(shape_from_value(&arg)?);
            }
            if implicit_proto.is_none() {
                implicit_proto = Some(arg.clone());
            }
            idx += 1;
        }

        let shape = if saw_dims_arg {
            if dims.is_empty() {
                vec![0, 0]
            } else if dims.len() == 1 {
                vec![dims[0], dims[0]]
            } else {
                dims
            }
        } else if let Some(shape) = shape_source {
            tracing::warn!(
                shape = ?shape,
                "zeros: falling back to shape source; no dimension arguments parsed"
            );
            shape
        } else {
            vec![1, 1]
        };

        tracing::trace!(
            "zeros: resolved output shape {:?} (saw_dims_arg={})",
            shape,
            saw_dims_arg
        );

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
        } else if let Some(proto) = implicit_proto {
            OutputTemplate::Like(proto)
        } else {
            OutputTemplate::Double
        };

        Ok(Self { shape, template })
    }
}

async fn build_output(parsed: ParsedZeros) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => zeros_double(&parsed.shape),
        OutputTemplate::Single => zeros_single(&parsed.shape),
        OutputTemplate::Logical => zeros_logical(&parsed.shape),
        OutputTemplate::Like(proto) => zeros_like(&proto, &parsed.shape).await,
    }
}

fn value_tag(value: &Value) -> &'static str {
    match value {
        Value::Num(_) => "Num",
        Value::Int(_) => "Int",
        Value::Bool(_) => "Bool",
        Value::Tensor(_) => "Tensor",
        Value::LogicalArray(_) => "LogicalArray",
        Value::GpuTensor(_) => "GpuTensor",
        Value::Complex(_, _) => "Complex",
        Value::ComplexTensor(_) => "ComplexTensor",
        Value::String(_) => "String",
        Value::StringArray(_) => "StringArray",
        Value::CharArray(_) => "CharArray",
        Value::Cell(_) => "Cell",
        Value::Struct(_) => "Struct",
        Value::Object(_) => "Object",
        Value::HandleObject(_) => "HandleObject",
        Value::Listener(_) => "Listener",
        Value::FunctionHandle(_) => "FunctionHandle",
        Value::Closure(_) => "Closure",
        Value::ClassRef(_) => "ClassRef",
        Value::MException(_) => "MException",
    }
}

fn zeros_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = zeros_gpu_alloc(shape, NumericDType::F64)? {
            return Ok(value);
        }
    }
    let tensor = tensor::zeros(shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn zeros_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = zeros_gpu_alloc(shape, NumericDType::F32)? {
            return Ok(value);
        }
    }
    let tensor = tensor::zeros_with_dtype(shape, NumericDType::F32)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn force_host_allocation(shape: &[usize]) -> bool {
    tensor::element_count(shape) <= 1
}

fn zeros_logical(shape: &[usize]) -> crate::BuiltinResult<Value> {
    Ok(Value::LogicalArray(LogicalArray::zeros(shape.to_vec())))
}

#[async_recursion::async_recursion(?Send)]
async fn zeros_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => zeros_logical(shape),
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            let tensor = ComplexTensor::zeros(shape.to_vec());
            Ok(Value::ComplexTensor(tensor))
        }
        Value::GpuTensor(handle) => zeros_like_gpu(handle, shape).await,
        Value::Tensor(t) => match t.dtype {
            NumericDType::F32 => zeros_single(shape),
            NumericDType::F64 => zeros_double(shape),
        },
        Value::Num(_) | Value::Int(_) => zeros_double(shape),
        Value::CharArray(_) | Value::Cell(_) => zeros_double(shape),
        _ => zeros_double(shape),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn zeros_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
        let dtype = dtype_from_precision(precision);
        let attempt = if handle.shape == shape {
            provider.zeros_like(handle)
        } else {
            provider.zeros(shape)
        };
        if let Ok(gpu) = attempt {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        } else {
            log_zeros_fallback(shape, dtype, "provider-like-error");
        }
        // Fallback: build a host tensor with dtype matching provider precision and upload
        let host = tensor::zeros_with_dtype(shape, dtype)?;
        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        } else {
            log_zeros_fallback(shape, dtype, "upload-error");
        }
    } else {
        log_zeros_fallback(shape, NumericDType::F32, "no-provider-like");
    }

    let gathered = crate::dispatcher::gather_if_needed_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(|e| format!("zeros: {e}"))?;
    log_zeros_fallback(shape, NumericDType::F32, "gather-fallback");
    zeros_like(&gathered, shape).await
}

fn zeros_gpu_alloc(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        log_zeros_fallback(shape, dtype, "no-provider");
        return Ok(None);
    };
    let precision = match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
    };
    if provider.precision() != precision {
        log_zeros_fallback(shape, dtype, "precision-mismatch");
        return Ok(None);
    }
    match provider.zeros(shape) {
        Ok(handle) => {
            runmat_accelerate_api::set_handle_precision(&handle, precision);
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(err) => {
            log::warn!("zeros: provider zeros failed ({err}); falling back to host tensor path");
            log_zeros_fallback(shape, dtype, "provider-error");
            Ok(None)
        }
    }
}

fn zeros_fallback_debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("RUNMAT_DEBUG_ZEROS_FALLBACK"),
            Ok(value)
                if value == "1"
                    || value.eq_ignore_ascii_case("true")
                    || value.eq_ignore_ascii_case("yes")
        )
    })
}

fn log_zeros_fallback(shape: &[usize], dtype: NumericDType, reason: &str) {
    if !zeros_fallback_debug_enabled() {
        return;
    }
    let elems = tensor::element_count(shape);
    tracing::debug!(
        dtype = ?dtype,
        elems,
        shape = ?shape,
        reason,
        "[zeros_debug] fallback"
    );
}

fn dtype_from_precision(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            Some(text.to_ascii_lowercase())
        }
        _ => None,
    }
}

async fn extract_dims(value: &Value) -> crate::BuiltinResult<Option<Vec<usize>>> {
    if matches!(value, Value::LogicalArray(_)) {
        return Ok(None);
    }
    let gpu_scalar = match value {
        Value::GpuTensor(handle) => tensor::element_count(&handle.shape) == 1,
        _ => false,
    };
    match tensor::dims_from_value_async(value).await {
        Ok(dims) => Ok(dims),
        Err(err) => {
            if matches!(value, Value::Tensor(_))
                || (matches!(value, Value::GpuTensor(_)) && !gpu_scalar)
            {
                Ok(None)
            } else {
                Err(builtin_error(format!("zeros: {err}")))
            }
        }
    }
}

fn shape_from_value(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::ComplexTensor(t) => Ok(t.shape.clone()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(normalize_scalar_shape(&h.shape)),
        Value::CharArray(ca) => Ok(vec![ca.rows, ca.cols]),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(format!("zeros: unsupported prototype {other:?}")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_default_scalar() {
        let result = block_on(zeros_builtin(Vec::new())).expect("zeros");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn zeros_type_defaults_to_num() {
        assert_eq!(zeros_type(&[], &ResolveContext::new(Vec::new())), Type::Num);
    }

    #[test]
    fn zeros_type_infers_rank_from_scalar_dim() {
        assert_eq!(
            zeros_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[test]
    fn zeros_type_infers_rank_from_size_vector() {
        let size_vec = Type::Tensor {
            shape: Some(vec![Some(1), Some(3)]),
        };
        assert_eq!(
            zeros_type(&[size_vec], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![None, None, None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_square_from_single_dimension() {
        let args = vec![Value::Num(3.0)];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![3, 3]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_rectangular_from_dims() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_eq!(tensor.data.len(), 8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_from_size_vector() {
        let size_vec = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let args = vec![Value::Tensor(size_vec)];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_logical_output() {
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("logical")];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 2]);
                assert!(logical.data.iter().all(|&x| x == 0));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_tensor_infers_shape() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_complex_scalar() {
        let args = vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert!(t.data.iter().all(|&(re, im)| re == 0.0 && im == 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_uses_shape_argument_when_combined_with_like() {
        let shape_source = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let proto = Tensor::new(vec![7.0, 8.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Tensor(shape_source.clone()),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 3]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_without_explicit_shape_uses_prototype_shape() {
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(proto)];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_empty_input_returns_empty_matrix() {
        let empty = Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let result = block_on(zeros_builtin(vec![Value::Tensor(empty)])).expect("zeros");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_conflicting_like_and_logical_is_error() {
        let proto = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Num(2.0),
            Value::from("logical"),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        assert!(block_on(zeros_builtin(args)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_gpu_like_alloc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(zeros_builtin(args)).expect("zeros");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered.data.iter().all(|&x| x == 0.0));
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn zeros_wgpu_single_allocates_gpu_without_like() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let value = zeros_single(&[2, 2]).expect("zeros single");
        match value {
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle)).expect("gather to host");
                assert_eq!(gathered.shape, vec![2, 2]);
                assert!(gathered.data.iter().all(|&x| x == 0.0));
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }
}
