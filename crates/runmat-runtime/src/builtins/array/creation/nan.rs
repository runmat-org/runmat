//! MATLAB-compatible `nan` array constructor with GPU-aware semantics.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, HostTensorView, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericDType, SparseTensor, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::array::type_resolvers::tensor_type_from_rank;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionExprContext,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, shape::normalize_scalar_shape, tensor};
use runmat_builtins::ResolveContext;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::nan")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "nan",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("fill"), ProviderHook::Custom("fill_like")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Allocates provider-resident NaN-filled arrays through constant-fill hooks when profitable; otherwise falls back to host tensors.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("nan").build()
}

fn nan_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    nan_error_with_message(error.message, error)
}

fn nan_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    nan_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn nan_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("nan");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn nan_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Num;
    }
    if args.iter().any(|arg| matches!(arg, Type::String)) {
        return Type::Unknown;
    }
    tensor_type_from_rank(args, ctx)
}

const NAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "NaN-filled output array.",
}];

const NAN_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const NAN_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const NAN_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const NAN_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const NAN_SIG_PROTOTYPE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prototype",
    ty: BuiltinParamType::LikePrototype,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Prototype value when no numeric dimension arguments are provided.",
}];

const NAN_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Class name override (double|single|gpuArray).",
    },
];

const NAN_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype array used for class/device.",
    },
];

const NAN_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "A = nan()",
        inputs: &NAN_SIG_EMPTY_INPUTS,
        outputs: &NAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = nan(n)",
        inputs: &NAN_SIG_N_INPUTS,
        outputs: &NAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = nan(size_vector)",
        inputs: &NAN_SIG_SIZE_VECTOR_INPUTS,
        outputs: &NAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = nan(m, n, ...)",
        inputs: &NAN_SIG_DIMS_INPUTS,
        outputs: &NAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = nan(prototype)",
        inputs: &NAN_SIG_PROTOTYPE_INPUTS,
        outputs: &NAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = nan(..., typename)",
        inputs: &NAN_SIG_CLASS_INPUTS,
        outputs: &NAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = nan(..., \"like\", prototype)",
        inputs: &NAN_SIG_LIKE_INPUTS,
        outputs: &NAN_OUTPUT,
    },
];

const NAN_ERROR_LIKE_EXPECTED_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAN.LIKE_EXPECTED_PROTOTYPE",
    identifier: None,
    when: "The 'like' keyword is provided without a prototype argument.",
    message: "nan: expected prototype after 'like'",
};

const NAN_ERROR_CLASS_CONFLICT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAN.CLASS_CONFLICT",
    identifier: None,
    when: "A class keyword and a 'like' prototype are both provided.",
    message: "nan: cannot combine 'like' with other class specifiers",
};

const NAN_ERROR_UNRECOGNIZED_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAN.UNRECOGNIZED_OPTION",
    identifier: None,
    when: "A trailing option string is not a supported class keyword.",
    message: "nan: unrecognised option",
};

const NAN_ERROR_LIKE_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAN.LIKE_DUPLICATE",
    identifier: None,
    when: "The 'like' keyword is specified more than once.",
    message: "nan: multiple 'like' specifications are not supported",
};

const NAN_ERRORS: [BuiltinErrorDescriptor; 4] = [
    NAN_ERROR_LIKE_EXPECTED_PROTOTYPE,
    NAN_ERROR_CLASS_CONFLICT,
    NAN_ERROR_UNRECOGNIZED_OPTION,
    NAN_ERROR_LIKE_DUPLICATE,
];

pub const NAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NAN_ERRORS,
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::nan")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "nan",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let literal = match ctx.scalar_ty {
                ScalarType::F32 => "bitcast<f32>(0x7fc00000u)".to_string(),
                ScalarType::F64 => "bitcast<f64>(0x7ff8000000000000u)".to_string(),
                ScalarType::I32 | ScalarType::Bool => {
                    return Err(crate::builtins::common::spec::FusionError::Message(
                        "nan: integer and logical fusion output is unsupported",
                    ));
                }
            };
            Ok(literal)
        },
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion planner materialises NaN constructors as canonical IEEE NaN literals.",
};

#[runtime_builtin(
    name = "nan",
    category = "array/creation",
    summary = "Create arrays filled with NaN values.",
    keywords = "nan,array,single,gpu,like",
    accel = "array_construct",
    type_resolver(nan_type),
    descriptor(crate::builtins::array::creation::nan::NAN_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::nan"
)]
async fn nan_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedNan::parse(rest).await?;
    build_output(parsed).await
}

struct ParsedNan {
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Single,
    GpuArray,
    Like(Value),
}

impl ParsedNan {
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
                            return Err(nan_error(&NAN_ERROR_LIKE_DUPLICATE));
                        }
                        if class_override.is_some() {
                            return Err(nan_error(&NAN_ERROR_CLASS_CONFLICT));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(nan_error(&NAN_ERROR_LIKE_EXPECTED_PROTOTYPE));
                        };
                        like_proto = Some(proto.clone());
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source = Some(shape_from_value(&proto)?);
                        }
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err(nan_error_with_detail(
                                &NAN_ERROR_CLASS_CONFLICT,
                                "double class override",
                            ));
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        if like_proto.is_some() {
                            return Err(nan_error_with_detail(
                                &NAN_ERROR_CLASS_CONFLICT,
                                "single class override",
                            ));
                        }
                        class_override = Some(OutputTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    "gpuArray" | "gpuarray" => {
                        if like_proto.is_some() {
                            return Err(nan_error_with_detail(
                                &NAN_ERROR_CLASS_CONFLICT,
                                "gpuArray class override",
                            ));
                        }
                        class_override = Some(OutputTemplate::GpuArray);
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(nan_error_with_detail(
                            &NAN_ERROR_UNRECOGNIZED_OPTION,
                            format!("'{other}'"),
                        ));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg).await? {
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

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
            shape
        } else {
            vec![1, 1]
        };

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

async fn build_output(parsed: ParsedNan) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => nan_double(&parsed.shape),
        OutputTemplate::Single => nan_single(&parsed.shape),
        OutputTemplate::GpuArray => nan_gpu(&parsed.shape).await,
        OutputTemplate::Like(proto) => nan_like(&proto, &parsed.shape).await,
    }
}

fn nan_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = nan_gpu_alloc(shape, NumericDType::F64)? {
            return Ok(value);
        }
    }
    nan_tensor(shape, NumericDType::F64).map(tensor::tensor_into_value)
}

fn nan_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = nan_gpu_alloc(shape, NumericDType::F32)? {
            return Ok(value);
        }
    }
    nan_tensor(shape, NumericDType::F32).map(tensor::tensor_into_value)
}

fn force_host_allocation(shape: &[usize]) -> bool {
    tensor::element_count(shape) <= 1
}

async fn nan_gpu(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let precision = provider.precision();
        match provider.fill(shape, f64::NAN) {
            Ok(handle) => {
                runmat_accelerate_api::set_handle_precision(&handle, precision);
                return Ok(Value::GpuTensor(handle));
            }
            Err(err) => {
                log::debug!("nan_gpu: provider.fill failed ({err}); falling back to host upload");
            }
        }
        let host = nan_tensor(shape, dtype_from_precision(precision))?;
        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        }
    }
    nan_double(shape)
}

#[async_recursion::async_recursion(?Send)]
async fn nan_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::ComplexTensor(_) | Value::Complex(_, _) => nan_complex(shape),
        Value::GpuTensor(handle) => nan_like_gpu(handle, shape).await,
        Value::Tensor(t) => match t.dtype {
            NumericDType::F32 => nan_single(shape),
            NumericDType::F64 | NumericDType::U8 | NumericDType::U16 => nan_double(shape),
        },
        Value::SparseTensor(_) => nan_double(shape),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => nan_double(shape),
        Value::LogicalArray(_) => nan_double(shape),
        Value::CharArray(_) | Value::Cell(_) => nan_double(shape),
        _ => nan_double(shape),
    }
}

fn nan_complex(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let data = vec![(f64::NAN, 0.0); len];
    ComplexTensor::new(data, shape.to_vec())
        .map(complex_tensor_into_value)
        .map_err(|e| builtin_error(format!("nan: {e}")))
}

#[async_recursion::async_recursion(?Send)]
async fn nan_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(provider) =
        runmat_accelerate_api::provider_for_handle(handle).or_else(runmat_accelerate_api::provider)
    {
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
        let storage = runmat_accelerate_api::handle_storage(handle);
        if handle.shape != shape && storage == GpuTensorStorage::ComplexInterleaved {
            let len = tensor::element_count(shape);
            let tensor = ComplexTensor::new(vec![(f64::NAN, 0.0); len], shape.to_vec())
                .map_err(|e| builtin_error(format!("nan: {e}")))?;
            match gpu_helpers::upload_complex_tensor(provider, &tensor) {
                Ok(gpu) => {
                    runmat_accelerate_api::set_handle_precision(&gpu, precision);
                    return Ok(Value::GpuTensor(gpu));
                }
                Err(_) => return Ok(complex_tensor_into_value(tensor)),
            }
        }
        let attempt = if handle.shape == shape {
            provider.fill_like(handle, f64::NAN)
        } else {
            provider.fill(shape, f64::NAN)
        };
        if let Ok(gpu) = attempt {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        }

        let host = nan_tensor(shape, dtype_from_precision(precision))?;
        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        }
    }

    let gathered = crate::dispatcher::gather_if_needed_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(|e| builtin_error(format!("nan: {e}")))?;
    nan_like(&gathered, shape).await
}

fn nan_gpu_alloc(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    let precision = match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
        NumericDType::U8 | NumericDType::U16 => return Ok(None),
    };
    if provider.precision() != precision {
        return Ok(None);
    }
    match provider.fill(shape, f64::NAN) {
        Ok(handle) => {
            runmat_accelerate_api::set_handle_precision(&handle, precision);
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(err) => {
            log::warn!("nan: provider fill failed ({err}); falling back to host tensor path");
            Ok(None)
        }
    }
}

fn nan_tensor(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Tensor> {
    Tensor::new_with_dtype(
        vec![f64::NAN; tensor::element_count(shape)],
        shape.to_vec(),
        dtype,
    )
    .map_err(|e| builtin_error(format!("nan: {e}")))
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
                Err(builtin_error(format!("nan: {err}")))
            }
        }
    }
}

fn shape_from_value(value: &Value) -> crate::BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::SparseTensor(SparseTensor { rows, cols, .. }) => Ok(vec![*rows, *cols]),
        Value::ComplexTensor(t) => Ok(t.shape.clone()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(normalize_scalar_shape(&h.shape)),
        Value::CharArray(ca) => Ok(vec![ca.rows, ca.cols]),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(builtin_error(format!(
            "nan: unsupported prototype {other:?}"
        ))),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn clear_accel_provider_state() -> test_support::AccelTestGuard {
        test_support::accel_test_lock()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_default_scalar() {
        let _guard = clear_accel_provider_state();
        let result = block_on(nan_builtin(Vec::new())).expect("nan");
        match result {
            Value::Num(value) => assert!(value.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[test]
    fn nan_type_defaults_to_num() {
        assert_eq!(nan_type(&[], &ResolveContext::new(Vec::new())), Type::Num);
    }

    #[test]
    fn nan_type_infers_rank_from_scalar_dim() {
        assert_eq!(
            nan_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_square_from_single_dimension() {
        let _guard = clear_accel_provider_state();
        let result = block_on(nan_builtin(vec![Value::Num(3.0)])).expect("nan");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![3, 3]);
        assert!(tensor.data.iter().all(|value| value.is_nan()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_rectangular_from_dims() {
        let _guard = clear_accel_provider_state();
        let result = block_on(nan_builtin(vec![Value::Num(2.0), Value::Num(4.0)])).expect("nan");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 4]);
        assert!(tensor.data.iter().all(|value| value.is_nan()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_from_size_vector() {
        let _guard = clear_accel_provider_state();
        let size_vec = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let result = block_on(nan_builtin(vec![Value::Tensor(size_vec)])).expect("nan");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 3, 4]);
        assert!(tensor.data.iter().all(|value| value.is_nan()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_single_output_marks_dtype() {
        let _guard = clear_accel_provider_state();
        let result = block_on(nan_builtin(vec![
            Value::Num(2.0),
            Value::Num(2.0),
            Value::from("single"),
        ]))
        .expect("nan");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.dtype, NumericDType::F32);
        assert!(tensor.data.iter().all(|value| value.is_nan()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_like_tensor_infers_shape() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = block_on(nan_builtin(vec![Value::Tensor(proto)])).expect("nan");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert!(tensor.data.iter().all(|value| value.is_nan()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_like_complex_scalar() {
        let _guard = clear_accel_provider_state();
        let result = block_on(nan_builtin(vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ]))
        .expect("nan");
        match result {
            Value::ComplexTensor(tensor) => {
                assert_eq!(tensor.shape, vec![3, 3]);
                assert!(tensor.data.iter().all(|(re, im)| re.is_nan() && *im == 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_like_uses_shape_argument_when_combined_with_like() {
        let _guard = clear_accel_provider_state();
        let shape_source = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let proto = Tensor::new_with_dtype(vec![7.0, 8.0], vec![1, 2], NumericDType::F32).unwrap();
        let result = block_on(nan_builtin(vec![
            Value::Tensor(shape_source),
            Value::from("like"),
            Value::Tensor(proto),
        ]))
        .expect("nan");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.dtype, NumericDType::F32);
        assert!(tensor.data.iter().all(|value| value.is_nan()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_like_without_explicit_shape_uses_prototype_shape() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result =
            block_on(nan_builtin(vec![Value::from("like"), Value::Tensor(proto)])).expect("nan");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert!(tensor.data.iter().all(|value| value.is_nan()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_conflicting_like_and_class_is_error() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Num(2.0),
            Value::from("single"),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        assert!(block_on(nan_builtin(args)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_gpu_like_alloc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = block_on(nan_builtin(vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ]))
            .expect("nan");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered.data.iter().all(|value| value.is_nan()));
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }
}
