//! MATLAB-compatible `inf` array constructor with GPU-aware semantics.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, HostTensorView, ProviderPrecision};
use runmat_builtins::ResolveContext;
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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::inf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "inf",
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
    notes: "Allocates provider-resident Inf-filled arrays through constant-fill hooks when profitable; otherwise falls back to host tensors.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("inf").build()
}

fn inf_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    inf_error_with_message(error.message, error)
}

fn inf_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    inf_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn inf_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("inf");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn inf_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Num;
    }
    if args.iter().any(|arg| matches!(arg, Type::String)) {
        return Type::Unknown;
    }
    tensor_type_from_rank(args, ctx)
}

const INF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Inf-filled output array.",
}];

const INF_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const INF_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const INF_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const INF_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const INF_SIG_PROTOTYPE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prototype",
    ty: BuiltinParamType::LikePrototype,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Prototype value when no numeric dimension arguments are provided.",
}];

const INF_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const INF_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
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

const INF_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "A = inf()",
        inputs: &INF_SIG_EMPTY_INPUTS,
        outputs: &INF_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = inf(n)",
        inputs: &INF_SIG_N_INPUTS,
        outputs: &INF_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = inf(size_vector)",
        inputs: &INF_SIG_SIZE_VECTOR_INPUTS,
        outputs: &INF_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = inf(m, n, ...)",
        inputs: &INF_SIG_DIMS_INPUTS,
        outputs: &INF_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = inf(prototype)",
        inputs: &INF_SIG_PROTOTYPE_INPUTS,
        outputs: &INF_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = inf(..., typename)",
        inputs: &INF_SIG_CLASS_INPUTS,
        outputs: &INF_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = inf(..., \"like\", prototype)",
        inputs: &INF_SIG_LIKE_INPUTS,
        outputs: &INF_OUTPUT,
    },
];

const INF_ERROR_LIKE_EXPECTED_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INF.LIKE_EXPECTED_PROTOTYPE",
    identifier: None,
    when: "The 'like' keyword is provided without a prototype argument.",
    message: "inf: expected prototype after 'like'",
};

const INF_ERROR_CLASS_CONFLICT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INF.CLASS_CONFLICT",
    identifier: None,
    when: "A class keyword and a 'like' prototype are both provided.",
    message: "inf: cannot combine 'like' with other class specifiers",
};

const INF_ERROR_UNRECOGNIZED_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INF.UNRECOGNIZED_OPTION",
    identifier: None,
    when: "A trailing option string is not a supported class keyword.",
    message: "inf: unrecognised option",
};

const INF_ERROR_LIKE_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INF.LIKE_DUPLICATE",
    identifier: None,
    when: "The 'like' keyword is specified more than once.",
    message: "inf: multiple 'like' specifications are not supported",
};

const INF_ERRORS: [BuiltinErrorDescriptor; 4] = [
    INF_ERROR_LIKE_EXPECTED_PROTOTYPE,
    INF_ERROR_CLASS_CONFLICT,
    INF_ERROR_UNRECOGNIZED_OPTION,
    INF_ERROR_LIKE_DUPLICATE,
];

pub const INF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INF_ERRORS,
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::inf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "inf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let literal = match ctx.scalar_ty {
                ScalarType::F32 => "bitcast<f32>(0x7f800000u)".to_string(),
                ScalarType::F64 => "bitcast<f64>(0x7ff0000000000000u)".to_string(),
                ScalarType::I32 | ScalarType::Bool => {
                    return Err(crate::builtins::common::spec::FusionError::Message(
                        "inf: integer and logical fusion output is unsupported",
                    ));
                }
            };
            Ok(literal)
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion planner materialises Inf constructors as canonical IEEE positive-infinity literals.",
};

#[runtime_builtin(
    name = "inf",
    category = "array/creation",
    summary = "Create arrays filled with positive infinity values.",
    keywords = "inf,infinity,array,single,gpu,like",
    accel = "array_construct",
    type_resolver(inf_type),
    descriptor(crate::builtins::array::creation::inf::INF_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::inf"
)]
async fn inf_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedInf::parse(rest).await?;
    build_output(parsed).await
}

struct ParsedInf {
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

impl ParsedInf {
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
                            return Err(inf_error(&INF_ERROR_LIKE_DUPLICATE));
                        }
                        if class_override.is_some() {
                            return Err(inf_error(&INF_ERROR_CLASS_CONFLICT));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(inf_error(&INF_ERROR_LIKE_EXPECTED_PROTOTYPE));
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
                            return Err(inf_error_with_detail(
                                &INF_ERROR_CLASS_CONFLICT,
                                "double class override",
                            ));
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        if like_proto.is_some() {
                            return Err(inf_error_with_detail(
                                &INF_ERROR_CLASS_CONFLICT,
                                "single class override",
                            ));
                        }
                        class_override = Some(OutputTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    "gpuArray" | "gpuarray" => {
                        if like_proto.is_some() {
                            return Err(inf_error_with_detail(
                                &INF_ERROR_CLASS_CONFLICT,
                                "gpuArray class override",
                            ));
                        }
                        class_override = Some(OutputTemplate::GpuArray);
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(inf_error_with_detail(
                            &INF_ERROR_UNRECOGNIZED_OPTION,
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

async fn build_output(parsed: ParsedInf) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => inf_double(&parsed.shape),
        OutputTemplate::Single => inf_single(&parsed.shape),
        OutputTemplate::GpuArray => inf_gpu(&parsed.shape).await,
        OutputTemplate::Like(proto) => inf_like(&proto, &parsed.shape).await,
    }
}

fn inf_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = inf_gpu_alloc(shape, NumericDType::F64)? {
            return Ok(value);
        }
    }
    inf_tensor(shape, NumericDType::F64).map(tensor::tensor_into_value)
}

fn inf_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = inf_gpu_alloc(shape, NumericDType::F32)? {
            return Ok(value);
        }
    }
    inf_tensor(shape, NumericDType::F32).map(tensor::tensor_into_value)
}

fn force_host_allocation(shape: &[usize]) -> bool {
    tensor::element_count(shape) <= 1
}

async fn inf_gpu(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let precision = provider.precision();
        match provider.fill(shape, f64::INFINITY) {
            Ok(handle) => {
                runmat_accelerate_api::set_handle_precision(&handle, precision);
                return Ok(Value::GpuTensor(handle));
            }
            Err(err) => {
                log::debug!("inf_gpu: provider.fill failed ({err}); falling back to host upload");
            }
        }
        let host = inf_tensor(shape, dtype_from_precision(precision))?;
        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        }
    }
    inf_double(shape)
}

#[async_recursion::async_recursion(?Send)]
async fn inf_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::ComplexTensor(_) | Value::Complex(_, _) => inf_complex(shape),
        Value::GpuTensor(handle) => inf_like_gpu(handle, shape).await,
        Value::Tensor(t) => match t.dtype {
            NumericDType::F32 => inf_single(shape),
            NumericDType::F64 | NumericDType::U8 | NumericDType::U16 => inf_double(shape),
        },
        Value::SparseTensor(_) => inf_double(shape),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => inf_double(shape),
        Value::LogicalArray(_) => inf_double(shape),
        Value::CharArray(_) | Value::Cell(_) => inf_double(shape),
        _ => inf_double(shape),
    }
}

fn inf_complex(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let data = vec![(f64::INFINITY, 0.0); len];
    ComplexTensor::new(data, shape.to_vec())
        .map(complex_tensor_into_value)
        .map_err(|e| builtin_error(format!("inf: {e}")))
}

#[async_recursion::async_recursion(?Send)]
async fn inf_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(provider) =
        runmat_accelerate_api::provider_for_handle(handle).or_else(runmat_accelerate_api::provider)
    {
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
        let storage = runmat_accelerate_api::handle_storage(handle);
        if handle.shape != shape && storage == GpuTensorStorage::ComplexInterleaved {
            let len = tensor::element_count(shape);
            let tensor = ComplexTensor::new(vec![(f64::INFINITY, 0.0); len], shape.to_vec())
                .map_err(|e| builtin_error(format!("inf: {e}")))?;
            if let Ok(gpu) = gpu_helpers::upload_complex_tensor(provider, &tensor) {
                runmat_accelerate_api::set_handle_precision(&gpu, precision);
                return Ok(Value::GpuTensor(gpu));
            }
        }
        let attempt = if handle.shape == shape {
            provider.fill_like(handle, f64::INFINITY)
        } else {
            provider.fill(shape, f64::INFINITY)
        };
        if let Ok(gpu) = attempt {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        }

        let host = inf_tensor(shape, dtype_from_precision(precision))?;
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
        .map_err(|e| builtin_error(format!("inf: {e}")))?;
    inf_like(&gathered, shape).await
}

fn inf_gpu_alloc(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Option<Value>> {
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
    match provider.fill(shape, f64::INFINITY) {
        Ok(handle) => {
            runmat_accelerate_api::set_handle_precision(&handle, precision);
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(err) => {
            log::warn!("inf: provider fill failed ({err}); falling back to host tensor path");
            Ok(None)
        }
    }
}

fn inf_tensor(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Tensor> {
    Tensor::new_with_dtype(
        vec![f64::INFINITY; tensor::element_count(shape)],
        shape.to_vec(),
        dtype,
    )
    .map_err(|e| builtin_error(format!("inf: {e}")))
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
                Err(builtin_error(format!("inf: {err}")))
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
            "inf: unsupported prototype {other:?}"
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

    fn assert_all_pos_inf(tensor: &Tensor) {
        assert!(tensor
            .data
            .iter()
            .all(|value| value.is_infinite() && value.is_sign_positive()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_default_scalar() {
        let _guard = clear_accel_provider_state();
        let result = block_on(inf_builtin(Vec::new())).expect("inf");
        match result {
            Value::Num(value) => assert!(value.is_infinite() && value.is_sign_positive()),
            other => panic!("expected scalar Inf, got {other:?}"),
        }
    }

    #[test]
    fn inf_type_defaults_to_num() {
        assert_eq!(inf_type(&[], &ResolveContext::new(Vec::new())), Type::Num);
    }

    #[test]
    fn inf_type_infers_rank_from_scalar_dim() {
        assert_eq!(
            inf_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_square_from_single_dimension() {
        let _guard = clear_accel_provider_state();
        let result = block_on(inf_builtin(vec![Value::Num(3.0)])).expect("inf");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![3, 3]);
        assert_all_pos_inf(&tensor);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_rectangular_from_dims() {
        let _guard = clear_accel_provider_state();
        let result = block_on(inf_builtin(vec![Value::Num(2.0), Value::Num(4.0)])).expect("inf");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_all_pos_inf(&tensor);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_from_size_vector() {
        let _guard = clear_accel_provider_state();
        let size_vec = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let result = block_on(inf_builtin(vec![Value::Tensor(size_vec)])).expect("inf");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 3, 4]);
        assert_all_pos_inf(&tensor);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_single_output_marks_dtype() {
        let _guard = clear_accel_provider_state();
        let result = block_on(inf_builtin(vec![
            Value::Num(2.0),
            Value::Num(2.0),
            Value::from("single"),
        ]))
        .expect("inf");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.dtype, NumericDType::F32);
        assert_all_pos_inf(&tensor);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_like_tensor_infers_shape() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = block_on(inf_builtin(vec![Value::Tensor(proto)])).expect("inf");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_all_pos_inf(&tensor);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_like_complex_scalar() {
        let _guard = clear_accel_provider_state();
        let result = block_on(inf_builtin(vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ]))
        .expect("inf");
        match result {
            Value::ComplexTensor(tensor) => {
                assert_eq!(tensor.shape, vec![3, 3]);
                assert!(tensor
                    .data
                    .iter()
                    .all(|(re, im)| re.is_infinite() && re.is_sign_positive() && *im == 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_like_uses_shape_argument_when_combined_with_like() {
        let _guard = clear_accel_provider_state();
        let shape_source = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let proto = Tensor::new_with_dtype(vec![7.0, 8.0], vec![1, 2], NumericDType::F32).unwrap();
        let result = block_on(inf_builtin(vec![
            Value::Tensor(shape_source),
            Value::from("like"),
            Value::Tensor(proto),
        ]))
        .expect("inf");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.dtype, NumericDType::F32);
        assert_all_pos_inf(&tensor);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_like_without_explicit_shape_uses_prototype_shape() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result =
            block_on(inf_builtin(vec![Value::from("like"), Value::Tensor(proto)])).expect("inf");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_all_pos_inf(&tensor);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_conflicting_like_and_class_is_error() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Num(2.0),
            Value::from("single"),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        assert!(block_on(inf_builtin(args)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_gpu_like_alloc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = block_on(inf_builtin(vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ]))
            .expect("inf");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert_all_pos_inf(&gathered);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }
}
