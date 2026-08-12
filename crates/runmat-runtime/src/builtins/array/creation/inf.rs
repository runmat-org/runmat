//! MATLAB-compatible `inf` array constructor with GPU-aware semantics.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_builtins::ResolveContext;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericDType, SparseTensor, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::array::type_resolvers::tensor_type_from_rank;
use crate::builtins::common::random_args::{
    complex_tensor_into_value, extract_constructor_dimensions, normalize_constructor_shape,
    validate_constructor_gpu_output,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionExprContext,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

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

const INF_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
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

const INF_COLUMN_SIZE_VECTOR_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "inf-column-size-vector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "inf with a column size vector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:InfColumnSizeVectorExtension"),
};
const INF_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "inf-resident-size-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "inf with a resident size control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:InfResidentSizeControlExtension"),
};
pub const INF_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    INF_COLUMN_SIZE_VECTOR_EXTENSION,
    INF_RESIDENT_SIZE_EXTENSION,
];

const INF_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz1...szN/sz",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are exact structural size controls; negative signed values clamp to zero and trailing singleton dimensions normalize away.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "X = inf(integer_n[, integer_sz2, ...])",
        inputs: &INF_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The default output is host double; typename can select single and explicit gpuArray syntax selects residency.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = inf(integer_sz)",
        inputs: &INF_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented size vector is a row vector of exact integer values.",
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

const INF_ERROR_INTEGER_LIKE_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INF.INTEGER_LIKE_PROTOTYPE",
    identifier: None,
    when: "The 'like' prototype has an integer data type that cannot represent Inf.",
    message: "inf: integer 'like' prototypes are not supported",
};

const INF_ERRORS: [BuiltinErrorDescriptor; 5] = [
    INF_ERROR_LIKE_EXPECTED_PROTOTYPE,
    INF_ERROR_CLASS_CONFLICT,
    INF_ERROR_UNRECOGNIZED_OPTION,
    INF_ERROR_LIKE_DUPLICATE,
    INF_ERROR_INTEGER_LIKE_PROTOTYPE,
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
    extensions(INF_EXTENSIONS),
    integer_capabilities(crate::builtins::array::creation::inf::INTEGER_CAPABILITIES),
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
    GpuArray(NumericDType),
    Like(Value),
}

impl ParsedInf {
    async fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<OutputTemplate> = None;
        let mut saw_size_vector = false;

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
                        if class_override.is_some() {
                            return Err(inf_error(&INF_ERROR_CLASS_CONFLICT));
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
                        if class_override.is_some() {
                            return Err(inf_error(&INF_ERROR_CLASS_CONFLICT));
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
                        let dtype = match class_override.take() {
                            Some(OutputTemplate::Single) => NumericDType::F32,
                            Some(OutputTemplate::Double) | None => NumericDType::F64,
                            Some(_) => unreachable!("inf class override is floating"),
                        };
                        class_override = Some(OutputTemplate::GpuArray(dtype));
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

            if matches!(arg, Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &INF_RESIDENT_SIZE_EXTENSION,
                    "inf",
                )?;
            }
            if let Some(parsed_dims) = extract_constructor_dimensions(&arg, "inf")
                .await
                .map_err(builtin_error)?
            {
                if parsed_dims.is_column_vector {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &INF_COLUMN_SIZE_VECTOR_EXTENSION,
                        "inf",
                    )?;
                }
                if parsed_dims.values.len() > 1 {
                    if saw_size_vector || saw_dims_arg {
                        return Err(builtin_error(
                            "inf: a size vector must be the only dimension argument",
                        ));
                    }
                    saw_size_vector = true;
                } else if saw_size_vector {
                    return Err(builtin_error(
                        "inf: a size vector must be the only dimension argument",
                    ));
                }
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims.values;
                } else {
                    dims.extend(parsed_dims.values);
                }
                idx += 1;
                continue;
            }
            return Err(builtin_error(format!(
                "inf: unsupported dimension or option {arg:?}"
            )));
        }

        let shape = if saw_dims_arg {
            normalize_constructor_shape(dims)
        } else {
            vec![1, 1]
        };

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
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
        OutputTemplate::GpuArray(dtype) => inf_gpu(&parsed.shape, dtype).await,
        OutputTemplate::Like(proto) => inf_like(&proto, &parsed.shape).await,
    }
}

fn inf_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    inf_tensor(shape, NumericDType::F64).map(tensor::tensor_into_value)
}

fn inf_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    inf_tensor(shape, NumericDType::F32).map(tensor::tensor_into_value)
}

async fn inf_gpu(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let precision = match dtype {
            NumericDType::F32 => ProviderPrecision::F32,
            NumericDType::F64 => ProviderPrecision::F64,
            _ => unreachable!("inf GPU output is floating"),
        };
        if provider.precision() != precision {
            return Err(builtin_error(
                "inf: active provider cannot preserve requested gpuArray precision",
            ));
        }
        match provider.fill(shape, f64::INFINITY) {
            Ok(handle) => {
                if let Ok(handle) = validate_constructor_gpu_output(
                    "inf",
                    provider,
                    handle,
                    shape,
                    GpuTensorStorage::Real,
                    Some(precision),
                    None,
                    false,
                ) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
            Err(err) => {
                log::debug!("inf_gpu: provider.fill failed ({err}); falling back to host upload");
            }
        }
        let host = inf_tensor(shape, dtype_from_precision(precision))?;
        if let Ok(gpu) = gpu_helpers::upload_tensor(provider, &host) {
            if let Ok(gpu) = validate_constructor_gpu_output(
                "inf",
                provider,
                gpu,
                shape,
                GpuTensorStorage::Real,
                Some(precision),
                None,
                false,
            ) {
                return Ok(Value::GpuTensor(gpu));
            }
        }
    }
    Err(builtin_error(
        "inf: gpuArray output requires an active provider",
    ))
}

#[async_recursion::async_recursion(?Send)]
async fn inf_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::ComplexTensor(tensor) => inf_complex(shape, tensor.numeric_dtype()),
        Value::Complex(_, _) => inf_complex(shape, NumericDType::F64),
        Value::GpuTensor(handle) => inf_like_gpu(handle, shape).await,
        Value::Tensor(t) => match t.numeric_dtype() {
            NumericDType::F32 => inf_single(shape),
            NumericDType::F64 => inf_double(shape),
            NumericDType::I8
            | NumericDType::I16
            | NumericDType::I32
            | NumericDType::I64
            | NumericDType::U8
            | NumericDType::U16
            | NumericDType::U32
            | NumericDType::U64 => Err(inf_error(&INF_ERROR_INTEGER_LIKE_PROTOTYPE)),
        },
        Value::SparseTensor(sparse) if sparse.integer_storage().is_some() => {
            Err(inf_error(&INF_ERROR_INTEGER_LIKE_PROTOTYPE))
        }
        Value::SparseTensor(sparse) => match sparse.numeric_dtype() {
            Some(NumericDType::F32) => inf_sparse(shape, NumericDType::F32),
            Some(NumericDType::F64) => inf_sparse(shape, NumericDType::F64),
            None => Err(builtin_error(
                "inf: 'like' prototype must be single or double",
            )),
            Some(_) => unreachable!("integer sparse prototypes are rejected above"),
        },
        Value::Int(_) => Err(inf_error(&INF_ERROR_INTEGER_LIKE_PROTOTYPE)),
        Value::Num(_) => inf_double(shape),
        _ => Err(builtin_error(
            "inf: 'like' prototype must be single or double",
        )),
    }
}

fn inf_complex(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let data = vec![(f64::INFINITY, 0.0); len];
    ComplexTensor::from_f64_values_with_dtype(data, shape.to_vec(), dtype)
        .map(complex_tensor_into_value)
        .map_err(|e| builtin_error(format!("inf: {e}")))
}

fn inf_sparse(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Value> {
    if shape.len() > 2 {
        return Err(builtin_error(
            "inf: sparse 'like' output must be two-dimensional",
        ));
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| builtin_error("inf: sparse output size overflow"))?;
    let col_ptrs = (0..=cols).map(|column| column * rows).collect();
    let row_indices = (0..cols).flat_map(|_| 0..rows).collect();
    let sparse = match dtype {
        NumericDType::F32 => {
            SparseTensor::new_f32(rows, cols, col_ptrs, row_indices, vec![f32::INFINITY; len])
        }
        NumericDType::F64 => {
            SparseTensor::new(rows, cols, col_ptrs, row_indices, vec![f64::INFINITY; len])
        }
        _ => unreachable!("inf sparse output is floating"),
    }
    .map_err(|error| builtin_error(format!("inf: {error}")))?;
    Ok(Value::SparseTensor(sparse))
}

#[async_recursion::async_recursion(?Send)]
async fn inf_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(handle).is_some() {
        return Err(inf_error(&INF_ERROR_INTEGER_LIKE_PROTOTYPE));
    }
    if runmat_accelerate_api::handle_is_logical(handle) {
        return Err(builtin_error(
            "inf: 'like' prototype must be single or double",
        ));
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) {
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
        let storage = runmat_accelerate_api::handle_storage(handle);
        if storage == GpuTensorStorage::ComplexInterleaved {
            let len = tensor::element_count(shape);
            let tensor = ComplexTensor::from_f64_values_with_dtype(
                vec![(f64::INFINITY, 0.0); len],
                shape.to_vec(),
                dtype_from_precision(precision),
            )
            .map_err(|e| builtin_error(format!("inf: {e}")))?;
            match gpu_helpers::upload_complex_tensor(provider, &tensor) {
                Ok(gpu) => {
                    if let Ok(gpu) = validate_constructor_gpu_output(
                        "inf",
                        provider,
                        gpu,
                        shape,
                        GpuTensorStorage::ComplexInterleaved,
                        Some(precision),
                        None,
                        false,
                    ) {
                        return Ok(Value::GpuTensor(gpu));
                    }
                }
                Err(_) => {}
            }
            return Err(builtin_error(
                "inf: provider cannot preserve explicit complex gpuArray output",
            ));
        }
        let attempt = if handle.shape == shape {
            provider.fill_like(handle, f64::INFINITY)
        } else {
            provider.fill(shape, f64::INFINITY)
        };
        if let Ok(gpu) = attempt {
            if let Ok(gpu) = validate_constructor_gpu_output(
                "inf",
                provider,
                gpu,
                shape,
                GpuTensorStorage::Real,
                Some(precision),
                None,
                false,
            ) {
                return Ok(Value::GpuTensor(gpu));
            }
        }

        let host = inf_tensor(shape, dtype_from_precision(precision))?;
        if let Ok(gpu) = gpu_helpers::upload_tensor(provider, &host) {
            if let Ok(gpu) = validate_constructor_gpu_output(
                "inf",
                provider,
                gpu,
                shape,
                GpuTensorStorage::Real,
                Some(precision),
                None,
                false,
            ) {
                return Ok(Value::GpuTensor(gpu));
            }
        }
    }

    Err(builtin_error(
        "inf: provider cannot preserve explicit gpuArray output",
    ))
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
            .materialize_f64()
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

    #[test]
    fn inf_integer_dimensions_clamp_negative_and_normalize_trailing_singletons() {
        let _guard = clear_accel_provider_state();
        let result = block_on(inf_builtin(vec![
            Value::Int(runmat_builtins::IntValue::I16(-2)),
            Value::Int(runmat_builtins::IntValue::U8(3)),
            Value::Int(runmat_builtins::IntValue::U64(1)),
        ]))
        .expect("inf integer dimensions");
        let Value::Tensor(tensor) = result else {
            panic!("expected empty host tensor");
        };
        assert_eq!(tensor.shape, vec![0, 3]);
    }

    #[test]
    fn inf_without_explicit_gpu_intent_remains_host_resident() {
        test_support::with_test_provider(|_| {
            let result = block_on(inf_builtin(vec![Value::Num(2.0)])).expect("inf");
            assert!(matches!(result, Value::Tensor(tensor) if tensor.shape == vec![2, 2]));
        });
    }

    #[test]
    fn inf_column_size_vector_follows_compatibility_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let size =
            Tensor::new_integer(runmat_builtins::IntegerStorage::U16(vec![2, 3]), vec![2, 1])
                .expect("column size vector");
        let error = block_on(inf_builtin(vec![Value::Tensor(size)])).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:InfColumnSizeVectorExtension")
        );
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
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert_all_pos_inf(&tensor);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_implicit_prototype_is_rejected() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert!(block_on(inf_builtin(vec![Value::Tensor(proto)])).is_err());
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
                    .materialize_f64()
                    .iter()
                    .all(|(re, im)| re.is_infinite() && re.is_sign_positive() && *im == 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn inf_like_complex_single_preserves_native_single() {
        let prototype =
            ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).expect("complex single");
        let result = block_on(inf_builtin(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::ComplexTensor(prototype),
        ]))
        .expect("inf complex single like");
        let Value::ComplexTensor(output) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_like_uses_shape_argument_when_combined_with_like() {
        let _guard = clear_accel_provider_state();
        let shape_source = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let proto = Tensor::new_with_dtype(vec![7.0, 8.0], vec![1, 2], NumericDType::F32).unwrap();
        let result = block_on(inf_builtin(vec![
            Value::Tensor(shape_source),
            Value::from("like"),
            Value::Tensor(proto),
        ]))
        .expect("inf");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert_all_pos_inf(&tensor);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_like_without_explicit_shape_returns_scalar() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result =
            block_on(inf_builtin(vec![Value::from("like"), Value::Tensor(proto)])).expect("inf");
        assert!(
            matches!(result, Value::Num(value) if value.is_infinite() && value.is_sign_positive())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_like_single_sparse_preserves_sparse_single() {
        let prototype = SparseTensor::zeros_f32(1, 1);
        let result = block_on(inf_builtin(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::SparseTensor(prototype),
        ]))
        .expect("inf single sparse like");
        let Value::SparseTensor(output) = result else {
            panic!("expected sparse tensor");
        };
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
        assert!(output
            .as_f32_slice()
            .expect("single storage")
            .iter()
            .all(|value| value.is_infinite() && value.is_sign_positive()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inf_rejects_every_integer_like_prototype() {
        let _guard = clear_accel_provider_state();
        let prototypes = [
            runmat_builtins::IntegerStorage::I8(vec![0]),
            runmat_builtins::IntegerStorage::I16(vec![0]),
            runmat_builtins::IntegerStorage::I32(vec![0]),
            runmat_builtins::IntegerStorage::I64(vec![0]),
            runmat_builtins::IntegerStorage::U8(vec![0]),
            runmat_builtins::IntegerStorage::U16(vec![0]),
            runmat_builtins::IntegerStorage::U32(vec![0]),
            runmat_builtins::IntegerStorage::U64(vec![0]),
        ];

        for storage in prototypes {
            let proto = Tensor::new_integer(storage, vec![1, 1]).expect("integer prototype");
            let err = block_on(inf_builtin(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::Tensor(proto),
            ]))
            .unwrap_err();
            assert!(err.message().contains("integer 'like' prototypes"));
        }

        let prototype = SparseTensor::zeros_with_integer_storage(
            1,
            1,
            &runmat_builtins::IntegerStorage::U64(Vec::new()),
        );
        let err = block_on(inf_builtin(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::SparseTensor(prototype),
        ]))
        .expect_err("integer sparse like");
        assert!(err.message().contains("integer 'like' prototypes"));
    }

    #[test]
    fn inf_rejects_resident_integer_gpu_like_prototype() {
        test_support::with_test_provider(|provider| {
            let values = [1_u64];
            let prototype = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&values),
                    shape: &[1, 1],
                })
                .expect("integer prototype");
            let err = block_on(inf_builtin(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(prototype),
            ]))
            .expect_err("integer gpu like");
            assert!(err.message().contains("integer 'like' prototypes"));
        });
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
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
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

    #[test]
    fn inf_same_shape_complex_gpu_like_stays_complex_and_resident() {
        test_support::with_f32_test_provider(|provider| {
            let prototype = ComplexTensor::from_f32(vec![(1.0, -1.0); 4], vec![2, 2])
                .expect("complex single prototype");
            let handle = gpu_helpers::upload_complex_tensor(provider, &prototype).expect("upload");
            let result = block_on(inf_builtin(vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ]))
            .expect("inf complex gpu like");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            );
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
        });
    }
}
