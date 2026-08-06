//! MATLAB-compatible `arrayfun` builtin with GPU-aware semantics.
//!
//! This implementation applies a scalar MATLAB function to every element of one or
//! more array inputs. Supported builtin callbacks can dispatch directly for
//! `gpuArray` inputs; other callbacks gather authoritative typed storage and
//! re-upload uniform numeric or logical output.

use crate::builtins::acceleration::gpu::type_resolvers::arrayfun_type;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{broadcast, gpu_helpers, tensor};
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, user_functions,
    BuiltinResult, RuntimeError,
};
use runmat_accelerate_api::{set_handle_logical, GpuTensorHandle};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, Closure, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
    LogicalArray, NumericScalar, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::arrayfun")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "arrayfun",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Unary { name: "unary_sin" },
        ProviderHook::Unary { name: "unary_cos" },
        ProviderHook::Unary { name: "unary_abs" },
        ProviderHook::Unary { name: "unary_exp" },
        ProviderHook::Unary { name: "unary_log" },
        ProviderHook::Unary { name: "unary_sqrt" },
        ProviderHook::Binary {
            name: "elem_add",
            commutative: true,
        },
        ProviderHook::Binary {
            name: "elem_sub",
            commutative: false,
        },
        ProviderHook::Binary {
            name: "elem_mul",
            commutative: true,
        },
        ProviderHook::Binary {
            name: "elem_div",
            commutative: false,
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers that implement the listed kernels can run supported callbacks entirely on the GPU; unsupported callbacks fall back to the host path with re-upload.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::acceleration::gpu::arrayfun"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "arrayfun",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion barrier because the callback can run arbitrary MATLAB code.",
};

const BUILTIN_NAME: &str = "arrayfun";

pub(crate) const ARRAYFUN_TEXT_CALLABLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "arrayfun-text-callable",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "arrayfun with a character-vector or string-scalar callable is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ArrayfunTextCallableExtension"),
    };

pub(crate) const ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "arrayfun-host-scalar-expansion",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "host arrayfun with scalar expansion across differently sized inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ArrayfunHostScalarExpansionExtension"),
    };

pub(crate) const ARRAYFUN_GPU_OPTIONS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "arrayfun-gpu-options",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "gpuArray arrayfun with UniformOutput or ErrorHandler options is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ArrayfunGpuOptionsExtension"),
    };

pub const ARRAYFUN_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ARRAYFUN_TEXT_CALLABLE_EXTENSION,
    ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION,
    ARRAYFUN_GPU_OPTIONS_EXTENSION,
];

const ARRAYFUN_INTEGER_ARRAY_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A1",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Each element is passed to func in the exact integer class stored by A1.",
    },
    BuiltinIntegerInputCapability {
        name: "An",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Additional arrays may independently use any real integer class; callback overload semantics decide whether a class combination is valid.",
    },
];

const ARRAYFUN_INTEGER_CALLBACK_RESULT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "func or ErrorHandler scalar result",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Uniform output requires the same scalar class on every invocation and concatenates exact same-class integer results.",
    }];

const ARRAYFUN_REJECTED_INTEGER_UNIFORM_OUTPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "UniformOutput",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public control is logical true or false; typed-integer controls are rejected.",
    }];

pub const ARRAYFUN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = arrayfun(func, integer_A1, integer_An...)",
        inputs: &ARRAYFUN_INTEGER_ARRAY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "arrayfun itself performs structural element extraction without conversion. The callback defines arithmetic, overflow, and result class. Host inputs must have equal size; the documented gpuArray overload permits compatible sizes.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "integer_B = arrayfun(func_returning_integer, A1, An...)",
        inputs: &ARRAYFUN_INTEGER_CALLBACK_RESULT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Uniform same-class scalar integer callback results are collected in authoritative native storage. With gpuArray input, supported results remain resident after direct execution or exact gather fallback and typed re-upload.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = arrayfun(func, A1, An..., \"UniformOutput\", typed_integer)",
        inputs: &ARRAYFUN_REJECTED_INTEGER_UNIFORM_OUTPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "All eight typed-integer scalar and tensor control classes reject rather than being coerced to logical.",
    },
];

const ARRAYFUN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise callback result (uniform array or cell array).",
}];

const ARRAYFUN_INPUTS_BASE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or callable name.",
    },
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional input arrays.",
    },
];

const ARRAYFUN_INPUTS_UNIFORM: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or callable name.",
    },
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional input arrays.",
    },
    BuiltinParamDescriptor {
        name: "UniformOutput",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: Some("\"UniformOutput\""),
        description: "Name-value key that toggles uniform output collection.",
    },
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: Some("true"),
        description: "Logical true/false value for UniformOutput.",
    },
];

const ARRAYFUN_INPUTS_HANDLER: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or callable name.",
    },
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional input arrays.",
    },
    BuiltinParamDescriptor {
        name: "ErrorHandler",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: Some("\"ErrorHandler\""),
        description: "Name-value key that provides fallback callback on per-element failures.",
    },
    BuiltinParamDescriptor {
        name: "handler",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Callback invoked with error struct and original scalar arguments.",
    },
];

const ARRAYFUN_INPUTS_OPTIONS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or callable name.",
    },
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional input arrays.",
    },
    BuiltinParamDescriptor {
        name: "nameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option pairs including UniformOutput and ErrorHandler.",
    },
];

const ARRAYFUN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "B = arrayfun(func, A1, An...)",
        inputs: &ARRAYFUN_INPUTS_BASE,
        outputs: &ARRAYFUN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = arrayfun(func, A1, An..., \"UniformOutput\", tf)",
        inputs: &ARRAYFUN_INPUTS_UNIFORM,
        outputs: &ARRAYFUN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = arrayfun(func, A1, An..., \"ErrorHandler\", handler)",
        inputs: &ARRAYFUN_INPUTS_HANDLER,
        outputs: &ARRAYFUN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = arrayfun(func, A1, An..., nameValue...)",
        inputs: &ARRAYFUN_INPUTS_OPTIONS,
        outputs: &ARRAYFUN_OUTPUT,
    },
];

const ARRAYFUN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARRAYFUN.INVALID_INPUT",
    identifier: Some("RunMat:arrayfun:InvalidInput"),
    when: "Inputs, callable forms, or option tails violate arrayfun argument requirements.",
    message: "arrayfun: invalid input arguments",
};

const ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARRAYFUN.UNIFORM_OUTPUT_OPTION",
    identifier: Some("RunMat:arrayfun:UniformOutputOption"),
    when: "UniformOutput option value is not interpretable as logical true/false.",
    message: "arrayfun: UniformOutput must be logical true or false",
};

const ARRAYFUN_ERROR_CALLBACK_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARRAYFUN.CALLBACK_FAILED",
    identifier: Some("RunMat:arrayfun:CallbackFailed"),
    when: "Callback invocation fails and no ErrorHandler recovers the element.",
    message: "arrayfun: callback execution failed",
};

const ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARRAYFUN.UNIFORM_OUTPUT_TYPE",
    identifier: Some("RunMat:arrayfun:UniformOutputType"),
    when: "UniformOutput=true callback result is not a supported scalar type.",
    message: "arrayfun: callback must return scalar values for UniformOutput=true",
};

const ARRAYFUN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARRAYFUN.INTERNAL",
    identifier: Some("RunMat:arrayfun:InternalError"),
    when: "Internal shape/index/materialization/upload path fails.",
    message: "arrayfun: internal error",
};

const ARRAYFUN_ERROR_UNDEFINED_FUNCTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARRAYFUN.UNDEFINED_FUNCTION",
    identifier: Some("RunMat:UndefinedFunction"),
    when: "External callable identity cannot be resolved in semantic/runtime boundaries.",
    message: "arrayfun: undefined function",
};

const ARRAYFUN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    ARRAYFUN_ERROR_INVALID_INPUT,
    ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION,
    ARRAYFUN_ERROR_CALLBACK_FAILED,
    ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
    ARRAYFUN_ERROR_INTERNAL,
    ARRAYFUN_ERROR_UNDEFINED_FUNCTION,
];

pub const ARRAYFUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARRAYFUN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ARRAYFUN_ERRORS,
};

fn arrayfun_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    arrayfun_error_with_message(error.message, error)
}

fn arrayfun_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn arrayfun_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    arrayfun_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn arrayfun_error_with_source(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
    source: RuntimeError,
) -> RuntimeError {
    let identifier = source.identifier().map(str::to_string);
    let mut builder = build_runtime_error(message.into())
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = identifier.as_deref().or(error.identifier) {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn arrayfun_flow(message: impl Into<String>) -> RuntimeError {
    arrayfun_error_with_message(message, &ARRAYFUN_ERROR_INVALID_INPUT)
}

fn arrayfun_internal(message: impl Into<String>) -> RuntimeError {
    arrayfun_error_with_message(message, &ARRAYFUN_ERROR_INTERNAL)
}

fn arrayfun_flow_with_source(message: impl Into<String>, source: RuntimeError) -> RuntimeError {
    arrayfun_error_with_source(message, &ARRAYFUN_ERROR_CALLBACK_FAILED, source)
}

fn format_handler_error(err: &RuntimeError) -> String {
    if let Some(identifier) = err.identifier() {
        if err.message().is_empty() {
            return identifier.to_string();
        }
        if err.message().starts_with(identifier) {
            return err.message().to_string();
        }
        return format!("{identifier}: {}", err.message());
    }
    err.message().to_string()
}

#[runtime_builtin(
    name = "arrayfun",
    category = "acceleration/gpu",
    summary = "Apply a function element-wise across array inputs.",
    keywords = "arrayfun,gpu,array,map,functional",
    accel = "host",
    type_resolver(arrayfun_type),
    descriptor(crate::builtins::acceleration::gpu::arrayfun::ARRAYFUN_DESCRIPTOR),
    extensions(crate::builtins::acceleration::gpu::arrayfun::ARRAYFUN_EXTENSIONS),
    integer_capabilities(
        crate::builtins::acceleration::gpu::arrayfun::ARRAYFUN_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::acceleration::gpu::arrayfun"
)]
async fn arrayfun_builtin(func: Value, mut rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(
        &func,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ARRAYFUN_TEXT_CALLABLE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let callable = Callable::from_function(func)?;

    let mut uniform_output = true;
    let mut uniform_output_explicit = false;
    let mut error_handler: Option<Callable> = None;

    while rest.len() >= 2 {
        let key_candidate = rest[rest.len() - 2].clone();
        let Some(name) = extract_string(&key_candidate) else {
            break;
        };
        let value = rest.pop().expect("value present");
        rest.pop();
        match name.trim().to_ascii_lowercase().as_str() {
            "uniformoutput" => {
                uniform_output = parse_uniform_output(value)?;
                uniform_output_explicit = true;
            }
            "errorhandler" => {
                if matches!(
                    &value,
                    Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
                ) {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &ARRAYFUN_TEXT_CALLABLE_EXTENSION,
                        BUILTIN_NAME,
                    )?;
                }
                error_handler = Some(Callable::from_function(value)?);
            }
            other => {
                return Err(arrayfun_flow(format!(
                    "arrayfun: unknown name-value argument '{other}'"
                )))
            }
        }
    }

    if rest.is_empty() {
        return Err(arrayfun_flow("arrayfun: expected at least one input array"));
    }

    let inputs_snapshot = rest.clone();
    let has_gpu_input = inputs_snapshot
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)));
    let gpu_device_id = inputs_snapshot.iter().find_map(|v| {
        if let Value::GpuTensor(h) = v {
            Some(h.device_id)
        } else {
            None
        }
    });
    if has_gpu_input && (uniform_output_explicit || error_handler.is_some()) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ARRAYFUN_GPU_OPTIONS_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    if uniform_output {
        if let Some(gpu_result) =
            try_gpu_fast_path(&callable, &inputs_snapshot, error_handler.as_ref()).await?
        {
            return Ok(gpu_result);
        }
    }

    let mut inputs: Vec<ArrayInput> = Vec::with_capacity(rest.len());

    for raw in rest {
        if matches!(raw, Value::Cell(_)) {
            return Err(arrayfun_flow(
                "arrayfun: cell inputs are not supported (use cellfun instead)",
            ));
        }
        if matches!(raw, Value::Struct(_)) {
            return Err(arrayfun_flow("arrayfun: struct inputs are not supported"));
        }

        let host_value = gather_if_needed_async(&raw).await?;
        let data = ArrayData::from_value(host_value)?;
        let shape = data.shape_vec();
        let strides = broadcast::compute_strides(&shape);
        inputs.push(ArrayInput {
            data,
            shape,
            strides,
        });
    }

    let first_shape = inputs
        .first()
        .map(|input| input.shape.clone())
        .unwrap_or_default();
    let base_shape = if has_gpu_input {
        inputs
            .iter()
            .skip(1)
            .try_fold(first_shape, |shape, input| {
                broadcast::broadcast_shapes(BUILTIN_NAME, &shape, &input.shape)
                    .map_err(arrayfun_flow)
            })?
    } else {
        let non_scalar_shapes: Vec<&[usize]> = inputs
            .iter()
            .filter(|input| input.len() != 1)
            .map(|input| input.shape.as_slice())
            .collect();
        let target_shape = non_scalar_shapes
            .first()
            .copied()
            .unwrap_or(first_shape.as_slice());
        if non_scalar_shapes
            .iter()
            .skip(1)
            .any(|shape| *shape != target_shape)
        {
            return Err(arrayfun_flow(
                "arrayfun: host input does not match the size of the first array",
            ));
        }
        if inputs.iter().any(|input| input.shape != target_shape) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        target_shape.to_vec()
    };

    let total_len = base_shape.iter().product();

    if total_len == 0 {
        if uniform_output {
            return Ok(empty_uniform(&base_shape));
        } else {
            return make_cell_with_shape(Vec::new(), base_shape)
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")));
        }
    }

    let mut collector = if uniform_output {
        Some(UniformCollector::Pending)
    } else {
        None
    };

    let mut cell_outputs: Vec<Value> = Vec::new();
    let mut args: Vec<Value> = Vec::with_capacity(inputs.len());

    for idx in 0..total_len {
        args.clear();
        for input in &inputs {
            args.push(input.value_at(idx, &base_shape)?);
        }

        let result = match callable.call(&args).await {
            Ok(value) => value,
            Err(err) => {
                let handler = match error_handler.as_ref() {
                    Some(handler) => handler,
                    None => {
                        return Err(arrayfun_flow_with_source(
                            format!("arrayfun: {}", err.message()),
                            err,
                        ))
                    }
                };
                let err_message = format_handler_error(&err);
                let err_value = make_error_struct(&err_message, idx);
                let mut handler_args = Vec::with_capacity(1 + args.len());
                handler_args.push(err_value);
                handler_args.extend(args.clone());
                handler.call(&handler_args).await?
            }
        };

        let host_result = gather_if_needed_async(&result).await?;

        if let Some(collector) = collector.as_mut() {
            collector.push(&host_result)?;
        } else {
            cell_outputs.push(host_result);
        }
    }

    if let Some(collector) = collector {
        let uniform = collector.finish(&base_shape)?;
        maybe_upload_uniform(uniform, has_gpu_input, gpu_device_id)
    } else {
        make_cell_with_shape(cell_outputs, base_shape)
            .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))
    }
}

fn maybe_upload_uniform(
    value: Value,
    has_gpu_input: bool,
    gpu_device_id: Option<u32>,
) -> BuiltinResult<Value> {
    if !has_gpu_input {
        return Ok(value);
    }
    #[cfg(all(test, feature = "wgpu"))]
    {
        if matches!(gpu_device_id, Some(id) if id != 0) {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let _ = gpu_device_id; // may be used only in cfg(test)
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(value),
    };

    match value {
        Value::Tensor(tensor) => {
            let handle = gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::LogicalArray(logical) => {
            let data: Vec<f64> = logical
                .data
                .iter()
                .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                .collect();
            let tensor = Tensor::new(data, logical.shape.clone())
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
            let handle = gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
            set_handle_logical(&handle, true);
            Ok(Value::GpuTensor(handle))
        }
        other => Ok(other),
    }
}

fn empty_uniform(shape: &[usize]) -> Value {
    if shape.is_empty() {
        return Value::Tensor(Tensor::zeros(vec![0, 0]));
    }
    let total: usize = shape.iter().product();
    let tensor = Tensor::new(vec![0.0; total], shape.to_vec())
        .unwrap_or_else(|_| Tensor::zeros(shape.to_vec()));
    Value::Tensor(tensor)
}

fn parse_uniform_output(value: Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(b),
        Value::LogicalArray(logical) if logical.len() == 1 => Ok(logical.data[0] != 0),
        Value::Num(0.0) => Ok(false),
        Value::Num(1.0) => Ok(true),
        Value::Tensor(tensor)
            if tensor.len() == 1
                && tensor.numeric_dtype() == runmat_builtins::NumericDType::F64 =>
        {
            match tensor.numeric_value_at(0) {
                Some(NumericScalar::F64(0.0)) => Ok(false),
                Some(NumericScalar::F64(1.0)) => Ok(true),
                _ => Err(arrayfun_error(&ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION)),
            }
        }
        other => Err(arrayfun_error_with_detail(
            &ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION,
            format!("got {other:?}"),
        )),
    }
}

fn extract_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}

struct ArrayInput {
    data: ArrayData,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl ArrayInput {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn value_at(&self, idx: usize, output_shape: &[usize]) -> BuiltinResult<Value> {
        let source_index =
            broadcast::broadcast_index(idx, output_shape, &self.shape, &self.strides);
        self.data.value_at(source_index)
    }
}

enum ArrayData {
    Tensor(Tensor),
    Logical(LogicalArray),
    Complex(ComplexTensor),
    Char(CharArray),
    String(StringArray),
    Scalar(Value),
}

impl ArrayData {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(t) => Ok(ArrayData::Tensor(t)),
            Value::LogicalArray(l) => Ok(ArrayData::Logical(l)),
            Value::ComplexTensor(c) => Ok(ArrayData::Complex(c)),
            Value::CharArray(ca) => Ok(ArrayData::Char(ca)),
            Value::StringArray(sa) => Ok(ArrayData::String(sa)),
            Value::Num(_)
            | Value::Bool(_)
            | Value::Int(_)
            | Value::Complex(_, _)
            | Value::String(_) => {
                Ok(ArrayData::Scalar(value))
            }
            other => Err(arrayfun_flow(format!(
                "arrayfun: unsupported input type {other:?} (expected numeric, logical, complex, char, or string arrays)"
            ))),
        }
    }

    fn len(&self) -> usize {
        match self {
            ArrayData::Tensor(t) => t.len(),
            ArrayData::Logical(l) => l.data.len(),
            ArrayData::Complex(c) => c.len(),
            ArrayData::Char(ca) => ca.rows * ca.cols,
            ArrayData::String(sa) => sa.data.len(),
            ArrayData::Scalar(_) => 1,
        }
    }

    fn shape_vec(&self) -> Vec<usize> {
        match self {
            ArrayData::Tensor(t) => {
                if t.shape.is_empty() {
                    vec![1, 1]
                } else {
                    t.shape.clone()
                }
            }
            ArrayData::Logical(l) => {
                if l.shape.is_empty() {
                    vec![1, 1]
                } else {
                    l.shape.clone()
                }
            }
            ArrayData::Complex(c) => {
                if c.shape.is_empty() {
                    vec![1, 1]
                } else {
                    c.shape.clone()
                }
            }
            ArrayData::Char(ca) => vec![ca.rows, ca.cols],
            ArrayData::String(sa) => {
                if sa.shape.is_empty() {
                    vec![1, 1]
                } else {
                    sa.shape.clone()
                }
            }
            ArrayData::Scalar(_) => vec![1, 1],
        }
    }

    fn value_at(&self, idx: usize) -> BuiltinResult<Value> {
        match self {
            ArrayData::Tensor(t) => match t
                .numeric_value_at(idx)
                .ok_or_else(|| arrayfun_flow("arrayfun: index out of bounds"))?
            {
                NumericScalar::F64(value) => Ok(Value::Num(value)),
                NumericScalar::F32(value) => Ok(Value::Tensor(
                    Tensor::from_f32(vec![value], vec![1, 1])
                        .map_err(|error| arrayfun_internal(format!("arrayfun: {error}")))?,
                )),
                value => Ok(Value::Int(value.into_int_value().ok_or_else(|| {
                    arrayfun_internal("arrayfun: integer scalar classification failed")
                })?)),
            },
            ArrayData::Logical(l) => Ok(Value::Bool(
                *l.data
                    .get(idx)
                    .ok_or_else(|| arrayfun_flow("arrayfun: index out of bounds"))?
                    != 0,
            )),
            ArrayData::Complex(c) => {
                let (real, imag) = c
                    .numeric_value_at(idx)
                    .ok_or_else(|| arrayfun_flow("arrayfun: index out of bounds"))?;
                match (real, imag) {
                    (NumericScalar::F64(real), NumericScalar::F64(imag)) => {
                        Ok(Value::Complex(real, imag))
                    }
                    (NumericScalar::F32(real), NumericScalar::F32(imag)) => {
                        Ok(Value::ComplexTensor(
                            ComplexTensor::from_f32(vec![(real, imag)], vec![1, 1])
                                .map_err(|error| arrayfun_internal(format!("arrayfun: {error}")))?,
                        ))
                    }
                    (real, imag) => {
                        let real = real.into_int_value().ok_or_else(|| {
                            arrayfun_internal(
                                "arrayfun: complex scalar components have inconsistent classes",
                            )
                        })?;
                        let imag = imag.into_int_value().ok_or_else(|| {
                            arrayfun_internal(
                                "arrayfun: complex scalar components have inconsistent classes",
                            )
                        })?;
                        let storage = IntegerComplexStorage::new(
                            IntegerStorage::from_scalar(real),
                            IntegerStorage::from_scalar(imag),
                        )
                        .map_err(|error| arrayfun_internal(format!("arrayfun: {error}")))?;
                        Ok(Value::ComplexTensor(
                            ComplexTensor::new_integer(storage, vec![1, 1])
                                .map_err(|error| arrayfun_internal(format!("arrayfun: {error}")))?,
                        ))
                    }
                }
            }
            ArrayData::Char(ca) => {
                if ca.rows == 0 || ca.cols == 0 {
                    return Ok(Value::CharArray(
                        CharArray::new(Vec::new(), 0, 0)
                            .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?,
                    ));
                }
                let rows = ca.rows;
                let cols = ca.cols;
                let row = idx % rows;
                let col = idx / rows;
                let data_idx = row * cols + col;
                let ch = *ca
                    .data
                    .get(data_idx)
                    .ok_or_else(|| arrayfun_flow("arrayfun: index out of bounds"))?;
                let char_array = CharArray::new(vec![ch], 1, 1)
                    .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
                Ok(Value::CharArray(char_array))
            }
            ArrayData::String(sa) => {
                Ok(Value::String(sa.data.get(idx).cloned().ok_or_else(
                    || arrayfun_flow("arrayfun: index out of bounds"),
                )?))
            }
            ArrayData::Scalar(v) => Ok(v.clone()),
        }
    }
}

#[derive(Clone)]
enum Callable {
    Builtin { name: String },
    ExternalName { name: String },
    Closure(Closure),
}

impl Callable {
    fn resolved_semantic_handle(name: &str) -> Option<Self> {
        let function = user_functions::resolve_semantic_function_by_name(name)?;
        Some(Callable::Closure(Closure {
            function_name: name.to_string(),
            bound_function: Some(function),
            captures: Vec::new(),
        }))
    }

    fn from_function(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(text) => Self::from_text(&text),
            Value::CharArray(ca) => {
                if ca.rows != 1 {
                    Err(arrayfun_flow(
                        "arrayfun: function name must be a character vector or string scalar",
                    ))
                } else {
                    let text: String = ca.data.iter().collect();
                    Self::from_text(&text)
                }
            }
            Value::StringArray(sa) if sa.data.len() == 1 => Self::from_text(&sa.data[0]),
            Value::FunctionHandle(name) => Self::from_text(&name),
            Value::ExternalFunctionHandle(name) => {
                if let Some(callable) = Self::resolved_semantic_handle(&name) {
                    Ok(callable)
                } else if crate::is_well_formed_qualified_name(&name) {
                    Ok(Callable::ExternalName { name })
                } else {
                    Ok(Callable::Builtin { name })
                }
            }
            Value::BoundFunctionHandle { name, function } => Ok(Callable::Closure(Closure {
                function_name: name,
                bound_function: Some(function),
                captures: Vec::new(),
            })),
            Value::Closure(mut closure) => {
                if closure.bound_function.is_none() {
                    if let Some(function) =
                        user_functions::resolve_semantic_function_by_name(&closure.function_name)
                    {
                        closure.bound_function = Some(function);
                    }
                }
                Ok(Callable::Closure(closure))
            }
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => Err(arrayfun_flow(
                "arrayfun: expected function handle or builtin name, not a scalar value",
            )),
            other => Err(arrayfun_flow(format!(
                "arrayfun: expected function handle or builtin name, got {other:?}"
            ))),
        }
    }

    fn from_text(text: &str) -> BuiltinResult<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(arrayfun_flow(
                "arrayfun: expected function handle or builtin name, got empty string",
            ));
        }
        if let Some(rest) = trimmed.strip_prefix('@') {
            let name = rest.trim();
            if name.is_empty() {
                Err(arrayfun_flow("arrayfun: empty function handle"))
            } else {
                if let Some(callable) = Self::resolved_semantic_handle(name) {
                    return Ok(callable);
                }
                if crate::is_well_formed_qualified_name(name) {
                    return Ok(Callable::ExternalName {
                        name: name.to_string(),
                    });
                }
                Ok(Callable::Builtin {
                    name: name.to_string(),
                })
            }
        } else {
            let name = trimmed.to_ascii_lowercase();
            if let Some(callable) = Self::resolved_semantic_handle(&name) {
                return Ok(callable);
            }
            if crate::is_well_formed_qualified_name(&name) {
                return Ok(Callable::ExternalName { name });
            }
            Ok(Callable::Builtin { name })
        }
    }

    fn builtin_name(&self) -> Option<&str> {
        match self {
            Callable::Builtin { name } => Some(name.as_str()),
            Callable::ExternalName { .. } | Callable::Closure(_) => None,
        }
    }

    async fn call(&self, args: &[Value]) -> crate::BuiltinResult<Value> {
        match self {
            Callable::Builtin { name } => {
                let request = user_functions::CallableRequest::resolved(
                    runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name.clone())),
                    runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
                    args.to_vec(),
                    1,
                );
                if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
                    return result;
                }
                crate::call_builtin_async(name, args).await
            }
            Callable::ExternalName { name } => {
                let identity = crate::external_callable_identity_for_name(name);
                let request = user_functions::CallableRequest::resolved(
                    identity.clone(),
                    runmat_hir::CallableFallbackPolicy::ExternalBoundary,
                    args.to_vec(),
                    1,
                );
                if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
                    return result;
                }
                Err(arrayfun_error_with_message(
                    format!("Undefined function for callable identity {identity:?}"),
                    &ARRAYFUN_ERROR_UNDEFINED_FUNCTION,
                ))
            }
            Callable::Closure(c) => {
                let mut merged = c.captures.clone();
                merged.extend_from_slice(args);
                if let Some(function) = c.bound_function {
                    let request =
                        user_functions::CallableRequest::semantic(function, merged.clone(), 1);
                    if let Some(result) =
                        user_functions::try_call_semantic_descriptor(request).await
                    {
                        return result;
                    }
                    return Err(arrayfun_error_with_detail(
                        &ARRAYFUN_ERROR_CALLBACK_FAILED,
                        format!(
                            "semantic closure '{}' ({function}) is unavailable",
                            c.function_name
                        ),
                    ));
                }
                if let Some(function) =
                    user_functions::resolve_semantic_function_by_name(&c.function_name)
                {
                    let request =
                        user_functions::CallableRequest::semantic(function, merged.clone(), 1);
                    if let Some(result) =
                        user_functions::try_call_semantic_descriptor(request).await
                    {
                        return result;
                    }
                }
                crate::call_builtin_async(&c.function_name, &merged).await
            }
        }
    }
}

async fn try_gpu_fast_path(
    callable: &Callable,
    inputs: &[Value],
    error_handler: Option<&Callable>,
) -> BuiltinResult<Option<Value>> {
    if inputs.is_empty() || error_handler.is_some() {
        return Ok(None);
    }
    if !inputs
        .iter()
        .all(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(None);
    }

    #[cfg(all(test, feature = "wgpu"))]
    {
        if inputs
            .iter()
            .any(|v| matches!(v, Value::GpuTensor(h) if h.device_id != 0))
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    let Some(name_raw) = callable.builtin_name() else {
        return Ok(None);
    };
    let name = name_raw.to_ascii_lowercase();

    let mut handles: Vec<GpuTensorHandle> = Vec::with_capacity(inputs.len());
    for value in inputs {
        if let Value::GpuTensor(handle) = value {
            handles.push(handle.clone());
        }
    }

    if handles.len() >= 2 {
        let base_shape = handles[0].shape.clone();
        if handles
            .iter()
            .skip(1)
            .any(|handle| handle.shape != base_shape)
        {
            return Ok(None);
        }
    }

    let result = match name.as_str() {
        "sin" if handles.len() == 1 => provider.unary_sin(&handles[0]).await,
        "cos" if handles.len() == 1 => provider.unary_cos(&handles[0]).await,
        "abs" if handles.len() == 1 => provider.unary_abs(&handles[0]).await,
        "exp" if handles.len() == 1 => provider.unary_exp(&handles[0]).await,
        "log" if handles.len() == 1 => provider.unary_log(&handles[0]).await,
        "sqrt" if handles.len() == 1 => provider.unary_sqrt(&handles[0]).await,
        "plus" if handles.len() == 2 => provider.elem_add(&handles[0], &handles[1]).await,
        "minus" if handles.len() == 2 => provider.elem_sub(&handles[0], &handles[1]).await,
        "times" if handles.len() == 2 => provider.elem_mul(&handles[0], &handles[1]).await,
        "rdivide" if handles.len() == 2 => provider.elem_div(&handles[0], &handles[1]).await,
        "ldivide" if handles.len() == 2 => provider.elem_div(&handles[1], &handles[0]).await,
        _ => return Ok(None),
    };

    match result {
        Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
        Err(_) => Ok(None),
    }
}

enum UniformCollector {
    Pending,
    F64(Vec<f64>),
    F32(Vec<f32>),
    Integer {
        prototype: IntegerStorage,
        values: Vec<IntValue>,
    },
    Logical(Vec<u8>),
    ComplexF64(Vec<(f64, f64)>),
    ComplexF32(Vec<(f32, f32)>),
    IntegerComplex {
        real_prototype: IntegerStorage,
        imag_prototype: IntegerStorage,
        real_values: Vec<IntValue>,
        imag_values: Vec<IntValue>,
    },
    Char(Vec<char>),
}

fn heterogeneous_uniform_output() -> RuntimeError {
    arrayfun_error_with_detail(
        &ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
        "callback outputs with UniformOutput=true must have the same data type on every invocation",
    )
}

impl UniformCollector {
    fn push(&mut self, value: &Value) -> BuiltinResult<()> {
        match self {
            UniformCollector::Pending => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    *self = UniformCollector::Logical(vec![b as u8]);
                    Ok(())
                }
                ClassifiedValue::F64(value) => {
                    *self = UniformCollector::F64(vec![value]);
                    Ok(())
                }
                ClassifiedValue::F32(value) => {
                    *self = UniformCollector::F32(vec![value]);
                    Ok(())
                }
                ClassifiedValue::Integer(value) => {
                    *self = UniformCollector::Integer {
                        prototype: IntegerStorage::from_scalar(value.clone()),
                        values: vec![value],
                    };
                    Ok(())
                }
                ClassifiedValue::ComplexF64(value) => {
                    *self = UniformCollector::ComplexF64(vec![value]);
                    Ok(())
                }
                ClassifiedValue::ComplexF32(value) => {
                    *self = UniformCollector::ComplexF32(vec![value]);
                    Ok(())
                }
                ClassifiedValue::IntegerComplex(real, imag) => {
                    *self = UniformCollector::IntegerComplex {
                        real_prototype: IntegerStorage::from_scalar(real.clone()),
                        imag_prototype: IntegerStorage::from_scalar(imag.clone()),
                        real_values: vec![real],
                        imag_values: vec![imag],
                    };
                    Ok(())
                }
                ClassifiedValue::Char(ch) => {
                    *self = UniformCollector::Char(vec![ch]);
                    Ok(())
                }
            },
            UniformCollector::Logical(bits) => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    bits.push(b as u8);
                    Ok(())
                }
                _ => Err(heterogeneous_uniform_output()),
            },
            UniformCollector::F64(data) => match classify_value(value)? {
                ClassifiedValue::F64(value) => {
                    data.push(value);
                    Ok(())
                }
                ClassifiedValue::ComplexF64(value) => {
                    let mut entries: Vec<(f64, f64)> = std::mem::take(data)
                        .into_iter()
                        .map(|value| (value, 0.0))
                        .collect();
                    entries.push(value);
                    *self = UniformCollector::ComplexF64(entries);
                    Ok(())
                }
                _ => Err(heterogeneous_uniform_output()),
            },
            UniformCollector::F32(data) => match classify_value(value)? {
                ClassifiedValue::F32(value) => {
                    data.push(value);
                    Ok(())
                }
                ClassifiedValue::ComplexF32(value) => {
                    let mut entries: Vec<(f32, f32)> = std::mem::take(data)
                        .into_iter()
                        .map(|value| (value, 0.0))
                        .collect();
                    entries.push(value);
                    *self = UniformCollector::ComplexF32(entries);
                    Ok(())
                }
                _ => Err(heterogeneous_uniform_output()),
            },
            UniformCollector::Integer { prototype, values } => match classify_value(value)? {
                ClassifiedValue::Integer(value) => {
                    let candidate = IntegerStorage::from_scalar(value.clone());
                    if candidate.numeric_dtype() != prototype.numeric_dtype() {
                        return Err(heterogeneous_uniform_output());
                    }
                    values.push(value);
                    Ok(())
                }
                _ => Err(heterogeneous_uniform_output()),
            },
            UniformCollector::ComplexF64(data) => match classify_value(value)? {
                ClassifiedValue::F64(value) => {
                    data.push((value, 0.0));
                    Ok(())
                }
                ClassifiedValue::ComplexF64(value) => {
                    data.push(value);
                    Ok(())
                }
                _ => Err(heterogeneous_uniform_output()),
            },
            UniformCollector::ComplexF32(data) => match classify_value(value)? {
                ClassifiedValue::F32(value) => {
                    data.push((value, 0.0));
                    Ok(())
                }
                ClassifiedValue::ComplexF32(value) => {
                    data.push(value);
                    Ok(())
                }
                _ => Err(heterogeneous_uniform_output()),
            },
            UniformCollector::IntegerComplex {
                real_prototype,
                imag_prototype,
                real_values,
                imag_values,
            } => match classify_value(value)? {
                ClassifiedValue::IntegerComplex(real, imag) => {
                    if IntegerStorage::from_scalar(real.clone()).numeric_dtype()
                        != real_prototype.numeric_dtype()
                        || IntegerStorage::from_scalar(imag.clone()).numeric_dtype()
                            != imag_prototype.numeric_dtype()
                    {
                        return Err(heterogeneous_uniform_output());
                    }
                    real_values.push(real);
                    imag_values.push(imag);
                    Ok(())
                }
                _ => Err(heterogeneous_uniform_output()),
            },
            UniformCollector::Char(chars) => match classify_value(value)? {
                ClassifiedValue::Char(ch) => {
                    chars.push(ch);
                    Ok(())
                }
                _ => Err(heterogeneous_uniform_output()),
            },
        }
    }

    fn finish(self, shape: &[usize]) -> BuiltinResult<Value> {
        match self {
            UniformCollector::Pending => {
                let total = shape.iter().product();
                let tensor = Tensor::new(vec![0.0; total], shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::F64(data) => {
                let tensor = Tensor::new(data, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::F32(data) => {
                let tensor = Tensor::from_f32(data, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::Integer { prototype, values } => {
                let storage = prototype
                    .from_same_class_values(values)
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                let tensor = Tensor::new_integer(storage, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::Logical(bits) => {
                let logical = LogicalArray::new(bits, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::LogicalArray(logical))
            }
            UniformCollector::ComplexF64(entries) => {
                let tensor = ComplexTensor::new(entries, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::ComplexTensor(tensor))
            }
            UniformCollector::ComplexF32(entries) => {
                let tensor = ComplexTensor::from_f32(entries, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::ComplexTensor(tensor))
            }
            UniformCollector::IntegerComplex {
                real_prototype,
                imag_prototype,
                real_values,
                imag_values,
            } => {
                let real = real_prototype
                    .from_same_class_values(real_values)
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                let imag = imag_prototype
                    .from_same_class_values(imag_values)
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                let storage = IntegerComplexStorage::new(real, imag)
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                let tensor = ComplexTensor::new_integer(storage, shape.to_vec())
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::ComplexTensor(tensor))
            }
            UniformCollector::Char(chars) => {
                let normalized_shape = if shape.is_empty() {
                    vec![1, 1]
                } else {
                    shape.to_vec()
                };

                if normalized_shape.len() > 2 {
                    return Err(arrayfun_error_with_detail(
                        &ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
                        "character outputs with UniformOutput=true must be 2-D",
                    ));
                }

                let rows = normalized_shape.first().copied().unwrap_or(1);
                let cols = normalized_shape.get(1).copied().unwrap_or(1);
                let expected = rows.checked_mul(cols).ok_or_else(|| {
                    arrayfun_internal("arrayfun: character output size exceeds platform limits")
                })?;

                if expected != chars.len() {
                    return Err(arrayfun_error_with_detail(
                        &ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
                        "callback returned the wrong number of characters",
                    ));
                }

                let mut row_major = vec!['\0'; expected];
                for col in 0..cols {
                    for row in 0..rows {
                        let col_major_idx = row + col * rows;
                        let row_major_idx = row * cols + col;
                        row_major[row_major_idx] = chars[col_major_idx];
                    }
                }

                let array = CharArray::new(row_major, rows, cols)
                    .map_err(|e| arrayfun_internal(format!("arrayfun: {e}")))?;
                Ok(Value::CharArray(array))
            }
        }
    }
}

enum ClassifiedValue {
    Logical(bool),
    F64(f64),
    F32(f32),
    Integer(IntValue),
    ComplexF64((f64, f64)),
    ComplexF32((f32, f32)),
    IntegerComplex(IntValue, IntValue),
    Char(char),
}

fn classify_value(value: &Value) -> BuiltinResult<ClassifiedValue> {
    match value {
        Value::Bool(b) => Ok(ClassifiedValue::Logical(*b)),
        Value::LogicalArray(la) if la.len() == 1 => Ok(ClassifiedValue::Logical(la.data[0] != 0)),
        Value::Int(value) => Ok(ClassifiedValue::Integer(value.clone())),
        Value::Num(n) => Ok(ClassifiedValue::F64(*n)),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            match t.numeric_value_at(0).ok_or_else(|| {
                arrayfun_internal("arrayfun: scalar tensor has no numeric storage value")
            })? {
                NumericScalar::F64(value) => Ok(ClassifiedValue::F64(value)),
                NumericScalar::F32(value) => Ok(ClassifiedValue::F32(value)),
                value => Ok(ClassifiedValue::Integer(
                    value.into_int_value().ok_or_else(|| {
                        arrayfun_internal("arrayfun: integer scalar classification failed")
                    })?,
                )),
            }
        }
        Value::Complex(re, im) => Ok(ClassifiedValue::ComplexF64((*re, *im))),
        Value::ComplexTensor(t) if tensor::is_scalar_complex_tensor(t) => {
            let (real, imag) = t.numeric_value_at(0).ok_or_else(|| {
                arrayfun_internal("arrayfun: scalar complex tensor has no storage value")
            })?;
            match (real, imag) {
                (NumericScalar::F64(real), NumericScalar::F64(imag)) => {
                    Ok(ClassifiedValue::ComplexF64((real, imag)))
                }
                (NumericScalar::F32(real), NumericScalar::F32(imag)) => {
                    Ok(ClassifiedValue::ComplexF32((real, imag)))
                }
                (real, imag) => Ok(ClassifiedValue::IntegerComplex(
                    real.into_int_value().ok_or_else(|| {
                        arrayfun_internal(
                            "arrayfun: complex callback result has inconsistent component classes",
                        )
                    })?,
                    imag.into_int_value().ok_or_else(|| {
                        arrayfun_internal(
                            "arrayfun: complex callback result has inconsistent component classes",
                        )
                    })?,
                )),
            }
        }
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            let ch = ca.data.first().copied().unwrap_or('\0');
            Ok(ClassifiedValue::Char(ch))
        }
        other => Err(arrayfun_error_with_detail(
            &ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
            format!(
                "callback must return scalar numeric, logical, character, or complex values for UniformOutput=true (got {other:?})"
            ),
        )),
    }
}

fn make_error_struct(raw_error: &str, linear_index: usize) -> Value {
    let (identifier, message) = split_error_message(raw_error);
    let mut st = runmat_builtins::StructValue::new();
    st.fields
        .insert("identifier".to_string(), Value::String(identifier));
    st.fields
        .insert("message".to_string(), Value::String(message));
    st.fields
        .insert("index".to_string(), Value::Num((linear_index + 1) as f64));
    Value::Struct(st)
}

fn split_error_message(raw: &str) -> (String, String) {
    let trimmed = raw.trim();
    let mut indices = trimmed.match_indices(':');
    if let Some((_, _)) = indices.next() {
        if let Some((second_idx, _)) = indices.next() {
            let identifier = trimmed[..second_idx].trim().to_string();
            let message = trimmed[second_idx + 1..].trim().to_string();
            if !identifier.is_empty() && identifier.contains(':') {
                return (
                    identifier,
                    if message.is_empty() {
                        trimmed.to_string()
                    } else {
                        message
                    },
                );
            }
        } else if trimmed.len() >= 7
            && (trimmed[..7].eq_ignore_ascii_case("matlab:")
                || trimmed[..7].eq_ignore_ascii_case("runmat:"))
        {
            return (trimmed.to_string(), String::new());
        }
    }
    (
        "RunMat:arrayfun:FunctionError".to_string(),
        trimmed.to_string(),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        ComplexTensor, IntegerComplexStorage, IntegerStorage, ResolveContext, Tensor, Type, Value,
    };
    use std::sync::Arc;

    fn call(func: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(arrayfun_builtin(func, rest))
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    #[test]
    fn typed_integer_input_elements_are_extracted_from_exact_storage() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("integer tensor");
        let input = ArrayInput {
            data: ArrayData::Tensor(tensor),
            shape: vec![1, 1],
            strides: vec![1, 1],
        };

        assert_eq!(
            input.value_at(0, &[1, 1]).expect("value"),
            Value::Int(runmat_builtins::IntValue::U64(9_007_199_254_740_993))
        );
    }

    #[test]
    fn uniform_classifier_preserves_typed_integer_tensor_storage() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("integer tensor");

        match classify_value(&Value::Tensor(tensor)).expect("classify") {
            ClassifiedValue::Integer(value) => {
                assert_eq!(value, IntValue::U64(9_007_199_254_740_993));
            }
            _ => panic!("expected integer classification"),
        }
    }

    #[test]
    fn uniform_classifier_preserves_wide_integer_scalar() {
        match classify_value(&Value::Int(IntValue::I64(i64::MIN))).expect("classify") {
            ClassifiedValue::Integer(value) => assert_eq!(value, IntValue::I64(i64::MIN)),
            _ => panic!("expected integer classification"),
        }
    }

    #[test]
    fn uniform_collector_preserves_every_integer_class() {
        let cases = [
            (
                IntValue::I8(i8::MIN),
                IntValue::I8(i8::MAX),
                IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            ),
            (
                IntValue::I16(i16::MIN),
                IntValue::I16(i16::MAX),
                IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            ),
            (
                IntValue::I32(i32::MIN),
                IntValue::I32(i32::MAX),
                IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            ),
            (
                IntValue::I64(i64::MIN),
                IntValue::I64(i64::MAX),
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            ),
            (
                IntValue::U8(u8::MIN),
                IntValue::U8(u8::MAX),
                IntegerStorage::U8(vec![u8::MIN, u8::MAX]),
            ),
            (
                IntValue::U16(u16::MIN),
                IntValue::U16(u16::MAX),
                IntegerStorage::U16(vec![u16::MIN, u16::MAX]),
            ),
            (
                IntValue::U32(u32::MIN),
                IntValue::U32(u32::MAX),
                IntegerStorage::U32(vec![u32::MIN, u32::MAX]),
            ),
            (
                IntValue::U64(9_007_199_254_740_993),
                IntValue::U64(u64::MAX),
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            ),
        ];

        for (first, second, expected) in cases {
            let mut collector = UniformCollector::Pending;
            collector.push(&Value::Int(first)).expect("first value");
            collector.push(&Value::Int(second)).expect("second value");
            let Value::Tensor(tensor) = collector.finish(&[1, 2]).expect("finish") else {
                panic!("expected integer tensor");
            };
            assert_eq!(tensor.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn uniform_collector_rejects_mixed_integer_classes() {
        let mut collector = UniformCollector::Pending;
        collector
            .push(&Value::Int(IntValue::U8(1)))
            .expect("first value");
        let error = collector
            .push(&Value::Int(IntValue::U16(2)))
            .expect_err("mixed integer classes must fail");
        assert_eq!(
            error.identifier(),
            ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE.identifier
        );
    }

    #[test]
    fn arrayfun_preserves_every_integer_input_and_uniform_output_class() {
        for (storage, callback) in [
            (IntegerStorage::I8(vec![i8::MIN, i8::MAX]), "int8"),
            (IntegerStorage::I16(vec![i16::MIN, i16::MAX]), "int16"),
            (IntegerStorage::I32(vec![i32::MIN, i32::MAX]), "int32"),
            (IntegerStorage::I64(vec![i64::MIN, i64::MAX]), "int64"),
            (IntegerStorage::U8(vec![0, u8::MAX]), "uint8"),
            (IntegerStorage::U16(vec![0, u16::MAX]), "uint16"),
            (IntegerStorage::U32(vec![0, u32::MAX]), "uint32"),
            (
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                "uint64",
            ),
        ] {
            let result = call(
                Value::FunctionHandle(callback.to_string()),
                vec![Value::Tensor(
                    Tensor::new_integer(storage.clone(), vec![1, 2]).expect("integer input"),
                )],
            )
            .expect("arrayfun integer identity cast");
            let Value::Tensor(result) = result else {
                panic!("expected typed integer tensor");
            };
            assert_eq!(result.integer_storage(), Some(&storage));
        }
    }

    #[test]
    fn arrayfun_preserves_native_single_and_complex_single_scalars() {
        let input =
            ArrayData::Tensor(Tensor::from_f32(vec![0.25, -1.5], vec![1, 2]).expect("single"));
        let first = input.value_at(0).expect("first single");
        let Value::Tensor(first) = first else {
            panic!("single element must remain a scalar tensor");
        };
        assert_eq!(first.as_f32_slice(), Some(&[0.25][..]));

        let mut real_collector = UniformCollector::Pending;
        real_collector
            .push(&Value::Tensor(first))
            .expect("single result");
        real_collector
            .push(&Value::Tensor(
                Tensor::from_f32(vec![-1.5], vec![1, 1]).expect("single scalar"),
            ))
            .expect("second single result");
        let Value::Tensor(real) = real_collector.finish(&[1, 2]).expect("single output") else {
            panic!("expected native single output");
        };
        assert_eq!(real.as_f32_slice(), Some(&[0.25, -1.5][..]));

        let mut complex_collector = UniformCollector::Pending;
        for value in [(1.25_f32, -2.5_f32), (3.5_f32, 4.75_f32)] {
            complex_collector
                .push(&Value::ComplexTensor(
                    ComplexTensor::from_f32(vec![value], vec![1, 1])
                        .expect("complex single scalar"),
                ))
                .expect("complex single result");
        }
        let Value::ComplexTensor(complex) =
            complex_collector.finish(&[1, 2]).expect("complex output")
        else {
            panic!("expected complex single output");
        };
        assert_eq!(
            complex.as_f32_slice(),
            Some(&[(1.25, -2.5), (3.5, 4.75)][..])
        );
    }

    #[test]
    fn uniform_collector_preserves_exact_complex_integer_storage() {
        let mut collector = UniformCollector::Pending;
        for (real, imag) in [
            (
                IntValue::I64(9_007_199_254_740_993),
                IntValue::I64(-9_007_199_254_740_993),
            ),
            (IntValue::I64(i64::MAX), IntValue::I64(i64::MIN)),
        ] {
            let storage = IntegerComplexStorage::new(
                IntegerStorage::from_scalar(real),
                IntegerStorage::from_scalar(imag),
            )
            .expect("integer complex scalar");
            collector
                .push(&Value::ComplexTensor(
                    ComplexTensor::new_integer(storage, vec![1, 1])
                        .expect("integer complex tensor"),
                ))
                .expect("integer complex result");
        }
        let Value::ComplexTensor(output) =
            collector.finish(&[1, 2]).expect("integer complex output")
        else {
            panic!("expected integer complex tensor");
        };
        let storage = output.integer_storage().expect("exact component storage");
        assert_eq!(
            storage.real,
            IntegerStorage::I64(vec![9_007_199_254_740_993, i64::MAX])
        );
        assert_eq!(
            storage.imag,
            IntegerStorage::I64(vec![-9_007_199_254_740_993, i64::MIN])
        );
    }

    #[test]
    fn uniform_collector_rejects_different_noninteger_classes() {
        for second in [
            Value::Num(1.0),
            Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single")),
            Value::CharArray(CharArray::new(vec!['1'], 1, 1).expect("char")),
        ] {
            let mut collector = UniformCollector::Pending;
            collector.push(&Value::Bool(true)).expect("logical");
            let error = collector
                .push(&second)
                .expect_err("heterogeneous uniform result must reject");
            assert_eq!(
                error.identifier(),
                ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE.identifier
            );
        }
    }

    #[test]
    fn arrayfun_rejects_every_typed_integer_uniform_output_control() {
        let input = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("input"));
        for storage in [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ] {
            for control in [
                Value::Int(storage.value_at(0).expect("scalar")),
                Value::Tensor(
                    Tensor::new_integer(storage.clone(), vec![1, 1]).expect("typed control"),
                ),
            ] {
                let error = call(
                    Value::FunctionHandle("sin".to_string()),
                    vec![input.clone(), Value::from("UniformOutput"), control],
                )
                .expect_err("typed integer control must reject");
                assert_eq!(
                    error.identifier(),
                    ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION.identifier
                );
            }
        }
    }

    #[test]
    fn arrayfun_text_callable_and_host_scalar_expansion_are_mode_gated() {
        let input = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("input"));
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = call(Value::from("sin"), vec![input.clone()])
                .expect_err("text callable must reject in compatible mode");
            assert_eq!(
                error.identifier(),
                ARRAYFUN_TEXT_CALLABLE_EXTENSION.error_identifier
            );
            let error = call(
                Value::FunctionHandle("atan2".to_string()),
                vec![input.clone(), Value::Num(1.0)],
            )
            .expect_err("host scalar expansion must reject in compatible mode");
            assert_eq!(
                error.identifier(),
                ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION.error_identifier
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(call(Value::from("sin"), vec![input.clone()]).is_ok());
            assert!(call(
                Value::FunctionHandle("atan2".to_string()),
                vec![input, Value::Num(1.0)],
            )
            .is_ok());
        }
    }

    #[test]
    fn arrayfun_error_struct_has_only_documented_fields() {
        let Value::Struct(error) = make_error_struct("RunMat:test:Failure: detail", 4) else {
            panic!("expected error struct");
        };
        let mut fields: Vec<_> = error.fields.keys().map(String::as_str).collect();
        fields.sort_unstable();
        assert_eq!(fields, vec!["identifier", "index", "message"]);
        assert_eq!(error.fields.get("index"), Some(&Value::Num(5.0)));
    }

    #[test]
    fn uniform_classifier_reads_typed_complex_integer_tensor_storage_exactly() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I64(vec![9_007_199_254_740_993]),
            IntegerStorage::I64(vec![-9_007_199_254_740_993]),
        )
        .expect("complex integer storage");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex tensor");

        match classify_value(&Value::ComplexTensor(tensor)).expect("classify") {
            ClassifiedValue::IntegerComplex(re, im) => {
                assert_eq!(re, IntValue::I64(9_007_199_254_740_993));
                assert_eq!(im, IntValue::I64(-9_007_199_254_740_993));
            }
            _ => panic!("expected exact integer-complex classification"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_basic_sin() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let expected: Vec<f64> = values(&tensor).into_iter().map(f64::sin).collect();
        let result = call(
            Value::FunctionHandle("sin".to_string()),
            vec![Value::Tensor(tensor.clone())],
        )
        .expect("arrayfun");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(values(&out), expected);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn arrayfun_semantic_function_handle_uses_semantic_invoker() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 78);
                assert_eq!(requested_outputs, 1);
                let [Value::Num(value)] = args else {
                    panic!("expected scalar numeric argument, got {args:?}");
                };
                let value = *value;
                Box::pin(async move { Ok(Value::Num(value + 10.0)) })
            },
        )));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        let handle = Value::BoundFunctionHandle {
            name: "arrayfun_target".to_string(),
            function: 78,
        };

        let result = call(handle, vec![Value::Tensor(tensor)]).expect("semantic arrayfun");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(values(&out), vec![11.0, 12.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn arrayfun_name_only_callback_uses_semantic_resolver() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_arrayfun_target").then_some(80)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 80);
                assert_eq!(requested_outputs, 1);
                let [Value::Num(value)] = args else {
                    panic!("expected scalar numeric argument, got {args:?}");
                };
                let value = *value;
                Box::pin(async move { Ok(Value::Num(value + 20.0)) })
            }),
        ));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");

        let result = call(
            Value::String("resolved_arrayfun_target".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect("resolved name-only arrayfun");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(values(&out), vec![21.0, 22.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn arrayfun_qualified_text_callback_classifies_as_external_name() {
        let callable =
            Callable::from_text("pkg.callback").expect("qualified arrayfun callback should parse");
        assert!(matches!(
            callable,
            Callable::ExternalName { name } if name == "pkg.callback"
        ));
    }

    #[test]
    fn arrayfun_external_handle_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.callback").then_some(87)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 87);
                assert_eq!(requested_outputs, 1);
                let [Value::Num(value)] = args else {
                    panic!("expected scalar numeric argument, got {args:?}");
                };
                let value = *value;
                Box::pin(async move { Ok(Value::Num(value + 30.0)) })
            }),
        ));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");

        let result = call(
            Value::ExternalFunctionHandle("pkg.callback".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect("resolved external-handle arrayfun");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(values(&out), vec![31.0, 32.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn arrayfun_single_segment_external_handle_uses_runtime_name_resolution() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "callback").then_some(887)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 887);
                assert_eq!(requested_outputs, 1);
                let [Value::Num(value)] = args else {
                    panic!("expected scalar numeric argument, got {args:?}");
                };
                let value = *value;
                Box::pin(async move { Ok(Value::Num(value + 40.0)) })
            }),
        ));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");

        let result = call(
            Value::ExternalFunctionHandle("callback".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect("single-segment external-handle arrayfun should resolve via runtime-name policy");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(values(&out), vec![41.0, 42.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn arrayfun_external_handle_prefers_semantic_handle_binding_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.callback").then_some(87)
            })));
        let callable =
            Callable::from_function(Value::ExternalFunctionHandle("pkg.callback".to_string()))
                .expect("external handle should parse");
        assert!(matches!(
            callable,
            Callable::Closure(Closure {
                function_name,
                bound_function: Some(87),
                ..
            }) if function_name == "pkg.callback"
        ));
    }

    #[test]
    fn arrayfun_name_only_closure_prefers_semantic_handle_binding_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.callback").then_some(187)
            })));
        let callable = Callable::from_function(Value::Closure(Closure {
            function_name: "pkg.callback".to_string(),
            bound_function: None,
            captures: vec![Value::Num(5.0)],
        }))
        .expect("closure callback should parse");
        assert!(matches!(
            callable,
            Callable::Closure(Closure {
                function_name,
                bound_function: Some(187),
                captures
            }) if function_name == "pkg.callback" && captures == vec![Value::Num(5.0)]
        ));
    }

    #[test]
    fn arrayfun_name_only_closure_call_uses_semantic_resolver_when_unbound() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.callback").then_some(287)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 287);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(5.0), Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(9.0)) })
            }),
        ));
        let callable = Callable::Closure(Closure {
            function_name: "pkg.callback".to_string(),
            bound_function: None,
            captures: vec![Value::Num(5.0)],
        });
        let value = block_on(callable.call(&[Value::Num(4.0)])).expect("closure call");
        assert_eq!(value, Value::Num(9.0));
    }

    #[test]
    fn arrayfun_external_handle_errors_as_undefined_when_unresolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_| None)));
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");

        let err = call(
            Value::ExternalFunctionHandle("pkg.callback".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect_err("unresolved external callback should error");
        assert_eq!(
            err.identifier(),
            ARRAYFUN_ERROR_UNDEFINED_FUNCTION.identifier,
            "unexpected error: {}",
            err.message()
        );
        assert!(
            err.message().contains("ExternalName(QualifiedName"),
            "unexpected error: {err:?}"
        );
        assert!(
            !err.message().contains("Undefined function 'pkg.callback'"),
            "well-formed external callback should report typed identity: {err:?}"
        );
    }

    #[test]
    fn arrayfun_malformed_external_handle_errors_as_undefined_when_unresolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_| None)));
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");

        let err = call(
            Value::ExternalFunctionHandle("pkg..callback".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect_err("malformed unresolved external callback should error");
        assert_eq!(
            err.identifier(),
            ARRAYFUN_ERROR_UNDEFINED_FUNCTION.identifier,
            "unexpected error: {}",
            err.message()
        );
        assert!(
            err.message().contains("pkg..callback"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn arrayfun_type_tracks_function_returns() {
        let func = Type::Function {
            params: vec![Type::Num],
            returns: Box::new(Type::Num),
        };
        assert_eq!(
            arrayfun_type(&[func, Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::tensor()
        );
    }

    #[test]
    fn arrayfun_type_uses_logical_returns() {
        let func = Type::Function {
            params: vec![Type::Num],
            returns: Box::new(Type::Bool),
        };
        assert_eq!(
            arrayfun_type(&[func, Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::logical()
        );
    }

    #[test]
    fn arrayfun_type_with_text_args_stays_unknown() {
        let func = Type::Function {
            params: vec![Type::Num],
            returns: Box::new(Type::Num),
        };
        assert_eq!(
            arrayfun_type(
                &[func, Type::tensor(), Type::String, Type::Bool],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Unknown
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_additional_scalar_argument() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.5, 1.0, -1.0], vec![3, 1]).unwrap();
        let expected: Vec<f64> = values(&tensor).into_iter().map(|y| y.atan2(1.0)).collect();
        let result = call(
            Value::FunctionHandle("atan2".to_string()),
            vec![Value::Tensor(tensor), Value::Num(1.0)],
        )
        .expect("arrayfun");
        match result {
            Value::Tensor(out) => {
                assert_eq!(values(&out), expected);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_uniform_false_returns_cell() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let expected: Vec<Value> = values(&tensor)
            .into_iter()
            .map(|x| Value::Num(x.sin()))
            .collect();
        let result = call(
            Value::FunctionHandle("sin".to_string()),
            vec![
                Value::Tensor(tensor),
                Value::String("UniformOutput".into()),
                Value::Bool(false),
            ],
        )
        .expect("arrayfun");
        let Value::Cell(cell) = result else {
            panic!("expected cell, got something else");
        };
        assert_eq!(cell.rows, 2);
        assert_eq!(cell.cols, 1);
        for (row, value) in expected.iter().enumerate() {
            assert_eq!(cell.get(row, 0).unwrap(), *value);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_uniform_output_option_identifier() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = call(
            Value::FunctionHandle("sin".to_string()),
            vec![
                Value::Tensor(tensor),
                Value::String("UniformOutput".into()),
                Value::String("maybe".into()),
            ],
        )
        .expect_err("expected invalid uniform output option");
        assert_eq!(
            err.identifier(),
            ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_unknown_name_value_identifier() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = call(
            Value::FunctionHandle("sin".to_string()),
            vec![
                Value::Tensor(tensor),
                Value::String("MysteryFlag".into()),
                Value::Bool(true),
            ],
        )
        .expect_err("expected unknown name-value error");
        assert_eq!(err.identifier(), ARRAYFUN_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_size_mismatch_errors() {
        let taller = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let shorter = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let err = call(
            Value::FunctionHandle("sin".to_string()),
            vec![Value::Tensor(taller), Value::Tensor(shorter)],
        )
        .expect_err("expected size mismatch error");
        let err = err.to_string();
        assert!(
            err.contains("does not match"),
            "expected size mismatch error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_error_handler_recovers() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let handler = Value::Closure(Closure {
            function_name: "__arrayfun_test_handler".into(),
            bound_function: None,
            captures: vec![Value::Num(42.0)],
        });
        let result = call(
            Value::FunctionHandle("nonexistent_builtin".into()),
            vec![
                Value::Tensor(tensor),
                Value::String("ErrorHandler".into()),
                handler,
            ],
        )
        .expect("arrayfun error handler");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(values(&out), vec![42.0, 42.0, 42.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_error_without_handler_propagates_identifier() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = call(
            Value::FunctionHandle("nonexistent_builtin".into()),
            vec![Value::Tensor(tensor)],
        )
        .expect_err("expected unresolved function error");
        assert_eq!(
            err.identifier(),
            ARRAYFUN_ERROR_UNDEFINED_FUNCTION.identifier,
            "unexpected error: {}",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_uniform_logical_result() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 0.0, f64::INFINITY], vec![4, 1]).unwrap();
        let result = call(
            Value::FunctionHandle("isfinite".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect("arrayfun isfinite");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![4, 1]);
                assert_eq!(la.data, vec![1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_uniform_character_result() {
        let tensor = Tensor::new(vec![65.0, 66.0, 67.0], vec![1, 3]).unwrap();
        let result = call(
            Value::FunctionHandle("char".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect("arrayfun char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 3);
                assert_eq!(ca.data, vec!['A', 'B', 'C']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn arrayfun_gpu_options_are_mode_gated() {
        test_support::with_test_provider(|provider| {
            let handle = gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![0.0, 1.0], vec![2, 1]).expect("input"),
            )
            .expect("upload");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = call(
                Value::FunctionHandle("sin".to_string()),
                vec![
                    Value::GpuTensor(handle.clone()),
                    Value::from("UniformOutput"),
                    Value::Bool(false),
                ],
            )
            .expect_err("gpu options must reject in compatible mode");
            assert_eq!(
                error.identifier(),
                ARRAYFUN_GPU_OPTIONS_EXTENSION.error_identifier
            );
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn arrayfun_provider_fallback_preserves_every_integer_class() {
        test_support::with_test_provider(|provider| {
            for (storage, callback) in [
                (IntegerStorage::I8(vec![i8::MIN, i8::MAX]), "int8"),
                (IntegerStorage::I16(vec![i16::MIN, i16::MAX]), "int16"),
                (IntegerStorage::I32(vec![i32::MIN, i32::MAX]), "int32"),
                (IntegerStorage::I64(vec![i64::MIN, i64::MAX]), "int64"),
                (IntegerStorage::U8(vec![0, u8::MAX]), "uint8"),
                (IntegerStorage::U16(vec![0, u16::MAX]), "uint16"),
                (IntegerStorage::U32(vec![0, u32::MAX]), "uint32"),
                (
                    IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                    "uint64",
                ),
            ] {
                let input = Tensor::new_integer(storage.clone(), vec![1, 2]).expect("input");
                let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
                let result = call(
                    Value::FunctionHandle(callback.to_string()),
                    vec![Value::GpuTensor(handle.clone())],
                )
                .expect("provider arrayfun");
                let Value::GpuTensor(output) = result else {
                    panic!("expected resident output");
                };
                let gathered =
                    test_support::gather(Value::GpuTensor(output.clone())).expect("gather output");
                assert_eq!(gathered.integer_storage(), Some(&storage));
                let _ = provider.free(&handle);
                let _ = provider.free(&output);
            }
        });
    }

    #[test]
    fn arrayfun_gpu_overload_uses_documented_compatible_size_expansion_exactly() {
        test_support::with_test_provider(|provider| {
            let row_storage = IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                9_007_199_254_740_994,
                9_007_199_254_740_995,
            ]);
            let row = Tensor::new_integer(row_storage, vec![1, 3]).expect("row");
            let row_handle = gpu_helpers::upload_tensor(provider, &row).expect("upload row");
            let column =
                Tensor::new_integer(IntegerStorage::U64(vec![10, 20]), vec![2, 1]).expect("column");
            let result = call(
                Value::FunctionHandle("plus".to_string()),
                vec![Value::GpuTensor(row_handle.clone()), Value::Tensor(column)],
            )
            .expect("compatible gpu arrayfun");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident output");
            };
            let gathered = test_support::gather(Value::GpuTensor(output.clone())).expect("gather");
            assert_eq!(gathered.shape, vec![2, 3]);
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    9_007_199_254_741_003,
                    9_007_199_254_741_013,
                    9_007_199_254_741_004,
                    9_007_199_254_741_014,
                    9_007_199_254_741_005,
                    9_007_199_254_741_015,
                ]))
            );
            let _ = provider.free(&row_handle);
            let _ = provider.free(&output);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_uniform_false_gpu_returns_cell() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = call(
                Value::FunctionHandle("sin".to_string()),
                vec![
                    Value::GpuTensor(handle),
                    Value::String("UniformOutput".into()),
                    Value::Bool(false),
                ],
            )
            .expect("arrayfun");
            match result {
                Value::Cell(cell) => {
                    assert_eq!(cell.rows, 2);
                    assert_eq!(cell.cols, 1);
                    let first = cell.get(0, 0).expect("first cell");
                    let second = cell.get(1, 0).expect("second cell");
                    match (first, second) {
                        (Value::Num(a), Value::Num(b)) => {
                            assert!((a - 0.0f64.sin()).abs() < 1e-12);
                            assert!((b - 1.0f64.sin()).abs() < 1e-12);
                        }
                        other => panic!("expected numeric cells, got {other:?}"),
                    }
                }
                other => panic!("expected cell, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn arrayfun_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = call(
                Value::FunctionHandle("sin".to_string()),
                vec![Value::GpuTensor(handle)],
            )
            .expect("arrayfun");
            match result {
                Value::GpuTensor(gpu) => {
                    let gathered = test_support::gather(Value::GpuTensor(gpu.clone())).unwrap();
                    let expected: Vec<f64> = values(&tensor).into_iter().map(f64::sin).collect();
                    assert_eq!(values(&gathered), expected);
                    let _ = provider.free(&gpu);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn arrayfun_wgpu_sin_matches_cpu() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };

        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = call(
            Value::FunctionHandle("sin".into()),
            vec![Value::GpuTensor(handle.clone())],
        )
        .expect("arrayfun sin gpu");
        let Value::GpuTensor(out_handle) = result else {
            panic!("expected GPU tensor result");
        };
        let gathered = test_support::gather(Value::GpuTensor(out_handle.clone())).unwrap();
        let expected: Vec<f64> = values(&tensor).into_iter().map(f64::sin).collect();
        assert_eq!(gathered.shape, tensor.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (actual, expect) in values(&gathered).iter().zip(expected.iter()) {
            assert!(
                (actual - expect).abs() < tol,
                "expected {expect}, got {actual}"
            );
        }
        let _ = provider.free(&handle);
        let _ = provider.free(&out_handle);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn arrayfun_wgpu_plus_matches_cpu() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };

        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
        let handle_a = gpu_helpers::upload_tensor(provider, &a).expect("upload a");
        let handle_b = gpu_helpers::upload_tensor(provider, &b).expect("upload b");
        let result = call(
            Value::FunctionHandle("plus".into()),
            vec![
                Value::GpuTensor(handle_a.clone()),
                Value::GpuTensor(handle_b.clone()),
            ],
        )
        .expect("arrayfun plus gpu");

        let Value::GpuTensor(out_handle) = result else {
            panic!("expected GPU tensor result");
        };
        let gathered = test_support::gather(Value::GpuTensor(out_handle.clone())).unwrap();
        let expected: Vec<f64> = values(&a)
            .iter()
            .zip(values(&b).iter())
            .map(|(x, y)| x + y)
            .collect();
        assert_eq!(gathered.shape, a.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (actual, expect) in values(&gathered).iter().zip(expected.iter()) {
            assert!(
                (actual - expect).abs() < tol,
                "expected {expect}, got {actual}"
            );
        }
        let _ = provider.free(&handle_a);
        let _ = provider.free(&handle_b);
        let _ = provider.free(&out_handle);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn arrayfun_wgpu_fallback_preserves_every_integer_class() {
        let _guard = test_support::accel_test_lock();
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        for (storage, callback) in [
            (IntegerStorage::I8(vec![i8::MIN, i8::MAX]), "int8"),
            (IntegerStorage::I16(vec![i16::MIN, i16::MAX]), "int16"),
            (IntegerStorage::I32(vec![i32::MIN, i32::MAX]), "int32"),
            (IntegerStorage::I64(vec![i64::MIN, i64::MAX]), "int64"),
            (IntegerStorage::U8(vec![0, u8::MAX]), "uint8"),
            (IntegerStorage::U16(vec![0, u16::MAX]), "uint16"),
            (IntegerStorage::U32(vec![0, u32::MAX]), "uint32"),
            (
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                "uint64",
            ),
        ] {
            let handle = gpu_helpers::upload_tensor(
                provider,
                &Tensor::new_integer(storage.clone(), vec![1, 2]).expect("input"),
            )
            .expect("upload");
            let result = call(
                Value::FunctionHandle(callback.to_string()),
                vec![Value::GpuTensor(handle.clone())],
            )
            .expect("wgpu arrayfun");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident output");
            };
            let gathered = test_support::gather(Value::GpuTensor(output.clone())).expect("gather");
            assert_eq!(gathered.integer_storage(), Some(&storage));
            let _ = provider.free(&handle);
            let _ = provider.free(&output);
        }

        let row = Tensor::new_integer(
            IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                9_007_199_254_740_994,
                9_007_199_254_740_995,
            ]),
            vec![1, 3],
        )
        .expect("row");
        let row_handle = gpu_helpers::upload_tensor(provider, &row).expect("upload row");
        let column =
            Tensor::new_integer(IntegerStorage::U64(vec![10, 20]), vec![2, 1]).expect("column");
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            vec![Value::GpuTensor(row_handle.clone()), Value::Tensor(column)],
        )
        .expect("compatible wgpu arrayfun");
        let Value::GpuTensor(output) = result else {
            panic!("expected resident output");
        };
        let gathered = test_support::gather(Value::GpuTensor(output.clone())).expect("gather");
        assert_eq!(gathered.shape, vec![2, 3]);
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_741_003,
                9_007_199_254_741_013,
                9_007_199_254_741_004,
                9_007_199_254_741_014,
                9_007_199_254_741_005,
                9_007_199_254_741_015,
            ]))
        );
        let _ = provider.free(&row_handle);
        let _ = provider.free(&output);
    }

    #[test]
    fn arrayfun_metadata_classifies_integer_and_extension_forms() {
        assert_eq!(ARRAYFUN_INTEGER_CAPABILITIES.len(), 3);
        assert_eq!(
            ARRAYFUN_INTEGER_CAPABILITIES[1].output_class,
            BuiltinIntegerOutputClassRule::PreserveInput
        );
        assert_eq!(
            ARRAYFUN_INTEGER_CAPABILITIES[2].inputs[0].availability,
            BuiltinIntegerInputAvailability::Rejected
        );
        assert_eq!(
            ARRAYFUN_EXTENSIONS,
            [
                ARRAYFUN_TEXT_CALLABLE_EXTENSION,
                ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION,
                ARRAYFUN_GPU_OPTIONS_EXTENSION
            ]
        );
    }

    const ARRAYFUN_TEST_HELPER_ERRORS: [BuiltinErrorDescriptor; 0] = [];
    const ARRAYFUN_TEST_HELPER_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "out",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Helper output value.",
    }];
    const ARRAYFUN_TEST_HANDLER_INPUTS: [BuiltinParamDescriptor; 3] = [
        BuiltinParamDescriptor {
            name: "seed",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Seed value.",
        },
        BuiltinParamDescriptor {
            name: "err",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Error context placeholder.",
        },
        BuiltinParamDescriptor {
            name: "rest",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Variadic,
            default: None,
            description: "Additional values.",
        },
    ];
    const ARRAYFUN_TEST_HANDLER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
        [BuiltinSignatureDescriptor {
            label: "out = __arrayfun_test_handler(seed, err, ...)",
            inputs: &ARRAYFUN_TEST_HANDLER_INPUTS,
            outputs: &ARRAYFUN_TEST_HELPER_OUT,
        }];
    const ARRAYFUN_TEST_HANDLER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &ARRAYFUN_TEST_HANDLER_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::HiddenInternal,
        errors: &ARRAYFUN_TEST_HELPER_ERRORS,
    };

    #[runmat_macros::runtime_builtin(
        name = "__arrayfun_test_handler",
        descriptor(
            crate::builtins::acceleration::gpu::arrayfun::tests::ARRAYFUN_TEST_HANDLER_DESCRIPTOR
        ),
        type_resolver(arrayfun_type),
        builtin_path = "crate::builtins::acceleration::gpu::arrayfun::tests"
    )]
    async fn arrayfun_test_handler(
        seed: Value,
        _err: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        let _ = rest;
        Ok(seed)
    }
}
