//! MATLAB-compatible `double` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexTensor, LogicalArray, NumericStorage, SparseTensor, StringArray,
    SymbolicArray, Tensor, Value,
};

use crate::builtins::common::{
    gpu_helpers,
    random_args::keyword_of,
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
        FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
        ResidencyPolicy, ScalarType, ShapeRequirements,
    },
    tensor,
};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::double")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "double",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_double",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Casts inputs to float64. Providers without native float64 support gather to host; float64-capable providers keep results on device.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::double")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "double",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion treats double as an identity when the execution scalar type is already float64.",
};

const BUILTIN_NAME: &str = "double";

const DOUBLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double-precision output value.",
}];

const DOUBLE_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar/array value to convert.",
}];

const DOUBLE_INPUTS_X_LIKE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar/array value to convert.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Literal string \"like\".",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output class/device prototype.",
    },
];

const DOUBLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = double(X)",
        inputs: &DOUBLE_INPUTS_X,
        outputs: &DOUBLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = double(X, \"like\", prototype)",
        inputs: &DOUBLE_INPUTS_X_LIKE,
        outputs: &DOUBLE_OUTPUT,
    },
];

const DOUBLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOUBLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:double:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "double: invalid argument",
};

const DOUBLE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOUBLE.INVALID_INPUT",
    identifier: Some("RunMat:double:InvalidInput"),
    when: "Input value or prototype cannot be converted to double.",
    message: "double: invalid input",
};

const DOUBLE_ERROR_GPU_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOUBLE.GPU_UNSUPPORTED",
    identifier: Some("RunMat:double:GpuUnsupported"),
    when: "GPU output via \"like\" is requested but no compatible float64 provider is active.",
    message: "double: gpu output not supported",
};

const DOUBLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOUBLE.INTERNAL",
    identifier: Some("RunMat:double:Internal"),
    when: "Internal conversion, gather, or provider upload failed.",
    message: "double: internal error",
};

const DOUBLE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    DOUBLE_ERROR_INVALID_ARGUMENT,
    DOUBLE_ERROR_INVALID_INPUT,
    DOUBLE_ERROR_GPU_UNSUPPORTED,
    DOUBLE_ERROR_INTERNAL,
];

const DOUBLE_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "double-like-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "double(X, 'like', prototype) is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DoubleLikePrototypeExtension"),
};

const DOUBLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DOUBLE_LIKE_EXTENSION];

const DOUBLE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "R2026a documents conversion from every built-in integer class; conversion to IEEE binary64 can round wide int64/uint64 values.",
    }];

pub const DOUBLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = double(integer_X)",
        inputs: &DOUBLE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "All eight integer classes convert elementwise to double. Host conversion reads authoritative integer storage; resident conversion uses the owning provider when it can produce true F64 and otherwise gathers.",
    }];

pub const DOUBLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DOUBLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DOUBLE_ERRORS,
};

fn double_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    double_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn double_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn conversion_error(type_name: &str) -> RuntimeError {
    double_error_with_detail(
        &DOUBLE_ERROR_INVALID_INPUT,
        format!("conversion to double from {type_name} is not possible"),
    )
}

#[runtime_builtin(
    name = "double",
    category = "math/elementwise",
    summary = "Convert values to double precision.",
    keywords = "double,float64,cast,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::double::DOUBLE_DESCRIPTOR),
    extensions(DOUBLE_EXTENSIONS),
    integer_capabilities(DOUBLE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::double"
)]
async fn double_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    if matches!(template, OutputTemplate::Like(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DOUBLE_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let converted = match value {
        Value::Num(n) => Ok(Value::Num(n)),
        Value::Int(i) => Ok(Value::Num(i.to_f64())),
        Value::Bool(flag) => Ok(Value::Num(if flag { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) => double_from_tensor(tensor),
        Value::SparseTensor(sparse) => double_from_sparse_tensor(sparse),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        Value::ComplexTensor(tensor) => double_from_complex_tensor(tensor),
        Value::LogicalArray(array) => double_from_logical(array),
        Value::CharArray(chars) => double_from_char_array(chars),
        Value::GpuTensor(handle) => double_from_gpu(handle).await,
        Value::String(text) => Ok(Value::Num(parse_string_double(&text))),
        Value::StringArray(array) => double_from_string_array(array),
        Value::Symbolic(expr) => expr
            .numeric_constant_value()
            .map(Value::Num)
            .ok_or_else(|| conversion_error("sym")),
        Value::SymbolicArray(array) => double_from_symbolic_array(array),
        Value::Cell(_) => Err(conversion_error("cell")),
        Value::Struct(_) => Err(conversion_error("struct")),
        Value::ObjectArray(array) => Err(conversion_error(array.class_name())),
        Value::Object(obj) => Err(conversion_error(&obj.class_name)),
        Value::HandleObject(handle) => Err(conversion_error(&handle.class_name)),
        Value::Listener(_) => Err(conversion_error("event.listener")),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Err(conversion_error("function_handle")),
        Value::ClassRef(_) => Err(conversion_error("meta.class")),
        Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_) => Err(conversion_error("MException")),
        Value::Foreign(_) => Err(conversion_error("foreign")),
        Value::OutputList(_) => Err(conversion_error("OutputList")),
    }?;
    apply_output_template(converted, &template).await
}

fn parse_string_double(text: &str) -> f64 {
    text.trim().parse::<f64>().unwrap_or(f64::NAN)
}

fn double_from_string_array(array: StringArray) -> BuiltinResult<Value> {
    let shape = array.shape.clone();
    let data = array
        .data
        .iter()
        .map(|text| parse_string_double(text))
        .collect();
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))
}

fn double_from_logical(array: LogicalArray) -> BuiltinResult<Value> {
    let tensor = tensor::logical_to_tensor(&array)
        .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn double_from_symbolic_array(array: SymbolicArray) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(array.data.len());
    for expr in array.data {
        data.push(
            expr.numeric_constant_value()
                .ok_or_else(|| conversion_error("sym"))?,
        );
    }
    Tensor::new(data, array.shape)
        .map(Value::Tensor)
        .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))
}

fn double_from_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let values = tensor
        .into_numeric_storage()
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))?
        .materialize_f64();
    Tensor::from_numeric_storage(NumericStorage::F64(values), shape)
        .map(Value::Tensor)
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))
}

fn double_from_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    ComplexTensor::new(tensor.materialize_f64(), shape)
        .map(Value::ComplexTensor)
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))
}

fn double_from_sparse_tensor(sparse: SparseTensor) -> BuiltinResult<Value> {
    let values = sparse.materialize_f64();
    SparseTensor::new(
        sparse.rows,
        sparse.cols,
        sparse.col_ptrs,
        sparse.row_indices,
        values,
    )
    .map(Value::SparseTensor)
    .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))
}

fn double_from_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

async fn double_from_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = resolved_actual_double_owner(&handle);

    if let Some(provider) = provider {
        if provider.precision() == ProviderPrecision::F64
            && runmat_accelerate_api::handle_integer_type(&handle).is_none()
            && !runmat_accelerate_api::handle_is_logical(&handle)
            && runmat_accelerate_api::handle_storage(&handle)
                == runmat_accelerate_api::GpuTensorStorage::Real
        {
            match provider.unary_double(&handle).await {
                Ok(result) if valid_double_gpu_output(&result, &handle, provider, false) => {
                    return Ok(Value::GpuTensor(result));
                }
                Ok(result) => {
                    free_rejected_double_handle(&result, &[&handle]);
                }
                Err(err) => {
                    trace!("double: provider unary_double unavailable ({err}); falling back to host conversion");
                }
            }
        } else {
            trace!(
                "double: provider precision {:?} cannot store float64 values; gathering to host",
                provider.precision()
            );
        }
    }

    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))?;
    let value = match gathered {
        Value::Tensor(tensor) => double_from_tensor(tensor)?,
        Value::ComplexTensor(tensor) => double_from_complex_tensor(tensor)?,
        Value::LogicalArray(array) => double_from_logical(array)?,
        other => {
            return Err(double_error_with_detail(
                &DOUBLE_ERROR_INTERNAL,
                format!("gather returned unsupported value {other:?}"),
            ))
        }
    };
    match (provider, value) {
        (Some(provider), Value::Tensor(tensor)) => {
            if provider.precision() == ProviderPrecision::F64 {
                let upload_data = tensor
                    .clone()
                    .into_numeric_storage()
                    .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))?
                    .materialize_f64();
                let view = HostTensorView {
                    data: &upload_data,
                    shape: &tensor.shape,
                };
                match provider.upload(&view) {
                    Ok(new_handle)
                        if valid_double_gpu_output(&new_handle, &handle, provider, false) =>
                    {
                        return Ok(Value::GpuTensor(new_handle));
                    }
                    Ok(new_handle) => {
                        free_rejected_double_handle(&new_handle, &[&handle]);
                    }
                    Err(err) => {
                        trace!("double: provider upload failed after gather ({err})");
                    }
                }
            } else {
                trace!(
                    "double: provider precision {:?} does not support float64 outputs; returning host tensor",
                    provider.precision()
                );
            }
            Ok(Value::Tensor(tensor))
        }
        (Some(provider), Value::ComplexTensor(tensor)) => {
            if provider.precision() == ProviderPrecision::F64 {
                match gpu_helpers::upload_complex_tensor(provider, &tensor) {
                    Ok(new_handle)
                        if valid_double_gpu_output(&new_handle, &handle, provider, true) =>
                    {
                        return Ok(gpu_helpers::complex_gpu_value(new_handle));
                    }
                    Ok(new_handle) => {
                        free_rejected_double_handle(&new_handle, &[&handle]);
                    }
                    Err(err) => trace!("double: complex upload failed after gather ({err})"),
                }
            }
            Ok(Value::ComplexTensor(tensor))
        }
        (Some(provider), value) => {
            trace!(
                "double: provider precision {:?} does not support float64 outputs; returning host value",
                provider.precision()
            );
            Ok(value)
        }
        (None, value) => Ok(value),
    }
}

fn valid_double_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    complex: bool,
) -> bool {
    let expected_storage = if complex {
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    } else {
        runmat_accelerate_api::GpuTensorStorage::Real
    };
    output.shape == input.shape
        && output.device_id == input.device_id
        && !gpu_handles_alias(output, input)
        && runmat_accelerate_api::handle_precision(output) == Some(ProviderPrecision::F64)
        && runmat_accelerate_api::handle_storage(output) == expected_storage
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && resolved_actual_double_owner(output).is_some_and(|owner| std::ptr::eq(owner, provider))
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> BuiltinResult<OutputTemplate> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err(double_error_with_detail(
                    &DOUBLE_ERROR_INVALID_ARGUMENT,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(double_error_with_detail(
                    &DOUBLE_ERROR_INVALID_ARGUMENT,
                    "unrecognised argument for double",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(double_error_with_detail(
                    &DOUBLE_ERROR_INVALID_ARGUMENT,
                    "unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(double_error_with_detail(
            &DOUBLE_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        )),
    }
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => match proto {
            Value::GpuTensor(prototype) => convert_to_gpu(value, prototype).await,
            Value::Tensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_) => convert_to_host_like(value).await,
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(double_error_with_detail(
                &DOUBLE_ERROR_INVALID_INPUT,
                "complex prototypes for 'like' are not supported yet",
            )),
            _ => Err(double_error_with_detail(
                &DOUBLE_ERROR_INVALID_INPUT,
                "unsupported prototype for 'like'; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

async fn convert_to_gpu(value: Value, prototype: &GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = resolved_actual_double_owner(prototype).ok_or_else(|| {
        double_error_with_detail(
            &DOUBLE_ERROR_GPU_UNSUPPORTED,
            "GPU output requested via 'like' but the prototype owner is unavailable",
        )
    })?;
    if provider.precision() != ProviderPrecision::F64 {
        return Err(double_error_with_detail(
            &DOUBLE_ERROR_GPU_UNSUPPORTED,
            "active acceleration provider does not support float64 storage",
        ));
    }
    let value = match value {
        Value::GpuTensor(handle) => {
            let same_owner = resolved_actual_double_owner(&handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider));
            if same_owner
                && handle.device_id == prototype.device_id
                && runmat_accelerate_api::handle_precision(&handle) == Some(ProviderPrecision::F64)
                && runmat_accelerate_api::handle_integer_type(&handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(&handle)
            {
                return Ok(Value::GpuTensor(handle));
            }
            convert_to_host_like(Value::GpuTensor(handle)).await?
        }
        other => other,
    };
    match value {
        Value::Tensor(tensor) => upload_double_like_real(provider, prototype, &tensor),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
            upload_double_like_real(provider, prototype, &tensor)
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
            upload_double_like_real(provider, prototype, &tensor)
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
            upload_double_like_real(provider, prototype, &tensor)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            upload_double_like_real(provider, prototype, &tensor)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
            upload_double_like_complex(provider, prototype, &tensor)
        }
        Value::ComplexTensor(tensor) => upload_double_like_complex(provider, prototype, &tensor),
        other => Err(double_error_with_detail(
            &DOUBLE_ERROR_INVALID_INPUT,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
        )),
    }
}

fn upload_double_like_real(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    prototype: &GpuTensorHandle,
    tensor: &Tensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_tensor(provider, tensor)
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))?;
    if valid_double_like_output(&handle, prototype, provider, &tensor.shape, false) {
        Ok(Value::GpuTensor(handle))
    } else {
        free_rejected_double_handle(&handle, &[prototype]);
        Err(double_error_with_detail(
            &DOUBLE_ERROR_INTERNAL,
            "provider returned malformed GPU output for 'like'",
        ))
    }
}

fn upload_double_like_complex(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    prototype: &GpuTensorHandle,
    tensor: &ComplexTensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_complex_tensor(provider, tensor)
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))?;
    if valid_double_like_output(&handle, prototype, provider, &tensor.shape, true) {
        Ok(gpu_helpers::complex_gpu_value(handle))
    } else {
        free_rejected_double_handle(&handle, &[prototype]);
        Err(double_error_with_detail(
            &DOUBLE_ERROR_INTERNAL,
            "provider returned malformed complex GPU output for 'like'",
        ))
    }
}

fn valid_double_like_output(
    output: &GpuTensorHandle,
    prototype: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
    complex: bool,
) -> bool {
    let expected_storage = if complex {
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    } else {
        runmat_accelerate_api::GpuTensorStorage::Real
    };
    output.shape == expected_shape
        && output.device_id == prototype.device_id
        && !gpu_handles_alias(output, prototype)
        && runmat_accelerate_api::handle_precision(output) == Some(ProviderPrecision::F64)
        && runmat_accelerate_api::handle_storage(output) == expected_storage
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && resolved_actual_double_owner(output).is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn resolved_actual_double_owner(
    handle: &GpuTensorHandle,
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    runmat_accelerate_api::provider_for_handle(handle)
        .filter(|owner| owner.device_id() == handle.device_id)
}

fn gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

fn free_rejected_double_handle(handle: &GpuTensorHandle, protected: &[&GpuTensorHandle]) {
    if protected
        .iter()
        .any(|protected| gpu_handles_alias(handle, protected))
    {
        trace!("double: rejected handle aliases a caller-owned input; not freeing it");
        return;
    }
    if let Some(owner) = resolved_actual_double_owner(handle) {
        if let Err(error) = owner.free(handle) {
            trace!("double: failed to free rejected handle through its owner ({error})");
        }
    } else {
        trace!("double: rejected handle has no resolvable owner; leaving cleanup to its producer");
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value_async(&proxy)
                .await
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))
        }
        other => Ok(other),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::ProviderPrecision;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView, HostTensorView};
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{
        IntValue, IntegerComplexStorage, IntegerStorage, NumericDType, SparseTensor, SymbolicArray,
        SymbolicExpr,
    };

    fn double_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::double_builtin(value, rest))
    }

    #[test]
    fn double_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = DOUBLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = double(X)"));
        assert!(labels.contains(&"Y = double(X, \"like\", prototype)"));
    }

    #[test]
    fn double_type_preserves_tensor_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn double_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_scalar_num_is_identity() {
        let value = Value::Num(std::f64::consts::PI);
        let result = double_builtin(value, Vec::new()).expect("double");
        match result {
            Value::Num(n) => assert_eq!(n, std::f64::consts::PI),
            other => panic!("expected scalar Num, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_promotes_all_integer_scalar_classes() {
        for (value, expected) in [
            (IntValue::I8(-8), -8.0),
            (IntValue::I16(-16), -16.0),
            (IntValue::I32(-32), -32.0),
            (IntValue::I64(i64::MIN), i64::MIN as f64),
            (IntValue::U8(8), 8.0),
            (IntValue::U16(16), 16.0),
            (IntValue::U32(32), 32.0),
            (IntValue::U64(u64::MAX), u64::MAX as f64),
        ] {
            let result = double_builtin(Value::Int(value), Vec::new()).expect("double");
            match result {
                Value::Num(actual) => assert_eq!(actual, expected),
                other => panic!("expected scalar Num, got {other:?}"),
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_converts_symbolic_constants() {
        let result = double_builtin(Value::Symbolic(SymbolicExpr::constant(42.5)), Vec::new())
            .expect("double");

        assert_eq!(result, Value::Num(42.5));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_converts_symbolic_array_constants() {
        let array = SymbolicArray::new(
            vec![SymbolicExpr::constant(1.5), SymbolicExpr::constant(2.5)],
            vec![1, 2],
        )
        .unwrap();

        let result = double_builtin(Value::SymbolicArray(array), Vec::new()).expect("double");

        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.materialize_f64(), vec![1.5, 2.5]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_rejects_symbolic_array_variables() {
        let array = SymbolicArray::new(vec![SymbolicExpr::variable("x")], vec![1, 1]).unwrap();

        let err = double_builtin(Value::SymbolicArray(array), Vec::new())
            .expect_err("symbolic variable should not convert");

        assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("conversion to double from sym"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_rejects_symbolic_variables() {
        let err = double_builtin(Value::Symbolic(SymbolicExpr::variable("x")), Vec::new())
            .expect_err("symbolic variable should not convert");

        assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("conversion to double from sym"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_logical_array_returns_tensor() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
        let result = double_builtin(Value::LogicalArray(logical), Vec::new()).expect("double");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), vec![0.0, 1.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_char_array_converts_to_codes() {
        let chars = CharArray::new_row("AB");
        let result = double_builtin(Value::CharArray(chars), Vec::new()).expect("double");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![65.0, 66.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_complex_scalar_is_identity() {
        let result = double_builtin(Value::Complex(1.5, -2.5), Vec::new()).expect("double");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.5);
                assert_eq!(im, -2.5);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![1.25, 2.5, 3.75, 4.5], vec![2, 2]).unwrap();
        let result = double_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("double");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, tensor.shape);
                assert_eq!(
                    t.into_numeric_storage().expect("double storage"),
                    NumericStorage::F64(vec![1.25, 2.5, 3.75, 4.5])
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_tensor_reads_typed_integer_storage_and_clears_class() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
                .unwrap();

        let result = double_builtin(Value::Tensor(tensor), Vec::new()).expect("double");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(
                    t.into_numeric_storage().expect("double storage"),
                    NumericStorage::F64(vec![(1_u64 << 63) as f64, u64::MAX as f64])
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_complex_tensor_reads_typed_integer_storage_and_clears_class() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![-2, 3]),
            IntegerStorage::I16(vec![5, -7]),
        )
        .unwrap();
        let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();

        let result = double_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("double");
        match result {
            Value::ComplexTensor(t) => {
                assert!(t.integer_storage().is_none());
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![(-2.0, 5.0), (3.0, -7.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_sparse_tensor_preserves_sparsity_and_converts_storage() {
        let sparse =
            SparseTensor::new_f32(3, 2, vec![0, 1, 2], vec![1, 2], vec![4.0, -1.0]).unwrap();
        let result = double_builtin(Value::SparseTensor(sparse), Vec::new()).expect("double");
        let Value::SparseTensor(output) = result else {
            panic!("expected sparse tensor");
        };
        assert_eq!(output.shape(), vec![3, 2]);
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F64));
        assert_eq!(output.as_f64_slice(), Some(&[4.0, -1.0][..]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_sparse_tensor_reads_typed_integer_storage_and_clears_class() {
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![1, 0],
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        )
        .unwrap();
        let result = double_builtin(Value::SparseTensor(sparse), Vec::new()).expect("double");
        let Value::SparseTensor(output) = result else {
            panic!("expected sparse tensor");
        };
        assert_eq!(output.shape(), vec![2, 2]);
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F64));
        assert!(output.integer_storage().is_none());
        assert_eq!(
            output.as_f64_slice(),
            Some(&[i64::MIN as f64, i64::MAX as f64][..])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_converts_string_scalars_and_arrays() {
        assert_eq!(
            double_builtin(Value::String(" 12.5 ".into()), Vec::new()).unwrap(),
            Value::Num(12.5)
        );
        let invalid = double_builtin(Value::String("not a number".into()), Vec::new()).unwrap();
        assert!(matches!(invalid, Value::Num(value) if value.is_nan()));

        let strings = StringArray::new(
            vec!["1".into(), "-2.25".into(), "missing".into(), "Inf".into()],
            vec![2, 2],
        )
        .unwrap();
        let Value::Tensor(output) =
            double_builtin(Value::StringArray(strings), Vec::new()).unwrap()
        else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![2, 2]);
        let values = output.materialize_f64();
        assert_eq!(values[0..2], [1.0, -2.25]);
        assert!(values[2].is_nan());
        assert_eq!(values[3], f64::INFINITY);
    }

    #[test]
    fn double_like_is_a_gated_runmat_extension() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = double_builtin(Value::Num(1.0), vec![Value::from("like"), Value::Num(0.0)])
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DoubleLikePrototypeExtension")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = double_builtin(Value::GpuTensor(handle), Vec::new()).expect("double");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.materialize_f64(), tensor.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_gpu_integer_handle_converts_to_double_resident_output() {
        test_support::with_test_provider(|provider| {
            let data = [1_u64 << 63, u64::MAX];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&data),
                    shape: &[1, 2],
                })
                .expect("upload integer");

            let result = double_builtin(Value::GpuTensor(handle), Vec::new()).expect("double");
            let Value::GpuTensor(result_handle) = result else {
                panic!("expected f64-capable provider to keep double output resident");
            };
            assert!(runmat_accelerate_api::handle_integer_type(&result_handle).is_none());

            let gathered =
                test_support::gather(Value::GpuTensor(result_handle)).expect("gather double");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
            assert!(gathered.integer_storage().is_none());
            assert_eq!(
                gathered.materialize_f64(),
                vec![(1_u64 << 63) as f64, u64::MAX as f64]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_like_gpu_prototype_keeps_residency() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let proto = provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &[1, 1],
                })
                .expect("upload");
            let result = double_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("double");
            match result {
                Value::GpuTensor(h) => {
                    let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather");
                    assert_eq!(gathered.materialize_f64(), tensor.materialize_f64());
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn double_like_rejects_hostile_upload_metadata_and_frees_only_resolved_owner() {
        test_support::with_test_provider(|provider| {
            let prototype = provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &[1, 1],
                })
                .expect("prototype upload");

            let make_output = || {
                provider
                    .upload(&HostTensorView {
                        data: &[1.0, 2.0],
                        shape: &[2, 1],
                    })
                    .expect("result upload")
            };

            let wrong_shape = {
                let mut handle = make_output();
                handle.shape = vec![1, 2];
                handle
            };
            assert!(!valid_double_like_output(
                &wrong_shape,
                &prototype,
                provider,
                &[2, 1],
                false,
            ));
            provider
                .free(&wrong_shape)
                .expect("free wrong-shape result");

            let wrong_storage = make_output();
            runmat_accelerate_api::set_handle_storage(
                &wrong_storage,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            );
            assert!(!valid_double_like_output(
                &wrong_storage,
                &prototype,
                provider,
                &[2, 1],
                false,
            ));
            provider
                .free(&wrong_storage)
                .expect("free wrong-storage result");

            let wrong_precision = make_output();
            runmat_accelerate_api::set_handle_precision(
                &wrong_precision,
                runmat_accelerate_api::ProviderPrecision::F32,
            );
            assert!(!valid_double_like_output(
                &wrong_precision,
                &prototype,
                provider,
                &[2, 1],
                false,
            ));
            provider
                .free(&wrong_precision)
                .expect("free wrong-precision result");

            let integer = make_output();
            runmat_accelerate_api::set_handle_integer_type(
                &integer,
                runmat_accelerate_api::IntegerElementType::U8,
            );
            assert!(!valid_double_like_output(
                &integer,
                &prototype,
                provider,
                &[2, 1],
                false,
            ));
            provider.free(&integer).expect("free integer result");

            let logical = make_output();
            runmat_accelerate_api::set_handle_logical(&logical, true);
            assert!(!valid_double_like_output(
                &logical,
                &prototype,
                provider,
                &[2, 1],
                false,
            ));
            provider.free(&logical).expect("free logical result");

            let owned_rejection = make_output();
            runmat_accelerate_api::set_handle_storage(
                &owned_rejection,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            );
            free_rejected_double_handle(&owned_rejection, &[]);
            assert!(block_on(provider.download(&owned_rejection)).is_err());

            assert!(!valid_double_like_output(
                &prototype,
                &prototype,
                provider,
                &[1, 1],
                false,
            ));
            assert!(!valid_double_gpu_output(
                &prototype, &prototype, provider, false,
            ));
            free_rejected_double_handle(&prototype, &[&prototype]);
            assert!(block_on(provider.download(&prototype)).is_ok());

            let unowned_rejection = runmat_accelerate_api::GpuTensorHandle {
                device_id: prototype.device_id.wrapping_add(10_000),
                buffer_id: prototype.buffer_id,
                shape: vec![2, 1],
            };
            assert!(!valid_double_like_output(
                &unowned_rejection,
                &prototype,
                provider,
                &[2, 1],
                false,
            ));
            assert!(resolved_actual_double_owner(&unowned_rejection).is_none());
            free_rejected_double_handle(&unowned_rejection, &[]);
            assert!(block_on(provider.download(&prototype)).is_ok());
            provider.free(&prototype).expect("free prototype");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_like_host_gathers_gpu_input() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = double_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("double");
            match result {
                Value::Num(n) => assert_eq!(n, 3.0),
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![1, 1]);
                    assert_eq!(t.materialize_f64(), vec![3.0]);
                }
                other => panic!("expected scalar host value, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_like_missing_prototype_errors() {
        let err =
            double_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("expected prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_like_rejects_extra_arguments() {
        let err = double_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn double_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let tensor = Tensor::new(vec![1.0, 2.5, -3.75, 4.125], vec![2, 2]).unwrap();
        let cpu_value = double_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();

        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = double_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();

        let gathered = test_support::gather(gpu_value.clone()).expect("gather");
        match cpu_value {
            Value::Tensor(ref ct) => {
                assert_eq!(gathered.shape, ct.shape);
                assert_eq!(gathered.materialize_f64(), ct.materialize_f64());
            }
            Value::Num(n) => {
                assert_eq!(gathered.materialize_f64(), vec![n]);
            }
            other => panic!("unexpected CPU reference value {other:?}"),
        }

        if provider.precision() == ProviderPrecision::F64 {
            assert!(
                matches!(gpu_value, Value::GpuTensor(_)),
                "expected GPU residency under f64 precision"
            );
        } else {
            assert!(
                !matches!(gpu_value, Value::GpuTensor(_)),
                "expected host fallback when f64 unsupported"
            );
        }
    }
}
