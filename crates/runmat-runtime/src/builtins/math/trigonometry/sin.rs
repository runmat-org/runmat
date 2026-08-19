//! MATLAB-compatible `sin` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, Tensor, Value};

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::symbolic::symbolic_function;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::SymbolicFunction;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::sin")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sin",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sin" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute sin in-place on the device; runtimes gather to host when unary_sin is unavailable.",
};

const BUILTIN_NAME: &str = "sin";

pub const SIN_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sin-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sin with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinIntegerInputExtension"),
};
pub const SIN_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sin-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sin with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinLogicalInputExtension"),
};
pub const SIN_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sin-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sin with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinCharacterInputExtension"),
};
pub const SIN_LIKE_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sin-like-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sin with a like output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinLikeOutputExtension"),
};
pub const SIN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    SIN_INTEGER_INPUT_EXTENSION,
    SIN_LOGICAL_INPUT_EXTENSION,
    SIN_CHARACTER_INPUT_EXTENSION,
    SIN_LIKE_OUTPUT_EXTENSION,
];
const SIN_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight real integer classes are admitted only when exactly representable at the binary64 transcendental boundary.",
    }];
pub const SIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = sin(integer_X)",
        inputs: &SIN_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat mode validates authoritative integer storage before conversion; automatic residency may gather and explicit output residency follows the existing provider policy.",
    }];

const SIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise sine result.",
}];

const SIN_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar, array, char array, complex value, or gpuArray.",
}];

const SIN_INPUTS_X_LIKE_P: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar, array, char array, complex value, or gpuArray.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Output template selector keyword.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype determining host vs gpuArray output residency.",
    },
];

const SIN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = sin(X)",
        inputs: &SIN_INPUTS_X,
        outputs: &SIN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = sin(X, \"like\", P)",
        inputs: &SIN_INPUTS_X_LIKE_P,
        outputs: &SIN_OUTPUT,
    },
];

const SIN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.INVALID_INPUT",
    identifier: Some("RunMat:sin:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric/logical/char/complex data.",
    message: "sin: invalid input",
};

const SIN_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.INVALID_OPTION",
    identifier: Some("RunMat:sin:InvalidOption"),
    when: "Optional arguments after X are malformed or unsupported.",
    message: "sin: invalid option",
};

const SIN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.ARG_COUNT",
    identifier: Some("RunMat:sin:ArgCount"),
    when: "Too many input arguments were supplied.",
    message: "sin: too many input arguments",
};

const SIN_ERROR_LIKE_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.LIKE_PROTOTYPE",
    identifier: Some("RunMat:sin:LikePrototype"),
    when: "The \"like\" prototype is unsupported for this output conversion path.",
    message: "sin: invalid \"like\" prototype",
};

const SIN_ERROR_GPU_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.GPU_UNAVAILABLE",
    identifier: Some("RunMat:sin:GpuUnavailable"),
    when: "GPU output was requested via \"like\" but no active provider is available.",
    message: "sin: GPU provider unavailable",
};

const SIN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.INTERNAL",
    identifier: Some("RunMat:sin:Internal"),
    when: "Internal tensor conversion/allocation/provider flow failed.",
    message: "sin: internal error",
};

const SIN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    SIN_ERROR_INVALID_INPUT,
    SIN_ERROR_INVALID_OPTION,
    SIN_ERROR_ARG_COUNT,
    SIN_ERROR_LIKE_PROTOTYPE,
    SIN_ERROR_GPU_UNAVAILABLE,
    SIN_ERROR_INTERNAL,
];

pub const SIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SIN_ERRORS,
};

fn sin_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sin_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::sin")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sin",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("sin({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `sin` calls; providers may override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "sin",
    category = "math/trigonometry",
    summary = "Compute element-wise sine values in radians.",
    keywords = "sin,sine,trigonometry,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::trigonometry::sin::SIN_DESCRIPTOR),
    extensions(SIN_EXTENSIONS),
    integer_capabilities(SIN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::trigonometry::sin"
)]
async fn sin_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let output = parse_output_template(&rest)?;
    ensure_sin_extensions(&value, &rest).await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "sin")?;
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Sin) {
        return apply_output_template(symbolic, &output).await;
    }
    let base = match value {
        Value::GpuTensor(handle) => sin_gpu(handle).await?,
        Value::Complex(re, im) => Value::Complex(sin_complex_re(re, im), sin_complex_im(re, im)),
        Value::ComplexTensor(ct) => sin_complex_tensor(ct)?,
        Value::CharArray(ca) => sin_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err(sin_error_with_detail(
                &SIN_ERROR_INVALID_INPUT,
                "expected numeric input, got string",
            ))
        }
        other => sin_real(other)?,
    };
    apply_output_template(base, &output).await
}

async fn ensure_sin_extensions(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        value,
        &SIN_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
        "X",
    )
    .await?;
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SIN_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SIN_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SIN_LIKE_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn sin_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_sin(&handle).await {
            return Ok(gpu_helpers::resident_gpu_value(out));
        }
    }
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    match gathered {
        Value::Complex(re, im) => Ok(Value::Complex(
            sin_complex_re(re, im),
            sin_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => sin_complex_tensor(ct),
        Value::Tensor(tensor) => sin_tensor(tensor).map(tensor::tensor_into_value),
        Value::Num(n) => Ok(Value::Num(n.sin())),
        other => Err(sin_error_with_detail(
            &SIN_ERROR_INVALID_INPUT,
            format!("unsupported gathered gpuArray value {other:?}"),
        )),
    }
}

fn sin_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("sin", value)
        .map_err(|e| sin_error_with_detail(&SIN_ERROR_INVALID_INPUT, e))?;
    sin_tensor(tensor).map(tensor::tensor_into_value)
}

fn sin_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&v| v.sin())
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))
}

fn sin_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .materialize_f64()
        .iter()
        .map(|&(re, im)| (sin_complex_re(re, im), sin_complex_im(re, im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(tensor))
}

fn sin_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).sin())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn sin_complex_re(re: f64, im: f64) -> f64 {
    re.sin() * im.cosh()
}

#[inline]
fn sin_complex_im(re: f64, im: f64) -> f64 {
    re.cos() * im.sinh()
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
                Err(sin_error_with_detail(
                    &SIN_ERROR_INVALID_OPTION,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(sin_error_with_detail(
                    &SIN_ERROR_INVALID_OPTION,
                    "unrecognised argument for sin",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(sin_error_with_detail(
                    &SIN_ERROR_INVALID_OPTION,
                    "unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(sin_error(&SIN_ERROR_ARG_COUNT)),
    }
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => match proto {
            Value::GpuTensor(handle) => {
                if runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                {
                    convert_to_gpu_complex(value).await
                } else {
                    convert_to_gpu(value)
                }
            }
            Value::Tensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_) => convert_to_host_like(value).await,
            Value::Complex(_, _) | Value::ComplexTensor(_) => convert_to_host_complex(value).await,
            _ => Err(sin_error_with_detail(
                &SIN_ERROR_LIKE_PROTOTYPE,
                "unsupported prototype; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        sin_error_with_detail(
            &SIN_ERROR_GPU_UNAVAILABLE,
            "GPU output requested via 'like' but no acceleration provider is active",
        )
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let data = tensor::tensor_values_f64_cow(&tensor);
            let view = HostTensorView {
                data: data.as_ref(),
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(sin_error_with_detail(
            &SIN_ERROR_LIKE_PROTOTYPE,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(sin_error_with_detail(
            &SIN_ERROR_INTERNAL,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu_complex(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        sin_error_with_detail(
            &SIN_ERROR_GPU_UNAVAILABLE,
            "complex GPU output requested via 'like' but no acceleration provider is active",
        )
    })?;
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(&handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                Ok(Value::GpuTensor(handle))
            } else if let Some(handle_provider) =
                runmat_accelerate_api::provider_for_handle(&handle)
            {
                match handle_provider.complex_from_real(&handle).await {
                    Ok(out) => Ok(Value::GpuTensor(out)),
                    Err(_) => {
                        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                            .await
                            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                        convert_to_gpu_complex(gathered).await
                    }
                }
            } else {
                Err(sin_error_with_detail(
                    &SIN_ERROR_GPU_UNAVAILABLE,
                    "complex GPU output requested but the input handle has no provider",
                ))
            }
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::ComplexTensor(tensor) => {
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => convert_to_gpu_complex(Value::Complex(n, 0.0)).await,
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(&tensor);
            let data = values.iter().map(|&re| (re, 0.0)).collect::<Vec<_>>();
            let complex = ComplexTensor::new(data, tensor.shape.clone())
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::ComplexTensor(complex)).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::Tensor(tensor)).await
        }
        Value::Int(i) => convert_to_gpu_complex(Value::Num(i.to_f64())).await,
        Value::Bool(b) => convert_to_gpu_complex(Value::Num(if b { 1.0 } else { 0.0 })).await,
        other => Err(sin_error_with_detail(
            &SIN_ERROR_INTERNAL,
            format!("cannot convert value {other:?} to complex GPU output via 'like'"),
        )),
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value_async(&proxy).await
        }
        other => Ok(other),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_host_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(&tensor);
            let data = values.iter().map(|&re| (re, 0.0)).collect::<Vec<_>>();
            let complex = ComplexTensor::new(data, tensor.shape.clone())
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            Ok(complex_tensor_into_value(complex))
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            convert_to_host_complex(gathered).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            convert_to_host_complex(Value::Tensor(tensor)).await
        }
        Value::Int(i) => convert_to_host_complex(Value::Num(i.to_f64())).await,
        Value::Bool(b) => convert_to_host_complex(Value::Num(if b { 1.0 } else { 0.0 })).await,
        other => Err(sin_error_with_detail(
            &SIN_ERROR_INTERNAL,
            format!("cannot convert value {other:?} to complex output via 'like'"),
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntValue, Tensor};

    use crate::builtins::common::{gpu_helpers, test_support};

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn sin_descriptor_signatures_cover_like_overload() {
        let labels: Vec<&str> = SIN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = sin(X)"));
        assert!(labels.contains(&"Y = sin(X, \"like\", P)"));
    }

    #[test]
    fn sin_type_preserves_tensor_shape() {
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
    fn sin_type_scalar_tensor_returns_num() {
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
    fn sin_scalar() {
        let value = Value::Num(std::f64::consts::PI / 2.0);
        let result = block_on(sin_builtin(value, Vec::new())).expect("sin");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let result = block_on(sin_builtin(Value::Tensor(tensor), Vec::new())).expect("sin");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.materialize_f64()[0] - 0.0).abs() < 1e-12);
                assert!((t.materialize_f64()[1] - 0.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_reads_typed_integer_tensor_storage_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![0, 1, 2]), vec![3, 1])
                .expect("integer tensor");

        let result = block_on(sin_builtin(Value::Tensor(tensor), Vec::new())).expect("sin");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, 1.0f64.sin(), 2.0f64.sin()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sin_host_complex_conversion_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![-3, 0, 5]),
            vec![3, 1],
        )
        .expect("integer tensor");

        let result =
            block_on(convert_to_host_complex(Value::Tensor(tensor))).expect("complex conversion");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor result");
        };
        assert_eq!(out.shape, vec![3, 1]);
        assert_eq!(
            out.materialize_f64(),
            vec![(-3.0, 0.0), (0.0, 0.0), (5.0, 0.0)]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_int_value_promotes() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Value::Int(IntValue::I32(1));
        let result = block_on(sin_builtin(value, Vec::new())).expect("sin");
        match result {
            Value::Num(v) => assert!((v - 1.0_f64.sin()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_complex_scalar() {
        let result = block_on(sin_builtin(Value::Complex(1.0, 2.0), Vec::new())).expect("sin");
        match result {
            Value::Complex(re, im) => {
                assert!((re - (1.0f64.sin() * 2.0f64.cosh())).abs() < 1e-12);
                assert!((im - (1.0f64.cos() * 2.0f64.sinh())).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_char_array_roundtrip() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = block_on(sin_builtin(Value::CharArray(chars), Vec::new())).expect("sin");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).sin();
                    assert!((t.materialize_f64()[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = block_on(sin_builtin(Value::GpuTensor(handle), Vec::new())).expect("sin");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.sin()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_missing_prototype_errors() {
        let err = block_on(sin_builtin(Value::Num(1.0), vec![Value::from("like")]))
            .expect_err("expected error");
        assert_eq!(err.identifier(), SIN_ERROR_INVALID_OPTION.identifier);
        let message = error_message(err);
        assert!(message.contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_complex_prototype_returns_complex() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = block_on(sin_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        ))
        .expect("sin");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 1.0_f64.sin()).abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_gpu_prototype() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = block_on(sin_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            ))
            .expect("sin");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.sin()).collect();
                    assert_eq!(gathered.shape, vec![4, 1]);
                    assert_eq!(gathered.materialize_f64(), expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn sin_gpu_complex_input_preserves_complex_output() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(0.5, 0.75), (2.0, -0.25)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            let result = block_on(sin_builtin(Value::GpuTensor(handle), Vec::new())).expect("sin");
            let out = match result {
                Value::GpuTensor(handle) => {
                    assert_eq!(
                        runmat_accelerate_api::handle_storage(&handle),
                        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                    );
                    match block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle)))
                        .expect("gather")
                    {
                        Value::ComplexTensor(out) => out,
                        other => panic!("expected gathered complex tensor, got {other:?}"),
                    }
                }
                Value::ComplexTensor(out) => out,
                other => panic!("expected complex output, got {other:?}"),
            };
            assert_eq!(out.shape, vec![1, 2]);
            for (idx, &(re, im)) in input.materialize_f64().iter().enumerate() {
                assert!((out.materialize_f64()[idx].0 - sin_complex_re(re, im)).abs() < 1e-12);
                assert!((out.materialize_f64()[idx].1 - sin_complex_im(re, im)).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn sin_like_complex_gpu_prototype_uploads_complex_result() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = block_on(sin_builtin(
                Value::Tensor(input.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            ))
            .expect("sin");
            let Value::GpuTensor(handle) = result else {
                panic!("expected complex gpu tensor, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap();
            let Value::ComplexTensor(out) = gathered else {
                panic!("expected gathered complex tensor, got {gathered:?}");
            };
            assert_eq!(out.shape, vec![2, 1]);
            for (idx, &re) in input.materialize_f64().iter().enumerate() {
                assert!((out.materialize_f64()[idx].0 - re.sin()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[test]
    fn sin_like_complex_gpu_prototype_converts_resident_real_gpu_result() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let input_view = HostTensorView {
                data: &input.materialize_f64(),
                shape: &input.shape,
            };
            let input_handle = provider.upload(&input_view).expect("upload input");
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = block_on(sin_builtin(
                Value::GpuTensor(input_handle),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            ))
            .expect("sin");
            let Value::GpuTensor(handle) = result else {
                panic!("expected complex gpu tensor, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap();
            let Value::ComplexTensor(out) = gathered else {
                panic!("expected gathered complex tensor, got {gathered:?}");
            };
            assert_eq!(out.shape, vec![2, 1]);
            for (idx, &re) in input.materialize_f64().iter().enumerate() {
                assert!((out.materialize_f64()[idx].0 - re.sin()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_host_with_gpu_input_gathers() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = block_on(sin_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            ))
            .expect("sin");
            match result {
                Value::Tensor(t) => {
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.sin()).collect();
                    assert_eq!(t.shape, vec![2, 1]);
                    assert_eq!(t.materialize_f64(), expected);
                }
                Value::GpuTensor(_) => panic!("expected host result"),
                Value::Num(_) => panic!("expected vector output"),
                other => panic!("unexpected result {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_rejects_extra_arguments() {
        let err = block_on(sin_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        ))
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_keyword_case_insensitive() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result = block_on(sin_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        ))
        .expect("sin");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> =
                    tensor.materialize_f64().iter().map(|&v| v.sin()).collect();
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.materialize_f64(), expected);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_char_array_keyword() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let keyword = CharArray::new_row("like");
        let result = block_on(sin_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        ))
        .expect("sin");
        match result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sin_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let cpu = sin_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.materialize_f64(),
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(sin_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
