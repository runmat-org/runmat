//! MATLAB-compatible `cos` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexStorage, ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

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
use runmat_builtins::SymbolicFunction;

const BUILTIN_NAME: &str = "cos";

pub const COS_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cos-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cos with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CosIntegerInputExtension"),
};
pub const COS_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cos-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cos with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CosLogicalInputExtension"),
};
pub const COS_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cos-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cos with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CosCharacterInputExtension"),
};
pub const COS_LIKE_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cos-like-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cos with a like output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CosLikeOutputExtension"),
};
pub const COS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    COS_INTEGER_INPUT_EXTENSION,
    COS_LOGICAL_INPUT_EXTENSION,
    COS_CHARACTER_INPUT_EXTENSION,
    COS_LIKE_OUTPUT_EXTENSION,
];
const COS_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "X", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "All eight real integer classes are admitted only when exactly representable at the binary64 transcendental boundary." }];
pub const COS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "Y = cos(integer_X)", inputs: &COS_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "RunMat mode checks authoritative integer storage before conversion; host output is double and resident fallback returns through the owning provider." }];

const COS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise cosine result.",
}];

const COS_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar, array, char array, complex value, or gpuArray.",
}];

const COS_INPUTS_X_LIKE_P: [BuiltinParamDescriptor; 3] = [
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

const COS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = cos(X)",
        inputs: &COS_INPUTS_X,
        outputs: &COS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = cos(X, \"like\", P)",
        inputs: &COS_INPUTS_X_LIKE_P,
        outputs: &COS_OUTPUT,
    },
];

const COS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.INVALID_INPUT",
    identifier: Some("RunMat:cos:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric/logical/char/complex data.",
    message: "cos: invalid input",
};

const COS_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.INVALID_OPTION",
    identifier: Some("RunMat:cos:InvalidOption"),
    when: "Optional arguments after X are malformed or unsupported.",
    message: "cos: invalid option",
};

const COS_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.ARG_COUNT",
    identifier: Some("RunMat:cos:ArgCount"),
    when: "Too many input arguments were supplied.",
    message: "cos: too many input arguments",
};

const COS_ERROR_LIKE_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.LIKE_PROTOTYPE",
    identifier: Some("RunMat:cos:LikePrototype"),
    when: "The \"like\" prototype is unsupported for this output conversion path.",
    message: "cos: invalid \"like\" prototype",
};

const COS_ERROR_GPU_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.GPU_UNAVAILABLE",
    identifier: Some("RunMat:cos:GpuUnavailable"),
    when: "GPU output was requested via \"like\" but no active provider is available.",
    message: "cos: GPU provider unavailable",
};

const COS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.INTERNAL",
    identifier: Some("RunMat:cos:Internal"),
    when: "Internal tensor conversion/allocation/provider flow failed.",
    message: "cos: internal error",
};

const COS_ERRORS: [BuiltinErrorDescriptor; 6] = [
    COS_ERROR_INVALID_INPUT,
    COS_ERROR_INVALID_OPTION,
    COS_ERROR_ARG_COUNT,
    COS_ERROR_LIKE_PROTOTYPE,
    COS_ERROR_GPU_UNAVAILABLE,
    COS_ERROR_INTERNAL,
];

pub const COS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COS_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::cos")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cos",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_cos" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute cosine directly on device; runtimes gather to host when unary_cos is unavailable.",
};

fn cos_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cos_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::cos")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cos",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("cos({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `cos` calls; providers can override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "cos",
    category = "math/trigonometry",
    summary = "Compute cosine element-wise.",
    keywords = "cos,cosine,trigonometry,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::trigonometry::cos::COS_DESCRIPTOR),
    extensions(COS_EXTENSIONS),
    integer_capabilities(COS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::trigonometry::cos"
)]
async fn cos_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    ensure_cos_extensions(&value, &rest)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "cos")?;
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Cos) {
        return apply_output_template(symbolic, &template).await;
    }
    let base = match value {
        Value::GpuTensor(handle) => cos_gpu(handle).await?,
        Value::Complex(re, im) => Value::Complex(cos_complex_re(re, im), cos_complex_im(re, im)),
        Value::ComplexTensor(ct) => cos_complex_tensor(ct)?,
        Value::CharArray(ca) => cos_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err(cos_error_with_detail(
                &COS_ERROR_INVALID_INPUT,
                "expected numeric input, got string",
            ))
        }
        other => cos_real(other)?,
    };
    apply_output_template(base, &template).await
}

fn ensure_cos_extensions(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if is_real_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COS_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COS_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COS_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COS_LIKE_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    ensure_integer_exact_f64(value)
}

fn is_real_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some())
}

fn ensure_integer_exact_f64(value: &Value) -> BuiltinResult<()> {
    let exact = integer_is_exact_f64;
    let ok = match value {
        Value::Int(v) => exact(v),
        Value::Tensor(t) => t
            .integer_storage()
            .is_none_or(|s| s.exact_values().iter().all(exact)),
        _ => true,
    };
    if ok {
        Ok(())
    } else {
        Err(cos_error_with_detail(
            &COS_ERROR_INVALID_INPUT,
            "integer input must be exactly representable as double",
        ))
    }
}

pub(crate) fn integer_is_exact_f64(value: &runmat_builtins::IntValue) -> bool {
    let magnitude = match value {
        runmat_builtins::IntValue::I8(value) => u64::from(value.unsigned_abs()),
        runmat_builtins::IntValue::I16(value) => u64::from(value.unsigned_abs()),
        runmat_builtins::IntValue::I32(value) => u64::from(value.unsigned_abs()),
        runmat_builtins::IntValue::I64(value) => value.unsigned_abs(),
        runmat_builtins::IntValue::U8(value) => u64::from(*value),
        runmat_builtins::IntValue::U16(value) => u64::from(*value),
        runmat_builtins::IntValue::U32(value) => u64::from(*value),
        runmat_builtins::IntValue::U64(value) => *value,
    };
    if magnitude == 0 {
        return true;
    }
    let significant_bits = u64::BITS - magnitude.leading_zeros();
    significant_bits <= f64::MANTISSA_DIGITS
        || magnitude.trailing_zeros() >= significant_bits - f64::MANTISSA_DIGITS
}

async fn cos_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle);
    let requires_exact_host_path = runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle);
    if !requires_exact_host_path {
        if let Some(provider) = provider {
            match provider.unary_cos(&handle).await {
                Ok(out) if native_unary_output_matches(&handle, &out, provider) => {
                    return Ok(gpu_helpers::resident_gpu_value(out));
                }
                Ok(out) => free_rejected_native_output(&out, provider),
                Err(_) => {}
            }
        }
    }
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    ensure_integer_exact_f64(&gathered)?;
    let host = match gathered {
        Value::Complex(re, im) => Ok(Value::Complex(
            cos_complex_re(re, im),
            cos_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => cos_complex_tensor(ct),
        Value::Tensor(tensor) => cos_tensor(tensor).map(tensor::tensor_into_value),
        Value::Num(n) => Ok(Value::Num(n.cos())),
        other => cos_real(other),
    }?;
    if let Some(provider) = provider {
        upload_gpu_output(provider, host)
    } else {
        Ok(host)
    }
}

fn native_unary_output_matches(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    output.device_id == input.device_id
        && runmat_accelerate_api::handle_precision(output)
            == Some(
                runmat_accelerate_api::handle_precision(input)
                    .unwrap_or_else(|| provider.precision()),
            )
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::handle_storage(input)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn free_rejected_native_output(
    output: &GpuTensorHandle,
    invoked_provider: &dyn runmat_accelerate_api::AccelProvider,
) {
    let owner = runmat_accelerate_api::provider_for_handle(output).unwrap_or(invoked_provider);
    let _ = owner.free(output);
}

fn cos_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("cos", value)
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INVALID_INPUT, e))?;
    cos_tensor(tensor).map(tensor::tensor_into_value)
}

fn cos_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&v| v.cos())
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&v| v.cos())
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))
}

fn cos_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let tensor = match ct.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| {
                    (
                        cos_complex_re(f64::from(re), f64::from(im)) as f32,
                        cos_complex_im(f64::from(re), f64::from(im)) as f32,
                    )
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| (cos_complex_re(re, im), cos_complex_im(re, im)))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(tensor))
}

fn upload_gpu_output(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    value: Value,
) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => {
            let tensor = Tensor::new(vec![value], vec![1, 1])
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            upload_real_gpu_output(provider, tensor)
        }
        Value::Tensor(tensor) => upload_real_gpu_output(provider, tensor),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            upload_complex_gpu_output(provider, tensor)
        }
        Value::ComplexTensor(tensor) => upload_complex_gpu_output(provider, tensor),
        other => Err(cos_error_with_detail(
            &COS_ERROR_INTERNAL,
            format!("cannot restore GPU output {other:?}"),
        )),
    }
}

fn upload_real_gpu_output(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    tensor: Tensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_tensor(provider, &tensor)
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
    Ok(gpu_helpers::resident_gpu_value(handle))
}

fn upload_complex_gpu_output(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    tensor: ComplexTensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)?;
    Ok(gpu_helpers::complex_gpu_value(handle))
}

fn cos_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).cos())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn cos_complex_re(re: f64, im: f64) -> f64 {
    re.cos() * im.cosh()
}

#[inline]
fn cos_complex_im(re: f64, im: f64) -> f64 {
    -re.sin() * im.sinh()
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
                Err(cos_error_with_detail(
                    &COS_ERROR_INVALID_OPTION,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(cos_error_with_detail(
                    &COS_ERROR_INVALID_OPTION,
                    "unrecognised argument for cos",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(cos_error_with_detail(
                    &COS_ERROR_INVALID_OPTION,
                    "unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(cos_error(&COS_ERROR_ARG_COUNT)),
    }
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => match proto {
            Value::GpuTensor(handle) => {
                let provider =
                    runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
                        cos_error_with_detail(
                            &COS_ERROR_GPU_UNAVAILABLE,
                            "GPU prototype for 'like' has no owning acceleration provider",
                        )
                    })?;
                if runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                {
                    convert_to_gpu_complex(value, provider).await
                } else {
                    convert_to_gpu(value, provider).await
                }
            }
            Value::Tensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_) => convert_to_host_like(value).await,
            Value::Complex(_, _) | Value::ComplexTensor(_) => convert_to_host_complex(value).await,
            _ => Err(cos_error_with_detail(
                &COS_ERROR_LIKE_PROTOTYPE,
                "unsupported prototype; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu(
    value: Value,
    provider: &dyn runmat_accelerate_api::AccelProvider,
) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) if gpu_handle_is_owned_by(&handle, provider) => {
            Ok(Value::GpuTensor(handle))
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            convert_to_gpu(gathered, provider).await
        }
        Value::Tensor(tensor) => {
            let data = tensor::tensor_values_f64_cow(&tensor);
            let view = HostTensorView {
                data: data.as_ref(),
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor), provider).await
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64()), provider).await,
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 }), provider).await,
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor), provider).await
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(cos_error_with_detail(
            &COS_ERROR_LIKE_PROTOTYPE,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(cos_error_with_detail(
            &COS_ERROR_INTERNAL,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu_complex(
    value: Value,
    provider: &dyn runmat_accelerate_api::AccelProvider,
) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) if gpu_handle_is_owned_by(&handle, provider) => {
            if runmat_accelerate_api::handle_storage(&handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                Ok(Value::GpuTensor(handle))
            } else {
                match provider.complex_from_real(&handle).await {
                    Ok(out) => Ok(Value::GpuTensor(out)),
                    Err(_) => {
                        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                            .await
                            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                        convert_to_gpu_complex(gathered, provider).await
                    }
                }
            }
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            convert_to_gpu_complex(gathered, provider).await
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::ComplexTensor(tensor) => {
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => convert_to_gpu_complex(Value::Complex(n, 0.0), provider).await,
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(&tensor);
            let data = values.iter().map(|&re| (re, 0.0)).collect::<Vec<_>>();
            let complex = ComplexTensor::new(data, tensor.shape.clone())
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::ComplexTensor(complex), provider).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::Tensor(tensor), provider).await
        }
        Value::Int(i) => convert_to_gpu_complex(Value::Num(i.to_f64()), provider).await,
        Value::Bool(b) => {
            convert_to_gpu_complex(Value::Num(if b { 1.0 } else { 0.0 }), provider).await
        }
        other => Err(cos_error_with_detail(
            &COS_ERROR_INTERNAL,
            format!("cannot convert value {other:?} to complex GPU output via 'like'"),
        )),
    }
}

fn gpu_handle_is_owned_by(
    handle: &GpuTensorHandle,
    provider: &dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    handle.device_id == provider.device_id()
        && runmat_accelerate_api::provider_for_handle(handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
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
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
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
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_host_complex(Value::Tensor(tensor)).await
        }
        Value::Int(i) => convert_to_host_complex(Value::Num(i.to_f64())).await,
        Value::Bool(b) => convert_to_host_complex(Value::Num(if b { 1.0 } else { 0.0 })).await,
        other => Err(cos_error_with_detail(
            &COS_ERROR_INTERNAL,
            format!("cannot convert value {other:?} to complex output via 'like'"),
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorStorage, HostTensorOwned,
        HostTensorView, ProviderPrecision,
    };
    use runmat_builtins::{IntValue, NumericDType, ResolveContext, StringArray, Tensor, Type};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
    use std::sync::Mutex;

    use crate::builtins::common::{gpu_helpers, test_support};

    fn cos_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::cos_builtin(value, rest))
    }

    struct MalformedCosProvider {
        device_id: u32,
        next_buffer: AtomicU64,
        malformed: AtomicU8,
        allocations: AtomicUsize,
        frees: AtomicUsize,
        buffers: Mutex<HashMap<u64, HostTensorOwned>>,
    }

    impl MalformedCosProvider {
        fn new() -> Self {
            Self {
                device_id: runmat_accelerate_api::next_device_id(),
                next_buffer: AtomicU64::new(8_700_000_000_000_000_000),
                malformed: AtomicU8::new(0),
                allocations: AtomicUsize::new(0),
                frees: AtomicUsize::new(0),
                buffers: Mutex::new(HashMap::new()),
            }
        }

        fn allocate(
            &self,
            data: Vec<f64>,
            shape: Vec<usize>,
            device_id: u32,
            precision: ProviderPrecision,
            storage: GpuTensorStorage,
        ) -> GpuTensorHandle {
            let buffer_id = self.next_buffer.fetch_add(1, Ordering::Relaxed);
            self.buffers.lock().unwrap().insert(
                buffer_id,
                HostTensorOwned {
                    data,
                    shape: shape.clone(),
                    storage,
                },
            );
            self.allocations.fetch_add(1, Ordering::Relaxed);
            let handle = GpuTensorHandle {
                shape,
                device_id,
                buffer_id,
                descriptor: runmat_accelerate_api::GpuTensorDescriptor::numeric(
                    match precision {
                        ProviderPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                        ProviderPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                    },
                    storage,
                ),
            };
            runmat_accelerate_api::set_handle_precision(&handle, precision);
            runmat_accelerate_api::set_handle_storage(&handle, storage);
            handle
        }
    }

    impl AccelProvider for MalformedCosProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(self.allocate(
                host.data.to_vec(),
                host.shape.to_vec(),
                self.device_id,
                ProviderPrecision::F64,
                GpuTensorStorage::Real,
            ))
        }

        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                self.buffers
                    .lock()
                    .unwrap()
                    .get(&handle.buffer_id)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("unknown test buffer"))
            })
        }

        fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
            if self
                .buffers
                .lock()
                .unwrap()
                .remove(&handle.buffer_id)
                .is_some()
            {
                self.frees.fetch_add(1, Ordering::Relaxed);
            }
            runmat_accelerate_api::clear_handle_precision(handle);
            runmat_accelerate_api::clear_handle_storage(handle);
            Ok(())
        }

        fn device_info(&self) -> String {
            "malformed-cos-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            self.device_id
        }

        fn unary_cos<'a>(
            &'a self,
            input: &'a GpuTensorHandle,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move {
                let malformed = self.malformed.load(Ordering::Relaxed);
                let device_id = if malformed == 2 {
                    self.device_id.wrapping_add(10_000)
                } else {
                    self.device_id
                };
                let precision = if malformed == 0 {
                    ProviderPrecision::F32
                } else {
                    ProviderPrecision::F64
                };
                let storage = if malformed == 1 {
                    GpuTensorStorage::ComplexInterleaved
                } else {
                    GpuTensorStorage::Real
                };
                Ok(self.allocate(
                    vec![99.0; input.shape.iter().product()],
                    input.shape.clone(),
                    device_id,
                    precision,
                    storage,
                ))
            })
        }
    }

    #[test]
    fn cos_descriptor_signatures_cover_like_overload() {
        let labels: Vec<&str> = COS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = cos(X)"));
        assert!(labels.contains(&"Y = cos(X, \"like\", P)"));
        assert_eq!(COS_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    }

    #[test]
    fn cos_integer_gate_all_classes_boundary_and_single_precision() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(super::cos_builtin(Value::Int(IntValue::I8(1)), Vec::new()))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            err.identifier(),
            COS_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(_strict);

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for value in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ] {
            assert!(block_on(super::cos_builtin(Value::Int(value), Vec::new())).is_ok());
        }
        assert!(block_on(super::cos_builtin(
            Value::Int(IntValue::U64((1_u64 << 53) + 1)),
            Vec::new(),
        ))
        .is_err());
        assert!(block_on(super::cos_builtin(
            Value::Int(IntValue::U64(1_u64 << 54)),
            Vec::new(),
        ))
        .is_ok());

        let single = Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let Value::Tensor(single_out) =
            block_on(super::cos_builtin(Value::Tensor(single), Vec::new())).unwrap()
        else {
            panic!("expected single tensor")
        };
        assert_eq!(single_out.numeric_dtype(), NumericDType::F32);
        let complex = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(complex_out) = block_on(super::cos_builtin(
            Value::ComplexTensor(complex),
            Vec::new(),
        ))
        .unwrap() else {
            panic!("expected single complex tensor")
        };
        assert_eq!(complex_out.numeric_dtype(), NumericDType::F32);
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn cos_type_preserves_tensor_shape() {
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
    fn cos_type_scalar_tensor_returns_num() {
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
    fn cos_scalar_zero() {
        let result = cos_builtin(Value::Num(0.0), Vec::new()).expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let result = cos_builtin(Value::Tensor(tensor), Vec::new()).expect("cos");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.materialize_f64()[0] - 1.0).abs() < 1e-12);
                assert!((t.materialize_f64()[1] + 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(vec![0, 1, 2]),
            vec![3, 1],
        )
        .expect("integer tensor");

        let result = cos_builtin(Value::Tensor(tensor), Vec::new()).expect("cos");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [1.0, 1.0f64.cos(), 2.0f64.cos()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cos_host_complex_conversion_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I64(vec![-3, 0, 5]),
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
    fn cos_int_value_promotes() {
        let value = Value::Int(IntValue::I32(1));
        let result = cos_builtin(value, Vec::new()).expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.cos()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_complex_scalar() {
        let result = cos_builtin(Value::Complex(1.0, 2.0), Vec::new()).expect("cos");
        match result {
            Value::Complex(re, im) => {
                assert!((re - (1.0f64.cos() * 2.0f64.cosh())).abs() < 1e-12);
                assert!((im + (1.0f64.sin() * 2.0f64.sinh())).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_char_array_roundtrip() {
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = cos_builtin(Value::CharArray(chars), Vec::new()).expect("cos");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).cos();
                    assert!((t.materialize_f64()[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cos_builtin(Value::GpuTensor(handle), Vec::new()).expect("cos");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[test]
    fn cos_rejects_and_frees_malformed_native_outputs_before_owner_restoration() {
        let _guard = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(MalformedCosProvider::new()));
        unsafe {
            runmat_accelerate_api::register_provider(provider);
        }

        for malformed in 0..3_u8 {
            provider.malformed.store(malformed, Ordering::Relaxed);
            let input = provider
                .upload(&HostTensorView {
                    data: &[0.0, 1.0],
                    shape: &[2, 1],
                })
                .expect("input upload");
            let Value::GpuTensor(output) = block_on(super::cos_gpu(input.clone()))
                .expect("malformed native output must fall back and restore")
            else {
                panic!("fallback must restore residency")
            };
            assert_eq!(output.device_id, provider.device_id());
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(ProviderPrecision::F64)
            );
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                GpuTensorStorage::Real
            );
            let gathered = block_on(provider.download(&output)).expect("restored output");
            assert_eq!(gathered.data, vec![1.0, 1.0_f64.cos()]);

            let completed = usize::from(malformed) + 1;
            assert_eq!(provider.allocations.load(Ordering::Relaxed), completed * 3);
            assert_eq!(provider.frees.load(Ordering::Relaxed), completed * 3 - 2);
            provider.free(&input).expect("free input");
            provider.free(&output).expect("free restored output");
            assert_eq!(provider.frees.load(Ordering::Relaxed), completed * 3);
        }
        assert_eq!(
            provider.allocations.load(Ordering::Relaxed),
            provider.frees.load(Ordering::Relaxed)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_missing_prototype_errors() {
        let err =
            cos_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert_eq!(err.identifier(), COS_ERROR_INVALID_OPTION.identifier);
        let message = error_message(err);
        assert!(message.contains("prototype"));
    }

    #[test]
    fn cos_validates_like_syntax_before_extension_gate() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(super::cos_builtin(
            Value::Int(IntValue::U8(1)),
            vec![Value::from("like")],
        ))
        .expect_err("malformed like syntax");
        assert_eq!(error.identifier(), COS_ERROR_INVALID_OPTION.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_complex_prototype_returns_complex() {
        let result = cos_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("cos");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 1.0_f64.cos()).abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn cos_gpu_complex_input_preserves_complex_output() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(0.5, 0.75), (2.0, -0.25)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            let result = cos_builtin(Value::GpuTensor(handle), Vec::new()).expect("cos");
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
                assert!((out.materialize_f64()[idx].0 - cos_complex_re(re, im)).abs() < 1e-12);
                assert!((out.materialize_f64()[idx].1 - cos_complex_im(re, im)).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn cos_like_complex_gpu_prototype_uploads_complex_result() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = cos_builtin(
                Value::Tensor(input.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("cos");
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
                assert!((out.materialize_f64()[idx].0 - re.cos()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[test]
    fn cos_like_complex_gpu_prototype_converts_resident_real_gpu_result() {
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
            let result = cos_builtin(
                Value::GpuTensor(input_handle),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("cos");
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
                assert!((out.materialize_f64()[idx].0 - re.cos()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = cos_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("cos");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
                    assert_eq!(gathered.shape, vec![4, 1]);
                    assert_eq!(gathered.materialize_f64(), expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn cos_like_gpu_prototype_controls_owner_across_providers() {
        let _guard = test_support::accel_test_lock();
        let input_provider = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        let prototype_provider = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        unsafe {
            runmat_accelerate_api::register_provider(input_provider);
            runmat_accelerate_api::register_provider(prototype_provider);
        }
        let input = input_provider
            .upload(&HostTensorView {
                data: &[0.0, 1.0],
                shape: &[2, 1],
            })
            .expect("upload input");
        let prototype = prototype_provider
            .upload(&HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            })
            .expect("upload prototype");

        let result = cos_builtin(
            Value::GpuTensor(input),
            vec![Value::from("like"), Value::GpuTensor(prototype)],
        )
        .expect("mixed-provider like conversion");
        let Value::GpuTensor(output) = result else {
            panic!("expected GPU output")
        };
        assert_eq!(output.device_id, prototype_provider.device_id());
        let output_owner =
            runmat_accelerate_api::provider_for_handle(&output).expect("registered output owner");
        assert!(std::ptr::eq(
            output_owner,
            prototype_provider as &dyn runmat_accelerate_api::AccelProvider
        ));
        let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather output");
        assert_eq!(gathered.materialize_f64(), vec![1.0, 1.0_f64.cos()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_host_with_gpu_input_gathers() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cos_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("cos");
            match result {
                Value::Tensor(t) => {
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
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
    fn cos_like_rejects_extra_arguments() {
        let err = cos_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_keyword_case_insensitive() {
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result = cos_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> =
                    tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.materialize_f64(), expected);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_char_array_keyword() {
        let keyword = CharArray::new_row("like");
        let result = cos_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_string_array_keyword() {
        let keyword = StringArray::new(vec!["LIKE".to_string()], vec![1]).unwrap();
        let result = cos_builtin(
            Value::Num(0.0),
            vec![Value::StringArray(keyword), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_unrecognised_option_errors() {
        let err =
            cos_builtin(Value::Num(0.0), vec![Value::from("invalid")]).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("unrecognised argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cos_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let cpu = cos_real(Value::Tensor(t.clone())).unwrap();
        let view = HostTensorView {
            data: &t.materialize_f64(),
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(cos_gpu(h)).unwrap();
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
