//! MATLAB-compatible `ldivide` builtin with GPU-aware semantics for RunMat.

use async_recursion::async_recursion;
use num_complex::{Complex32, Complex64};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexStorage, ComplexTensor, IntegerStorage, NumericDType, NumericStorage, Tensor,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::{
    reject_integer_logical_operands, try_integer_binary, IntegerBinaryOp,
};
use crate::builtins::math::symbolic::{symbolic_binary, SymbolicBinaryOp};
use crate::builtins::math::type_resolvers::numeric_binary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::ldivide")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ldivide",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "elem_div",
            commutative: false,
        },
        ProviderHook::Custom("scalar_div"),
        ProviderHook::Custom("scalar_rdiv"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses validated exact-owner elem_div for supported same-class resident pairs, including provider-supported implicit expansion and typed-integer division; scalar_div/scalar_rdiv retain floating scalar cases, while exact integer scalar values are uploaded in the resident integer class. Unsupported or mixed-class cases use a class-preserving host fallback before 'like' prototypes are honoured.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::ldivide")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ldivide",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let divisor = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let numerator = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("({numerator} / {divisor})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a plain quotient; providers can override with specialised kernels when desirable.",
};

const BUILTIN_NAME: &str = "ldivide";

pub const LDIVIDE_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ldivide-like-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ldivide with a 'like' output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LdivideLikePrototypeExtension"),
};
pub const LDIVIDE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [LDIVIDE_LIKE_EXTENSION];
const LDIVIDE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "A", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "An integer divisor requires the dividend to use the same integer class or to be scalar double; complex integer division is rejected." },
    BuiltinIntegerInputCapability { name: "B", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "An integer dividend requires the divisor to use the same integer class or to be scalar double; compatible dimensions expand implicitly." },
];
pub const LDIVIDE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "C = ldivide(A, B)", inputs: &LDIVIDE_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::BroadcastCompatible, notes: "Integer quotients round to nearest with half ties away from zero and saturate to the integer class. Resident fallback reads authoritative storage and restores the result to the owning provider." }];

const LDIVIDE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise left quotient result.",
}];

const LDIVIDE_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left divisor operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right dividend operand.",
    },
];

const LDIVIDE_INPUTS_A_B_LIKE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left divisor operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right dividend operand.",
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

const LDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = ldivide(A, B)",
        inputs: &LDIVIDE_INPUTS_A_B,
        outputs: &LDIVIDE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = ldivide(A, B, \"like\", prototype)",
        inputs: &LDIVIDE_INPUTS_A_B_LIKE,
        outputs: &LDIVIDE_OUTPUT,
    },
];

const LDIVIDE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LDIVIDE.INVALID_ARGUMENT",
    identifier: Some("RunMat:ldivide:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "ldivide: invalid argument",
};

const LDIVIDE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LDIVIDE.INVALID_INPUT",
    identifier: Some("RunMat:ldivide:InvalidInput"),
    when: "Operands or prototypes cannot be converted into supported numeric/logical forms.",
    message: "ldivide: invalid input",
};

const LDIVIDE_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LDIVIDE.SIZE_MISMATCH",
    identifier: Some("RunMat:ldivide:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "ldivide: array sizes are not compatible for broadcasting",
};

const LDIVIDE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LDIVIDE.INTERNAL",
    identifier: Some("RunMat:ldivide:Internal"),
    when: "Provider interaction, gather/upload, or internal tensor construction failed.",
    message: "ldivide: internal error",
};

const LDIVIDE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    LDIVIDE_ERROR_INVALID_ARGUMENT,
    LDIVIDE_ERROR_INVALID_INPUT,
    LDIVIDE_ERROR_SIZE_MISMATCH,
    LDIVIDE_ERROR_INTERNAL,
];

pub const LDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LDIVIDE_ERRORS,
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn ldivide_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ldivide_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "ldivide",
    category = "math/elementwise",
    summary = "Compute element-wise left division (B ./ A).",
    keywords = "ldivide,element-wise left division,gpu,.\\",
    accel = "elementwise",
    type_resolver(numeric_binary_type),
    extensions(LDIVIDE_EXTENSIONS),
    integer_capabilities(LDIVIDE_INTEGER_CAPABILITIES),
    descriptor(crate::builtins::math::elementwise::ldivide::LDIVIDE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::ldivide"
)]
async fn ldivide_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::is_typed_complex_integer(&lhs)
        || crate::builtins::common::validation::is_typed_complex_integer(&rhs)
        || rest
            .iter()
            .any(crate::builtins::common::validation::is_typed_complex_integer)
    {
        return Err(builtin_error("complex integer arithmetic is not supported"));
    }
    reject_integer_logical_operands(&lhs, &rhs, BUILTIN_NAME).map_err(builtin_error)?;
    validate_integer_admission(&lhs, &rhs)?;
    let template = parse_output_template(&rest)?;
    if matches!(template, OutputTemplate::Like(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LDIVIDE_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let source = preferred_ldivide_source(&lhs, &rhs);
    let native = match (&lhs, &rhs) {
        (Value::GpuTensor(divisor), Value::GpuTensor(numerator)) => {
            try_ldivide_gpu_pair(divisor, numerator).await?
        }
        (Value::GpuTensor(divisor), numerator) => {
            try_ldivide_gpu_host_left(divisor, numerator).await?
        }
        (divisor, Value::GpuTensor(numerator)) => {
            try_ldivide_gpu_host_right(divisor, numerator).await?
        }
        _ => None,
    };
    let base = if let Some(handle) = native {
        let provenance = [&lhs, &rhs]
            .into_iter()
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => runmat_accelerate_api::handle_provenance(handle),
                _ => None,
            })
            .find(|provenance| *provenance == runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        runmat_accelerate_api::set_handle_provenance(&handle, provenance);
        gpu_helpers::resident_gpu_value(handle)
    } else if let Some(source) = source.as_ref() {
        let lhs = gather_ldivide_operand(lhs).await?;
        let rhs = gather_ldivide_operand(rhs).await?;
        let host = ldivide_host(lhs, rhs)?;
        restore_ldivide_residency(source, host)?
    } else {
        ldivide_host(lhs, rhs)?
    };
    apply_output_template(base, &template).await
}

fn preferred_ldivide_source(lhs: &Value, rhs: &Value) -> Option<GpuTensorHandle> {
    let handles = [lhs, rhs].map(|value| match value {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    });
    handles
        .iter()
        .flatten()
        .find(|handle| runmat_accelerate_api::handle_is_explicit(handle))
        .or_else(|| handles.iter().flatten().next())
        .map(|handle| (*handle).clone())
}

fn integer_class(value: &Value) -> Option<&'static str> {
    match value {
        Value::Int(value) => Some(value.class_name()),
        Value::Tensor(tensor) => tensor.integer_storage().map(IntegerStorage::class_name),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).map(|ty| match ty {
                runmat_accelerate_api::IntegerElementType::I8 => "int8",
                runmat_accelerate_api::IntegerElementType::I16 => "int16",
                runmat_accelerate_api::IntegerElementType::I32 => "int32",
                runmat_accelerate_api::IntegerElementType::I64 => "int64",
                runmat_accelerate_api::IntegerElementType::U8 => "uint8",
                runmat_accelerate_api::IntegerElementType::U16 => "uint16",
                runmat_accelerate_api::IntegerElementType::U32 => "uint32",
                runmat_accelerate_api::IntegerElementType::U64 => "uint64",
            })
        }
        _ => None,
    }
}

fn is_scalar_double(value: &Value) -> bool {
    match value {
        Value::Num(_) => true,
        Value::Tensor(tensor) => {
            tensor::is_scalar_tensor(tensor) && tensor.numeric_dtype() == NumericDType::F64
        }
        Value::GpuTensor(handle) => {
            is_scalar_shape(&handle.shape)
                && runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F64)
        }
        _ => false,
    }
}

fn validate_integer_admission(lhs: &Value, rhs: &Value) -> BuiltinResult<()> {
    match (integer_class(lhs), integer_class(rhs)) {
        (None, None) => Ok(()),
        (Some(left), Some(right)) if left == right => Ok(()),
        (Some(_), Some(_)) => Err(builtin_error(
            "ldivide: integer operands must have the same integer class",
        )),
        (Some(_), None) if is_scalar_double(rhs) => Ok(()),
        (None, Some(_)) if is_scalar_double(lhs) => Ok(()),
        _ => Err(builtin_error(
            "ldivide: integer arrays can only be combined with scalar double values",
        )),
    }
}

async fn gather_ldivide_operand(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME)),
        _ => Ok(value),
    }
}

fn restore_ldivide_residency(source: &GpuTensorHandle, mut host: Value) -> BuiltinResult<Value> {
    if matches!(host, Value::Num(_) | Value::Int(_) | Value::Bool(_)) {
        host = Value::Tensor(
            tensor::value_into_tensor_for(BUILTIN_NAME, host)
                .map_err(|error| builtin_error(format!("ldivide: {error}")))?,
        );
    }
    let restored = gpu_helpers::restore_class_preserving_value(source, host, BUILTIN_NAME)?;
    if runmat_accelerate_api::handle_is_explicit(source) && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(builtin_error(
            "ldivide: provider cannot preserve explicit gpuArray output",
        ));
    }
    Ok(restored)
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> BuiltinResult<OutputTemplate> {
    if args.is_empty() {
        return Ok(OutputTemplate::Default);
    }
    if args.len() == 1 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Err(ldivide_error_with_detail(
                &LDIVIDE_ERROR_INVALID_ARGUMENT,
                "expected prototype after 'like'",
            ));
        }
        return Err(ldivide_error_with_detail(
            &LDIVIDE_ERROR_INVALID_ARGUMENT,
            "unsupported option; only 'like' is accepted",
        ));
    }
    if args.len() == 2 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Ok(OutputTemplate::Like(args[1].clone()));
        }
        return Err(ldivide_error_with_detail(
            &LDIVIDE_ERROR_INVALID_ARGUMENT,
            "unsupported option; only 'like' is accepted",
        ));
    }
    Err(ldivide_error_with_detail(
        &LDIVIDE_ERROR_INVALID_ARGUMENT,
        "too many input arguments",
    ))
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => apply_like_template(value, proto).await,
    }
}

#[derive(Clone, Copy)]
enum PrototypeClass {
    Real,
    Complex,
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

struct LikeAnalysis {
    device: DevicePreference,
    class: PrototypeClass,
}

async fn apply_like_template(value: Value, prototype: &Value) -> BuiltinResult<Value> {
    let analysed = analyse_like_prototype(prototype).await?;
    match analysed.class {
        PrototypeClass::Real => match analysed.device {
            DevicePreference::Host => ensure_device(value, DevicePreference::Host).await,
            DevicePreference::Gpu => ensure_device(value, DevicePreference::Gpu).await,
        },
        PrototypeClass::Complex => {
            let host_value = ensure_device(value, DevicePreference::Host).await?;
            real_to_complex(host_value).await
        }
    }
}

async fn ensure_device(value: Value, device: DevicePreference) -> BuiltinResult<Value> {
    match device {
        DevicePreference::Host => convert_to_host_like(value).await,
        DevicePreference::Gpu => convert_to_gpu(value),
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = value {
        let temp = Value::GpuTensor(handle);
        gpu_helpers::gather_value_async(&temp)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))
            .map_err(|e| builtin_error(format!("ldivide: {e}")))
    } else {
        Ok(value)
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(ldivide_error_with_detail(
            &LDIVIDE_ERROR_INVALID_ARGUMENT,
            "GPU output requested via 'like' but no acceleration provider is active",
        ));
    };
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let handle = gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|e| builtin_error(format!("ldivide: failed to upload GPU result: {e}")))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => {
            let tensor = Tensor::new_integer(IntegerStorage::from_scalar(i), vec![1, 1])
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(ldivide_error_with_detail(
            &LDIVIDE_ERROR_INVALID_ARGUMENT,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        Value::String(_)
        | Value::StringArray(_)
        | Value::SparseTensor(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::Symbolic(_) => Err(ldivide_error_with_detail(
            &LDIVIDE_ERROR_INVALID_ARGUMENT,
            "unsupported prototype conversion to GPU output",
        )),
        Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(ldivide_error_with_detail(
            &LDIVIDE_ERROR_INVALID_ARGUMENT,
            "unsupported prototype conversion to GPU output",
        )),
    }
}

#[async_recursion(?Send)]
async fn analyse_like_prototype(proto: &Value) -> BuiltinResult<LikeAnalysis> {
    match proto {
        Value::GpuTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Gpu,
            class: PrototypeClass::Real,
        }),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Real,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Complex,
        }),
        other => {
            let gathered = gather_like_prototype(other).await?;
            analyse_like_prototype(&gathered).await
        }
    }
}

async fn gather_like_prototype(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(value)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))
            .map_err(|e| builtin_error(format!("ldivide: {e}"))),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_) => Ok(value.clone()),
        _ => Err(ldivide_error_with_detail(
            &LDIVIDE_ERROR_INVALID_ARGUMENT,
            format!("unsupported prototype for 'like' ({value:?})"),
        )),
    }
}

#[async_recursion(?Send)]
async fn real_to_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(t) => {
            let shape = t.shape.clone();
            let storage = t
                .into_numeric_storage()
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
            let storage = match storage {
                NumericStorage::F64(values) => {
                    ComplexStorage::F64(values.into_iter().map(|value| (value, 0.0)).collect())
                }
                NumericStorage::F32(values) => {
                    ComplexStorage::F32(values.into_iter().map(|value| (value, 0.0)).collect())
                }
                storage => promote_integer_real_storage_to_complex(storage),
            };
            let tensor = ComplexTensor::from_complex_storage(storage, shape)
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
            real_to_complex(Value::Tensor(tensor)).await
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            real_to_complex(Value::Tensor(tensor)).await
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            real_to_complex(gathered).await
        }
        other => Err(ldivide_error_with_detail(
            &LDIVIDE_ERROR_INVALID_INPUT,
            format!("cannot convert value {other:?} to complex output"),
        )),
    }
}

fn promote_integer_real_storage_to_complex(storage: NumericStorage) -> ComplexStorage {
    ComplexStorage::F64(
        storage
            .materialize_f64()
            .into_iter()
            .map(|value| (value, 0.0))
            .collect(),
    )
}

async fn try_ldivide_gpu_pair(
    divisor: &GpuTensorHandle,
    numerator: &GpuTensorHandle,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let Some(provider) = common_ldivide_owner(divisor, numerator) else {
        return Ok(None);
    };
    if !ldivide_pair_metadata_supported(divisor, numerator) {
        return Ok(None);
    }
    let expected_shape = BroadcastPlan::new(&divisor.shape, &numerator.shape)
        .map_err(|_| ldivide_error(&LDIVIDE_ERROR_SIZE_MISMATCH))?
        .output_shape()
        .to_vec();
    let output = match provider.elem_div(numerator, divisor).await {
        Ok(output) => output,
        Err(_) => return Ok(None),
    };
    if validate_ldivide_output(&output, divisor, Some(numerator), provider, &expected_shape) {
        Ok(Some(output))
    } else {
        gpu_helpers::free_unprotected_exact_owner(&output, &[divisor, numerator]);
        Ok(None)
    }
}

async fn try_ldivide_gpu_host_left(
    divisor: &GpuTensorHandle,
    numerator: &Value,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let Some(provider) = gpu_helpers::exact_provider_for_handle(divisor) else {
        return Ok(None);
    };
    if let Value::Num(scalar) = numerator {
        if runmat_accelerate_api::handle_integer_type(divisor).is_some() {
            if let Some(uploaded) =
                gpu_helpers::upload_exact_integer_scalar_like(provider, divisor, *scalar)
            {
                let result = provider.elem_div(&uploaded, divisor).await.ok();
                let valid = result.as_ref().is_some_and(|output| {
                    validate_ldivide_output(
                        output,
                        divisor,
                        Some(&uploaded),
                        provider,
                        &divisor.shape,
                    )
                });
                if let Some(output) = result.as_ref().filter(|_| !valid) {
                    gpu_helpers::free_unprotected_exact_owner(output, &[divisor, &uploaded]);
                }
                gpu_helpers::free_unprotected_exact_owner(&uploaded, &[divisor]);
                return Ok(result.filter(|_| valid));
            }
        } else if ldivide_float_handle_supported(divisor) {
            if let Ok(output) = provider.scalar_rdiv(divisor, *scalar) {
                if validate_ldivide_output(&output, divisor, None, provider, &divisor.shape) {
                    return Ok(Some(output));
                }
                gpu_helpers::free_unprotected_exact_owner(&output, &[divisor]);
            }
        }
    }
    Ok(None)
}

async fn try_ldivide_gpu_host_right(
    divisor: &Value,
    numerator: &GpuTensorHandle,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let Some(provider) = gpu_helpers::exact_provider_for_handle(numerator) else {
        return Ok(None);
    };
    if let Value::Num(scalar) = divisor {
        if runmat_accelerate_api::handle_integer_type(numerator).is_some() {
            if let Some(uploaded) =
                gpu_helpers::upload_exact_integer_scalar_like(provider, numerator, *scalar)
            {
                let result = provider.elem_div(numerator, &uploaded).await.ok();
                let valid = result.as_ref().is_some_and(|output| {
                    validate_ldivide_output(
                        output,
                        numerator,
                        Some(&uploaded),
                        provider,
                        &numerator.shape,
                    )
                });
                if let Some(output) = result.as_ref().filter(|_| !valid) {
                    gpu_helpers::free_unprotected_exact_owner(output, &[numerator, &uploaded]);
                }
                gpu_helpers::free_unprotected_exact_owner(&uploaded, &[numerator]);
                return Ok(result.filter(|_| valid));
            }
        } else if ldivide_float_handle_supported(numerator) {
            if let Ok(output) = provider.scalar_div(numerator, *scalar) {
                if validate_ldivide_output(&output, numerator, None, provider, &numerator.shape) {
                    return Ok(Some(output));
                }
                gpu_helpers::free_unprotected_exact_owner(&output, &[numerator]);
            }
        }
    }
    Ok(None)
}

fn common_ldivide_owner(
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    let left_owner = gpu_helpers::exact_provider_for_handle(left)?;
    let right_owner = gpu_helpers::exact_provider_for_handle(right)?;
    (std::ptr::eq(left_owner, right_owner) && left.device_id == right.device_id)
        .then_some(left_owner)
}

fn ldivide_float_handle_supported(handle: &GpuTensorHandle) -> bool {
    runmat_accelerate_api::handle_storage(handle) == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && runmat_accelerate_api::handle_precision(handle).is_some()
}

fn ldivide_pair_metadata_supported(left: &GpuTensorHandle, right: &GpuTensorHandle) -> bool {
    if runmat_accelerate_api::handle_storage(left) != runmat_accelerate_api::GpuTensorStorage::Real
        || runmat_accelerate_api::handle_storage(right)
            != runmat_accelerate_api::GpuTensorStorage::Real
        || runmat_accelerate_api::handle_is_logical(left)
        || runmat_accelerate_api::handle_is_logical(right)
    {
        return false;
    }
    match (
        runmat_accelerate_api::handle_integer_type(left),
        runmat_accelerate_api::handle_integer_type(right),
    ) {
        (Some(left), Some(right)) => left == right,
        (None, None) => {
            runmat_accelerate_api::handle_precision(left)
                == runmat_accelerate_api::handle_precision(right)
                && runmat_accelerate_api::handle_precision(left).is_some()
        }
        _ => false,
    }
}

fn validate_ldivide_output(
    output: &GpuTensorHandle,
    class_source: &GpuTensorHandle,
    other_source: Option<&GpuTensorHandle>,
    owner: &dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
) -> bool {
    output.shape == expected_shape
        && output.device_id == class_source.device_id
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|output_owner| std::ptr::eq(output_owner, owner))
        && !gpu_helpers::same_gpu_handle(output, class_source)
        && other_source.is_none_or(|source| !gpu_helpers::same_gpu_handle(output, source))
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output)
            == runmat_accelerate_api::handle_integer_type(class_source)
        && runmat_accelerate_api::handle_is_logical(output)
            == runmat_accelerate_api::handle_is_logical(class_source)
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(class_source)
}

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            Some(ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0))
        }
        _ => None,
    }
}

fn scalar_complex_value(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Complex(re, im) => Some((*re, *im)),
        _ => None,
    }
}

fn scalar_ldivide_value(divisor: &Value, numerator: &Value) -> Option<Value> {
    if matches!(divisor, Value::Tensor(_) | Value::ComplexTensor(_))
        || matches!(numerator, Value::Tensor(_) | Value::ComplexTensor(_))
    {
        return None;
    }
    let num = scalar_complex_value(numerator)
        .or_else(|| scalar_real_value(numerator).map(|v| (v, 0.0)))?;
    let div =
        scalar_complex_value(divisor).or_else(|| scalar_real_value(divisor).map(|v| (v, 0.0)))?;
    let (nr, ni) = num;
    let (dr, di) = div;
    if ni != 0.0 || di != 0.0 {
        let quotient = Complex64::new(nr, ni) / Complex64::new(dr, di);
        return Some(Value::Complex(quotient.re, quotient.im));
    }
    Some(Value::Num(nr / dr))
}

fn ldivide_host(divisor: Value, numerator: Value) -> BuiltinResult<Value> {
    if let Some(result) = symbolic_binary(&numerator, &divisor, SymbolicBinaryOp::Div) {
        return Ok(result);
    }
    if let Some(result) =
        try_integer_binary(&numerator, &divisor, IntegerBinaryOp::Divide, BUILTIN_NAME)
            .map_err(builtin_error)?
    {
        return Ok(result);
    }
    if let Some(result) = scalar_ldivide_value(&divisor, &numerator) {
        return Ok(result);
    }
    match (classify_operand(divisor)?, classify_operand(numerator)?) {
        (LdivideOperand::Real(div), LdivideOperand::Real(num)) => ldivide_real_real(&div, &num),
        (LdivideOperand::Complex(div), LdivideOperand::Complex(num)) => {
            ldivide_complex_complex(&div, &num)
        }
        (LdivideOperand::Complex(div), LdivideOperand::Real(num)) => {
            ldivide_complex_real(&div, &num)
        }
        (LdivideOperand::Real(div), LdivideOperand::Complex(num)) => {
            ldivide_real_complex(&div, &num)
        }
    }
}

fn ldivide_real_real(divisor: &Tensor, numerator: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| ldivide_error_with_detail(&LDIVIDE_ERROR_SIZE_MISMATCH, &err))?;
    let numerator = numerator
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    let divisor = divisor
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    let output = match (numerator, divisor) {
        (NumericStorage::F32(numerator), NumericStorage::F32(divisor)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, numerator_index, divisor_index) in plan.iter() {
                output[output_index] = numerator[numerator_index] / divisor[divisor_index];
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(numerator), NumericStorage::F64(divisor)) => {
            let mut output = vec![0.0f64; plan.len()];
            for (output_index, numerator_index, divisor_index) in plan.iter() {
                output[output_index] = numerator[numerator_index] / divisor[divisor_index];
            }
            NumericStorage::F64(output)
        }
        (NumericStorage::F32(numerator), NumericStorage::F64(divisor)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, numerator_index, divisor_index) in plan.iter() {
                output[output_index] =
                    (f64::from(numerator[numerator_index]) / divisor[divisor_index]) as f32;
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(numerator), NumericStorage::F32(divisor)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, numerator_index, divisor_index) in plan.iter() {
                output[output_index] =
                    (numerator[numerator_index] / f64::from(divisor[divisor_index])) as f32;
            }
            NumericStorage::F32(output)
        }
        _ => {
            return Err(builtin_error(
                "ldivide: integer operands did not use the exact integer arithmetic path",
            ))
        }
    };
    let tensor = Tensor::from_numeric_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn ldivide_complex_complex(
    divisor: &ComplexTensor,
    numerator: &ComplexTensor,
) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| ldivide_error_with_detail(&LDIVIDE_ERROR_SIZE_MISMATCH, &err))?;
    let output = match (numerator.complex_storage(), divisor.complex_storage()) {
        (ComplexStorage::F64(numerator), ComplexStorage::F64(divisor)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, numerator_index, divisor_index) in plan.iter() {
                output[output_index] =
                    divide_complex_f64(numerator[numerator_index], divisor[divisor_index]);
            }
            ComplexStorage::F64(output)
        }
        (ComplexStorage::F32(numerator), ComplexStorage::F32(divisor)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, numerator_index, divisor_index) in plan.iter() {
                output[output_index] =
                    divide_complex_f32(numerator[numerator_index], divisor[divisor_index]);
            }
            ComplexStorage::F32(output)
        }
        (ComplexStorage::F32(numerator), ComplexStorage::F64(divisor)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, numerator_index, divisor_index) in plan.iter() {
                let numerator = (
                    f64::from(numerator[numerator_index].0),
                    f64::from(numerator[numerator_index].1),
                );
                let value = divide_complex_f64(numerator, divisor[divisor_index]);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output)
        }
        (ComplexStorage::F64(numerator), ComplexStorage::F32(divisor)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, numerator_index, divisor_index) in plan.iter() {
                let divisor = (
                    f64::from(divisor[divisor_index].0),
                    f64::from(divisor[divisor_index].1),
                );
                let value = divide_complex_f64(numerator[numerator_index], divisor);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output)
        }
        _ => {
            return Err(builtin_error(
                "ldivide: complex integer arithmetic is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn ldivide_complex_real(divisor: &ComplexTensor, numerator: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| ldivide_error_with_detail(&LDIVIDE_ERROR_SIZE_MISMATCH, &err))?;
    let numerator = numerator
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    let output = divide_real_by_complex_storage(&numerator, divisor.complex_storage(), &plan)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn ldivide_real_complex(divisor: &Tensor, numerator: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| ldivide_error_with_detail(&LDIVIDE_ERROR_SIZE_MISMATCH, &err))?;
    let divisor = divisor
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    let output = divide_complex_by_real_storage(numerator.complex_storage(), &divisor, &plan)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn divide_complex_by_real_storage(
    complex: &ComplexStorage,
    real: &NumericStorage,
    plan: &BroadcastPlan,
) -> BuiltinResult<ComplexStorage> {
    Ok(match (complex, real) {
        (ComplexStorage::F64(complex), NumericStorage::F64(real)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, complex_index, real_index) in plan.iter() {
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (value.0 / scalar, value.1 / scalar);
            }
            ComplexStorage::F64(output)
        }
        (ComplexStorage::F32(complex), NumericStorage::F32(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, complex_index, real_index) in plan.iter() {
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (value.0 / scalar, value.1 / scalar);
            }
            ComplexStorage::F32(output)
        }
        (ComplexStorage::F32(complex), NumericStorage::F64(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, complex_index, real_index) in plan.iter() {
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (
                    (f64::from(value.0) / scalar) as f32,
                    (f64::from(value.1) / scalar) as f32,
                );
            }
            ComplexStorage::F32(output)
        }
        (ComplexStorage::F64(complex), NumericStorage::F32(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, complex_index, real_index) in plan.iter() {
                let value = complex[complex_index];
                let scalar = f64::from(real[real_index]);
                output[output_index] = ((value.0 / scalar) as f32, (value.1 / scalar) as f32);
            }
            ComplexStorage::F32(output)
        }
        _ => {
            return Err(builtin_error(
                "ldivide: integer operands did not use the exact integer arithmetic path",
            ))
        }
    })
}

fn divide_real_by_complex_storage(
    real: &NumericStorage,
    complex: &ComplexStorage,
    plan: &BroadcastPlan,
) -> BuiltinResult<ComplexStorage> {
    Ok(match (real, complex) {
        (NumericStorage::F64(real), ComplexStorage::F64(complex)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, real_index, complex_index) in plan.iter() {
                output[output_index] =
                    divide_complex_f64((real[real_index], 0.0), complex[complex_index]);
            }
            ComplexStorage::F64(output)
        }
        (NumericStorage::F32(real), ComplexStorage::F32(complex)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, real_index, complex_index) in plan.iter() {
                output[output_index] =
                    divide_complex_f32((real[real_index], 0.0), complex[complex_index]);
            }
            ComplexStorage::F32(output)
        }
        (NumericStorage::F32(real), ComplexStorage::F64(complex)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, real_index, complex_index) in plan.iter() {
                let value =
                    divide_complex_f64((f64::from(real[real_index]), 0.0), complex[complex_index]);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output)
        }
        (NumericStorage::F64(real), ComplexStorage::F32(complex)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, real_index, complex_index) in plan.iter() {
                let complex = (
                    f64::from(complex[complex_index].0),
                    f64::from(complex[complex_index].1),
                );
                let value = divide_complex_f64((real[real_index], 0.0), complex);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output)
        }
        _ => {
            return Err(builtin_error(
                "ldivide: integer operands did not use the exact integer arithmetic path",
            ))
        }
    })
}

fn divide_complex_f64(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    let quotient = Complex64::new(lhs.0, lhs.1) / Complex64::new(rhs.0, rhs.1);
    (quotient.re, quotient.im)
}

fn divide_complex_f32(lhs: (f32, f32), rhs: (f32, f32)) -> (f32, f32) {
    let quotient = Complex32::new(lhs.0, lhs.1) / Complex32::new(rhs.0, rhs.1);
    (quotient.re, quotient.im)
}

enum LdivideOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> BuiltinResult<LdivideOperand> {
    match value {
        Value::Tensor(t) => Ok(LdivideOperand::Real(t)),
        Value::Num(n) => Ok(LdivideOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| builtin_error(format!("ldivide: {e}")))?,
        )),
        Value::Bool(b) => Ok(LdivideOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?,
        )),
        Value::LogicalArray(logical) => Ok(LdivideOperand::Real(
            tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?,
        )),
        Value::CharArray(chars) => Ok(LdivideOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(LdivideOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?,
        )),
        Value::ComplexTensor(ct) => Ok(LdivideOperand::Complex(ct)),
        Value::GpuTensor(_) => Err(ldivide_error(&LDIVIDE_ERROR_INTERNAL)),
        other => Err(ldivide_error_with_detail(
            &LDIVIDE_ERROR_INVALID_INPUT,
            format!(
                "unsupported operand type {:?}; expected numeric or logical data",
                other
            ),
        )),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| builtin_error(format!("ldivide: {e}")))
}

fn is_scalar_shape(shape: &[usize]) -> bool {
    shape.iter().copied().product::<usize>() <= 1
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, ResolveContext,
        SymbolicExpr, Tensor, Type,
    };

    const EPS: f64 = 1e-12;
    const GPU_EPS: f64 = 1e-6;

    fn ldivide_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ldivide_builtin(lhs, rhs, rest))
    }

    #[test]
    fn ldivide_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = LDIVIDE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"C = ldivide(A, B)"));
        assert!(labels.contains(&"C = ldivide(A, B, \"like\", prototype)"));
    }

    #[test]
    fn ldivide_parser_error_has_stable_identifier() {
        let err = ldivide_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
            .expect_err("expected parser error");
        assert_eq!(err.identifier(), LDIVIDE_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn ldivide_type_preserves_tensor_shape() {
        let out = numeric_binary_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
            ],
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
    fn ldivide_type_scalar_returns_num() {
        let out = numeric_binary_type(&[Type::Num, Type::Int], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_scalar_numbers() {
        let result =
            ldivide_builtin(Value::Num(7.0), Value::Num(2.0), Vec::new()).expect("ldivide");
        match result {
            Value::Num(v) => assert!((v - (2.0 / 7.0)).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_integer_arrays_preserve_exact_reversed_division() {
        let divisor = Tensor::new_integer(IntegerStorage::I64(vec![2, 2, 0]), vec![1, 3])
            .expect("integer tensor");
        let numerator = Tensor::new_integer(IntegerStorage::I64(vec![-3, 3, i64::MAX]), vec![1, 3])
            .expect("integer tensor");
        let result = ldivide_builtin(Value::Tensor(divisor), Value::Tensor(numerator), Vec::new())
            .expect("integer ldivide");
        let Value::Tensor(result) = result else {
            panic!("expected typed integer tensor");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::I64(vec![-2, 2, i64::MAX]))
        );
    }

    #[test]
    fn ldivide_mixed_floating_arrays_return_single() {
        let divisor = Tensor::from_f32(vec![2.0, 4.0], vec![1, 2]).unwrap();
        let numerator = Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap();

        let result =
            ldivide_builtin(Value::Tensor(divisor), Value::Tensor(numerator), Vec::new()).unwrap();
        let Value::Tensor(result) = result else {
            panic!("expected single tensor");
        };
        assert_eq!(
            result.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![5.0, 5.0])
        );
    }

    #[test]
    fn ldivide_like_complex_conversion_reads_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new_integer(IntegerStorage::U32(vec![4, 5]), vec![1, 2]).unwrap();

        let result =
            block_on(super::real_to_complex(Value::Tensor(tensor))).expect("complex conversion");

        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.materialize_f64(), vec![(4.0, 0.0), (5.0, 0.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ldivide_like_complex_conversion_preserves_single_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::from_f32(vec![4.0, 5.0], vec![1, 2]).unwrap();

        let result =
            block_on(super::real_to_complex(Value::Tensor(tensor))).expect("complex conversion");

        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor");
        };
        assert_eq!(result.as_f32_slice(), Some(&[(4.0, 0.0), (5.0, 0.0)][..]));
    }

    #[test]
    fn ldivide_symbolic_scalar_returns_symbolic_quotient() {
        let result = ldivide_builtin(
            Value::Num(2.0),
            Value::Symbolic(SymbolicExpr::variable("x")),
            Vec::new(),
        )
        .expect("ldivide");

        assert_eq!(result.to_string(), "x/2");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_matrix_scalar() {
        let tensor = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result =
            ldivide_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("ldivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [1.0, 0.5, 0.3333333333333333, 0.25];
                for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_row_column_broadcast() {
        let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0, 40.0], vec![1, 3]).unwrap();
        let result = ldivide_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
            .expect("broadcast ldivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = [
                    10.0,
                    5.0,
                    3.3333333333333335,
                    20.0,
                    10.0,
                    6.666666666666667,
                    40.0,
                    20.0,
                    13.333333333333334,
                ];
                for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < EPS);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_complex_inputs() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
        let result = ldivide_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex ldivide");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [(0.0, -1.0), (-0.28, -0.04)];
                for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < 1e-10 && (got.1 - exp.1).abs() < 1e-10);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ldivide_preserves_native_complex_single_storage() {
        let divisor = ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let numerator =
            ComplexTensor::from_f32(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
        let result = ldivide_builtin(
            Value::ComplexTensor(divisor),
            Value::ComplexTensor(numerator),
            Vec::new(),
        )
        .expect("complex ldivide");
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor");
        };
        assert_eq!(
            result.as_f32_slice(),
            Some(&[(0.0, -1.0), (-0.28, -0.04)][..])
        );
    }

    #[test]
    fn ldivide_mixed_complex_floating_inputs_return_single_without_scalar_collapse() {
        let single = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let double = ComplexTensor::new(vec![(2.0, -1.0)], vec![1, 1]).unwrap();
        for (divisor, numerator, expected) in [
            (single.clone(), double.clone(), (0.0, -1.0)),
            (double.clone(), single.clone(), (0.0, 1.0)),
        ] {
            let result = ldivide_builtin(
                Value::ComplexTensor(divisor),
                Value::ComplexTensor(numerator),
                Vec::new(),
            )
            .expect("complex ldivide");
            let Value::ComplexTensor(result) = result else {
                panic!("expected one-element complex single tensor");
            };
            assert_eq!(result.as_f32_slice(), Some(&[expected][..]));
        }
    }

    #[test]
    fn ldivide_mixed_real_complex_single_paths_preserve_single() {
        let complex = ComplexTensor::from_f32(vec![(1.0, 2.0), (-3.0, 1.0)], vec![1, 2]).unwrap();
        let real = Tensor::new(vec![2.0, 0.5], vec![1, 2]).unwrap();

        let result = ldivide_builtin(
            Value::ComplexTensor(complex.clone()),
            Value::Tensor(real.clone()),
            Vec::new(),
        )
        .expect("complex divisor ldivide");
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor");
        };
        assert_eq!(
            result.as_f32_slice(),
            Some(&[(0.4, -0.8), (-0.15, -0.05)][..])
        );

        let result = ldivide_builtin(
            Value::Tensor(real),
            Value::ComplexTensor(complex),
            Vec::new(),
        )
        .expect("complex numerator ldivide");
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor");
        };
        assert_eq!(result.as_f32_slice(), Some(&[(0.5, 1.0), (-6.0, 2.0)][..]));
    }

    #[test]
    fn ldivide_preserves_empty_complex_single_class() {
        let divisor = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
        let numerator = ComplexTensor::new(Vec::new(), vec![0, 2]).unwrap();
        let result = ldivide_builtin(
            Value::ComplexTensor(divisor),
            Value::ComplexTensor(numerator),
            Vec::new(),
        )
        .expect("complex ldivide");
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor");
        };
        assert_eq!(result.shape, vec![0, 2]);
        assert_eq!(result.as_f32_slice(), Some(&[][..]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_division_by_zero() {
        let tensor = Tensor::new(vec![0.0, 1.0, -2.0], vec![3, 1]).unwrap();
        let result =
            ldivide_builtin(Value::Tensor(tensor), Value::Num(0.0), Vec::new()).expect("ldivide");
        match result {
            Value::Tensor(t) => {
                assert!(t.materialize_f64()[0].is_nan());
                assert_eq!(t.materialize_f64()[1], 0.0);
                assert_eq!(t.materialize_f64()[2], -0.0);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_logical_inputs_promote() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![2, 2]).unwrap();
        let result = ldivide_builtin(
            Value::LogicalArray(logical),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect("logical ldivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [1.0, f64::INFINITY, 4.0, 8.0];
                for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    if exp.is_infinite() {
                        assert!(got.is_infinite());
                    } else {
                        assert!((got - exp).abs() < EPS);
                    }
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_char_array_promotes_to_double() {
        let chars = CharArray::new_row("AB");
        let result =
            ldivide_builtin(Value::CharArray(chars), Value::Num(2.0), Vec::new()).expect("ldivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.materialize_f64()[0] - (2.0 / 65.0)).abs() < EPS);
                assert!((t.materialize_f64()[1] - (2.0 / 66.0)).abs() < EPS);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_gpu_pair_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap();
            let rhs = Tensor::new(vec![2.0, 5.0, 10.0], vec![3, 1]).unwrap();
            let view_l = HostTensorView {
                data: &lhs.materialize_f64(),
                shape: &lhs.shape,
            };
            let view_r = HostTensorView {
                data: &rhs.materialize_f64(),
                shape: &rhs.shape,
            };
            let ha = provider.upload(&view_l).expect("upload lhs");
            let hb = provider.upload(&view_r).expect("upload rhs");
            let result = ldivide_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb), Vec::new())
                .expect("gpu ldivide");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            let expected = [0.2, 0.25, 0.3333333333333333];
            for (got, exp) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((got - exp).abs() < GPU_EPS);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_like_gpu_prototype_keeps_residency() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
            let rhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload proto");
            let result = ldivide_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("ldivide like gpu");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert!(gathered
                        .materialize_f64()
                        .iter()
                        .all(|v| (v - 0.5).abs() < GPU_EPS));
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_like_gpu_prototype_uploads_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let divisor = Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");

            let result = ldivide_builtin(
                Value::Tensor(divisor),
                Value::Num(20.0),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("ldivide like gpu");

            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert_eq!(gathered.materialize_f64(), vec![10.0, 5.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_like_host_gathers_gpu_value() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![8.0, 18.0], vec![2, 1]).unwrap();
            let rhs = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
            let view_l = HostTensorView {
                data: &lhs.materialize_f64(),
                shape: &lhs.shape,
            };
            let view_r = HostTensorView {
                data: &rhs.materialize_f64(),
                shape: &rhs.shape,
            };
            let ha = provider.upload(&view_l).expect("upload lhs");
            let hb = provider.upload(&view_r).expect("upload rhs");
            let result = ldivide_builtin(
                Value::GpuTensor(ha),
                Value::GpuTensor(hb),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("ldivide like host");
            let Value::Tensor(t) = result else {
                panic!("expected tensor result after host gather");
            };
            assert_eq!(t.shape, vec![2, 1]);
            let expected = [0.25, 1.0 / 6.0];
            for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((got - exp).abs() < GPU_EPS);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_like_complex_prototype_yields_complex() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let lhs = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = ldivide_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("ldivide like complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                let expected = [(0.5, 0.0), (0.5, 0.0)];
                for (got, exp) in ct.materialize_f64().iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < EPS);
                    assert!((got.1 - exp.1).abs() < EPS);
                }
            }
            Value::Complex(re, im) => {
                assert!((re - 0.5).abs() < EPS && im.abs() < EPS);
            }
            other => panic!("expected complex output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_like_missing_prototype_errors() {
        let lhs = Value::Num(2.0);
        let rhs = Value::Num(4.0);
        let err = ldivide_builtin(lhs, rhs, vec![Value::from("like")]).unwrap_err();
        assert!(
            err.message().contains("prototype"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_like_keyword_char_array() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let keyword = CharArray::new_row("LIKE");
            let lhs = Value::Num(2.0);
            let rhs = Value::Num(5.0);
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = ldivide_builtin(
                lhs,
                rhs,
                vec![Value::CharArray(keyword), Value::GpuTensor(proto)],
            )
            .expect("ldivide like char");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert!((gathered.materialize_f64()[0] - 2.5).abs() < GPU_EPS);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ldivide_wgpu_matches_cpu_elementwise() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let lhs = Tensor::new(vec![4.0, 9.0, 16.0, 25.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let cpu = ldivide_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();
        let view_l = HostTensorView {
            data: &lhs.materialize_f64(),
            shape: &lhs.shape,
        };
        let view_r = HostTensorView {
            data: &rhs.materialize_f64(),
            shape: &rhs.shape,
        };
        let ha = provider.upload(&view_l).unwrap();
        let hb = provider.upload(&view_r).unwrap();
        let gpu = block_on(try_ldivide_gpu_pair(&ha, &hb))
            .unwrap()
            .expect("native ldivide");
        let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
        match cpu {
            Value::Tensor(t) => {
                assert_eq!(gathered.materialize_f64().len(), t.materialize_f64().len());
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (ga, ca) in gathered
                    .materialize_f64()
                    .iter()
                    .zip(t.materialize_f64().iter())
                {
                    assert!((ga - ca).abs() < tol);
                }
            }
            Value::Num(n) => assert_eq!(gathered.materialize_f64(), vec![n]),
            other => panic!("unexpected cpu result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_int_inputs_preserve_integer_class() {
        let lhs = Value::Int(IntValue::I32(6));
        let rhs = Value::Int(IntValue::I32(4));
        let result = ldivide_builtin(lhs, rhs, Vec::new()).expect("ldivide");
        match result {
            Value::Int(IntValue::I32(v)) => assert_eq!(v, 1),
            other => panic!("expected int32 scalar, got {other:?}"),
        }
    }

    #[test]
    fn ldivide_like_is_gated_in_matlab_mode() {
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = ldivide_builtin(
            Value::Num(2.0),
            Value::Num(5.0),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect_err("like form must be gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:LdivideLikePrototypeExtension")
        );
    }

    #[test]
    fn ldivide_resident_integer_fallback_preserves_source_and_residency() {
        test_support::with_test_provider(|provider| {
            let divisor =
                Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![2, 1]).expect("divisor");
            let source = gpu_helpers::upload_tensor(provider, &divisor).expect("upload");
            runmat_accelerate_api::mark_handle_explicit(&source);
            let result = ldivide_builtin(
                Value::GpuTensor(source.clone()),
                Value::Num(10.0),
                Vec::new(),
            )
            .expect("resident ldivide");
            let Value::GpuTensor(output) = result else {
                panic!("explicit gpuArray result must remain resident");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            assert!(!gpu_helpers::same_gpu_handle(&source, &output));
            let original = test_support::gather(Value::GpuTensor(source)).expect("source survives");
            assert_eq!(
                original.integer_storage(),
                Some(&IntegerStorage::I32(vec![2, 4]))
            );
        });
    }

    #[test]
    fn ldivide_fallback_prefers_explicit_second_resident_operand() {
        test_support::with_test_provider(|provider| {
            let scalar = Tensor::new(vec![2.0], vec![1, 1]).expect("scalar");
            let automatic = gpu_helpers::upload_tensor(provider, &scalar).expect("automatic");
            runmat_accelerate_api::mark_handle_automatic(&automatic);
            let integer =
                Tensor::new_integer(IntegerStorage::I32(vec![6, 10]), vec![2, 1]).expect("integer");
            let explicit = gpu_helpers::upload_tensor(provider, &integer).expect("explicit");
            runmat_accelerate_api::mark_handle_explicit(&explicit);

            let result = ldivide_builtin(
                Value::GpuTensor(automatic),
                Value::GpuTensor(explicit.clone()),
                Vec::new(),
            )
            .expect("fallback ldivide");
            let Value::GpuTensor(output) = result else {
                panic!("explicit second operand must preserve resident output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(runmat_accelerate_api::IntegerElementType::I32)
            );
            let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::I32(vec![3, 5]))
            );
            let original = test_support::gather(Value::GpuTensor(explicit)).expect("source");
            assert_eq!(
                original.integer_storage(),
                Some(&IntegerStorage::I32(vec![6, 10]))
            );
        });
    }
}
