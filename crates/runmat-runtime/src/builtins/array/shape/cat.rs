//! MATLAB-compatible `cat` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::arg_tokens::{tokens_from_context, tokens_from_values, ArgToken};
use crate::builtins::common::concatenation::char_array_from_f64_with_prefix;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::elementwise::integer_cast::{integer_values, IntegerTarget};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::AccelProvider;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, IntValue, IntegerComplexStorage, LogicalArray,
    NumericDType, NumericStorage, StringArray, Tensor, Value,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::cat")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cat",
    op_kind: GpuOpKind::Custom("cat"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("cat")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Falls back to exact gather, host assembly, and upload for typed integers, mixed host/device inputs, runtime empty omission, or providers without a native concatenation kernel.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::cat")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cat",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Concatenation is a sink and terminates fusion pipelines.",
};

fn cat_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("cat").build()
}

const MAX_CAT_DIMENSION: usize = 1024;
const BUILTIN_NAME: &str = "cat";

const CAT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Concatenated output array.",
}];

const CAT_INPUTS_CORE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Concatenation dimension (1-based).",
    },
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "A2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input array.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional input arrays.",
    },
];

const CAT_INPUTS_LIKE: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Concatenation dimension (1-based).",
    },
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "A2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input array.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional input arrays.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Name-value key ('like').",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype that sets output class/device.",
    },
];

const CAT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = cat(dim, A1, A2, An...)",
        inputs: &CAT_INPUTS_CORE,
        outputs: &CAT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cat(dim, A1, A2, An..., \"like\", prototype)",
        inputs: &CAT_INPUTS_LIKE,
        outputs: &CAT_OUTPUT,
    },
];

const CAT_ERROR_TOO_FEW_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAT.TOO_FEW_INPUTS",
    identifier: Some("RunMat:cat:TooFewInputs"),
    when: "Fewer than two input arrays are supplied after dim.",
    message: "cat: at least two input arrays are required",
};

const CAT_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAT.INVALID_DIMENSION",
    identifier: Some("RunMat:cat:InvalidDimension"),
    when: "dim is non-numeric, non-integer, or less than one.",
    message: "cat: dimension must be numeric and >= 1",
};

const CAT_ERROR_GPU: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAT.GPU",
    identifier: Some("RunMat:cat:GpuError"),
    when: "A required gpuArray gather, upload, or provider operation fails.",
    message: "cat: gpuArray operation failed",
};

const CAT_ERROR_TYPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAT.TYPE_MISMATCH",
    identifier: Some("RunMat:cat:TypeMismatch"),
    when: "Inputs are from incompatible classes for concatenation.",
    message: "cat: incompatible input classes for concatenation",
};

const CAT_ERROR_INVALID_LIKE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAT.INVALID_LIKE",
    identifier: Some("RunMat:cat:InvalidLikePrototype"),
    when: "The 'like' name-value form is malformed or uses an unsupported prototype.",
    message: "cat: invalid 'like' prototype",
};

const CAT_ERRORS: [BuiltinErrorDescriptor; 5] = [
    CAT_ERROR_TOO_FEW_INPUTS,
    CAT_ERROR_INVALID_DIMENSION,
    CAT_ERROR_GPU,
    CAT_ERROR_TYPE_MISMATCH,
    CAT_ERROR_INVALID_LIKE,
];

pub const CAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CAT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CAT_ERRORS,
};

const CAT_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cat-like-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the trailing cat \"like\" prototype form is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CatLikePrototypeExtension"),
};

const CAT_RESIDENT_DIMENSION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cat-resident-dimension",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cat with a resident gpuArray dimension control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CatResidentDimensionExtension"),
};

pub const CAT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [CAT_LIKE_EXTENSION, CAT_RESIDENT_DIMENSION_EXTENSION];

const CAT_INTEGER_DIMENSION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A host scalar from any integer class is read exactly as the positive one-based concatenation dimension; scalar double remains accepted by the same structural parser.",
    }];

const CAT_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A1,...,An",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every ordered mixture of the eight real integer classes is accepted together with real single, double, or logical data; the leftmost participating integer class determines the numeric output class.",
    }];

const CAT_COMPLEX_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "complex A1,...,An",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight paired complex-integer classes concatenate structurally with exact real and imaginary components.",
    }];

const CAT_RESIDENT_INTEGER_DIMENSION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "gpuArray dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode may gather a resident scalar dimension exactly before structural concatenation; the documented dimension control is a host positive integer scalar.",
    }];

pub const CAT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = cat(integer_dim,A1,A2,...,An)",
        inputs: &CAT_INTEGER_DIMENSION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The integer dimension is a structural control only and does not affect output class. It is range-checked exactly before shape arithmetic; data inputs independently determine output class and residency.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = cat(dim,A1,A2,...,An) with real integer data",
        inputs: &CAT_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The leftmost nonempty integer class dominates real numeric and logical operands. Unlike integer values and floating or logical values convert to that class with ordinary rounding and saturation. Empty operands are omitted when a nonempty operand remains and therefore do not affect the result class. Resident data uses the provider hook where representable or exact gather/assemble/upload fallback.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = cat(dim,complex_integer_A1,...,An)",
        inputs: &CAT_COMPLEX_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed complex-integer concatenation applies the leftmost integer component class independently to real and imaginary parts, preserves exact paired storage, and restores resident output to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = cat(gpuArray(integer_dim),A1,A2,...,An)",
        inputs: &CAT_RESIDENT_INTEGER_DIMENSION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only resident dimension controls are gathered exactly after the extension gate; the scalar remains a structural control and data inputs independently determine output class and residency.",
    },
];

#[derive(Clone, Copy, Debug)]
struct ParsedCatTokens {
    dim: Option<usize>,
    like_index: Option<usize>,
}

fn parse_cat_tokens(tokens: &[ArgToken]) -> ParsedCatTokens {
    let dim = match tokens.first() {
        Some(ArgToken::Number(value)) => coerce_positive_dim(*value),
        Some(ArgToken::Integer(value)) => coerce_positive_integer_dim(value),
        _ => None,
    };
    let like_index = if tokens.len() >= 3 {
        match &tokens[tokens.len() - 2] {
            ArgToken::String(text) if text == "like" => Some(tokens.len() - 2),
            _ => None,
        }
    } else {
        None
    };
    ParsedCatTokens { dim, like_index }
}

fn coerce_positive_integer_dim(value: &IntValue) -> Option<usize> {
    value.try_to_usize().filter(|&dim| dim >= 1)
}

fn coerce_positive_dim(value: f64) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return None;
    }
    if rounded < 1.0 {
        return None;
    }
    Some(rounded as usize)
}

fn cell_element_type(inputs: &[Type]) -> Option<Box<Type>> {
    let mut element: Option<Type> = None;
    for ty in inputs {
        let Type::Cell { element_type, .. } = ty else {
            return None;
        };
        match (&element, element_type.as_deref()) {
            (None, Some(current)) => element = Some(current.clone()),
            (Some(existing), Some(current)) if existing == current => {}
            (Some(_), Some(_)) => return None,
            _ => {}
        }
    }
    element.map(Box::new)
}

fn cat_input_shape(ty: &Type) -> Option<Vec<Option<usize>>> {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            Some(shape.clone())
        }
        Type::Num | Type::Int | Type::Bool => Some(vec![Some(1), Some(1)]),
        _ => None,
    }
}

fn cat_concat_shape(inputs: &[Type], dim_1based: usize) -> Option<Vec<Option<usize>>> {
    if inputs.is_empty() || dim_1based == 0 {
        return None;
    }
    let mut shapes = Vec::with_capacity(inputs.len());
    for ty in inputs {
        shapes.push(cat_input_shape(ty)?);
    }
    let has_known_nonempty = shapes.iter().any(|shape| {
        shape.iter().all(Option::is_some)
            && shape.iter().all(|dimension| dimension.unwrap_or(0) > 0)
    });
    if has_known_nonempty {
        shapes.retain(|shape| !shape.iter().any(|dimension| matches!(dimension, Some(0))));
    }
    let rank = shapes
        .iter()
        .map(|shape| shape.len())
        .max()?
        .max(dim_1based);
    let mut padded = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let mut current = shape;
        while current.len() < rank {
            current.push(Some(1));
        }
        padded.push(current);
    }
    let mut output = vec![None; rank];
    let dim_zero = dim_1based - 1;
    for axis in 0..rank {
        if axis == dim_zero {
            let mut total: Option<usize> = Some(0);
            for shape in &padded {
                match (total, shape[axis]) {
                    (Some(acc), Some(value)) => total = acc.checked_add(value),
                    _ => {
                        total = None;
                        break;
                    }
                }
            }
            output[axis] = total;
        } else {
            let mut shared: Option<usize> = None;
            let mut mismatch = false;
            for shape in &padded {
                match (shared, shape[axis]) {
                    (None, value) => shared = value,
                    (Some(current), Some(value)) if current == value => {}
                    (Some(_), Some(_)) => {
                        mismatch = true;
                        break;
                    }
                    _ => {
                        shared = None;
                        break;
                    }
                }
            }
            output[axis] = if mismatch { None } else { shared };
        }
    }
    let min_len = dim_1based.max(2).min(output.len());
    while output.len() > min_len && matches!(output.last(), Some(Some(1))) {
        output.pop();
    }
    Some(output)
}

fn cat_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.len() < 3 {
        return Type::Unknown;
    }
    let parsed = parse_cat_tokens(&tokens_from_context(ctx));
    let inputs = &args[1..];
    let has_cell = inputs.iter().any(|arg| matches!(arg, Type::Cell { .. }));
    if has_cell {
        return Type::Cell {
            element_type: cell_element_type(inputs),
            length: None,
        };
    }
    let has_string = inputs.iter().any(|arg| matches!(arg, Type::String));
    if has_string {
        return Type::cell_of(Type::String);
    }
    let has_numeric = inputs
        .iter()
        .any(|arg| matches!(arg, Type::Tensor { .. } | Type::Num | Type::Int));
    let has_logical = inputs
        .iter()
        .any(|arg| matches!(arg, Type::Logical { .. } | Type::Bool));
    let dim = match parsed.dim {
        Some(value) if value > 0 => value,
        Some(_) => return Type::Unknown,
        None => {
            if has_numeric {
                return Type::tensor();
            }
            if has_logical {
                return Type::logical();
            }
            return Type::Unknown;
        }
    };
    let shape = cat_concat_shape(inputs, dim);
    if has_numeric {
        return Type::Tensor { shape };
    }
    if has_logical {
        return Type::Logical { shape };
    }
    Type::Unknown
}

fn cat_err(message: impl Into<String>) -> RuntimeError {
    cat_error(message)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CatCategory {
    Numeric,
    Logical,
    Complex,
    Char,
    String,
    Cell,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LikeDevice {
    Host,
    Gpu,
}

#[derive(Clone, Debug)]
struct LikeSpec {
    device: LikeDevice,
    category_hint: Option<CatCategory>,
    explicit: bool,
}

impl Default for LikeSpec {
    fn default() -> Self {
        Self {
            device: LikeDevice::Host,
            category_hint: None,
            explicit: false,
        }
    }
}

impl LikeSpec {
    fn from_prototype(proto: Value) -> BuiltinResult<Self> {
        match proto {
            Value::GpuTensor(_) => Ok(Self {
                device: LikeDevice::Gpu,
                category_hint: Some(CatCategory::Numeric),
                explicit: true,
            }),
            Value::Tensor(_) | Value::Num(_) | Value::Int(_) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Numeric),
                explicit: true,
            }),
            Value::LogicalArray(_) | Value::Bool(_) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Logical),
                explicit: true,
            }),
            Value::ComplexTensor(_) | Value::Complex(_, _) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Complex),
                explicit: true,
            }),
            other => Err(cat_err(format!(
                "cat: unsupported prototype for 'like' ({other:?}); provide a numeric or gpuArray prototype"
            ))),
        }
    }

    fn ensure_device(&self, category: CatCategory) -> BuiltinResult<()> {
        if matches!(self.device, LikeDevice::Gpu)
            && !matches!(category, CatCategory::Numeric | CatCategory::Complex)
        {
            return Err(cat_err(
                "cat: GPU 'like' prototypes are only supported for numeric or complex inputs",
            ));
        }
        Ok(())
    }
}

fn extract_like(mut inputs: Vec<Value>) -> BuiltinResult<(Vec<Value>, LikeSpec)> {
    if inputs.len() >= 2 {
        let tokens = tokens_from_values(&inputs);
        let parsed = parse_cat_tokens(&tokens);
        if parsed.like_index == Some(inputs.len() - 2) {
            let prototype = inputs.last().cloned().unwrap();
            if matches!(
                prototype,
                Value::CharArray(_) | Value::String(_) | Value::StringArray(_) | Value::Cell(_)
            ) {
                // Treat as data to avoid colliding with textual concatenation cases.
            } else if inputs.len() < 4 {
                // Removing the pair would leave fewer than two inputs; treat as data.
            } else {
                let spec = LikeSpec::from_prototype(prototype)?;
                inputs.pop();
                inputs.pop();
                return Ok((inputs, spec));
            }
        }
    }
    Ok((inputs, LikeSpec::default()))
}

#[runtime_builtin(
    name = "cat",
    category = "array/shape",
    summary = "Concatenate arrays along a chosen dimension.",
    keywords = "cat,concatenate,array,dimension,gpu",
    accel = "array_construct",
    type_resolver(cat_type),
    descriptor(crate::builtins::array::shape::cat::CAT_DESCRIPTOR),
    extensions(crate::builtins::array::shape::cat::CAT_EXTENSIONS),
    integer_capabilities(crate::builtins::array::shape::cat::CAT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::shape::cat"
)]
async fn cat_builtin(dim: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() < 2 {
        return Err(cat_err(CAT_ERROR_TOO_FEW_INPUTS.message));
    }
    let (inputs, like) = extract_like(rest)?;
    if inputs.len() < 2 {
        return Err(cat_err(CAT_ERROR_TOO_FEW_INPUTS.message));
    }
    if like.explicit {
        crate::compatibility::ensure_builtin_extension_enabled(&CAT_LIKE_EXTENSION, BUILTIN_NAME)?;
    }
    if matches!(dim, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CAT_RESIDENT_DIMENSION_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let dim_index = match dim {
        Value::Int(_) | Value::Num(_) | Value::Tensor(_) | Value::GpuTensor(_) => {
            match tensor::dimension_from_value_async(&dim, "cat", false)
                .await
                .map_err(cat_err)?
            {
                Some(index) => index,
                None => {
                    return Err(cat_err(format!(
                        "cat: dimension must be numeric, got {:?}",
                        dim
                    )))
                }
            }
        }
        _ => {
            return Err(cat_err(format!(
                "cat: dimension must be numeric, got {:?}",
                dim
            )))
        }
    };
    if dim_index > MAX_CAT_DIMENSION {
        return Err(cat_err(format!(
            "cat: dimension must be <= {MAX_CAT_DIMENSION}"
        )));
    }
    let dim_zero = dim_index - 1;

    if inputs.iter().any(|value| matches!(value, Value::Cell(_))) {
        let category = determine_category(&inputs, &like)?;
        debug_assert_eq!(category, CatCategory::Cell);
        return cat_cell_arrays(dim_zero, inputs);
    }

    if inputs.iter().any(|v| matches!(v, Value::GpuTensor(_))) {
        return cat_gpu_values(dim_zero, inputs, &like).await;
    }

    let category = determine_category(&inputs, &like)?;
    match category {
        CatCategory::String => cat_string_arrays(dim_zero, inputs),
        CatCategory::Char => cat_char_arrays(dim_zero, inputs),
        CatCategory::Cell => unreachable!("cell concatenation returned before residency dispatch"),
        CatCategory::Logical => cat_logical_arrays(dim_zero, inputs, &like),
        CatCategory::Complex => cat_complex_arrays(dim_zero, inputs, &like),
        CatCategory::Numeric => cat_numeric_tensors(dim_zero, inputs, &like),
    }
}

fn determine_category(inputs: &[Value], like: &LikeSpec) -> BuiltinResult<CatCategory> {
    let mut category = infer_category(inputs)?;
    if let Some(hint) = like.category_hint {
        category = match hint {
            CatCategory::Numeric => {
                if matches!(
                    category,
                    CatCategory::String
                        | CatCategory::Char
                        | CatCategory::Cell
                        | CatCategory::Complex
                ) {
                    return Err(cat_err(
                        "cat: 'like' prototype class does not match the input classes",
                    ));
                }
                CatCategory::Numeric
            }
            CatCategory::Logical => {
                if !matches!(category, CatCategory::Logical) {
                    return Err(cat_err(
                        "cat: 'like' logical prototypes require logical inputs",
                    ));
                }
                CatCategory::Logical
            }
            CatCategory::Complex => {
                if matches!(
                    category,
                    CatCategory::String | CatCategory::Char | CatCategory::Cell
                ) {
                    return Err(cat_err(
                        "cat: 'like' complex prototypes require numeric or complex inputs",
                    ));
                }
                CatCategory::Complex
            }
            CatCategory::Char | CatCategory::String | CatCategory::Cell => {
                return Err(cat_err(
                    "cat: 'like' prototypes for char, string, or cell arrays are not supported",
                ));
            }
        };
    }
    like.ensure_device(category)?;
    Ok(category)
}

fn infer_category(inputs: &[Value]) -> BuiltinResult<CatCategory> {
    // Direct cell concatenation encloses each complete non-cell operand in one
    // cell. Resolve this dominance before validating the enclosed value class.
    if inputs.iter().any(|value| matches!(value, Value::Cell(_))) {
        return Ok(CatCategory::Cell);
    }

    let mut has_string = false;
    let mut has_char = false;
    let mut has_complex = false;
    let mut has_logical = false;
    let mut all_logical = true;

    for value in inputs {
        match value {
            Value::Tensor(_) | Value::Num(_) | Value::Int(_) => {
                all_logical = false;
            }
            Value::LogicalArray(_) | Value::Bool(_) => {
                has_logical = true;
            }
            Value::ComplexTensor(_) | Value::Complex(_, _) => {
                has_complex = true;
                all_logical = false;
            }
            Value::String(_) | Value::StringArray(_) => {
                has_string = true;
                all_logical = false;
            }
            Value::CharArray(_) => {
                has_char = true;
                all_logical = false;
            }
            Value::Cell(_) => unreachable!("cell dominance returned above"),
            Value::GpuTensor(_) => {
                return Err(cat_err(
                    "cat: gpuArray inputs must be concatenated using the GPU path",
                ));
            }
            other => {
                return Err(cat_err(format!(
                    "cat: unsupported input type for concatenation: {other:?}"
                )));
            }
        }
        if !matches!(value, Value::LogicalArray(_) | Value::Bool(_)) {
            all_logical = false;
        }
    }

    if has_string && has_complex {
        return Err(cat_err(
            "cat: mixed string and complex concatenation is not supported",
        ));
    }
    if has_char && (has_complex || has_logical) {
        return Err(cat_err("cat: cannot mix char arrays with other classes"));
    }
    if has_complex && (has_string || has_char) {
        return Err(cat_err(
            "cat: cannot mix complex arrays with textual or cell arrays",
        ));
    }

    if has_string {
        Ok(CatCategory::String)
    } else if has_char {
        Ok(CatCategory::Char)
    } else if has_complex {
        Ok(CatCategory::Complex)
    } else if all_logical {
        Ok(CatCategory::Logical)
    } else {
        Ok(CatCategory::Numeric)
    }
}

fn finalize_numeric_output(tensor: Tensor, like: &LikeSpec) -> BuiltinResult<Value> {
    finalize_numeric_output_with_provider(tensor, like, None)
}

fn finalize_numeric_output_with_provider(
    tensor: Tensor,
    like: &LikeSpec,
    provider_override: Option<&'static dyn AccelProvider>,
) -> BuiltinResult<Value> {
    like.ensure_device(CatCategory::Numeric)?;
    match like.device {
        LikeDevice::Host => Ok(tensor::tensor_into_value(tensor)),
        LikeDevice::Gpu => {
            let provider = provider_override
                .or_else(runmat_accelerate_api::provider)
                .ok_or_else(|| {
                    cat_err(
                        "cat: GPU output requested via 'like' but no acceleration provider is active",
                    )
                })?;
            let handle = upload_numeric_tensor(provider, &tensor)?;
            Ok(Value::GpuTensor(handle))
        }
    }
}

fn upload_numeric_tensor(
    provider: &dyn AccelProvider,
    tensor: &Tensor,
) -> BuiltinResult<runmat_accelerate_api::GpuTensorHandle> {
    gpu_helpers::upload_tensor(provider, tensor)
        .map_err(|err| cat_err(format!("cat: failed to upload concatenated tensor: {err}")))
}

fn cat_numeric_tensors(
    dim_zero: usize,
    values: Vec<Value>,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    if let Some(target) = leftmost_integer_target(&values) {
        return cat_integer_tensors(target, dim_zero, values, like);
    }

    let mut tensors = Vec::with_capacity(values.len());
    for value in values {
        let tensor = tensor::value_into_tensor_for("cat", value).map_err(cat_err)?;
        tensors.push(tensor);
    }

    let tensor = concat_floating_tensors(dim_zero, tensors)?;
    finalize_numeric_output(tensor, like)
}

fn concat_floating_tensors(dim_zero: usize, tensors: Vec<Tensor>) -> BuiltinResult<Tensor> {
    let use_single = tensors
        .iter()
        .any(|tensor| tensor.numeric_dtype() == NumericDType::F32);
    let shapes: Vec<Vec<usize>> = tensors.iter().map(|tensor| tensor.shape.clone()).collect();
    let storages: Vec<NumericStorage> = tensors
        .into_iter()
        .map(|tensor| {
            tensor
                .into_numeric_storage()
                .map_err(|error| cat_err(format!("cat: {error}")))
        })
        .collect::<BuiltinResult<_>>()?;

    if use_single {
        let buffers: Vec<Vec<f32>> = storages
            .into_iter()
            .map(|storage| match storage {
                NumericStorage::F64(values) => {
                    Ok(values.into_iter().map(|value| value as f32).collect())
                }
                NumericStorage::F32(values) => Ok(values),
                storage => Err(cat_err(format!(
                    "cat: internal floating concatenation received {} storage",
                    storage.class_name()
                ))),
            })
            .collect::<BuiltinResult<_>>()?;
        let data_refs: Vec<&[f32]> = buffers.iter().map(Vec::as_slice).collect();
        let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
        Tensor::from_numeric_storage(NumericStorage::F32(data), shape)
            .map_err(|error| cat_err(format!("cat: {error}")))
    } else {
        let buffers: Vec<Vec<f64>> = storages
            .into_iter()
            .map(|storage| match storage {
                NumericStorage::F64(values) => Ok(values),
                storage => Err(cat_err(format!(
                    "cat: internal double concatenation received {} storage",
                    storage.class_name()
                ))),
            })
            .collect::<BuiltinResult<_>>()?;
        let data_refs: Vec<&[f64]> = buffers.iter().map(Vec::as_slice).collect();
        let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
        Tensor::from_numeric_storage(NumericStorage::F64(data), shape)
            .map_err(|error| cat_err(format!("cat: {error}")))
    }
}

/// MATLAB concatenation adopts the class of the leftmost integer operand,
/// converting every other numeric or logical input to that class.
fn leftmost_integer_target(values: &[Value]) -> Option<IntegerTarget> {
    let mut empty_integer_target = None;
    for value in values {
        match value {
            Value::Int(value) => return Some(IntegerTarget::from_int_value(value)),
            Value::Tensor(tensor) => {
                if let Some(storage) = tensor.integer_storage() {
                    let target = IntegerTarget::from_storage(storage);
                    if !is_empty_concat_shape(&tensor.shape) {
                        return Some(target);
                    }
                    empty_integer_target.get_or_insert(target);
                }
            }
            _ => {}
        }
    }
    empty_integer_target
}

fn cat_integer_tensors(
    target: IntegerTarget,
    dim_zero: usize,
    values: Vec<Value>,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    let tensor = build_integer_cat_tensor(target, dim_zero, values)?;
    finalize_numeric_output(tensor, like)
}

fn build_integer_cat_tensor(
    target: IntegerTarget,
    dim_zero: usize,
    values: Vec<Value>,
) -> BuiltinResult<Tensor> {
    let mut shapes = Vec::with_capacity(values.len());
    let mut value_buffers = Vec::with_capacity(values.len());

    for value in values {
        let (shape, values) = integer_concat_input(target, value)?;
        shapes.push(shape);
        value_buffers.push(values);
    }

    let data_refs: Vec<&[runmat_value::IntValue]> =
        value_buffers.iter().map(Vec::as_slice).collect();
    let (values, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    Tensor::new_integer(target.storage(values), shape)
        .map_err(|error| cat_err(format!("cat: {error}")))
}

fn integer_concat_input(
    target: IntegerTarget,
    value: Value,
) -> BuiltinResult<(Vec<usize>, Vec<runmat_value::IntValue>)> {
    match value {
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let values = match tensor
                .into_numeric_storage()
                .map_err(|error| cat_err(format!("cat: {error}")))?
            {
                NumericStorage::F64(values) => values
                    .into_iter()
                    .map(|value| target.cast_scalar(value))
                    .collect(),
                NumericStorage::F32(values) => values
                    .into_iter()
                    .map(|value| target.cast_scalar(f64::from(value)))
                    .collect(),
                storage => integer_values(
                    storage
                        .into_integer_storage()
                        .expect("non-floating numeric storage is integer"),
                )
                .iter()
                .map(|value| target.cast_int(value))
                .collect(),
            };
            Ok((shape, values))
        }
        Value::Int(value) => Ok((vec![1, 1], vec![target.cast_int(&value)])),
        Value::Num(value) => Ok((vec![1, 1], vec![target.cast_scalar(value)])),
        Value::Bool(value) => Ok((
            vec![1, 1],
            vec![target.cast_scalar(if value { 1.0 } else { 0.0 })],
        )),
        Value::LogicalArray(array) => Ok((
            array.shape,
            array
                .data
                .iter()
                .map(|&value| target.cast_scalar(if value == 0 { 0.0 } else { 1.0 }))
                .collect(),
        )),
        other => Err(cat_err(format!(
            "cat: cannot concatenate integer arrays with {other:?}"
        ))),
    }
}

fn cat_logical_arrays(
    dim_zero: usize,
    values: Vec<Value>,
    _like: &LikeSpec,
) -> BuiltinResult<Value> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        arrays.push(value_into_logical(value)?);
    }
    let shapes: Vec<Vec<usize>> = arrays.iter().map(|a| a.shape.clone()).collect();
    let data_refs: Vec<&[u8]> = arrays.iter().map(|a| a.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let logical = LogicalArray::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::LogicalArray(logical))
}

fn cat_complex_arrays(
    dim_zero: usize,
    values: Vec<Value>,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    if values.iter().any(|v| matches!(v, Value::GpuTensor(_))) {
        return Err(cat_err(
            "cat: complex concatenation requires host arrays; convert gpuArray values first",
        ));
    }

    if let Some(target) = leftmost_complex_integer_target(&values) {
        return cat_typed_complex_integer_arrays(target, dim_zero, values, like);
    }

    let mut tensors = Vec::with_capacity(values.len());
    for value in values {
        let tensor = match value {
            Value::ComplexTensor(ct) => ct,
            Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cat_err(format!("cat: {e}")))?,
            other => {
                let real = tensor::value_into_tensor_for("cat", other).map_err(cat_err)?;
                tensor_to_complex(real)?
            }
        };
        tensors.push(tensor);
    }

    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape.clone()).collect();
    let output_dtype = if tensors
        .iter()
        .all(|tensor| tensor.numeric_dtype() == NumericDType::F32)
    {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let materialized: Vec<Vec<(f64, f64)>> =
        tensors.iter().map(ComplexTensor::materialize_f64).collect();
    let data_refs: Vec<&[(f64, f64)]> = materialized.iter().map(Vec::as_slice).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let tensor = ComplexTensor::from_f64_values_with_dtype(data, shape, output_dtype)
        .map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn leftmost_complex_integer_target(values: &[Value]) -> Option<IntegerTarget> {
    let mut empty_target = None;
    for value in values {
        let Value::ComplexTensor(tensor) = value else {
            continue;
        };
        let Some(storage) = tensor.integer_storage() else {
            continue;
        };
        let target = IntegerTarget::from_storage(&storage.real);
        if !is_empty_concat_shape(&tensor.shape) {
            return Some(target);
        }
        empty_target.get_or_insert(target);
    }
    empty_target
}

fn cat_typed_complex_integer_arrays(
    target: IntegerTarget,
    dim_zero: usize,
    values: Vec<Value>,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    like.ensure_device(CatCategory::Complex)?;
    if matches!(like.device, LikeDevice::Gpu) {
        return Err(cat_err(
            "cat: the RunMat-only explicit GPU like form does not accept typed complex-integer output",
        ));
    }

    let mut shapes = Vec::with_capacity(values.len());
    let mut real_buffers = Vec::with_capacity(values.len());
    let mut imag_buffers = Vec::with_capacity(values.len());
    for value in values {
        let (shape, real, imag) = typed_complex_integer_concat_input(target, value)?;
        shapes.push(shape);
        real_buffers.push(real);
        imag_buffers.push(imag);
    }

    let real_refs: Vec<&[runmat_value::IntValue]> =
        real_buffers.iter().map(Vec::as_slice).collect();
    let imag_refs: Vec<&[runmat_value::IntValue]> =
        imag_buffers.iter().map(Vec::as_slice).collect();
    let (real, shape) = concat_column_major(dim_zero, &shapes, &real_refs, "cat")?;
    let (imag, imag_shape) = concat_column_major(dim_zero, &shapes, &imag_refs, "cat")?;
    debug_assert_eq!(shape, imag_shape);
    IntegerComplexStorage::new(target.storage(real), target.storage(imag))
        .and_then(|storage| ComplexTensor::new_integer(storage, shape))
        .map(Value::ComplexTensor)
        .map_err(|error| cat_err(format!("cat: {error}")))
}

fn typed_complex_integer_concat_input(
    target: IntegerTarget,
    value: Value,
) -> BuiltinResult<(
    Vec<usize>,
    Vec<runmat_value::IntValue>,
    Vec<runmat_value::IntValue>,
)> {
    let zero = || target.cast_scalar(0.0);
    match value {
        Value::ComplexTensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return Ok((
                    tensor.shape.clone(),
                    integer_values(storage.real.clone())
                        .iter()
                        .map(|value| target.cast_int(value))
                        .collect(),
                    integer_values(storage.imag.clone())
                        .iter()
                        .map(|value| target.cast_int(value))
                        .collect(),
                ));
            }
            let shape = tensor.shape.clone();
            let values = tensor.materialize_f64();
            Ok((
                shape,
                values
                    .iter()
                    .map(|&(real, _)| target.cast_scalar(real))
                    .collect(),
                values
                    .iter()
                    .map(|&(_, imag)| target.cast_scalar(imag))
                    .collect(),
            ))
        }
        Value::Complex(real, imag) => Ok((
            vec![1, 1],
            vec![target.cast_scalar(real)],
            vec![target.cast_scalar(imag)],
        )),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let real: Vec<runmat_value::IntValue> = match tensor
                .into_numeric_storage()
                .map_err(|error| cat_err(format!("cat: {error}")))?
            {
                NumericStorage::F64(values) => values
                    .into_iter()
                    .map(|value| target.cast_scalar(value))
                    .collect(),
                NumericStorage::F32(values) => values
                    .into_iter()
                    .map(|value| target.cast_scalar(f64::from(value)))
                    .collect(),
                storage => integer_values(
                    storage
                        .into_integer_storage()
                        .expect("non-floating numeric storage is integer"),
                )
                .iter()
                .map(|value| target.cast_int(value))
                .collect(),
            };
            let imag = vec![zero(); real.len()];
            Ok((shape, real, imag))
        }
        Value::Int(value) => Ok((vec![1, 1], vec![target.cast_int(&value)], vec![zero()])),
        Value::Num(value) => Ok((vec![1, 1], vec![target.cast_scalar(value)], vec![zero()])),
        Value::Bool(value) => Ok((
            vec![1, 1],
            vec![target.cast_scalar(if value { 1.0 } else { 0.0 })],
            vec![zero()],
        )),
        Value::LogicalArray(array) => {
            let real: Vec<_> = array
                .data
                .iter()
                .map(|&value| target.cast_scalar(if value == 0 { 0.0 } else { 1.0 }))
                .collect();
            let imag = vec![zero(); real.len()];
            Ok((array.shape, real, imag))
        }
        other => Err(cat_err(format!(
            "cat: cannot concatenate typed complex integer arrays with {other:?}"
        ))),
    }
}

fn cat_char_arrays(dim_zero: usize, values: Vec<Value>) -> BuiltinResult<Value> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        arrays.push(char_array_from_value(value)?);
    }
    let shapes: Vec<Vec<usize>> = arrays.iter().map(|array| array.shape.clone()).collect();
    let buffers: Vec<Vec<char>> = arrays.iter().map(CharArray::to_column_major).collect();
    let data_refs: Vec<&[char]> = buffers.iter().map(Vec::as_slice).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    CharArray::from_column_major(data, shape)
        .map(Value::CharArray)
        .map_err(|error| cat_err(format!("cat: {error}")))
}

fn char_array_from_value(value: Value) -> BuiltinResult<CharArray> {
    match value {
        Value::CharArray(ca) => Ok(ca),
        Value::Num(n) => char_array_from_f64_with_prefix(n, "cat"),
        Value::Int(i) => char_array_from_f64_with_prefix(i.to_f64(), "cat"),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let column_major = match tensor
                .into_numeric_storage()
                .map_err(|error| cat_err(format!("cat: {error}")))?
            {
                NumericStorage::F64(values) => values
                    .into_iter()
                    .map(char_from_numeric_code_point)
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericStorage::F32(values) => values
                    .into_iter()
                    .map(|value| char_from_numeric_code_point(f64::from(value)))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                storage => integer_values(
                    storage
                        .into_integer_storage()
                        .expect("non-floating numeric storage is integer"),
                )
                .into_iter()
                .map(|value| char_from_numeric_code_point(value.to_f64()))
                .collect::<BuiltinResult<Vec<_>>>()?,
            };
            CharArray::from_column_major(column_major, shape)
                .map_err(|error| cat_err(format!("cat: {error}")))
        }
        Value::Bool(_) | Value::LogicalArray(_) => Err(cat_err(
            "cat: character arrays cannot be concatenated with logical values",
        )),
        other => Err(cat_err(format!(
            "cat: expected char arrays or scalar code points, got {other:?}"
        ))),
    }
}

fn char_from_numeric_code_point(value: f64) -> BuiltinResult<char> {
    let scalar = char_array_from_f64_with_prefix(value, "cat")?;
    scalar
        .data
        .into_iter()
        .next()
        .ok_or_else(|| cat_err("cat: internal character conversion produced no value"))
}

fn cat_string_arrays(dim_zero: usize, values: Vec<Value>) -> BuiltinResult<Value> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        arrays.push(value_into_string_array(value)?);
    }
    let shapes: Vec<Vec<usize>> = arrays.iter().map(|a| a.shape.clone()).collect();
    let data_refs: Vec<&[String]> = arrays.iter().map(|a| a.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let array = StringArray::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::StringArray(array))
}

fn cat_cell_arrays(dim_zero: usize, values: Vec<Value>) -> BuiltinResult<Value> {
    let mut shapes = Vec::with_capacity(values.len());
    let mut buffers = Vec::with_capacity(values.len());
    for value in values {
        let (shape, data) = match value {
            Value::Cell(cell) => {
                let data = cell.to_column_major();
                (cell.shape, data)
            }
            other => (vec![1, 1], vec![other]),
        };
        shapes.push(shape);
        buffers.push(data);
    }
    let data_refs: Vec<&[Value]> = buffers.iter().map(Vec::as_slice).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    CellArray::from_column_major(data, shape)
        .map(Value::Cell)
        .map_err(|error| cat_err(format!("cat: {error}")))
}

async fn cat_gpu_values(
    dim_zero: usize,
    values: Vec<Value>,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    let has_complex_data = values.iter().any(|value| match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => true,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        }
        _ => false,
    });
    let output_like = if like.explicit {
        like.clone()
    } else {
        LikeSpec {
            device: LikeDevice::Gpu,
            category_hint: Some(if has_complex_data {
                CatCategory::Complex
            } else {
                CatCategory::Numeric
            }),
            explicit: false,
        }
    };
    if let Some(hint) = like.category_hint {
        if (has_complex_data && !matches!(hint, CatCategory::Complex))
            || (!has_complex_data && !matches!(hint, CatCategory::Numeric))
        {
            return Err(cat_err(
                "cat: 'like' prototype class does not match gpuArray inputs",
            ));
        }
    }
    output_like.ensure_device(if has_complex_data {
        CatCategory::Complex
    } else {
        CatCategory::Numeric
    })?;

    let all_gpu = values
        .iter()
        .all(|value| matches!(value, Value::GpuTensor(_)));
    let handles: Vec<_> = values
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect();

    let first_handle = handles
        .first()
        .map(|handle| (*handle).clone())
        .ok_or_else(|| cat_err("cat: no gpuArray inputs to concatenate"))?;
    let provider = runmat_accelerate_api::provider_for_handle(&first_handle)
        .ok_or_else(|| cat_err("cat: no acceleration provider is registered"))?;
    for handle in &handles {
        let Some(owner) = runmat_accelerate_api::provider_for_handle(handle) else {
            return Err(cat_err("cat: no acceleration provider is registered"));
        };
        if owner.device_id() != provider.device_id() {
            return Err(cat_err(
                "cat: all gpuArray inputs must belong to the same acceleration provider",
            ));
        }
    }
    let has_integer_handle = handles
        .iter()
        .any(|handle| runmat_accelerate_api::handle_integer_type(handle).is_some());
    let has_empty_handle = handles.iter().any(|handle| handle.shape.contains(&0));
    let has_nonempty_handle = handles.iter().any(|handle| !handle.shape.contains(&0));
    let must_apply_runtime_empty_omission = has_empty_handle && has_nonempty_handle;

    // Native provider hook
    if all_gpu && !has_integer_handle && !must_apply_runtime_empty_omission {
        let owned_handles: Vec<_> = handles.iter().map(|handle| (*handle).clone()).collect();
        if let Ok(result) = provider.cat(dim_zero + 1, &owned_handles) {
            return finalize_gpu_value(result, &output_like).await;
        }
    }

    let mut gathered_values = Vec::with_capacity(values.len());
    for value in values {
        match value {
            Value::GpuTensor(handle) => {
                gathered_values
                    .push(gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?);
            }
            value => gathered_values.push(value),
        }
    }

    let category = determine_category(&gathered_values, &output_like)?;
    if matches!(category, CatCategory::Complex) {
        let host_like = LikeSpec {
            device: LikeDevice::Host,
            category_hint: Some(CatCategory::Complex),
            explicit: false,
        };
        let result = cat_complex_arrays(dim_zero, gathered_values, &host_like)?;
        if matches!(output_like.device, LikeDevice::Host) {
            return Ok(result);
        }
        return gpu_helpers::restore_class_preserving_value(&first_handle, result, BUILTIN_NAME)
            .map_err(|error| cat_err(error.message()));
    }
    if !matches!(category, CatCategory::Numeric) {
        return Err(cat_err("cat: gpuArray concatenation requires numeric data"));
    }
    if let Some(target) = leftmost_integer_target(&gathered_values) {
        let tensor = build_integer_cat_tensor(target, dim_zero, gathered_values)?;
        return finalize_numeric_output_with_provider(tensor, &output_like, Some(provider));
    }

    let mut tensors = Vec::with_capacity(gathered_values.len());
    for value in gathered_values {
        tensors.push(tensor::value_into_tensor_for("cat", value).map_err(cat_err)?);
    }
    let tensor = concat_floating_tensors(dim_zero, tensors)?;
    if matches!(output_like.device, LikeDevice::Host) {
        return Ok(tensor::tensor_into_value(tensor));
    }

    match upload_numeric_tensor(provider, &tensor) {
        Ok(handle) => Ok(Value::GpuTensor(handle)),
        Err(_) => Ok(tensor::tensor_into_value(tensor)),
    }
}

fn concat_column_major<T: Clone>(
    dim_zero: usize,
    shapes: &[Vec<usize>],
    data: &[&[T]],
    context: &str,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    if shapes.is_empty() {
        return Err(cat_err(format!("{context}: no inputs to concatenate")));
    }
    let rank = shapes
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(1)
        .max(dim_zero + 1);

    let mut padded = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let mut current = shape.clone();
        while current.len() < rank {
            current.push(1);
        }
        padded.push(current);
    }

    for (idx, (shape, slice)) in padded.iter().zip(data.iter()).enumerate() {
        let expected = checked_product(shape)
            .ok_or_else(|| cat_err(format!("{context}: input {} exceeds maximum size", idx + 1)))?;
        if expected != slice.len() {
            return Err(cat_err(format!(
                "{context}: input {} has {} elements but the shape multiplies to {}",
                idx + 1,
                slice.len(),
                expected
            )));
        }
    }

    // MATLAB omits empty operands when a nonempty partner remains. Preserve
    // ordinary geometry when every operand is empty.
    let has_non_neutral = padded.iter().any(|shape| !is_empty_concat_shape(shape));
    let mut active_shapes: Vec<Vec<usize>> = Vec::with_capacity(padded.len());
    let mut active_data: Vec<&[T]> = Vec::with_capacity(data.len());
    if has_non_neutral {
        for (shape, slice) in padded.iter().zip(data.iter()) {
            if !is_empty_concat_shape(shape) {
                active_shapes.push(shape.clone());
                active_data.push(*slice);
            }
        }
    } else {
        active_shapes = padded;
        active_data.extend(data.iter().copied());
    }

    for axis in 0..rank {
        if axis == dim_zero {
            continue;
        }
        let reference = active_shapes[0][axis];
        for (idx, shape) in active_shapes.iter().enumerate().skip(1) {
            if shape[axis] != reference {
                return Err(cat_err(format!(
                    "{context}: dimension {} mismatch between input 1 (size {}) and input {} (size {})",
                    axis + 1,
                    reference,
                    idx + 1,
                    shape[axis]
                )));
            }
        }
    }

    let mut output_shape = active_shapes[0].clone();
    let mut concat_dim = 0usize;
    for shape in &active_shapes {
        concat_dim = concat_dim.checked_add(shape[dim_zero]).ok_or_else(|| {
            cat_err(format!(
                "{context}: concatenated dimension exceeds maximum size"
            ))
        })?;
    }
    output_shape[dim_zero] = concat_dim;

    let total = match checked_product(&output_shape) {
        Some(total) => total,
        None => {
            return Err(cat_err(format!(
                "{context}: resulting array exceeds maximum size"
            )))
        }
    };
    if total == 0 {
        return Ok((Vec::new(), normalize_shape(output_shape, dim_zero)));
    }

    let inner = if dim_zero == 0 {
        1
    } else {
        output_shape[..dim_zero].iter().product()
    };
    let outer = if dim_zero + 1 >= rank {
        1
    } else {
        output_shape[dim_zero + 1..].iter().product()
    };

    let mut output = Vec::with_capacity(total);
    for outer_idx in 0..outer {
        for (shape, slice) in active_shapes.iter().zip(active_data.iter()) {
            let mid = shape[dim_zero];
            let chunk = mid * inner;
            if chunk == 0 {
                continue;
            }
            let offset = outer_idx * chunk;
            output.extend_from_slice(&slice[offset..offset + chunk]);
        }
    }

    Ok((output, normalize_shape(output_shape, dim_zero)))
}

fn is_empty_concat_shape(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn normalize_shape(mut shape: Vec<usize>, dim_zero: usize) -> Vec<usize> {
    let min_len = (dim_zero + 1).max(2).min(shape.len());
    while shape.len() > min_len && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn checked_product(dims: &[usize]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn value_into_logical(value: Value) -> BuiltinResult<LogicalArray> {
    match value {
        Value::LogicalArray(array) => Ok(array),
        Value::Bool(flag) => LogicalArray::new(vec![if flag { 1 } else { 0 }], vec![1, 1])
            .map_err(|e| cat_err(format!("cat: {e}"))),
        other => Err(cat_err(format!(
            "cat: expected logical inputs, got {:?}",
            other
        ))),
    }
}

fn value_into_string_array(value: Value) -> BuiltinResult<StringArray> {
    match value {
        Value::StringArray(array) => Ok(array),
        Value::String(text) => {
            StringArray::new(vec![text], vec![1, 1]).map_err(|e| cat_err(format!("cat: {e}")))
        }
        Value::CharArray(chars) => {
            let pages = if chars.shape.len() <= 2 {
                1
            } else {
                checked_product(&chars.shape[2..])
                    .ok_or_else(|| cat_err("cat: character page count exceeds maximum size"))?
            };
            let mut data = Vec::with_capacity(chars.rows * pages);
            for page in 0..pages {
                let page_offset = page * chars.rows * chars.cols;
                for row in 0..chars.rows {
                    let start = page_offset + row * chars.cols;
                    let end = start + chars.cols;
                    let text: String = chars.data[start..end].iter().collect();
                    data.push(text);
                }
            }
            let mut shape = vec![chars.rows, 1];
            shape.extend(chars.shape.iter().skip(2).copied());
            StringArray::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))
        }
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let data = match tensor
                .into_numeric_storage()
                .map_err(|error| cat_err(format!("cat: {error}")))?
            {
                NumericStorage::F64(values) => {
                    values.into_iter().map(|value| value.to_string()).collect()
                }
                NumericStorage::F32(values) => {
                    values.into_iter().map(|value| value.to_string()).collect()
                }
                storage => integer_values(
                    storage
                        .into_integer_storage()
                        .expect("non-floating numeric storage is integer"),
                )
                .into_iter()
                .map(|value| value.decimal_string())
                .collect(),
            };
            StringArray::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))
        }
        Value::Num(value) => StringArray::new(vec![value.to_string()], vec![1, 1])
            .map_err(|e| cat_err(format!("cat: {e}"))),
        Value::Int(value) => StringArray::new(vec![value.decimal_string()], vec![1, 1])
            .map_err(|e| cat_err(format!("cat: {e}"))),
        Value::Bool(value) => {
            StringArray::new(vec![if value { "1" } else { "0" }.to_string()], vec![1, 1])
                .map_err(|e| cat_err(format!("cat: {e}")))
        }
        Value::LogicalArray(array) => StringArray::new(
            array
                .data
                .into_iter()
                .map(|value| if value == 0 { "0" } else { "1" }.to_string())
                .collect(),
            array.shape,
        )
        .map_err(|e| cat_err(format!("cat: {e}"))),
        other => Err(cat_err(format!(
            "cat: expected string arrays, got {:?}",
            other
        ))),
    }
}

fn tensor_to_complex(tensor: Tensor) -> BuiltinResult<ComplexTensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| cat_err(format!("cat: {error}")))?;
    let data = match storage {
        NumericStorage::F64(values) => values.into_iter().map(|real| (real, 0.0)).collect(),
        NumericStorage::F32(values) => values
            .into_iter()
            .map(|real| (f64::from(real), 0.0))
            .collect(),
        storage => integer_values(
            storage
                .into_integer_storage()
                .expect("non-floating numeric storage is integer"),
        )
        .into_iter()
        .map(|real| (real.to_f64(), 0.0))
        .collect(),
    };
    ComplexTensor::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))
}

async fn finalize_gpu_value(
    handle: runmat_accelerate_api::GpuTensorHandle,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    if matches!(like.device, LikeDevice::Host) {
        let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    Ok(Value::GpuTensor(handle))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn cat_builtin(dim: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::cat_builtin(dim, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{
        HostIntegerDataView, HostIntegerTensorView, HostTensorView, IntegerElementType,
    };
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    #[test]
    fn cat_type_prefers_cell() {
        let out = cat_type(
            &[
                Type::Int,
                Type::Cell {
                    element_type: Some(Box::new(Type::Num)),
                    length: None,
                },
                Type::Cell {
                    element_type: Some(Box::new(Type::Num)),
                    length: None,
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert!(matches!(out, Type::Cell { .. }));
    }

    #[test]
    fn cat_type_numeric_falls_back_tensor() {
        let out = cat_type(
            &[Type::Int, Type::Num, Type::Num],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::tensor());
    }

    #[test]
    fn cat_type_omits_known_empty_geometry_and_marks_mixed_cell_output() {
        let shape = cat_concat_shape(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(0), Some(4)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
            ],
            1,
        );
        assert_eq!(shape, Some(vec![Some(2), Some(3)]));

        let output = cat_type(
            &[
                Type::Int,
                Type::Cell {
                    element_type: Some(Box::new(Type::String)),
                    length: Some(1),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(2)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert!(matches!(
            output,
            Type::Cell {
                element_type: None,
                ..
            }
        ));
    }

    #[test]
    fn cat_integer_tokens_parse_exact_dimensions() {
        let parsed = parse_cat_tokens(&[ArgToken::Integer(IntValue::U16(2))]);
        assert_eq!(parsed.dim, Some(2));
        assert_eq!(parsed.like_index, None);

        let parsed = parse_cat_tokens(&[ArgToken::Integer(IntValue::I8(0))]);
        assert_eq!(parsed.dim, None);

        let parsed = parse_cat_tokens(&[ArgToken::Integer(IntValue::I8(-1))]);
        assert_eq!(parsed.dim, None);
    }

    #[test]
    fn cat_integer_capability_and_extension_metadata_is_explicit() {
        let builtin = runmat_builtins::builtin_function_by_name("cat").expect("registered cat");
        assert_eq!(builtin.integer_capabilities.len(), 4);
        assert_eq!(builtin.extensions.len(), 2);
        assert_eq!(
            builtin.integer_capabilities[0].overload,
            BuiltinIntegerOverloadKind::StructuralParameter
        );
        assert_eq!(
            builtin.integer_capabilities[1].overflow,
            BuiltinIntegerOverflowRule::Saturate
        );
        assert_eq!(
            builtin.integer_capabilities[2].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            builtin.integer_capabilities[3].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert_eq!(
            builtin.integer_capabilities[3].backend,
            BuiltinIntegerBackendRule::GatherFallback
        );
        assert_eq!(builtin.extensions, &CAT_EXTENSIONS);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_numeric_rows() {
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::Tensor(a), Value::Tensor(b)],
        )
        .expect("cat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 2]);
                assert_eq!(
                    t.materialize_f64(),
                    vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn cat_single_and_double_preserves_single_storage() {
        let single = Tensor::from_f32(vec![1.25, -2.5], vec![1, 2]).expect("single input tensor");
        let double = Tensor::new(vec![3.75], vec![1, 1]).expect("double input tensor");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(single), Value::Tensor(double)],
        )
        .expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![1, 3]);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            output.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![1.25, -2.5, 3.75])
        );
    }

    #[test]
    fn cat_dimension_scalar_tensor_reads_typed_integer_storage_exactly() {
        let dim = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("dim");
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let result =
            cat_builtin(Value::Tensor(dim), vec![Value::Tensor(a), Value::Tensor(b)]).expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![1, 3]);
        assert_eq!(output.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cat_dimension_scalar_double_tensor_is_accepted() {
        let dim = Tensor::new(vec![2.0], vec![1, 1]).expect("dim");
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let result =
            cat_builtin(Value::Tensor(dim), vec![Value::Tensor(a), Value::Tensor(b)]).expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![1, 3]);
        assert_eq!(output.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cat_dimension_scalar_tensor_rejects_negative_typed_integer_storage() {
        let dim = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("dim");
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let b = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let err =
            cat_builtin(Value::Tensor(dim), vec![Value::Tensor(a), Value::Tensor(b)]).unwrap_err();
        assert!(err.message().contains("dimension"));
    }

    #[test]
    fn cat_dimension_tensor_rejects_non_scalar_and_oversized_typed_integer_storage() {
        let vector_dim = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("dim");
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let b = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let err = cat_builtin(
            Value::Tensor(vector_dim),
            vec![Value::Tensor(a.clone()), Value::Tensor(b.clone())],
        )
        .unwrap_err();
        assert!(!err.message().is_empty());

        let too_large =
            Tensor::new_integer(IntegerStorage::U64(vec![usize::MAX as u64]), vec![1, 1])
                .expect("dim");
        let err = cat_builtin(
            Value::Tensor(too_large),
            vec![Value::Tensor(a.clone()), Value::Tensor(b.clone())],
        )
        .unwrap_err();
        assert!(err.message().contains("dimension"));

        let err = cat_builtin(
            Value::Int(IntValue::U64(usize::MAX as u64)),
            vec![Value::Tensor(a), Value::Tensor(b)],
        )
        .unwrap_err();
        assert!(err.message().contains("dimension"));
    }

    #[test]
    fn cat_preserves_exact_uint64_storage() {
        let left =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 4]), vec![1, 2]).expect("left");
        let right = Tensor::new_integer(IntegerStorage::U64(vec![5]), vec![1, 1]).expect("right");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(left), Value::Tensor(right)],
        )
        .expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![1, 3]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 4, 5]))
        );
    }

    #[test]
    fn cat_uses_leftmost_integer_class_and_conversion_rules() {
        let left = Tensor::new_integer(IntegerStorage::I8(vec![12]), vec![1, 1]).expect("left");
        let single = Tensor::from_f32(vec![3.5], vec![1, 1]).expect("single");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![
                Value::Tensor(left),
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Tensor(single),
                Value::Bool(true),
            ],
        )
        .expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I8(vec![12, i8::MAX, 4, 1]))
        );
    }

    #[test]
    fn cat_accepts_every_ordered_integer_class_pair_and_keeps_left_class() {
        let classes = vec![
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for left_storage in &classes {
            for right_storage in &classes {
                let left = Tensor::new_integer(left_storage.clone(), vec![1, 1]).expect("left");
                let right = Tensor::new_integer(right_storage.clone(), vec![1, 1]).expect("right");
                let result = cat_builtin(
                    Value::Int(IntValue::I32(2)),
                    vec![Value::Tensor(left), Value::Tensor(right)],
                )
                .expect("ordered integer pair");
                let Value::Tensor(output) = result else {
                    panic!("expected tensor");
                };
                let output_storage = output.integer_storage().expect("integer output");
                assert_eq!(
                    std::mem::discriminant(output_storage),
                    std::mem::discriminant(left_storage),
                    "left={left_storage:?}, right={right_storage:?}"
                );
                assert_eq!(output.shape, vec![1, 2]);
            }
        }
    }

    #[test]
    fn cat_selects_a_later_integer_class_after_double_input() {
        let right = Tensor::new_integer(IntegerStorage::I16(vec![8]), vec![1, 1]).expect("right");
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::Num(3.5), Value::Tensor(right)],
        )
        .expect("cat");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![2, 1]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I16(vec![4, 8]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_dimension_mismatch_errors() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3, 1]).unwrap();
        let err = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(a), Value::Tensor(b)],
        )
        .unwrap_err();
        assert!(err.message().contains("dimension 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_char_columns() {
        let left = CharArray::new("Run".chars().collect(), 1, 3).unwrap();
        let right = CharArray::new("Mat".chars().collect(), 1, 3).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::CharArray(left), Value::CharArray(right)],
        )
        .expect("cat");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 6);
                let text: String = arr.data.iter().collect();
                assert_eq!(text, "RunMat");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_string_and_char_mix_promotes_to_string_array() {
        let left = StringArray::new(vec!["wn = ".to_string()], vec![1, 1]).unwrap();
        let mid = CharArray::new_row("6");
        let right = StringArray::new(vec![" rad/s".to_string()], vec![1, 1]).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![
                Value::StringArray(left),
                Value::CharArray(mid),
                Value::StringArray(right),
            ],
        )
        .expect("cat");
        match result {
            Value::StringArray(arr) => {
                assert_eq!(arr.shape, vec![1, 3]);
                assert_eq!(
                    arr.data,
                    vec!["wn = ".to_string(), "6".to_string(), " rad/s".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn cat_string_dominates_exact_numeric_and_logical_inputs() {
        let prefix = StringArray::new(vec!["id".to_string()], vec![1, 1]).unwrap();
        let integers = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .expect("integer input");
        let logical = LogicalArray::new(vec![1], vec![1, 1]).expect("logical input");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![
                Value::StringArray(prefix),
                Value::Tensor(integers),
                Value::LogicalArray(logical),
            ],
        )
        .expect("mixed string cat");

        let Value::StringArray(output) = result else {
            panic!("expected string array");
        };
        assert_eq!(output.shape, vec![1, 4]);
        assert_eq!(
            output.data,
            vec![
                "id".to_string(),
                "9007199254740993".to_string(),
                u64::MAX.to_string(),
                "1".to_string(),
            ]
        );
    }

    #[test]
    fn cat_cell_wraps_each_complete_noncell_operand() {
        let first = CellArray::new(vec![Value::String("head".into())], 1, 1).expect("cell");
        let matrix =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3, 4]), vec![2, 2]).expect("matrix");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Cell(first), Value::Tensor(matrix.clone())],
        )
        .expect("mixed cell cat");

        let Value::Cell(output) = result else {
            panic!("expected cell array");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(output.data[0], Value::String("head".into()));
        assert_eq!(output.data[1], Value::Tensor(matrix));
    }

    #[test]
    fn cat_char_and_logical_rejects_both_scalar_and_array_forms() {
        let chars = CharArray::new_row("a");
        let scalar_error = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::CharArray(chars.clone()), Value::Bool(true)],
        )
        .unwrap_err();
        assert!(scalar_error.message().contains("char"));

        let logical = LogicalArray::new(vec![1], vec![1, 1]).expect("logical");
        let array_error = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::CharArray(chars), Value::LogicalArray(logical)],
        )
        .unwrap_err();
        assert!(array_error.message().contains("char"));
    }

    #[test]
    fn cat_supports_nd_character_and_cell_arrays() {
        let left_chars =
            CharArray::new_with_shape("abcd".chars().collect(), vec![1, 2, 2]).expect("left");
        let right_chars =
            CharArray::new_with_shape("efgh".chars().collect(), vec![1, 2, 2]).expect("right");
        let char_result = cat_builtin(
            Value::Int(IntValue::I32(3)),
            vec![Value::CharArray(left_chars), Value::CharArray(right_chars)],
        )
        .expect("nd char cat");
        let Value::CharArray(chars) = char_result else {
            panic!("expected char array");
        };
        assert_eq!(chars.shape, vec![1, 2, 4]);
        assert_eq!(chars.data.iter().collect::<String>(), "abcdefgh");

        let left_cells = CellArray::new_with_shape(
            vec![Value::Int(IntValue::I8(1)), Value::Int(IntValue::I8(2))],
            vec![1, 1, 2],
        )
        .expect("left cells");
        let right_cells = CellArray::new_with_shape(
            vec![Value::Int(IntValue::I8(3)), Value::Int(IntValue::I8(4))],
            vec![1, 1, 2],
        )
        .expect("right cells");
        let cell_result = cat_builtin(
            Value::Int(IntValue::I32(3)),
            vec![Value::Cell(left_cells), Value::Cell(right_cells)],
        )
        .expect("nd cell cat");
        let Value::Cell(cells) = cell_result else {
            panic!("expected cell array");
        };
        assert_eq!(cells.shape, vec![1, 1, 4]);
        assert_eq!(
            cells.data,
            vec![
                Value::Int(IntValue::I8(1)),
                Value::Int(IntValue::I8(2)),
                Value::Int(IntValue::I8(3)),
                Value::Int(IntValue::I8(4)),
            ]
        );
    }

    #[test]
    fn cat_omits_shaped_empty_geometry_and_uses_the_nonempty_integer_class() {
        let shaped_empty =
            Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 4]).expect("empty");
        let nonempty = Tensor::new_integer(IntegerStorage::U8(vec![1, 2, 3, 4, 5, 6]), vec![2, 3])
            .expect("nonempty");
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::Tensor(shaped_empty), Value::Tensor(nonempty)],
        )
        .expect("empty omission");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![2, 3]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U8(vec![1, 2, 3, 4, 5, 6]))
        );
    }

    #[test]
    fn cat_all_empty_inputs_keep_the_leftmost_integer_class() {
        let left =
            Tensor::new_integer(IntegerStorage::I16(Vec::new()), vec![0, 2]).expect("left empty");
        let right =
            Tensor::new_integer(IntegerStorage::U8(Vec::new()), vec![0, 3]).expect("right empty");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(left), Value::Tensor(right)],
        )
        .expect("all-empty concatenation");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![0, 5]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I16(Vec::new()))
        );
    }

    #[test]
    fn cat_complex_integer_empty_inputs_follow_the_same_class_selection_rule() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let empty = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I16(Vec::new()),
                IntegerStorage::I16(Vec::new()),
            )
            .expect("empty storage"),
            vec![1, 0],
        )
        .expect("empty complex integer");
        let value = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::U8(vec![7]), IntegerStorage::U8(vec![9]))
                .expect("value storage"),
            vec![1, 1],
        )
        .expect("complex integer value");
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::ComplexTensor(empty), Value::ComplexTensor(value)],
        )
        .expect("complex empty omission");
        let Value::ComplexTensor(output) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(
                &IntegerComplexStorage::new(
                    IntegerStorage::U8(vec![7]),
                    IntegerStorage::U8(vec![9]),
                )
                .expect("expected storage")
            )
        );
    }

    #[test]
    fn cat_mixed_host_and_gpu_complex_integers_preserves_exact_resident_output() {
        test_support::with_test_provider(|provider| {
            let device = ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![9_007_199_254_740_993]),
                    IntegerStorage::U64(vec![3]),
                )
                .expect("device storage"),
                vec![1, 1],
            )
            .expect("device value");
            let host = ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![5]),
                )
                .expect("host storage"),
                vec![1, 1],
            )
            .expect("host value");
            let device = gpu_helpers::upload_complex_tensor(provider, &device).expect("upload");
            let Value::GpuTensor(output) = cat_builtin(
                Value::Int(IntValue::I32(2)),
                vec![Value::GpuTensor(device), Value::ComplexTensor(host)],
            )
            .expect("cat") else {
                panic!("expected resident complex integer");
            };
            let Value::ComplexTensor(gathered) =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output)))
                    .expect("gather")
            else {
                panic!("expected complex integer");
            };
            assert_eq!(gathered.shape, vec![1, 2]);
            let storage = gathered.integer_storage().expect("integer storage");
            assert_eq!(
                storage.real,
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX])
            );
            assert_eq!(storage.imag, IntegerStorage::U64(vec![3, 5]));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_logical_preserves_type() {
        let top = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let bottom = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::LogicalArray(top), Value::LogicalArray(bottom)],
        )
        .expect("cat");
        match result {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.shape, vec![2, 3]);
                assert_eq!(arr.data, vec![1, 0, 0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let b = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let view_a = HostTensorView {
                data: &a.materialize_f64(),
                shape: &a.shape,
            };
            let view_b = HostTensorView {
                data: &b.materialize_f64(),
                shape: &b.shape,
            };
            let ha = provider.upload(&view_a).expect("upload a");
            let hb = provider.upload(&view_b).expect("upload b");
            let result = cat_builtin(
                Value::Int(IntValue::I32(1)),
                vec![Value::GpuTensor(ha), Value::GpuTensor(hb)],
            )
            .expect("cat");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_like_gpu_from_host_inputs() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &proto.materialize_f64(),
                shape: &proto.shape,
            };
            let proto_handle = provider.upload(&proto_view).expect("upload proto");

            let a = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let b = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let result = cat_builtin(
                Value::Int(IntValue::I32(1)),
                vec![
                    Value::Tensor(a),
                    Value::Tensor(b),
                    Value::from("like"),
                    Value::GpuTensor(proto_handle),
                ],
            )
            .expect("cat with like");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_like_gpu_from_host_integer_inputs_uploads_exact_integer_storage() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_handle = provider
                .upload(&HostTensorView {
                    data: &proto.materialize_f64(),
                    shape: &proto.shape,
                })
                .expect("upload proto");
            let left = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 4]), vec![1, 2])
                .expect("left");
            let right =
                Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                    .expect("right");

            let result = cat_builtin(
                Value::Int(IntValue::I32(2)),
                vec![
                    Value::Tensor(left),
                    Value::Tensor(right),
                    Value::from("like"),
                    Value::GpuTensor(proto_handle),
                ],
            )
            .expect("cat with integer gpu like");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U64)
            );
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    u64::MAX,
                    4,
                    9_007_199_254_740_993,
                ]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_gpu_integer_inputs_fallback_concatenates_exact_storage() {
        test_support::with_test_provider(|provider| {
            let left_values = [u64::MAX, 4];
            let right_values = [9_007_199_254_740_993_u64];
            let left = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&left_values),
                    shape: &[1, 2],
                })
                .expect("upload left");
            let right = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&right_values),
                    shape: &[1, 1],
                })
                .expect("upload right");

            let result = cat_builtin(
                Value::Int(IntValue::I32(2)),
                vec![Value::GpuTensor(left), Value::GpuTensor(right)],
            )
            .expect("cat gpu integers");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpu integer output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U64)
            );
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    u64::MAX,
                    4,
                    9_007_199_254_740_993,
                ]))
            );
        });
    }

    #[test]
    fn cat_mixed_host_and_gpu_integer_inputs_preserve_exact_storage() {
        test_support::with_test_provider(|provider| {
            let resident_values = [9_007_199_254_740_993_u64];
            let resident = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&resident_values),
                    shape: &[1, 1],
                })
                .expect("upload resident integer");
            let host = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("host integer");

            let result = cat_builtin(
                Value::Int(IntValue::I32(2)),
                vec![Value::GpuTensor(resident), Value::Tensor(host)],
            )
            .expect("mixed residency cat");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U64)
            );
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX,]))
            );
        });
    }

    #[test]
    fn cat_runmat_only_forms_gate_before_provider_or_data_conversion() {
        let left = Tensor::new(vec![1.0], vec![1, 1]).expect("left");
        let right = Tensor::new(vec![2.0], vec![1, 1]).expect("right");
        let like_error = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![
                Value::Tensor(left.clone()),
                Value::Tensor(right.clone()),
                Value::from("like"),
                Value::Num(0.0),
            ],
        )
        .expect_err("MATLAB mode rejects like form");
        assert_eq!(like_error.identifier(), CAT_LIKE_EXTENSION.error_identifier);

        let complex_storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![2]))
                .expect("complex storage");
        let complex =
            ComplexTensor::new_integer(complex_storage, vec![1, 1]).expect("complex integer");
        let Value::ComplexTensor(complex_output) = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![
                Value::ComplexTensor(complex.clone()),
                Value::ComplexTensor(complex),
            ],
        )
        .expect("typed complex integer concatenation is documented") else {
            panic!("expected complex tensor");
        };
        assert_eq!(complex_output.shape, vec![2, 1]);

        test_support::with_test_provider(|provider| {
            let dim_values = [1_u16];
            let dim = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&dim_values),
                    shape: &[1, 1],
                })
                .expect("upload dimension");
            let dim_error = cat_builtin(
                Value::GpuTensor(dim.clone()),
                vec![Value::Tensor(left.clone()), Value::Tensor(right.clone())],
            )
            .expect_err("MATLAB mode rejects resident dimension");
            assert_eq!(
                dim_error.identifier(),
                CAT_RESIDENT_DIMENSION_EXTENSION.error_identifier
            );

            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let result = cat_builtin(
                Value::GpuTensor(dim),
                vec![Value::Tensor(left), Value::Tensor(right)],
            )
            .expect("RunMat mode accepts resident dimension");
            let Value::Tensor(output) = result else {
                panic!("expected host output");
            };
            assert_eq!(output.materialize_f64(), vec![1.0, 2.0]);
        });
    }

    #[test]
    fn cat_gpu_fallback_omits_shaped_empty_geometry() {
        test_support::with_test_provider(|provider| {
            let empty = provider
                .upload(&HostTensorView {
                    data: &[],
                    shape: &[0, 4],
                })
                .expect("upload empty");
            let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let nonempty = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &[2, 3],
                })
                .expect("upload nonempty");
            let result = cat_builtin(
                Value::Int(IntValue::I32(1)),
                vec![Value::GpuTensor(empty), Value::GpuTensor(nonempty)],
            )
            .expect("gpu empty omission");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 3]);
            assert_eq!(gathered.materialize_f64(), values);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_like_logical_mismatch_errors() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let proto = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let err = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
                Value::from("like"),
                Value::LogicalArray(proto),
            ],
        )
        .unwrap_err();
        assert!(err.message().contains("logical"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cat_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();

        let cpu_result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(a.clone()), Value::Tensor(b.clone())],
        )
        .expect("cat cpu");
        let expected = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let view_a = HostTensorView {
            data: &a.materialize_f64(),
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.materialize_f64(),
            shape: &b.shape,
        };
        let ha = provider.upload(&view_a).expect("upload a");
        let hb = provider.upload(&view_b).expect("upload b");
        let gpu_value = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::GpuTensor(ha), Value::GpuTensor(hb)],
        )
        .expect("cat gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn cat_wgpu_mixed_residency_preserves_wide_integer_storage() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let resident_values = [9_007_199_254_740_993_u64];
        let resident = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&resident_values),
                shape: &[1, 1],
            })
            .expect("upload resident integer");
        let host = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("host integer");

        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::GpuTensor(resident), Value::Tensor(host)],
        )
        .expect("mixed WGPU cat");
        let Value::GpuTensor(handle) = result else {
            panic!("expected resident output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&handle),
            Some(IntegerElementType::U64)
        );
        let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX,]))
        );
    }
}
