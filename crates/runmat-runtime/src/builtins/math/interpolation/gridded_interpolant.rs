//! MATLAB-compatible `griddedInterpolant` builtin.
//!
//! This implements RunMat's gridded interpolant object surface for real dense
//! data. The object stores compact grid vectors and evaluates through `subsref`
//! so MATLAB syntax such as `F(xq)` and `F(xq,yq)` works through the normal
//! object indexing path.

use std::cell::Cell as LocalCell;
use std::collections::HashMap;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ClassDef, MethodDef, PropertyDef, ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    Access, CellArray, IntValue, IntegerStorage, NumericDType, ObjectInstance, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::{
    build_runtime_error, BuiltinResult, RuntimeError, OBJECT_INDEX_MEMBER, OBJECT_INDEX_PAREN,
    OBJECT_SUBSREF_METHOD,
};

use super::pp::{
    build_pchip_pp, build_spline_pp, eval_pp_scalar, Extrapolation, NumericSeries,
    PiecewisePolynomial,
};

const CLASS_NAME: &str = "griddedInterpolant";
const BUILTIN_SUBSREF: &str = "griddedInterpolant.subsref";
const GRID_VECTORS: &str = "GridVectors";
const VALUES: &str = "Values";
const METHOD: &str = "Method";
const EXTRAPOLATION_METHOD: &str = "ExtrapolationMethod";

thread_local! {
    static GRIDDED_INTERPOLANT_CLASS_REGISTERED: LocalCell<bool> = const { LocalCell::new(false) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InterpMethod {
    Linear,
    Nearest,
    Next,
    Previous,
    Pchip,
    Spline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtrapolationMethod {
    Linear,
    Nearest,
    Next,
    Previous,
    Pchip,
    Spline,
    None,
}

#[derive(Clone)]
struct InterpolantSpec {
    grid_vectors: Vec<Vec<f64>>,
    values: Tensor,
    method: InterpMethod,
    extrapolation: ExtrapolationMethod,
}

#[derive(Clone)]
struct Bracket {
    lo: usize,
    hi: usize,
    t: f64,
}

#[derive(Clone)]
struct QueryPlan {
    coords: QueryCoords,
    shape: Vec<usize>,
}

#[derive(Clone)]
enum QueryCoords {
    Explicit(Vec<Vec<f64>>),
    FullGrid {
        vectors: Vec<Vec<f64>>,
        total: usize,
    },
}

impl QueryPlan {
    fn len(&self) -> usize {
        match &self.coords {
            QueryCoords::Explicit(points) => points.len(),
            QueryCoords::FullGrid { total, .. } => *total,
        }
    }

    fn for_each_point(&self, mut f: impl FnMut(&[f64])) {
        match &self.coords {
            QueryCoords::Explicit(points) => {
                for point in points {
                    f(point);
                }
            }
            QueryCoords::FullGrid { vectors, total } => {
                for linear in 0..*total {
                    let subs = unravel_column_major(linear, &self.shape);
                    let point = subs
                        .iter()
                        .enumerate()
                        .map(|(dim, &idx)| vectors[dim][idx])
                        .collect::<Vec<_>>();
                    f(&point);
                }
            }
        }
    }
}

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "F",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Gridded interpolant object.",
}];

const SUBSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Indexed value or property.",
}];

const INPUTS_VARIADIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Grid vectors or full grids, values, interpolation method, and extrapolation method.",
}];

const SUBSREF_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "griddedInterpolant object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing payload.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant",
        inputs: &[],
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(V)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(x, v)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(gridVecs, V)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(X1, X2, ..., Xn, V)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(___, Method)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(___, Method, ExtrapolationMethod)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
];

const SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = griddedInterpolant.subsref(obj, kind, payload)",
    inputs: &SUBSREF_INPUTS,
    outputs: &SUBSREF_OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRIDDEDINTERPOLANT.INVALID_INPUT",
    identifier: Some("RunMat:griddedInterpolant:InvalidInput"),
    when: "Grid, value, method, or query inputs are invalid.",
    message: "griddedInterpolant: invalid input",
};

const ERROR_UNSUPPORTED_METHOD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRIDDEDINTERPOLANT.UNSUPPORTED_METHOD",
    identifier: Some("RunMat:griddedInterpolant:UnsupportedMethod"),
    when: "The requested interpolation or extrapolation method is not implemented by RunMat.",
    message: "griddedInterpolant: unsupported interpolation method",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRIDDEDINTERPOLANT.INTERNAL",
    identifier: Some("RunMat:griddedInterpolant:Internal"),
    when: "Object construction, property access, or interpolation output construction fails.",
    message: "griddedInterpolant: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ERROR_INVALID_INPUT,
    ERROR_UNSUPPORTED_METHOD,
    ERROR_INTERNAL,
];

pub const GRIDDED_INTERPOLANT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const GRIDDED_INTERPOLANT_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &ERRORS,
};

const GRIDDED_INTEGER_GRID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "griddedinterpolant-integer-grid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "griddedInterpolant with typed-integer sample grids is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GriddedInterpolantIntegerGridExtension"),
};

const GRIDDED_INTEGER_VALUES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "griddedinterpolant-integer-values",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "griddedInterpolant with typed-integer sample values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GriddedInterpolantIntegerValuesExtension"),
};

const GRIDDED_INTEGER_QUERY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "griddedinterpolant-integer-query",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "griddedInterpolant with typed-integer query coordinates is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GriddedInterpolantIntegerQueryExtension"),
};

pub const GRIDDED_INTERPOLANT_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    GRIDDED_INTEGER_GRID_EXTENSION,
    GRIDDED_INTEGER_VALUES_EXTENSION,
    GRIDDED_INTEGER_QUERY_EXTENSION,
];

const GRIDDED_INTEGER_GRID_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x, X_i, or gridVecs",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes must be exactly representable at the binary64 interpolation-coordinate boundary.",
    }];

const GRIDDED_INTEGER_VALUES_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "v, V, or Values",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes must be exactly representable before the object stores ordinary double Values.",
    }];

const GRIDDED_INTEGER_QUERY_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Xq, Xq_i, or query grid vectors",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All query syntaxes admit the same eight integer classes only when every coordinate is exactly representable as binary64.",
    }];

pub const GRIDDED_INTERPOLANT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "F = griddedInterpolant(integer_grid, V, ...)",
        inputs: &GRIDDED_INTEGER_GRID_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer sample grids cross one checked binary64 boundary; the constructor returns an interpolant object whose GridVectors are floating arrays.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "F = griddedInterpolant(grid, integer_V, ...)",
        inputs: &GRIDDED_INTEGER_VALUES_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer sample values cross one checked binary64 boundary, are stored as double Values, and produce double evaluations.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = F(integer_query)",
        inputs: &GRIDDED_INTEGER_QUERY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer query coordinates cross one checked binary64 boundary; output precision continues to follow the interpolant Values class.",
    },
];

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::interpolation::gridded_interpolant"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: CLASS_NAME,
    op_kind: GpuOpKind::Custom("gridded-interpolation"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "griddedInterpolant currently stores host tensors and evaluates through the runtime object indexing path.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::interpolation::gridded_interpolant"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: CLASS_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Interpolant object construction and evaluation are runtime sinks.",
};

fn gridded_interpolant_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn gridded_error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(CLASS_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    gridded_error(&ERROR_INVALID_INPUT, detail)
}

fn unsupported(detail: impl AsRef<str>) -> RuntimeError {
    gridded_error(&ERROR_UNSUPPORTED_METHOD, detail)
}

fn internal(detail: impl AsRef<str>) -> RuntimeError {
    gridded_error(&ERROR_INTERNAL, detail)
}

fn ensure_gridded_interpolant_class_registered() {
    GRIDDED_INTERPOLANT_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }
        let mut properties = HashMap::new();
        for name in [GRID_VECTORS, VALUES, METHOD, EXTRAPOLATION_METHOD] {
            properties.insert(
                name.to_string(),
                PropertyDef {
                    name: name.to_string(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: Access::Public,
                    set_access: Access::Public,
                    default_value: None,
                },
            );
        }
        let mut methods = HashMap::new();
        methods.insert(
            OBJECT_SUBSREF_METHOD.to_string(),
            MethodDef {
                name: OBJECT_SUBSREF_METHOD.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: BUILTIN_SUBSREF.to_string(),
                implicit_class_argument: None,
            },
        );
        runmat_builtins::register_class(ClassDef {
            name: CLASS_NAME.to_string(),
            parent: None,
            properties,
            methods,
        });
        registered.set(true);
    });
}

#[runtime_builtin(
    name = "griddedInterpolant",
    category = "math/interpolation",
    summary = "Create a gridded data interpolant object.",
    keywords = "griddedInterpolant,interpolation,grid,linear,nearest,object",
    type_resolver(gridded_interpolant_type),
    descriptor(
        crate::builtins::math::interpolation::gridded_interpolant::GRIDDED_INTERPOLANT_DESCRIPTOR
    ),
    extensions(
        crate::builtins::math::interpolation::gridded_interpolant::GRIDDED_INTERPOLANT_EXTENSIONS
    ),
    integer_capabilities(
        crate::builtins::math::interpolation::gridded_interpolant::GRIDDED_INTERPOLANT_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::math::interpolation::gridded_interpolant"
)]
async fn gridded_interpolant_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_constructor_integer_extensions(&args)?;
    let args = gather_values(args).await?;
    ensure_gridded_interpolant_class_registered();
    let spec = parse_constructor(args)?;
    Ok(Value::Object(spec_to_object(spec)?))
}

#[runtime_builtin(
    name = "griddedInterpolant.subsref",
    type_resolver(gridded_interpolant_type),
    descriptor(
        crate::builtins::math::interpolation::gridded_interpolant::GRIDDED_INTERPOLANT_SUBSREF_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::math::interpolation::gridded_interpolant"
)]
async fn gridded_interpolant_subsref(
    obj: Value,
    kind: String,
    payload: Value,
) -> BuiltinResult<Value> {
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => gridded_member(obj, payload),
        OBJECT_INDEX_PAREN => {
            let spec = object_to_spec(&obj)?;
            let query_args = payload_to_args(payload)?;
            ensure_query_integer_extensions(&query_args)?;
            let query_args = gather_values(query_args).await?;
            evaluate_interpolant(&spec, query_args)
        }
        other => Err(invalid(format!("unsupported indexing kind '{other}'"))),
    }
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(values.len());
    for value in values {
        gathered.push(crate::dispatcher::gather_if_needed_async(&value).await?);
    }
    Ok(gathered)
}

fn ensure_constructor_integer_extensions(args: &[Value]) -> BuiltinResult<()> {
    let mut numeric_len = args.len();
    let mut methods = 0usize;
    while numeric_len > 0 && methods < 2 {
        let Some(name) = text_from_value(&args[numeric_len - 1]) else {
            break;
        };
        if !is_method_name(&name) {
            break;
        }
        numeric_len -= 1;
        methods += 1;
    }
    match numeric_len {
        0 => Ok(()),
        1 => ensure_integer_extension(&args[0], &GRIDDED_INTEGER_VALUES_EXTENSION, "sample values"),
        2 => {
            ensure_integer_extension(&args[0], &GRIDDED_INTEGER_GRID_EXTENSION, "sample grid")?;
            ensure_integer_extension(&args[1], &GRIDDED_INTEGER_VALUES_EXTENSION, "sample values")
        }
        _ => {
            for grid in &args[..numeric_len - 1] {
                ensure_integer_extension(grid, &GRIDDED_INTEGER_GRID_EXTENSION, "sample grid")?;
            }
            ensure_integer_extension(
                &args[numeric_len - 1],
                &GRIDDED_INTEGER_VALUES_EXTENSION,
                "sample values",
            )
        }
    }
}

fn ensure_query_integer_extensions(args: &[Value]) -> BuiltinResult<()> {
    for query in args {
        ensure_integer_extension(query, &GRIDDED_INTEGER_QUERY_EXTENSION, "query coordinates")?;
    }
    Ok(())
}

fn ensure_integer_extension(
    value: &Value,
    extension: &BuiltinExtensionDescriptor,
    role: &str,
) -> BuiltinResult<()> {
    if value_contains_typed_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, CLASS_NAME)?;
        ensure_exact_integer_value(value, role)?;
    }
    Ok(())
}

fn value_contains_typed_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        Value::Cell(cell) => cell.data.iter().any(value_contains_typed_integer),
        _ => false,
    }
}

fn ensure_exact_integer_value(value: &Value, role: &str) -> BuiltinResult<()> {
    let exact = crate::builtins::math::trigonometry::cos::integer_is_exact_f64;
    let valid = match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        Value::Cell(cell) => cell
            .data
            .iter()
            .all(|value| ensure_exact_integer_value(value, role).is_ok()),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid(format!(
            "integer {role} must be exactly representable as double"
        )))
    }
}

fn parse_constructor(mut args: Vec<Value>) -> BuiltinResult<InterpolantSpec> {
    let (method, extrapolation) = parse_trailing_methods(&mut args)?;
    match args.len() {
        0 => Ok(InterpolantSpec {
            grid_vectors: Vec::new(),
            values: Tensor::new(Vec::new(), vec![0, 0]).map_err(internal)?,
            method,
            extrapolation,
        }),
        1 => {
            let values = normalize_default_values(numeric_tensor(args.remove(0), "V")?)?;
            let grid_vectors = default_grid_vectors(&values);
            validate_grid_value_shape(&grid_vectors, &values, method, extrapolation)?;
            Ok(InterpolantSpec {
                grid_vectors,
                values,
                method,
                extrapolation,
            })
        }
        2 => {
            let values_arg = args.pop().expect("values");
            let grid_arg = args.pop().expect("grid");
            let values = numeric_tensor(values_arg, "V")?;
            let grid_vectors = match grid_arg {
                Value::Cell(cell) => grid_vectors_from_cell(cell)?,
                other => vec![numeric_vector(other, "x")?],
            };
            let values = normalize_values_for_grid(&grid_vectors, values)?;
            validate_grid_value_shape(&grid_vectors, &values, method, extrapolation)?;
            Ok(InterpolantSpec {
                grid_vectors,
                values,
                method,
                extrapolation,
            })
        }
        _ => {
            let values = numeric_tensor(args.pop().expect("values"), "V")?;
            let grid_vectors = full_grid_vectors(args, &values)?;
            let values = normalize_values_for_grid(&grid_vectors, values)?;
            validate_grid_value_shape(&grid_vectors, &values, method, extrapolation)?;
            Ok(InterpolantSpec {
                grid_vectors,
                values,
                method,
                extrapolation,
            })
        }
    }
}

fn parse_trailing_methods(
    args: &mut Vec<Value>,
) -> BuiltinResult<(InterpMethod, ExtrapolationMethod)> {
    let mut names = Vec::new();
    while names.len() < 2 {
        let Some(name) = args.last().and_then(text_from_value) else {
            break;
        };
        if !is_method_name(&name) {
            break;
        }
        names.push(name);
        args.pop();
    }
    names.reverse();
    let method = if let Some(name) = names.first() {
        parse_interp_method(name)?
    } else {
        InterpMethod::Linear
    };
    let extrapolation = if let Some(name) = names.get(1) {
        parse_extrap_method(name)?
    } else {
        match method {
            InterpMethod::Linear => ExtrapolationMethod::Linear,
            InterpMethod::Nearest => ExtrapolationMethod::Nearest,
            InterpMethod::Next => ExtrapolationMethod::Next,
            InterpMethod::Previous => ExtrapolationMethod::Previous,
            InterpMethod::Pchip => ExtrapolationMethod::Pchip,
            InterpMethod::Spline => ExtrapolationMethod::Spline,
        }
    };
    Ok((method, extrapolation))
}

fn is_method_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "linear"
            | "nearest"
            | "next"
            | "previous"
            | "none"
            | "pchip"
            | "cubic"
            | "makima"
            | "spline"
    )
}

fn parse_interp_method(name: &str) -> BuiltinResult<InterpMethod> {
    match name.to_ascii_lowercase().as_str() {
        "linear" => Ok(InterpMethod::Linear),
        "nearest" => Ok(InterpMethod::Nearest),
        "next" => Ok(InterpMethod::Next),
        "previous" => Ok(InterpMethod::Previous),
        "pchip" => Ok(InterpMethod::Pchip),
        "cubic" => Err(unsupported(
            "method 'cubic' is not yet implemented for griddedInterpolant",
        )),
        "spline" => Ok(InterpMethod::Spline),
        "makima" => Err(unsupported(
            "method 'makima' is not yet implemented for griddedInterpolant",
        )),
        "none" => Err(invalid("'none' is only valid as an extrapolation method")),
        other => Err(invalid(format!("unknown interpolation method '{other}'"))),
    }
}

fn parse_extrap_method(name: &str) -> BuiltinResult<ExtrapolationMethod> {
    match name.to_ascii_lowercase().as_str() {
        "linear" => Ok(ExtrapolationMethod::Linear),
        "nearest" => Ok(ExtrapolationMethod::Nearest),
        "next" => Ok(ExtrapolationMethod::Next),
        "previous" => Ok(ExtrapolationMethod::Previous),
        "pchip" => Ok(ExtrapolationMethod::Pchip),
        "cubic" => Err(unsupported(
            "extrapolation method 'cubic' is not yet implemented for griddedInterpolant",
        )),
        "spline" => Ok(ExtrapolationMethod::Spline),
        "none" => Ok(ExtrapolationMethod::None),
        "makima" => Err(unsupported(
            "extrapolation method 'makima' is not yet implemented for griddedInterpolant",
        )),
        other => Err(invalid(format!("unknown extrapolation method '{other}'"))),
    }
}

fn text_from_value(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        _ => None,
    }
}

fn numeric_tensor(value: Value, name: &str) -> BuiltinResult<Tensor> {
    match value {
        Value::Num(x) => Tensor::new(vec![x], vec![1, 1]).map_err(internal),
        Value::Int(value) => {
            ensure_exact_integer_scalar(&value, name)?;
            let tensor = Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(internal)?;
            tensor_utils::integer_tensor_to_f64(tensor)
                .map_err(|err| invalid(format!("{name}: {err}")))
        }
        Value::Tensor(tensor) => match tensor.numeric_dtype() {
            NumericDType::F64 | NumericDType::F32 => Ok(tensor),
            _ => {
                ensure_exact_integer_tensor(&tensor, name)?;
                tensor_utils::integer_tensor_to_f64(tensor)
                    .map_err(|err| invalid(format!("{name}: {err}")))
            }
        },
        other => Err(invalid(format!(
            "{name} must be a real numeric array, got {other:?}"
        ))),
    }
}

fn numeric_vector(value: Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(x) => Ok(vec![x]),
        Value::Int(value) => {
            ensure_exact_integer_scalar(&value, name)?;
            Ok(vec![value.to_f64()])
        }
        Value::Tensor(tensor) if is_vector_shape(&tensor.shape) => {
            // Grid coordinates are evaluated in the interpolator's f64 domain.
            ensure_exact_integer_tensor(&tensor, name)?;
            Ok(tensor_utils::tensor_into_values_f64(tensor))
        }
        Value::Tensor(_) => Err(invalid(format!("{name} must be a vector"))),
        other => Err(invalid(format!(
            "{name} must be a real numeric vector, got {other:?}"
        ))),
    }
}

fn ensure_exact_integer_scalar(value: &IntValue, name: &str) -> BuiltinResult<()> {
    if crate::builtins::math::trigonometry::cos::integer_is_exact_f64(value) {
        Ok(())
    } else {
        Err(invalid(format!(
            "{name}: integer value must be exactly representable as double"
        )))
    }
}

fn ensure_exact_integer_tensor(tensor: &Tensor, name: &str) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    let exact = crate::builtins::math::trigonometry::cos::integer_is_exact_f64;
    if storage.exact_values().iter().all(exact) {
        Ok(())
    } else {
        Err(invalid(format!(
            "{name}: integer values must be exactly representable as double"
        )))
    }
}

fn normalize_default_values(values: Tensor) -> BuiltinResult<Tensor> {
    if is_vector_shape(&values.shape) && values.shape.first().copied().unwrap_or(1) == 1 {
        let len = tensor_utils::tensor_element_len(&values);
        values.reshape(vec![len, 1]).map_err(internal)
    } else {
        Ok(values)
    }
}

fn normalize_values_for_grid(grid_vectors: &[Vec<f64>], values: Tensor) -> BuiltinResult<Tensor> {
    if grid_vectors.len() == 1
        && is_vector_shape(&values.shape)
        && tensor_utils::tensor_element_len(&values) == grid_vectors[0].len()
        && values.shape.first().copied().unwrap_or(1) == 1
    {
        values
            .reshape(vec![grid_vectors[0].len(), 1])
            .map_err(internal)
    } else {
        Ok(values)
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.is_empty() || shape.iter().filter(|&&dim| dim > 1).count() <= 1
}

fn grid_vectors_from_cell(cell: CellArray) -> BuiltinResult<Vec<Vec<f64>>> {
    if cell.data.is_empty() {
        return Err(invalid("gridVecs must contain at least one grid vector"));
    }
    cell.data
        .into_iter()
        .enumerate()
        .map(|(idx, value)| numeric_vector(value, &format!("gridVecs{{{}}}", idx + 1)))
        .collect()
}

fn default_grid_vectors(values: &Tensor) -> Vec<Vec<f64>> {
    let dims = effective_grid_rank(&values.shape);
    (0..dims)
        .map(|dim| {
            let len = values.shape.get(dim).copied().unwrap_or(1);
            (1..=len).map(|value| value as f64).collect::<Vec<_>>()
        })
        .collect()
}

fn effective_grid_rank(shape: &[usize]) -> usize {
    match shape.len() {
        0 => 1,
        1 => 1,
        2 if shape[0] == 1 || shape[1] == 1 => 1,
        _ => shape.len().max(1),
    }
}

fn full_grid_vectors(grids: Vec<Value>, values: &Tensor) -> BuiltinResult<Vec<Vec<f64>>> {
    if grids.is_empty() {
        return Err(invalid("full-grid syntax requires at least one grid array"));
    }
    let mut tensors: Vec<Tensor> = Vec::with_capacity(grids.len());
    for (idx, grid) in grids.into_iter().enumerate() {
        let tensor = numeric_tensor(grid, &format!("X{}", idx + 1))?;
        if let Some(first) = tensors.first() {
            if tensor.shape != first.shape {
                return Err(invalid("full-grid arrays must have matching shapes"));
            }
        }
        tensors.push(tensor);
    }
    let grid_rank = tensors.len();
    let grid_shape = tensors[0].shape.clone();
    if grid_shape.len() != grid_rank {
        return Err(invalid(
            "full-grid arrays must have one dimension per grid variable",
        ));
    }
    if values.shape.len() < grid_rank || values.shape[..grid_rank] != grid_shape[..] {
        return Err(invalid(
            "full-grid arrays must match the leading shape of V",
        ));
    }
    let mut vectors = Vec::with_capacity(grid_rank);
    for (dim, tensor) in tensors.iter().enumerate() {
        let len = grid_shape[dim];
        let values = tensor_utils::tensor_values_f64_cow(tensor);
        let mut vector = Vec::with_capacity(len);
        for idx in 0..len {
            let mut subs = vec![0; grid_rank];
            subs[dim] = idx;
            vector.push(values[column_major_offset(&subs, &tensor.shape)]);
        }
        vectors.push(vector);
    }
    validate_rectilinear_full_grids(&tensors, &vectors, &grid_shape)?;
    Ok(vectors)
}

fn validate_rectilinear_full_grids(
    tensors: &[Tensor],
    vectors: &[Vec<f64>],
    shape: &[usize],
) -> BuiltinResult<()> {
    let total = shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, len| acc.checked_mul(len))
        .ok_or_else(|| invalid("full-grid size exceeds platform limits"))?;
    for linear in 0..total {
        let subs = unravel_column_major(linear, shape);
        for (dim, tensor) in tensors.iter().enumerate() {
            let actual = tensor_utils::tensor_value_f64(tensor, linear);
            let expected = vectors[dim][subs[dim]];
            if actual != expected {
                return Err(invalid(
                    "full-grid arrays must be rectilinear ndgrid-style arrays",
                ));
            }
        }
    }
    Ok(())
}

fn validate_grid_value_shape(
    grid_vectors: &[Vec<f64>],
    values: &Tensor,
    method: InterpMethod,
    extrapolation: ExtrapolationMethod,
) -> BuiltinResult<()> {
    if grid_vectors.is_empty() {
        return Ok(());
    }
    validate_method_support(grid_vectors, method, extrapolation)?;
    if grid_vectors.len() > values.shape.len().max(1) {
        return Err(invalid("number of grid vectors exceeds V rank"));
    }
    for (dim, grid) in grid_vectors.iter().enumerate() {
        if grid.len() < 2 {
            return Err(invalid("each grid vector must contain at least two points"));
        }
        if grid.iter().any(|value| !value.is_finite()) {
            return Err(invalid("grid vectors must contain finite values"));
        }
        if !grid.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(invalid("grid vectors must be strictly increasing"));
        }
        let expected = values.shape.get(dim).copied().unwrap_or(1);
        if grid.len() != expected {
            return Err(invalid(format!(
                "grid vector {} length {} does not match V dimension {} length {}",
                dim + 1,
                grid.len(),
                dim + 1,
                expected
            )));
        }
    }
    Ok(())
}

fn validate_method_support(
    grid_vectors: &[Vec<f64>],
    method: InterpMethod,
    extrapolation: ExtrapolationMethod,
) -> BuiltinResult<()> {
    let rank = grid_vectors.len();
    if is_step_method(method) && rank != 1 {
        return Err(unsupported(
            "next and previous griddedInterpolant methods are currently supported for 1-D grids",
        ));
    }
    if is_step_extrapolation(extrapolation) && rank != 1 {
        return Err(unsupported(
            "next and previous extrapolation methods are currently supported for 1-D grids",
        ));
    }
    if is_piecewise_method(method) && rank != 1 {
        return Err(unsupported(
            "pchip and spline griddedInterpolant methods are currently supported for 1-D grids",
        ));
    }
    if is_piecewise_extrapolation(extrapolation) && rank != 1 {
        return Err(unsupported(
            "pchip and spline extrapolation methods are currently supported for 1-D grids",
        ));
    }
    match extrapolation {
        ExtrapolationMethod::Pchip if method != InterpMethod::Pchip => {
            return Err(unsupported(
                "pchip extrapolation currently requires pchip interpolation",
            ))
        }
        ExtrapolationMethod::Spline if method != InterpMethod::Spline => {
            return Err(unsupported(
                "spline extrapolation currently requires spline interpolation",
            ))
        }
        _ => {}
    }
    if (method == InterpMethod::Pchip || extrapolation == ExtrapolationMethod::Pchip)
        && grid_vectors[0].len() < 4
    {
        return Err(invalid("pchip requires at least four grid points"));
    }
    Ok(())
}

fn spec_to_object(spec: InterpolantSpec) -> BuiltinResult<ObjectInstance> {
    let mut object = ObjectInstance::new(CLASS_NAME.to_string());
    let grid_cells = spec
        .grid_vectors
        .iter()
        .map(|grid| {
            Tensor::new(grid.clone(), vec![grid.len(), 1])
                .map(Value::Tensor)
                .map_err(internal)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    object.properties.insert(
        GRID_VECTORS.to_string(),
        Value::Cell(
            CellArray::new_with_shape(grid_cells, vec![1, spec.grid_vectors.len()])
                .map_err(internal)?,
        ),
    );
    object
        .properties
        .insert(VALUES.to_string(), Value::Tensor(spec.values));
    object.properties.insert(
        METHOD.to_string(),
        Value::String(interp_method_name(spec.method).to_string()),
    );
    object.properties.insert(
        EXTRAPOLATION_METHOD.to_string(),
        Value::String(extrap_method_name(spec.extrapolation).to_string()),
    );
    Ok(object)
}

fn object_to_spec(value: &Value) -> BuiltinResult<InterpolantSpec> {
    let Value::Object(object) = value else {
        return Err(invalid("receiver must be a griddedInterpolant object"));
    };
    if !object.is_class(CLASS_NAME) {
        return Err(invalid("receiver must be a griddedInterpolant object"));
    }
    let grid_vectors = match object.properties.get(GRID_VECTORS) {
        Some(Value::Cell(cell)) => {
            ensure_integer_extension(
                &Value::Cell(cell.clone()),
                &GRIDDED_INTEGER_GRID_EXTENSION,
                "sample grid",
            )?;
            cell.data
                .iter()
                .cloned()
                .map(|value| numeric_vector(value, GRID_VECTORS))
                .collect::<BuiltinResult<Vec<_>>>()?
        }
        _ => return Err(internal("object is missing GridVectors")),
    };
    let values = match object.properties.get(VALUES) {
        Some(Value::Tensor(tensor)) => {
            let value = Value::Tensor(tensor.clone());
            ensure_integer_extension(&value, &GRIDDED_INTEGER_VALUES_EXTENSION, "sample values")?;
            numeric_tensor(value, VALUES)?
        }
        _ => return Err(internal("object is missing Values")),
    };
    let method = match object.properties.get(METHOD).and_then(text_from_value) {
        Some(name) => parse_interp_method(&name)?,
        None => return Err(internal("object is missing Method")),
    };
    let extrapolation = match object
        .properties
        .get(EXTRAPOLATION_METHOD)
        .and_then(text_from_value)
    {
        Some(name) => parse_extrap_method(&name)?,
        None => return Err(internal("object is missing ExtrapolationMethod")),
    };
    validate_grid_value_shape(&grid_vectors, &values, method, extrapolation)?;
    Ok(InterpolantSpec {
        grid_vectors,
        values,
        method,
        extrapolation,
    })
}

fn gridded_member(obj: Value, payload: Value) -> BuiltinResult<Value> {
    let field = text_from_value(&payload).ok_or_else(|| invalid("property name must be text"))?;
    let Value::Object(object) = obj else {
        return Err(invalid("receiver must be a griddedInterpolant object"));
    };
    if !object.is_class(CLASS_NAME) {
        return Err(invalid("receiver must be a griddedInterpolant object"));
    }
    object
        .properties
        .get(&field)
        .cloned()
        .ok_or_else(|| invalid(format!("unknown griddedInterpolant property '{field}'")))
}

fn payload_to_args(payload: Value) -> BuiltinResult<Vec<Value>> {
    match payload {
        Value::Cell(cell) => Ok(cell.data),
        other => Err(invalid(format!(
            "indexing payload must be a cell array, got {other:?}"
        ))),
    }
}

fn evaluate_interpolant(spec: &InterpolantSpec, args: Vec<Value>) -> BuiltinResult<Value> {
    if spec.grid_vectors.is_empty() {
        return Err(invalid("cannot evaluate an empty griddedInterpolant"));
    }
    validate_grid_value_shape(
        &spec.grid_vectors,
        &spec.values,
        spec.method,
        spec.extrapolation,
    )?;
    ensure_query_integer_extensions(&args)?;
    let plan = parse_query_plan(&spec.grid_vectors, args)?;
    let grid_rank = spec.grid_vectors.len();
    let grid_size = spec
        .grid_vectors
        .iter()
        .map(Vec::len)
        .try_fold(1usize, |acc, len| acc.checked_mul(len))
        .ok_or_else(|| invalid("grid size exceeds platform limits"))?;
    let mut extra_shape = if spec.values.shape.len() > grid_rank {
        spec.values.shape[grid_rank..].to_vec()
    } else {
        Vec::new()
    };
    while extra_shape.last() == Some(&1) {
        extra_shape.pop();
    }
    let extra_count = extra_shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, len| acc.checked_mul(len))
        .ok_or_else(|| invalid("value size exceeds platform limits"))?;
    let mut out_shape = plan.shape.clone();
    out_shape.extend(extra_shape);
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let values = tensor_utils::tensor_values_f64_cow(&spec.values);
    if is_piecewise_method(spec.method) {
        return evaluate_piecewise_interpolant(
            spec,
            values.as_ref(),
            &plan,
            grid_size,
            extra_count,
            out_shape,
        );
    }
    let mut out = Vec::with_capacity(plan.len() * extra_count);
    for series in 0..extra_count {
        let series_offset = series * grid_size;
        plan.for_each_point(|point| {
            out.push(interpolate_point(
                spec,
                values.as_ref(),
                point,
                series_offset,
            ))
        });
    }
    finish_interpolant_output(out, out_shape, spec.values.numeric_dtype())
}

fn evaluate_piecewise_interpolant(
    spec: &InterpolantSpec,
    values: &[f64],
    plan: &QueryPlan,
    grid_size: usize,
    extra_count: usize,
    out_shape: Vec<usize>,
) -> BuiltinResult<Value> {
    let x = &spec.grid_vectors[0];
    let mut out = Vec::with_capacity(plan.len() * extra_count);
    for series in 0..extra_count {
        let series_offset = series * grid_size;
        let y = values[series_offset..series_offset + grid_size].to_vec();
        let series_data = NumericSeries {
            x: x.clone(),
            y,
            series: 1,
            trailing_shape: Vec::new(),
        };
        let pp =
            match spec.method {
                InterpMethod::Pchip => build_pchip_pp(&series_data, CLASS_NAME)
                    .map_err(|err| internal(err.message()))?,
                InterpMethod::Spline => build_spline_pp(&series_data, CLASS_NAME)
                    .map_err(|err| internal(err.message()))?,
                _ => unreachable!("piecewise evaluator only handles higher-order methods"),
            };
        plan.for_each_point(|point| {
            out.push(evaluate_piecewise_scalar(
                spec,
                values,
                &pp,
                point[0],
                series_offset,
            ));
        });
    }
    finish_interpolant_output(out, out_shape, spec.values.numeric_dtype())
}

fn finish_interpolant_output(
    out: Vec<f64>,
    out_shape: Vec<usize>,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    if out.len() == 1 && dtype == NumericDType::F64 {
        Ok(Value::Num(out[0]))
    } else {
        let tensor = match dtype {
            NumericDType::F64 => Tensor::new(out, out_shape),
            NumericDType::F32 => Tensor::from_f32(
                out.into_iter().map(|value| value as f32).collect(),
                out_shape,
            ),
            _ => Err("interpolant output class must be single or double".to_string()),
        }
        .map_err(internal)?;
        Ok(Value::Tensor(tensor))
    }
}

fn parse_query_plan(grid_vectors: &[Vec<f64>], args: Vec<Value>) -> BuiltinResult<QueryPlan> {
    let ndim = grid_vectors.len();
    if args.len() == 1 {
        if let Value::Cell(cell) = args[0].clone() {
            let query_vectors = grid_vectors_from_cell(cell)?;
            if query_vectors.len() != ndim {
                return Err(invalid(
                    "query grid vector count must match interpolant dimension",
                ));
            }
            return full_query_grid(query_vectors);
        }
        if ndim == 1 {
            let (values, shape) = numeric_query_values(args.into_iter().next().unwrap(), "Xq")?;
            return Ok(QueryPlan {
                coords: QueryCoords::Explicit(
                    values.into_iter().map(|value| vec![value]).collect(),
                ),
                shape,
            });
        }
        let tensor = numeric_tensor(args.into_iter().next().unwrap(), "Xq")?;
        if tensor.shape.len() != 2 {
            return Err(invalid(
                "matrix query must be a two-dimensional m-by-n array",
            ));
        }
        if tensor.cols() != ndim {
            return Err(invalid(
                "matrix query must have one column per grid dimension",
            ));
        }
        let mut points = Vec::with_capacity(tensor.rows());
        let values = tensor_utils::tensor_values_f64_cow(&tensor);
        for row in 0..tensor.rows() {
            let mut point = Vec::with_capacity(ndim);
            for col in 0..ndim {
                point.push(values[row + col * tensor.rows()]);
            }
            points.push(point);
        }
        return Ok(QueryPlan {
            coords: QueryCoords::Explicit(points),
            shape: vec![tensor.rows(), 1],
        });
    }
    if args.len() != ndim {
        return Err(invalid(format!(
            "expected {ndim} query arguments, got {}",
            args.len()
        )));
    }
    let mut values_by_dim = Vec::with_capacity(ndim);
    let mut query_shape = None;
    for (dim, arg) in args.into_iter().enumerate() {
        let (values, shape) = numeric_query_values(arg, &format!("Xq{}", dim + 1))?;
        if let Some(expected) = &query_shape {
            if expected != &shape {
                return Err(invalid("query arrays must have matching shapes"));
            }
        } else {
            query_shape = Some(shape);
        }
        values_by_dim.push(values);
    }
    let shape = query_shape.unwrap_or_else(|| vec![0, 0]);
    let count = values_by_dim.first().map_or(0, Vec::len);
    let mut points = Vec::with_capacity(count);
    for idx in 0..count {
        points.push((0..ndim).map(|dim| values_by_dim[dim][idx]).collect());
    }
    Ok(QueryPlan {
        coords: QueryCoords::Explicit(points),
        shape,
    })
}

fn numeric_query_values(value: Value, name: &str) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    match value {
        Value::Num(x) => Ok((vec![x], vec![1, 1])),
        Value::Int(value) => {
            ensure_exact_integer_scalar(&value, name)?;
            Ok((vec![value.to_f64()], vec![1, 1]))
        }
        Value::Tensor(tensor) => {
            ensure_exact_integer_tensor(&tensor, name)?;
            let shape = tensor.shape.clone();
            Ok((tensor_utils::tensor_into_values_f64(tensor), shape))
        }
        other => Err(invalid(format!("{name} must be numeric, got {other:?}"))),
    }
}

fn full_query_grid(query_vectors: Vec<Vec<f64>>) -> BuiltinResult<QueryPlan> {
    let shape = query_vectors.iter().map(Vec::len).collect::<Vec<_>>();
    let total = shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, len| acc.checked_mul(len))
        .ok_or_else(|| invalid("query grid size exceeds platform limits"))?;
    Ok(QueryPlan {
        coords: QueryCoords::FullGrid {
            vectors: query_vectors,
            total,
        },
        shape,
    })
}

fn interpolate_point(
    spec: &InterpolantSpec,
    values: &[f64],
    point: &[f64],
    series_offset: usize,
) -> f64 {
    if point.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    if point_outside_grid(spec, point) {
        return extrapolate_point(spec, values, point, series_offset);
    }
    match spec.method {
        InterpMethod::Linear => interpolate_linear(spec, values, point, series_offset),
        InterpMethod::Nearest => interpolate_nearest(spec, values, point, series_offset),
        InterpMethod::Next => interpolate_step(spec, values, point[0], series_offset, true),
        InterpMethod::Previous => interpolate_step(spec, values, point[0], series_offset, false),
        InterpMethod::Pchip | InterpMethod::Spline => {
            unreachable!("higher-order methods use evaluate_piecewise_interpolant")
        }
    }
}

fn point_outside_grid(spec: &InterpolantSpec, point: &[f64]) -> bool {
    point.iter().enumerate().any(|(dim, coord)| {
        *coord < spec.grid_vectors[dim][0]
            || *coord > spec.grid_vectors[dim][spec.grid_vectors[dim].len() - 1]
    })
}

fn extrapolate_point(
    spec: &InterpolantSpec,
    values: &[f64],
    point: &[f64],
    series_offset: usize,
) -> f64 {
    match spec.extrapolation {
        ExtrapolationMethod::None => f64::NAN,
        ExtrapolationMethod::Linear => interpolate_linear(spec, values, point, series_offset),
        ExtrapolationMethod::Nearest => interpolate_nearest(spec, values, point, series_offset),
        ExtrapolationMethod::Next => interpolate_step(spec, values, point[0], series_offset, true),
        ExtrapolationMethod::Previous => {
            interpolate_step(spec, values, point[0], series_offset, false)
        }
        ExtrapolationMethod::Pchip | ExtrapolationMethod::Spline => {
            unreachable!("piecewise extrapolation uses evaluate_piecewise_interpolant")
        }
    }
}

fn is_piecewise_method(method: InterpMethod) -> bool {
    matches!(method, InterpMethod::Pchip | InterpMethod::Spline)
}

fn is_step_method(method: InterpMethod) -> bool {
    matches!(method, InterpMethod::Next | InterpMethod::Previous)
}

fn is_step_extrapolation(method: ExtrapolationMethod) -> bool {
    matches!(
        method,
        ExtrapolationMethod::Next | ExtrapolationMethod::Previous
    )
}

fn is_piecewise_extrapolation(method: ExtrapolationMethod) -> bool {
    matches!(
        method,
        ExtrapolationMethod::Pchip | ExtrapolationMethod::Spline
    )
}

fn evaluate_piecewise_scalar(
    spec: &InterpolantSpec,
    values: &[f64],
    pp: &PiecewisePolynomial,
    coord: f64,
    series_offset: usize,
) -> f64 {
    if coord.is_nan() {
        return f64::NAN;
    }
    let grid = &spec.grid_vectors[0];
    let in_range = coord >= grid[0] && coord <= grid[grid.len() - 1];
    if !in_range {
        match spec.extrapolation {
            ExtrapolationMethod::None => return f64::NAN,
            ExtrapolationMethod::Nearest => {
                let idx = if coord < grid[0] { 0 } else { grid.len() - 1 };
                return values[series_offset + idx];
            }
            ExtrapolationMethod::Next => {
                return interpolate_step(spec, values, coord, series_offset, true);
            }
            ExtrapolationMethod::Previous => {
                return interpolate_step(spec, values, coord, series_offset, false);
            }
            ExtrapolationMethod::Linear => {
                return interpolate_linear(spec, values, &[coord], series_offset);
            }
            ExtrapolationMethod::Pchip | ExtrapolationMethod::Spline => {}
        }
    }
    eval_pp_scalar(pp, 0, coord, &Extrapolation::Extrapolate)
}

fn interpolate_linear(
    spec: &InterpolantSpec,
    values: &[f64],
    point: &[f64],
    series_offset: usize,
) -> f64 {
    let mut brackets = Vec::with_capacity(point.len());
    for (dim, &coord) in point.iter().enumerate() {
        let Some(bracket) = bracket_for_linear(&spec.grid_vectors[dim], coord, spec.extrapolation)
        else {
            return f64::NAN;
        };
        brackets.push(bracket);
    }
    if brackets.len() >= usize::BITS as usize {
        return f64::NAN;
    }
    let corners = 1usize << brackets.len();
    let mut acc = 0.0;
    for corner in 0..corners {
        let mut weight = 1.0;
        let mut subs = Vec::with_capacity(brackets.len());
        for (dim, bracket) in brackets.iter().enumerate() {
            if (corner >> dim) & 1 == 0 {
                weight *= 1.0 - bracket.t;
                subs.push(bracket.lo);
            } else {
                weight *= bracket.t;
                subs.push(bracket.hi);
            }
        }
        acc += weight * values[series_offset + column_major_offset(&subs, &spec.values.shape)];
    }
    acc
}

fn bracket_for_linear(
    grid: &[f64],
    coord: f64,
    extrapolation: ExtrapolationMethod,
) -> Option<Bracket> {
    if coord < grid[0] || coord > grid[grid.len() - 1] {
        match extrapolation {
            ExtrapolationMethod::None => return None,
            ExtrapolationMethod::Nearest => {
                let idx = if coord < grid[0] { 0 } else { grid.len() - 1 };
                return Some(Bracket {
                    lo: idx,
                    hi: idx,
                    t: 0.0,
                });
            }
            _ => {}
        }
    }
    let lo = if coord <= grid[0] {
        0
    } else if coord >= grid[grid.len() - 1] {
        grid.len() - 2
    } else {
        match grid.binary_search_by(|probe| probe.partial_cmp(&coord).unwrap()) {
            Ok(idx) => {
                return Some(Bracket {
                    lo: idx,
                    hi: idx,
                    t: 0.0,
                })
            }
            Err(idx) => idx - 1,
        }
    };
    let hi = lo + 1;
    let t = (coord - grid[lo]) / (grid[hi] - grid[lo]);
    Some(Bracket { lo, hi, t })
}

fn interpolate_nearest(
    spec: &InterpolantSpec,
    values: &[f64],
    point: &[f64],
    series_offset: usize,
) -> f64 {
    let mut subs = Vec::with_capacity(point.len());
    for (dim, &coord) in point.iter().enumerate() {
        let Some(idx) = nearest_index(&spec.grid_vectors[dim], coord, spec.extrapolation) else {
            return f64::NAN;
        };
        subs.push(idx);
    }
    values[series_offset + column_major_offset(&subs, &spec.values.shape)]
}

fn nearest_index(grid: &[f64], coord: f64, extrapolation: ExtrapolationMethod) -> Option<usize> {
    if coord < grid[0] {
        return (extrapolation != ExtrapolationMethod::None).then_some(0);
    }
    if coord > grid[grid.len() - 1] {
        return (extrapolation != ExtrapolationMethod::None).then_some(grid.len() - 1);
    }
    match grid.binary_search_by(|probe| probe.partial_cmp(&coord).unwrap()) {
        Ok(idx) => Some(idx),
        Err(idx) => {
            let lo = idx - 1;
            let hi = idx;
            if coord - grid[lo] < grid[hi] - coord {
                Some(lo)
            } else {
                Some(hi)
            }
        }
    }
}

fn interpolate_step(
    spec: &InterpolantSpec,
    values: &[f64],
    coord: f64,
    series_offset: usize,
    next: bool,
) -> f64 {
    let grid = &spec.grid_vectors[0];
    if coord < grid[0] || coord > grid[grid.len() - 1] {
        if spec.extrapolation == ExtrapolationMethod::None {
            return f64::NAN;
        }
        let idx = if coord < grid[0] { 0 } else { grid.len() - 1 };
        return values[series_offset + idx];
    }
    let idx = match grid.binary_search_by(|probe| probe.partial_cmp(&coord).unwrap()) {
        Ok(idx) => idx,
        Err(idx) if next => idx,
        Err(idx) => idx.saturating_sub(1),
    };
    values[series_offset + idx.min(grid.len() - 1)]
}

fn column_major_offset(subs: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (dim, &sub) in subs.iter().enumerate() {
        offset += sub * stride;
        stride *= shape.get(dim).copied().unwrap_or(1);
    }
    offset
}

fn unravel_column_major(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .map(|&dim| {
            let sub = if dim == 0 { 0 } else { linear % dim };
            if dim > 0 {
                linear /= dim;
            }
            sub
        })
        .collect()
}

fn interp_method_name(method: InterpMethod) -> &'static str {
    match method {
        InterpMethod::Linear => "linear",
        InterpMethod::Nearest => "nearest",
        InterpMethod::Next => "next",
        InterpMethod::Previous => "previous",
        InterpMethod::Pchip => "pchip",
        InterpMethod::Spline => "spline",
    }
}

fn extrap_method_name(method: ExtrapolationMethod) -> &'static str {
    match method {
        ExtrapolationMethod::Linear => "linear",
        ExtrapolationMethod::Nearest => "nearest",
        ExtrapolationMethod::Next => "next",
        ExtrapolationMethod::Previous => "previous",
        ExtrapolationMethod::Pchip => "pchip",
        ExtrapolationMethod::Spline => "spline",
        ExtrapolationMethod::None => "none",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, NumericStorage};

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).unwrap())
    }

    fn col(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
    }

    fn int_col(storage: IntegerStorage) -> Value {
        let len = storage.len();
        let tensor = Tensor::new_integer(storage, vec![len, 1]).unwrap();
        Value::Tensor(tensor)
    }

    fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).unwrap();
        Value::Tensor(tensor)
    }

    fn call(args: Vec<Value>) -> BuiltinResult<Value> {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(gridded_interpolant_builtin(args))
    }

    fn subsref(obj: Value, args: Vec<Value>) -> BuiltinResult<Value> {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let len = args.len();
        let payload = Value::Cell(CellArray::new_with_shape(args, vec![1, len]).unwrap());
        block_on(gridded_interpolant_subsref(
            obj,
            OBJECT_INDEX_PAREN.to_string(),
            payload,
        ))
    }

    #[test]
    fn gridded_integer_extensions_and_capabilities_are_declared() {
        assert_eq!(GRIDDED_INTERPOLANT_EXTENSIONS.len(), 3);
        assert_eq!(GRIDDED_INTERPOLANT_INTEGER_CAPABILITIES.len(), 3);
    }

    #[test]
    fn gridded_strict_mode_rejects_integer_values() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
        let values = int_col(IntegerStorage::I16(vec![1, 4, 9]));
        let error = block_on(gridded_interpolant_builtin(vec![values]))
            .expect_err("integer values are an extension");
        assert_eq!(
            error.identifier(),
            GRIDDED_INTEGER_VALUES_EXTENSION.error_identifier
        );
    }

    #[test]
    fn gridded_rejects_inexact_integer_grid_values_and_queries() {
        let inexact = 9_007_199_254_740_993_u64;
        let error = call(vec![
            int_col(IntegerStorage::U64(vec![inexact, inexact + 2])),
            col(&[1.0, 2.0]),
        ])
        .expect_err("inexact grid must reject");
        assert!(error.message().contains("exactly representable as double"));

        let error = call(vec![int_col(IntegerStorage::U64(vec![
            inexact,
            inexact + 2,
        ]))])
        .expect_err("inexact values must reject");
        assert!(error.message().contains("exactly representable as double"));

        let obj = call(vec![col(&[0.0, 1.0]), col(&[0.0, 1.0])]).expect("interpolant");
        let error = subsref(obj, vec![int_col(IntegerStorage::U64(vec![inexact]))])
            .expect_err("inexact query must reject");
        assert!(error.message().contains("exactly representable as double"));
    }

    #[test]
    fn gridded_integer_query_syntaxes_are_consistent() {
        let one_d = call(vec![col(&[0.0, 1.0, 2.0]), col(&[0.0, 10.0, 20.0])])
            .expect("one-dimensional interpolant");
        match subsref(one_d, vec![int_col(IntegerStorage::I16(vec![0, 2]))])
            .expect("integer vector query")
        {
            Value::Tensor(output) => assert_eq!(output.materialize_f64(), vec![0.0, 20.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let grid = CellArray::new_with_shape(vec![col(&[0.0, 1.0]), col(&[0.0, 2.0])], vec![1, 2])
            .expect("grid");
        let values = Tensor::new(vec![0.0, 10.0, 20.0, 30.0], vec![2, 2]).expect("values");
        let two_d = call(vec![Value::Cell(grid), Value::Tensor(values)])
            .expect("two-dimensional interpolant");
        match subsref(
            two_d,
            vec![
                int_col(IntegerStorage::I16(vec![0, 1])),
                int_col(IntegerStorage::U16(vec![0, 2])),
            ],
        )
        .expect("per-dimension integer query")
        {
            Value::Tensor(output) => assert_eq!(output.materialize_f64(), vec![0.0, 30.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn constructor_returns_object_with_properties() {
        let value = call(vec![row(&[10.0, 20.0, 40.0])]).unwrap();
        let Value::Object(obj) = value else {
            panic!("expected object");
        };
        assert_eq!(obj.class_name, CLASS_NAME);
        assert_eq!(
            obj.properties.get(METHOD),
            Some(&Value::String("linear".to_string()))
        );
        assert!(matches!(
            obj.properties.get(GRID_VECTORS),
            Some(Value::Cell(_))
        ));
    }

    #[test]
    fn one_dimensional_linear_interpolation_and_none_extrapolation() {
        let obj = call(vec![
            col(&[0.0, 1.0, 2.0]),
            col(&[0.0, 10.0, 40.0]),
            Value::String("linear".to_string()),
            Value::String("none".to_string()),
        ])
        .unwrap();
        match subsref(obj, vec![row(&[-1.0, 0.5, 1.5, 3.0])]).unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 4]);
                assert!(tensor.materialize_f64()[0].is_nan());
                assert_eq!(tensor.materialize_f64()[1], 5.0);
                assert_eq!(tensor.materialize_f64()[2], 25.0);
                assert!(tensor.materialize_f64()[3].is_nan());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn default_grid_and_nearest_method_work() {
        let obj = call(vec![
            row(&[10.0, 20.0, 40.0]),
            Value::String("nearest".to_string()),
        ])
        .unwrap();
        match subsref(obj, vec![row(&[1.2, 2.6])]).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![10.0, 40.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn single_values_reshape_and_interpolate_in_native_class() {
        let values = Tensor::from_f32(vec![0.0, 10.0, 40.0], vec![1, 3]).unwrap();
        let obj = call(vec![Value::Tensor(values)]).unwrap();
        match subsref(obj, vec![row(&[1.5, 2.5])]).unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
                assert_eq!(
                    tensor.into_numeric_storage().unwrap(),
                    NumericStorage::F32(vec![5.0, 25.0])
                );
            }
            other => panic!("expected single tensor, got {other:?}"),
        }
    }

    #[test]
    fn two_dimensional_grid_vectors_interpolate() {
        let grid = CellArray::new_with_shape(vec![col(&[0.0, 1.0]), col(&[0.0, 2.0])], vec![1, 2])
            .unwrap();
        let values = Tensor::new(vec![0.0, 10.0, 20.0, 30.0], vec![2, 2]).unwrap();
        let obj = call(vec![Value::Cell(grid), Value::Tensor(values)]).unwrap();
        match subsref(obj, vec![Value::Num(0.5), Value::Num(1.0)]).unwrap() {
            Value::Num(value) => assert_eq!(value, 15.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn grid_vectors_values_and_queries_read_typed_integer_storage_exactly() {
        let grid = CellArray::new_with_shape(
            vec![
                int_col(IntegerStorage::I16(vec![0, 1])),
                int_col(IntegerStorage::U16(vec![0, 2])),
            ],
            vec![1, 2],
        )
        .unwrap();
        let values = int_tensor(IntegerStorage::I16(vec![0, 10, 20, 30]), vec![2, 2]);
        let obj = call(vec![Value::Cell(grid), values]).unwrap();
        let query = int_tensor(IntegerStorage::I16(vec![0, 1, 0, 2]), vec![2, 2]);
        match subsref(obj, vec![query]).unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.materialize_f64(), vec![0.0, 30.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let x_grid = int_tensor(IntegerStorage::I16(vec![0, 1, 0, 1]), vec![2, 2]);
        let y_grid = int_tensor(IntegerStorage::U16(vec![0, 0, 2, 2]), vec![2, 2]);
        let values = int_tensor(IntegerStorage::I16(vec![0, 10, 20, 30]), vec![2, 2]);
        let obj = call(vec![x_grid, y_grid, values]).unwrap();
        match subsref(obj, vec![Value::Num(1.0), Value::Num(2.0)]).unwrap() {
            Value::Num(value) => assert_eq!(value, 30.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn one_dimensional_spline_uses_piecewise_polynomial_path() {
        let obj = call(vec![
            col(&[1.0, 2.0, 3.0]),
            int_col(IntegerStorage::I16(vec![1, 4, 9])),
            Value::String("spline".to_string()),
        ])
        .unwrap();
        match subsref(obj, vec![Value::Num(1.5)]).unwrap() {
            Value::Num(value) => assert!((value - 2.25).abs() < 1.0e-10),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn one_dimensional_pchip_preserves_samples_and_nearest_extrapolates() {
        let obj = call(vec![
            col(&[1.0, 2.0, 3.0, 4.0]),
            int_col(IntegerStorage::I16(vec![0, 10, 15, 18])),
            Value::String("pchip".to_string()),
            Value::String("nearest".to_string()),
        ])
        .unwrap();
        match subsref(obj, vec![row(&[0.0, 3.0, 5.0])]).unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.materialize_f64()[0], 0.0);
                assert_eq!(tensor.materialize_f64()[1], 15.0);
                assert_eq!(tensor.materialize_f64()[2], 18.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn extrapolation_method_is_independent_from_interpolation_method() {
        let obj = call(vec![
            col(&[0.0, 1.0, 2.0]),
            col(&[0.0, 10.0, 40.0]),
            Value::String("nearest".to_string()),
            Value::String("linear".to_string()),
        ])
        .unwrap();
        match subsref(obj, vec![row(&[-1.0, 0.4, 3.0])]).unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.materialize_f64()[0], -10.0);
                assert_eq!(tensor.materialize_f64()[1], 0.0);
                assert_eq!(tensor.materialize_f64()[2], 70.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_cubic_method_is_explicit() {
        let err = call(vec![
            row(&[10.0, 20.0, 40.0]),
            Value::String("cubic".to_string()),
        ])
        .expect_err("cubic is not implemented yet");
        assert_eq!(
            err.identifier(),
            Some("RunMat:griddedInterpolant:UnsupportedMethod")
        );
    }

    #[test]
    fn higher_order_methods_reject_nd_grids_explicitly() {
        let grid = CellArray::new_with_shape(vec![col(&[0.0, 1.0]), col(&[0.0, 2.0])], vec![1, 2])
            .unwrap();
        let values = Tensor::new(vec![0.0, 10.0, 20.0, 30.0], vec![2, 2]).unwrap();
        let err = call(vec![
            Value::Cell(grid),
            Value::Tensor(values),
            Value::String("spline".to_string()),
        ])
        .expect_err("N-D spline is not implemented");
        assert_eq!(
            err.identifier(),
            Some("RunMat:griddedInterpolant:UnsupportedMethod")
        );
    }

    #[test]
    fn pchip_requires_four_grid_points() {
        let err = call(vec![
            col(&[1.0, 2.0, 3.0]),
            col(&[1.0, 4.0, 9.0]),
            Value::String("pchip".to_string()),
        ])
        .expect_err("pchip requires four points");
        assert_eq!(
            err.identifier(),
            Some("RunMat:griddedInterpolant:InvalidInput")
        );
    }

    #[test]
    fn full_grid_constructor_rejects_non_rectilinear_arrays() {
        let x_grid = Tensor::new(vec![0.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let y_grid = Tensor::new(vec![0.0, 0.0, 2.0, 2.0], vec![2, 2]).unwrap();
        let values = Tensor::new(vec![0.0, 10.0, 20.0, 30.0], vec![2, 2]).unwrap();
        let err = call(vec![
            Value::Tensor(x_grid),
            Value::Tensor(y_grid),
            Value::Tensor(values),
        ])
        .expect_err("malformed full grid should be rejected");
        assert_eq!(
            err.identifier(),
            Some("RunMat:griddedInterpolant:InvalidInput")
        );
    }

    #[test]
    fn rowwise_matrix_query_requires_two_dimensions() {
        let grid = CellArray::new_with_shape(vec![col(&[0.0, 1.0]), col(&[0.0, 2.0])], vec![1, 2])
            .unwrap();
        let values = Tensor::new(vec![0.0, 10.0, 20.0, 30.0], vec![2, 2]).unwrap();
        let obj = call(vec![Value::Cell(grid), Value::Tensor(values)]).unwrap();
        let query = Tensor::new(vec![0.0, 0.0, 1.0, 2.0], vec![2, 2, 1]).unwrap();
        let err = subsref(obj, vec![Value::Tensor(query)])
            .expect_err("row-wise matrix query must be 2-D");
        assert_eq!(
            err.identifier(),
            Some("RunMat:griddedInterpolant:InvalidInput")
        );
    }

    #[test]
    fn mutated_object_properties_are_revalidated_before_evaluation() {
        let value = call(vec![row(&[1.0, 2.0, 3.0])]).unwrap();
        let Value::Object(mut obj) = value else {
            panic!("expected object");
        };
        obj.properties.insert(
            VALUES.to_string(),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        );
        let err = subsref(Value::Object(obj), vec![Value::Num(2.0)])
            .expect_err("invalid object state should be rejected");
        assert_eq!(
            err.identifier(),
            Some("RunMat:griddedInterpolant:InvalidInput")
        );
    }

    #[test]
    fn member_subsref_returns_public_properties() {
        let obj = call(vec![row(&[1.0, 2.0, 3.0])]).unwrap();
        let value = block_on(gridded_interpolant_subsref(
            obj,
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(VALUES.to_string()),
        ))
        .unwrap();
        assert!(matches!(value, Value::Tensor(_)));
    }

    #[test]
    fn unsupported_makima_method_is_explicit() {
        let err = call(vec![
            row(&[1.0, 2.0, 3.0]),
            Value::String("makima".to_string()),
        ])
        .expect_err("makima is not implemented yet");
        assert_eq!(
            err.identifier(),
            Some("RunMat:griddedInterpolant:UnsupportedMethod")
        );
    }
}
