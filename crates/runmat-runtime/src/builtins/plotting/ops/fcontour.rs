//! MATLAB-compatible `fcontour` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ContourFillPlot, ContourPlot};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::math::optim::common::call_function;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::common::SurfaceDataInput;
use super::contour::{
    apply_contour_options_with_integer_line_color_extension, build_contour_fill_plot,
    build_contour_plot, parse_level_spec, ContourArgs, ContourLevelSpec, ContourLineColor,
};
use super::fsurf::function_surface_ref;
use super::op_common::{apply_axes_target, split_leading_axes_handle, AxesTarget};
use super::plotting_error;
use super::state::{
    current_figure_handle, register_function_contour_handle, render_active_plot, PlotRenderOptions,
};
use super::style::value_as_string;

const BUILTIN_NAME: &str = "fcontour";
const DEFAULT_DOMAIN: Domain = Domain {
    x_min: -5.0,
    x_max: 5.0,
    y_min: -5.0,
    y_max: 5.0,
};
const DEFAULT_MESH_DENSITY: usize = 71;
const MAX_MESH_DENSITY: usize = 400;

pub(crate) const FCONTOUR_INTEGER_LINE_COLOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fcontour-integer-line-color",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fcontour with a typed-integer RGB LineColor is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FcontourIntegerLineColorExtension"),
    };
pub(crate) const FCONTOUR_POSITIONAL_LEVEL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fcontour-positional-level-spec",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fcontour with a positional level count or value vector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FcontourPositionalLevelSpecExtension"),
    };
pub(crate) const FCONTOUR_RESIDENT_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fcontour-resident-numeric-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fcontour with resident numeric arguments is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FcontourResidentNumericInputExtension"),
    };
pub const FCONTOUR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FCONTOUR_INTEGER_LINE_COLOR_EXTENSION,
    FCONTOUR_POSITIONAL_LEVEL_EXTENSION,
    FCONTOUR_RESIDENT_NUMERIC_EXTENSION,
];

const FCONTOUR_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered function contour.",
}];

const FCONTOUR_INPUTS_F: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "f",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function of two variables, z = f(x,y).",
}];

const FCONTOUR_INPUTS_F_DOMAIN: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function of two variables, z = f(x,y).",
    },
    BuiltinParamDescriptor {
        name: "xyinterval",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[-5 5 -5 5]"),
        description: "Two- or four-element domain vector.",
    },
];

const FCONTOUR_INPUTS_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function-handle and optional domain arguments.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "MeshDensity and contour name/value arguments.",
    },
];

const FCONTOUR_INPUTS_AX_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function-handle and optional domain arguments.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "MeshDensity and contour name/value arguments.",
    },
];

const FCONTOUR_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = fcontour(f)",
        inputs: &FCONTOUR_INPUTS_F,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fcontour(f, xyinterval)",
        inputs: &FCONTOUR_INPUTS_F_DOMAIN,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fcontour(___, Name, Value, ...)",
        inputs: &FCONTOUR_INPUTS_PROPS,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fcontour(ax, ___)",
        inputs: &FCONTOUR_INPUTS_AX_PROPS,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
];

pub const FCONTOUR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCONTOUR.INVALID_ARGUMENT",
    identifier: Some("RunMat:fcontour:InvalidArgument"),
    when: "Function handle, domain, mesh density, axes target, levels, or contour properties are invalid.",
    message: "fcontour: invalid argument",
};

pub const FCONTOUR_ERROR_EVALUATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCONTOUR.EVALUATION",
    identifier: Some("RunMat:fcontour:EvaluationFailed"),
    when: "A sampled function handle fails or does not return a scalar numeric value.",
    message: "fcontour: function evaluation failed",
};

pub const FCONTOUR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCONTOUR.INTERNAL",
    identifier: Some("RunMat:fcontour:Internal"),
    when: "Contour construction or rendering fails unexpectedly.",
    message: "fcontour: internal operation failed",
};

const FCONTOUR_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FCONTOUR_ERROR_INVALID_ARGUMENT,
    FCONTOUR_ERROR_EVALUATION,
    FCONTOUR_ERROR_INTERNAL,
];

pub const FCONTOUR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FCONTOUR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FCONTOUR_ERRORS,
};

const fn integer_input(
    name: &'static str,
    availability: BuiltinIntegerInputAvailability,
    notes: &'static str,
) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes,
    }
}

const INTEGER_DOMAIN_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "xyinterval",
    BuiltinIntegerInputAvailability::Documented,
    "The documented two- or four-element plotting interval accepts real numeric vectors; authoritative integer bounds are validated before the host graphics conversion boundary.",
)];
const INTEGER_MESH_DENSITY_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "MeshDensity",
    BuiltinIntegerInputAvailability::Documented,
    "MeshDensity is an exact scalar count with a documented effective minimum of three and a RunMat resource ceiling of 400.",
)];
const INTEGER_LEVEL_LIST_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "LevelList",
    BuiltinIntegerInputAvailability::Documented,
    "LevelList explicitly documents every built-in integer class; a scalar is one level and vector order is not constrained.",
)];
const INTEGER_LEVEL_STEP_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "LevelStep",
    BuiltinIntegerInputAvailability::Documented,
    "LevelStep explicitly documents every built-in integer class; zero selects automatic levels and positive values select spacing.",
)];
const INTEGER_LINE_WIDTH_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "LineWidth",
    BuiltinIntegerInputAvailability::Documented,
    "A positive integer line width crosses one checked f32 graphics-property boundary.",
)];
const INTEGER_FILL_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "Fill",
    BuiltinIntegerInputAvailability::Documented,
    "The documented numeric on/off form accepts only the exact scalar values zero and one.",
)];
const INTEGER_LINE_COLOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "LineColor",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode accepts a three-element typed-integer RGB vector containing only zero or one.",
    }];
const INTEGER_POSITIONAL_LEVEL_INPUT: [BuiltinIntegerInputCapability; 1] = [integer_input(
    "levels",
    BuiltinIntegerInputAvailability::RunMatOnly,
    "RunMat mode accepts the contour-style positional level count or value-vector shorthand after the function handle.",
)];

pub const FCONTOUR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 8] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(f, integer_xyinterval)",
        inputs: &INTEGER_DOMAIN_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Exact bounds cross one deliberate binary64 graphics-domain boundary; the result is an opaque graphics handle.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'MeshDensity', integer_density)",
        inputs: &INTEGER_MESH_DENSITY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The exact count controls bounded host callback sampling and does not become integer output.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'LevelList', integer_levels)",
        inputs: &INTEGER_LEVEL_LIST_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Authoritative level values cross one deliberate contour-geometry conversion boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'LevelStep', integer_step)",
        inputs: &INTEGER_LEVEL_STEP_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact scalar step crosses one checked f32 contour-geometry boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'LineWidth', integer_width)",
        inputs: &INTEGER_LINE_WIDTH_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact positive scalar crosses one checked f32 renderer-property boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'Fill', integer_on_off)",
        inputs: &INTEGER_FILL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Zero disables and one enables filled contour bands; the result remains graphics state.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(..., 'LineColor', integer_rgb)",
        inputs: &INTEGER_LINE_COLOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "This independently gated RunMat extension validates exact RGB components before graphics conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = fcontour(f, integer_level_count_or_values)",
        inputs: &INTEGER_POSITIONAL_LEVEL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "This independently gated RunMat shorthand reuses contour positional-level semantics.",
    },
];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::fcontour")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fcontour",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "fcontour samples arbitrary MATLAB function handles on the host, then renders through the existing contour plot pipeline.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::fcontour")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fcontour",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "fcontour performs callback sampling and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "fcontour",
    category = "plotting",
    summary = "Plot contour lines of a function over a 2-D domain.",
    keywords = "fcontour,function contour,contour,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::fcontour::FCONTOUR_DESCRIPTOR),
    extensions(crate::builtins::plotting::fcontour::FCONTOUR_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::fcontour::FCONTOUR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::fcontour"
)]
pub async fn fcontour_builtin(mut args: Vec<Value>) -> BuiltinResult<f64> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(fcontour_invalid("too many output arguments"));
    }
    preflight_fcontour_extensions(&args)?;
    for value in &mut args {
        if matches!(value, Value::GpuTensor(_)) {
            *value = crate::dispatcher::gather_if_needed_async(value)
                .await
                .map_err(map_fcontour_invalid)?;
        }
    }
    let parsed = parse_fcontour_args(args).map_err(map_fcontour_invalid)?;
    let sampled = sample_contour(&parsed).await?;
    render_fcontour(sampled, &parsed).map_err(map_fcontour_internal)
}

#[derive(Clone, Copy)]
struct Domain {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
}

struct ParsedFcontour {
    target_axes: AxesTarget,
    function: Value,
    domain: Domain,
    mesh_density: usize,
    level_spec: ContourLevelSpec,
    contour_options: Vec<Value>,
    display_name: Option<String>,
    fill: bool,
}

fn preflight_fcontour_extensions(args: &[Value]) -> BuiltinResult<()> {
    if args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FCONTOUR_RESIDENT_NUMERIC_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    for pair in args.windows(2) {
        let Some(key) = value_as_string(&pair[0]) else {
            continue;
        };
        if matches!(
            key.trim().to_ascii_lowercase().as_str(),
            "linecolor" | "color"
        ) && is_typed_integer_value(&pair[1])
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FCONTOUR_INTEGER_LINE_COLOR_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

fn fcontour_error_with_detail(
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

fn fcontour_invalid(detail: impl AsRef<str>) -> RuntimeError {
    fcontour_error_with_detail(&FCONTOUR_ERROR_INVALID_ARGUMENT, detail)
}

fn map_fcontour_invalid(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    fcontour_error_with_detail(&FCONTOUR_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_fcontour_eval(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    fcontour_error_with_detail(&FCONTOUR_ERROR_EVALUATION, err.message)
}

fn map_fcontour_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    fcontour_error_with_detail(&FCONTOUR_ERROR_INTERNAL, err.message)
}

fn parse_fcontour_args(args: Vec<Value>) -> BuiltinResult<ParsedFcontour> {
    if args.is_empty() {
        return Err(fcontour_invalid("expected a function handle"));
    }

    let (target_axes, mut values) = split_leading_axes_handle(args, BUILTIN_NAME)?;
    if values.is_empty() {
        return Err(fcontour_invalid(
            "expected a function handle after axes handle",
        ));
    }
    if !is_function_handle(&values[0]) {
        return Err(fcontour_invalid("expected a function handle"));
    }
    let function = values.remove(0);

    let mut domain = DEFAULT_DOMAIN;
    if values.first().is_some_and(is_domain_value) {
        domain = parse_domain(&values.remove(0))?;
    }

    let mut level_spec = ContourLevelSpec::Auto;
    if values.first().is_some_and(is_level_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FCONTOUR_POSITIONAL_LEVEL_EXTENSION,
            BUILTIN_NAME,
        )?;
        level_spec =
            parse_level_spec(values.remove(0), BUILTIN_NAME).map_err(map_fcontour_invalid)?;
    }

    let options = split_fcontour_options(values)?;
    if let Some(option_levels) = options.level_spec {
        level_spec = option_levels;
    }
    Ok(ParsedFcontour {
        target_axes,
        function,
        domain,
        mesh_density: options.mesh_density,
        level_spec,
        contour_options: options.contour_options,
        display_name: options.display_name,
        fill: options.fill,
    })
}

fn is_function_handle(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    )
}

fn is_domain_value(value: &Value) -> bool {
    let Ok(values) = numeric_vector(value) else {
        return false;
    };
    matches!(values.len(), 2 | 4)
}

fn is_level_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
    )
}

fn parse_domain(value: &Value) -> BuiltinResult<Domain> {
    if let Value::Tensor(tensor) = value {
        ensure_numeric_vector_shape(tensor, "domain")?;
        validate_exact_integer_domain_order(tensor)?;
    }
    let values = numeric_vector(value)?;
    match values.as_slice() {
        [lo, hi] => domain_from_values(*lo, *hi, *lo, *hi),
        [x_min, x_max, y_min, y_max] => domain_from_values(*x_min, *x_max, *y_min, *y_max),
        _ => Err(fcontour_invalid(
            "domain must be a two-element or four-element numeric vector",
        )),
    }
}

fn validate_exact_integer_domain_order(tensor: &Tensor) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    let ordered = match storage.len() {
        2 => integer_less(
            &storage.value_at(0).expect("validated integer storage"),
            &storage.value_at(1).expect("validated integer storage"),
        ),
        4 => {
            integer_less(
                &storage.value_at(0).expect("validated integer storage"),
                &storage.value_at(1).expect("validated integer storage"),
            ) && integer_less(
                &storage.value_at(2).expect("validated integer storage"),
                &storage.value_at(3).expect("validated integer storage"),
            )
        }
        _ => return Ok(()),
    };
    if ordered {
        Ok(())
    } else {
        Err(fcontour_invalid(
            "domain lower bounds must be less than upper bounds",
        ))
    }
}

fn integer_less(left: &IntValue, right: &IntValue) -> bool {
    match (left, right) {
        (IntValue::I8(left), IntValue::I8(right)) => left < right,
        (IntValue::I16(left), IntValue::I16(right)) => left < right,
        (IntValue::I32(left), IntValue::I32(right)) => left < right,
        (IntValue::I64(left), IntValue::I64(right)) => left < right,
        (IntValue::U8(left), IntValue::U8(right)) => left < right,
        (IntValue::U16(left), IntValue::U16(right)) => left < right,
        (IntValue::U32(left), IntValue::U32(right)) => left < right,
        (IntValue::U64(left), IntValue::U64(right)) => left < right,
        _ => false,
    }
}

fn domain_from_values(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> BuiltinResult<Domain> {
    if ![x_min, x_max, y_min, y_max]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(fcontour_invalid("domain limits must be finite"));
    }
    if x_min >= x_max || y_min >= y_max {
        return Err(fcontour_invalid(
            "domain lower bounds must be less than upper bounds",
        ));
    }
    Ok(Domain {
        x_min,
        x_max,
        y_min,
        y_max,
    })
}

fn numeric_vector(value: &Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        Value::Tensor(tensor) => Ok(tensor_utils::tensor_values_f64(tensor)),
        other => Err(fcontour_invalid(format!(
            "expected numeric vector, got {other:?}"
        ))),
    }
}

struct FcontourOptions {
    mesh_density: usize,
    display_name: Option<String>,
    contour_options: Vec<Value>,
    level_spec: Option<ContourLevelSpec>,
    fill: bool,
}

fn split_fcontour_options(args: Vec<Value>) -> BuiltinResult<FcontourOptions> {
    let mut mesh_density = DEFAULT_MESH_DENSITY;
    let mut display_name = None;
    let mut contour_options = Vec::new();
    let mut level_spec = None;
    let mut fill = false;
    let mut idx = 0usize;
    if let Some(token) = args.first().and_then(value_as_string) {
        let trimmed = token.trim();
        if !is_fcontour_special_option(trimmed) && !is_contour_pair_option(trimmed) {
            contour_options.push(args[0].clone());
            idx = 1;
        }
    }
    while idx < args.len() {
        let Some(key) = value_as_string(&args[idx]) else {
            return Err(fcontour_invalid("name-value option names must be strings"));
        };
        if idx + 1 >= args.len() {
            return Err(fcontour_invalid("name-value arguments must come in pairs"));
        }
        let normalized = key.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "meshdensity" => {
                mesh_density = parse_mesh_density(&args[idx + 1])?;
            }
            "displayname" => {
                display_name = Some(
                    value_as_string(&args[idx + 1])
                        .ok_or_else(|| fcontour_invalid("DisplayName must be text"))?
                        .to_string(),
                );
            }
            "levellist" => {
                level_spec = Some(parse_fcontour_level_list(&args[idx + 1])?);
            }
            "levelstep" => {
                level_spec = Some(parse_fcontour_level_step(&args[idx + 1])?);
            }
            "fill" => {
                fill = parse_fcontour_fill(&args[idx + 1])?;
            }
            "levels" | "color" => {
                return Err(fcontour_invalid(format!(
                    "unsupported property `{}`",
                    key.trim()
                )));
            }
            _ => {
                contour_options.push(args[idx].clone());
                contour_options.push(args[idx + 1].clone());
            }
        }
        idx += 2;
    }
    Ok(FcontourOptions {
        mesh_density,
        display_name,
        contour_options,
        level_spec,
        fill,
    })
}

fn is_fcontour_special_option(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "meshdensity" | "displayname" | "fill"
    )
}

fn is_contour_pair_option(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "levellist"
            | "levels"
            | "levelstep"
            | "linecolor"
            | "color"
            | "linewidth"
            | "levellistmode"
    )
}

fn ensure_numeric_vector_shape(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
    let shape = &tensor.shape;
    if shape.len() != 2 || (shape[0] != 1 && shape[1] != 1) {
        return Err(fcontour_invalid(format!(
            "{role} must be a row or column vector"
        )));
    }
    Ok(())
}

fn parse_fcontour_level_list(value: &Value) -> BuiltinResult<ContourLevelSpec> {
    let values = match value {
        Value::Num(value) => vec![*value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Tensor(tensor) => {
            ensure_numeric_vector_shape(tensor, "LevelList")?;
            tensor_utils::tensor_values_f64(tensor)
        }
        other => {
            return Err(fcontour_invalid(format!(
                "LevelList must be a real numeric vector, got {other:?}"
            )))
        }
    };
    if values.is_empty() {
        return Err(fcontour_invalid("LevelList must be nonempty"));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(fcontour_invalid("LevelList values must be finite"));
    }
    Ok(ContourLevelSpec::Values(values))
}

fn parse_fcontour_level_step(value: &Value) -> BuiltinResult<ContourLevelSpec> {
    let value = numeric_scalar(value, "LevelStep")?;
    if !value.is_finite() || value < 0.0 {
        return Err(fcontour_invalid(
            "LevelStep must be a nonnegative finite scalar",
        ));
    }
    if value == 0.0 {
        return Ok(ContourLevelSpec::Auto);
    }
    let step = value as f32;
    if !step.is_finite() {
        return Err(fcontour_invalid(
            "LevelStep is too large for contour geometry",
        ));
    }
    Ok(ContourLevelSpec::Step(step))
}

fn parse_fcontour_fill(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Int(value) if value.is_zero() => Ok(false),
        Value::Int(value) if integer_is_one(value) => Ok(true),
        Value::Num(value) if *value == 0.0 => Ok(false),
        Value::Num(value) if *value == 1.0 => Ok(true),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(value) = tensor.integer_storage().and_then(|s| s.value_at(0)) {
                return parse_fcontour_fill(&Value::Int(value));
            }
            parse_fcontour_fill(&Value::Num(tensor_utils::tensor_value_f64(tensor, 0)))
        }
        _ => Err(fcontour_invalid(
            "Fill must be the scalar value zero or one",
        )),
    }
}

fn integer_is_one(value: &IntValue) -> bool {
    matches!(
        value,
        IntValue::I8(1)
            | IntValue::I16(1)
            | IntValue::I32(1)
            | IntValue::I64(1)
            | IntValue::U8(1)
            | IntValue::U16(1)
            | IntValue::U32(1)
            | IntValue::U64(1)
    )
}

fn numeric_scalar(value: &Value, role: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Ok(tensor_utils::tensor_value_f64(tensor, 0))
        }
        _ => Err(fcontour_invalid(format!("{role} must be a numeric scalar"))),
    }
}

fn parse_mesh_density(value: &Value) -> BuiltinResult<usize> {
    if let Some(count) = exact_integer_scalar(value) {
        return parse_mesh_density_integer(&count);
    }
    let values = numeric_vector(value)?;
    if values.len() != 1 {
        return Err(fcontour_invalid("MeshDensity must be a scalar"));
    }
    let raw = values[0];
    if !raw.is_finite() {
        return Err(fcontour_invalid("MeshDensity must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1.0e-9
        || rounded < 3.0
        || rounded > usize::MAX as f64
        || (usize::BITS == 64 && rounded == usize::MAX as f64)
    {
        return Err(fcontour_invalid(
            "MeshDensity must be an integer greater than or equal to 3",
        ));
    }
    let density = rounded as usize;
    if density > MAX_MESH_DENSITY {
        return Err(fcontour_invalid(format!(
            "MeshDensity must be at most {MAX_MESH_DENSITY}"
        )));
    }
    Ok(density)
}

fn exact_integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn parse_mesh_density_integer(value: &IntValue) -> BuiltinResult<usize> {
    let Some(density) = value.try_to_usize() else {
        return Err(fcontour_invalid(
            "MeshDensity must be an integer greater than or equal to 3",
        ));
    };
    if density < 3 {
        return Err(fcontour_invalid(
            "MeshDensity must be an integer greater than or equal to 3",
        ));
    }
    if density > MAX_MESH_DENSITY {
        return Err(fcontour_invalid(format!(
            "MeshDensity must be at most {MAX_MESH_DENSITY}"
        )));
    }
    Ok(density)
}

struct SampledFcontour {
    contour: ContourPlot,
    fill: Option<ContourFillPlot>,
}

async fn sample_contour(parsed: &ParsedFcontour) -> BuiltinResult<SampledFcontour> {
    let x_axis = linspace(
        parsed.domain.x_min,
        parsed.domain.x_max,
        parsed.mesh_density,
    );
    let y_axis = linspace(
        parsed.domain.y_min,
        parsed.domain.y_max,
        parsed.mesh_density,
    );
    let mut grid = vec![vec![0.0; x_axis.len()]; y_axis.len()];
    for (row, &y) in y_axis.iter().enumerate() {
        for (col, &x) in x_axis.iter().enumerate() {
            grid[row][col] = call_contour_function(&parsed.function, x, y).await?;
        }
    }

    let dummy = Tensor::new(vec![0.0], vec![1, 1])
        .map_err(|err| plotting_error(BUILTIN_NAME, format!("fcontour: {err}")))?;
    let mut contour_args = ContourArgs {
        name: BUILTIN_NAME,
        x_axis: x_axis.clone(),
        y_axis: y_axis.clone(),
        z_input: SurfaceDataInput::Host(dummy),
        level_spec: parsed.level_spec.clone(),
        line_color: ContourLineColor::Auto,
        line_width: 1.0,
    };
    apply_contour_options_with_integer_line_color_extension(
        &mut contour_args,
        &parsed.contour_options,
        &FCONTOUR_INTEGER_LINE_COLOR_EXTENSION,
    )
    .map_err(map_fcontour_invalid)?;

    let fill = if parsed.fill {
        Some(
            build_contour_fill_plot(
                BUILTIN_NAME,
                &x_axis,
                &y_axis,
                &grid,
                ColorMap::Parula,
                0.0,
                &contour_args.level_spec,
            )
            .map_err(map_fcontour_internal)?,
        )
    } else {
        None
    };

    if matches!(contour_args.line_color, ContourLineColor::None) {
        let contour = build_contour_plot(
            BUILTIN_NAME,
            &x_axis,
            &y_axis,
            &grid,
            ColorMap::Parula,
            0.0,
            &ContourLevelSpec::Values(vec![0.0]),
            &ContourLineColor::None,
        )
        .map_err(map_fcontour_internal)?;
        return Ok(SampledFcontour { contour, fill });
    }

    let mut plot = build_contour_plot(
        BUILTIN_NAME,
        &x_axis,
        &y_axis,
        &grid,
        ColorMap::Parula,
        0.0,
        &contour_args.level_spec,
        &contour_args.line_color,
    )
    .map_err(map_fcontour_internal)?
    .with_line_width(contour_args.line_width)
    .with_label("Function Contours");
    if let Some(display_name) = &parsed.display_name {
        plot.label = Some(display_name.clone());
    }
    Ok(SampledFcontour {
        contour: plot,
        fill,
    })
}

async fn call_contour_function(function: &Value, x: f64, y: f64) -> BuiltinResult<f64> {
    let value = call_function(function, vec![Value::Num(x), Value::Num(y)])
        .await
        .map_err(map_fcontour_eval)?;
    let value = crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(map_fcontour_eval)?;
    contour_value_to_scalar(value).map_err(map_fcontour_eval)
}

fn contour_value_to_scalar(value: Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Bool(value) => Ok(if value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(&tensor) => {
            Ok(tensor_utils::tensor_value_f64(&tensor, 0))
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(if array.data[0] != 0 { 1.0 } else { 0.0 })
        }
        other => Err(fcontour_error_with_detail(
            &FCONTOUR_ERROR_EVALUATION,
            format!("function output must be a scalar real numeric value, got {other:?}"),
        )),
    }
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (stop - start) / (count as f64 - 1.0);
    (0..count).map(|idx| start + step * idx as f64).collect()
}

fn render_fcontour(sampled: SampledFcontour, parsed: &ParsedFcontour) -> BuiltinResult<f64> {
    apply_axes_target(parsed.target_axes, BUILTIN_NAME)?;
    let mut contour = Some(sampled.contour);
    let mut fill = sampled.fill;
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = current_figure_handle();
    let target_axes_index = parsed.target_axes.map(|(_, axes)| axes);
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Function Contour",
            x_label: "X",
            y_label: "Y",
            axis_equal: true,
            ..Default::default()
        },
        move |figure, axes| {
            let axes = target_axes_index.unwrap_or(axes);
            if let Some(fill) = fill.take() {
                figure.add_contour_fill_plot_on_axes(fill, axes);
            }
            let plot_index = figure
                .add_contour_plot_on_axes(contour.take().expect("fcontour consumed once"), axes);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = register_function_contour_handle(
        figure_handle,
        axes,
        plot_index,
        parsed.mesh_density,
        (parsed.domain.x_min, parsed.domain.x_max),
        (parsed.domain.y_min, parsed.domain.y_max),
        function_surface_ref(&parsed.function),
        parsed.fill,
    );
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use futures::executor::block_on;
    use runmat_plot::plots::PlotElement;
    use std::sync::Arc;

    fn every_integer_one() -> Vec<IntValue> {
        vec![
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ]
    }

    fn with_test_function(
        f: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
    ) -> crate::user_functions::FunctionInvokerGuard {
        let f = Arc::new(f);
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, args, _requested_outputs| {
                let f = Arc::clone(&f);
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected first scalar argument, got {other:?}"),
                };
                let y = match &args[1] {
                    Value::Num(value) => *value,
                    other => panic!("expected second scalar argument, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(f(x, y))) })
            },
        )))
    }

    #[test]
    fn fcontour_numeric_vector_reads_typed_integer_storage_exactly() {
        let domain = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(vec![-4, 4]),
            vec![1, 2],
        )
        .expect("typed domain vector");

        assert_eq!(
            numeric_vector(&Value::Tensor(domain)).expect("numeric vector"),
            vec![-4.0, 4.0]
        );
    }

    #[test]
    fn fcontour_samples_function_and_returns_function_contour_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let _invoker = with_test_function(|x, y| x * x + y);

        let handle = block_on(fcontour_builtin(vec![
            Value::BoundFunctionHandle {
                name: "surface".into(),
                function: 1,
            },
            Value::Tensor(
                runmat_builtins::Tensor::new(vec![0.0, 1.0, 0.0, 2.0], vec![1, 4]).unwrap(),
            ),
            Value::String("MeshDensity".into()),
            Value::Num(5.0),
            Value::String("LevelList".into()),
            Value::Tensor(runmat_builtins::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::String("LineWidth".into()),
            Value::Num(2.0),
            Value::String("DisplayName".into()),
            Value::String("parabola".into()),
        ]))
        .unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("functioncontour".into()));
        let mesh_density = get_builtin(vec![
            Value::Num(handle),
            Value::String("MeshDensity".into()),
        ])
        .unwrap();
        assert_eq!(mesh_density, Value::Num(5.0));
        let display_name = get_builtin(vec![
            Value::Num(handle),
            Value::String("DisplayName".into()),
        ])
        .unwrap();
        assert_eq!(display_name, Value::String("parabola".into()));

        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Contour(contour) = fig.plots().next().unwrap() else {
            panic!("expected contour plot");
        };
        assert_eq!(contour.label.as_deref(), Some("parabola"));
        assert_eq!(contour.line_width, 2.0);
    }

    #[test]
    fn fcontour_supports_axes_target_and_linespec() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let _invoker = with_test_function(|x, y| x - y);

        let handle = block_on(fcontour_builtin(vec![
            Value::Num(ax),
            Value::BoundFunctionHandle {
                name: "surface".into(),
                function: 1,
            },
            Value::Tensor(
                runmat_builtins::Tensor::new(vec![-1.0, 1.0, -1.0, 1.0], vec![1, 4]).unwrap(),
            ),
            Value::String("r--".into()),
            Value::String("MeshDensity".into()),
            Value::Num(3.0),
        ]))
        .unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("functioncontour".into()));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
    }

    #[test]
    fn fcontour_fill_adds_filled_bands_and_reports_handle_property() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        let _invoker = with_test_function(|x, y| x + y);

        let handle = block_on(fcontour_builtin(vec![
            Value::BoundFunctionHandle {
                name: "surface".into(),
                function: 1,
            },
            Value::String("MeshDensity".into()),
            Value::Int(IntValue::U8(3)),
            Value::String("Fill".into()),
            Value::Int(IntValue::U16(1)),
        ]))
        .unwrap();

        let fill = get_builtin(vec![Value::Num(handle), Value::String("Fill".into())]).unwrap();
        assert_eq!(fill, Value::Bool(true));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(fig
            .plots()
            .any(|plot| matches!(plot, PlotElement::ContourFill(_))));
        assert!(fig
            .plots()
            .any(|plot| matches!(plot, PlotElement::Contour(_))));
    }

    #[test]
    fn fcontour_rejects_invalid_mesh_density() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        let _invoker = with_test_function(|x, y| x + y);
        let err = block_on(fcontour_builtin(vec![
            Value::BoundFunctionHandle {
                name: "surface".into(),
                function: 1,
            },
            Value::String("MeshDensity".into()),
            Value::Num(1.0),
        ]))
        .expect_err("expected mesh-density validation error");
        assert_eq!(err.identifier(), FCONTOUR_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn fcontour_mesh_density_accepts_three_and_rejects_two() {
        assert_eq!(parse_mesh_density(&Value::Int(IntValue::U8(3))).unwrap(), 3);
        assert!(parse_mesh_density(&Value::Int(IntValue::U8(2))).is_err());
    }

    #[test]
    fn fcontour_domain_requires_row_or_column_vector_shape() {
        let column =
            Tensor::new_integer(runmat_builtins::IntegerStorage::I8(vec![-2, 2]), vec![2, 1])
                .unwrap();
        let parsed = parse_domain(&Value::Tensor(column)).unwrap();
        assert_eq!((parsed.x_min, parsed.x_max), (-2.0, 2.0));

        let matrix = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I8(vec![-2, 2, -3, 3]),
            vec![2, 2],
        )
        .unwrap();
        assert!(parse_domain(&Value::Tensor(matrix)).is_err());
    }

    #[test]
    fn fcontour_level_list_treats_scalar_as_one_level_and_preserves_descending_order() {
        match parse_fcontour_level_list(&Value::Int(IntValue::I16(7))).unwrap() {
            ContourLevelSpec::Values(values) => assert_eq!(values, vec![7.0]),
            other => panic!("expected explicit values, got {other:?}"),
        }
        let descending = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(vec![3, 2, 1]),
            vec![1, 3],
        )
        .unwrap();
        match parse_fcontour_level_list(&Value::Tensor(descending)).unwrap() {
            ContourLevelSpec::Values(values) => assert_eq!(values, vec![3.0, 2.0, 1.0]),
            other => panic!("expected explicit values, got {other:?}"),
        }
    }

    #[test]
    fn fcontour_level_list_rejects_matrix_shape() {
        let matrix = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U8(vec![1, 2, 3, 4]),
            vec![2, 2],
        )
        .unwrap();
        assert!(parse_fcontour_level_list(&Value::Tensor(matrix)).is_err());
    }

    #[test]
    fn fcontour_level_step_uses_zero_for_auto_and_positive_integer_for_spacing() {
        assert!(matches!(
            parse_fcontour_level_step(&Value::Int(IntValue::U64(0))).unwrap(),
            ContourLevelSpec::Auto
        ));
        match parse_fcontour_level_step(&Value::Int(IntValue::I8(2))).unwrap() {
            ContourLevelSpec::Step(step) => assert_eq!(step, 2.0),
            other => panic!("expected step, got {other:?}"),
        }
    }

    #[test]
    fn fcontour_fill_accepts_zero_or_one_for_every_integer_class() {
        for one in every_integer_one() {
            assert!(parse_fcontour_fill(&Value::Int(one)).unwrap());
        }
        for zero in [
            IntValue::I8(0),
            IntValue::I16(0),
            IntValue::I32(0),
            IntValue::I64(0),
            IntValue::U8(0),
            IntValue::U16(0),
            IntValue::U32(0),
            IntValue::U64(0),
        ] {
            assert!(!parse_fcontour_fill(&Value::Int(zero)).unwrap());
        }
        assert!(parse_fcontour_fill(&Value::Int(IntValue::U8(2))).is_err());
    }

    #[test]
    fn fcontour_positional_integer_levels_require_runmat_extension_mode() {
        let args = vec![
            Value::BoundFunctionHandle {
                name: "surface".into(),
                function: 1,
            },
            Value::Int(IntValue::U8(5)),
        ];
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = parse_fcontour_args(args.clone())
                .err()
                .expect("strict mode rejects extension");
            assert_eq!(
                err.identifier(),
                FCONTOUR_POSITIONAL_LEVEL_EXTENSION.error_identifier
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(matches!(
                parse_fcontour_args(args).unwrap().level_spec,
                ContourLevelSpec::Count(5)
            ));
        }
    }

    #[test]
    fn fcontour_declares_all_integer_roles_and_extensions() {
        assert_eq!(FCONTOUR_INTEGER_CAPABILITIES.len(), 8);
        assert_eq!(FCONTOUR_EXTENSIONS.len(), 3);
        assert!(FCONTOUR_INTEGER_CAPABILITIES
            .iter()
            .all(|capability| capability
                .inputs
                .iter()
                .all(|input| input.classes.len() == 8)));
    }

    #[test]
    fn fcontour_mesh_density_reads_typed_integer_tensor_exactly() {
        let exact = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![400]),
            vec![1, 1],
        )
        .expect("typed density");
        assert_eq!(parse_mesh_density(&Value::Tensor(exact)).unwrap(), 400);

        let too_large = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![401]),
            vec![1, 1],
        )
        .expect("large density");
        assert!(parse_mesh_density(&Value::Tensor(too_large)).is_err());

        let negative = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::I64(vec![-1]),
            vec![1, 1],
        )
        .expect("negative density");
        assert!(parse_mesh_density(&Value::Tensor(negative)).is_err());

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        assert!(parse_mesh_density(&Value::Num(boundary)).is_err());
    }

    #[test]
    fn fcontour_function_scalar_reads_typed_integer_storage_exactly() {
        let tensor = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(vec![12]),
            vec![1, 1],
        )
        .unwrap();

        assert_eq!(
            contour_value_to_scalar(Value::Tensor(tensor)).unwrap(),
            12.0
        );
    }
}
