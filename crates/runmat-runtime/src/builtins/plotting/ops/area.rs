use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::AreaPlot;
use runmat_value::{IntegerStorage, NumericScalar, Tensor, Value};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::common::gather_tensor_from_gpu;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

use super::op_common::line_inputs::NumericInput;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_line_style_args, LineStyleParseOptions};
use crate::build_runtime_error;

const BUILTIN_NAME: &str = "area";
type AreaSeries = Vec<(Vec<f64>, Vec<f64>, Option<Vec<f64>>)>;
const MATLAB_COLOR_ORDER: [glam::Vec4; 7] = [
    glam::Vec4::new(0.0, 0.447, 0.741, 0.4),
    glam::Vec4::new(0.85, 0.325, 0.098, 0.4),
    glam::Vec4::new(0.929, 0.694, 0.125, 0.4),
    glam::Vec4::new(0.494, 0.184, 0.556, 0.4),
    glam::Vec4::new(0.466, 0.674, 0.188, 0.4),
    glam::Vec4::new(0.301, 0.745, 0.933, 0.4),
    glam::Vec4::new(0.635, 0.078, 0.184, 0.4),
];

const AREA_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Area graphics handle or handle row vector, with one handle per plotted series.",
}];

const AREA_INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y data vector or matrix. Columns are rendered as stacked series.",
}];

const AREA_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates matching the row count of Y.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
];

const AREA_INPUTS_Y_STYLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
    BuiltinParamDescriptor {
        name: "lineSpec",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line/color shorthand such as '--r'.",
    },
];

const AREA_INPUTS_X_Y_STYLE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates matching the row count of Y.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
    BuiltinParamDescriptor {
        name: "lineSpec",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line/color shorthand such as '--r'.",
    },
];

const AREA_INPUTS_Y_BASE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
    BuiltinParamDescriptor {
        name: "basevalue",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Horizontal baseline value.",
    },
];

const AREA_INPUTS_X_Y_BASE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates matching the row count of Y.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
    BuiltinParamDescriptor {
        name: "basevalue",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Horizontal baseline value.",
    },
];

const AREA_INPUTS_Y_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style properties such as Color, LineWidth, and BaseValue.",
    },
];

const AREA_INPUTS_X_Y_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates matching the row count of Y.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style properties such as Color, LineWidth, and BaseValue.",
    },
];

const AREA_INPUTS_AX_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
];

const AREA_INPUTS_AX_X_Y: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates matching the row count of Y.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
];

const AREA_INPUTS_AX_Y_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style properties such as Color, LineWidth, and BaseValue.",
    },
];

const AREA_INPUTS_AX_X_Y_PROPS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates matching the row count of Y.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style properties such as Color, LineWidth, and BaseValue.",
    },
];

const AREA_INPUTS_AX_Y_BASE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
    BuiltinParamDescriptor {
        name: "basevalue",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Horizontal baseline value.",
    },
];

const AREA_INPUTS_AX_X_Y_BASE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates matching the row count of Y.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data vector or matrix. Columns are rendered as stacked series.",
    },
    BuiltinParamDescriptor {
        name: "basevalue",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Horizontal baseline value.",
    },
];

const AREA_SIGNATURES: [BuiltinSignatureDescriptor; 14] = [
    BuiltinSignatureDescriptor {
        label: "h = area(Y)",
        inputs: &AREA_INPUTS_Y,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(Y, LineSpec)",
        inputs: &AREA_INPUTS_Y_STYLE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(Y, basevalue)",
        inputs: &AREA_INPUTS_Y_BASE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(Y, Name, Value, ...)",
        inputs: &AREA_INPUTS_Y_PROPS,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(X, Y)",
        inputs: &AREA_INPUTS_X_Y,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(X, Y, LineSpec)",
        inputs: &AREA_INPUTS_X_Y_STYLE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(X, Y, basevalue)",
        inputs: &AREA_INPUTS_X_Y_BASE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(X, Y, Name, Value, ...)",
        inputs: &AREA_INPUTS_X_Y_PROPS,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, Y)",
        inputs: &AREA_INPUTS_AX_Y,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, Y, Name, Value, ...)",
        inputs: &AREA_INPUTS_AX_Y_PROPS,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, Y, basevalue)",
        inputs: &AREA_INPUTS_AX_Y_BASE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, X, Y)",
        inputs: &AREA_INPUTS_AX_X_Y,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, X, Y, Name, Value, ...)",
        inputs: &AREA_INPUTS_AX_X_Y_PROPS,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, X, Y, basevalue)",
        inputs: &AREA_INPUTS_AX_X_Y_BASE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
];

const AREA_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AREA.INVALID_ARGUMENT",
    identifier: Some("RunMat:area:InvalidArgument"),
    when: "Input data, style tokens, or name/value options are invalid.",
    message: "area: invalid argument",
};

const AREA_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AREA.INTERNAL",
    identifier: Some("RunMat:area:Internal"),
    when: "Renderer/GPU conversion fails during chart construction.",
    message: "area: internal operation failed",
};

const AREA_ERRORS: [BuiltinErrorDescriptor; 2] = [AREA_ERROR_INVALID_ARGUMENT, AREA_ERROR_INTERNAL];

const AREA_LINESPEC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "area-linespec",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "area with a compact positional LineSpec is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AreaLineSpecExtension"),
};

pub const AREA_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [AREA_LINESPEC_EXTENSION];

const AREA_Y_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Y",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Y independently accepts every built-in integer class as vector or matrix data.",
}];

const AREA_X_Y_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes:
            "X independently accepts every built-in integer class as vector or matrix coordinates.",
    },
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Y independently accepts every built-in integer class as vector or matrix data.",
    },
];

const AREA_BASE_VALUE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "basevalue",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The positional or BaseValue name-value control accepts a real numeric scalar.",
    }];

const AREA_LINE_WIDTH_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "LineWidth",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "LineWidth explicitly accepts every built-in integer class as a scalar point width.",
    }];

pub const AREA_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "a = area(integer_Y)",
        inputs: &AREA_Y_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "An implicit double X axis is generated from one through the vector length or matrix row count. Authoritative integer Y storage crosses one explicit client graphics boundary; resident integer data gathers because area accepts GPU arrays but executes on the client.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "a = area(integer_X, integer_Y)",
        inputs: &AREA_X_Y_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "X and Y retain authoritative typed storage through shape validation and exact integer X ordering before the deliberate client graphics conversion. The output is one or more opaque Area graphics handles rather than integer data.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "a = area(..., integer_basevalue)",
        inputs: &AREA_BASE_VALUE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The positional and BaseValue name-value forms read exactly one real numeric scalar and then cross the explicit f64 graphics-property boundary; a resident scalar gathers on the client.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "a = area(..., \"LineWidth\", integer_width)",
        inputs: &AREA_LINE_WIDTH_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "LineWidth is validated and converted once by the shared host graphics-style parser; it does not create an integer output. Applying and exposing the parsed outline width is a general Area graphics-property gap, not an integer-storage distinction.",
    },
];

pub const AREA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AREA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &AREA_ERRORS,
};

fn area_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    let message = format!("{}: {}", error.message, detail.as_ref());
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::area")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "area",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "area is a plotting sink; GPU inputs may remain on device when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::area")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "area",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "area performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "area",
    category = "plotting",
    summary = "Create filled area plots.",
    keywords = "area,plotting,stacked,fill",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::area::AREA_DESCRIPTOR),
    extensions(crate::builtins::plotting::area::AREA_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::area::AREA_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::area"
)]
pub fn area_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (target_axes, x_value, y_value, rest) = parse_area_args(args)?;
    if begins_with_linespec(&rest) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &AREA_LINESPEC_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let mut x_input = Some(area_numeric_input(x_value, "X")?);
    let mut y_input = Some(area_numeric_input(y_value, "Y")?);
    let parsed = parse_area_style_args(&rest)?;
    let direct_gpu_geometry = matches!(
        x_input.as_ref(),
        Some(NumericInput::Host(tensor)) if host_x_is_nondecreasing_vector(tensor)
    );

    let plot_handles = Rc::new(RefCell::new(Vec::new()));
    let plot_handles_slot = Rc::clone(&plot_handles);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Area",
            x_label: "X",
            y_label: "Y",
            ..Default::default()
        },
        move |figure, axes| {
            let axes = target_axes.unwrap_or(axes);
            if direct_gpu_geometry {
                if let Some(y_gpu) = y_input.as_ref().and_then(NumericInput::gpu_handle) {
                    if let Ok(plots) =
                        build_area_gpu_plots(x_input.as_ref().expect("x present"), y_gpu, &parsed)
                    {
                        for plot in plots {
                            let plot_index = figure.add_area_plot_on_axes(plot, axes);
                            plot_handles_slot.borrow_mut().push((axes, plot_index));
                        }
                        return Ok(());
                    }
                }
            }
            let x_tensor = x_input
                .take()
                .expect("x consumed")
                .into_tensor(BUILTIN_NAME)?;
            let y_tensor = y_input
                .take()
                .expect("y consumed")
                .into_tensor(BUILTIN_NAME)?;
            let series = area_series_from_tensors(&x_tensor, &y_tensor)?;
            for (idx, (x, upper, lower)) in series.iter().enumerate() {
                let mut plot = AreaPlot::new(x.clone(), upper.clone())
                    .map_err(|e| area_error_with_detail(&AREA_ERROR_INTERNAL, &e))?;
                plot.baseline = parsed.base_value;
                if let Some(lower) = lower.clone() {
                    plot = plot.with_lower_curve(lower);
                }
                let color = parsed
                    .color
                    .unwrap_or(MATLAB_COLOR_ORDER[idx % MATLAB_COLOR_ORDER.len()]);
                plot.color = color;
                plot.label = Some(
                    parsed
                        .label
                        .clone()
                        .unwrap_or_else(|| format!("Series {}", idx + 1)),
                );
                let plot_index = figure.add_area_plot_on_axes(plot, axes);
                plot_handles_slot.borrow_mut().push((axes, plot_index));
            }
            Ok(())
        },
    );
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if !(lower.contains("plotting is unavailable") || lower.contains("non-main thread")) {
            return Err(err);
        }
    }
    let handles = plot_handles
        .borrow()
        .iter()
        .map(|(axes, plot_index)| {
            crate::builtins::plotting::state::register_area_handle(
                figure_handle,
                *axes,
                *plot_index,
            )
        })
        .collect::<Vec<_>>();
    Ok(super::line::handles_value(handles))
}

fn build_area_gpu_plots(
    x: &NumericInput,
    y: &runmat_accelerate_api::GpuTensorHandle,
    parsed: &ParsedAreaStyle,
) -> crate::BuiltinResult<Vec<AreaPlot>> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(BUILTIN_NAME)?;
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(y).ok_or_else(|| {
        area_error_with_detail(&AREA_ERROR_INTERNAL, "unable to export GPU Y data")
    })?;
    let (rows, cols) = area_shape_from_gpu_shape(&y_ref.shape, y_ref.len);
    let scalar = runmat_plot::gpu::ScalarType::from_is_f64(
        y_ref.precision == runmat_accelerate_api::ProviderPrecision::F64,
    );
    let (x_axis, x_source, x_bounds) = match x {
        NumericInput::Gpu(handle) => {
            let x_ref = runmat_accelerate_api::export_wgpu_buffer(handle).ok_or_else(|| {
                area_error_with_detail(&AREA_ERROR_INTERNAL, "unable to export GPU X data")
            })?;
            if x_ref.len != rows {
                return Err(area_error_with_detail(
                    &AREA_ERROR_INVALID_ARGUMENT,
                    "X length must match rows of Y",
                ));
            }
            let bounds =
                super::gpu_helpers::axis_bounds(handle, BUILTIN_NAME).unwrap_or((0.0, 0.0));
            (
                runmat_plot::gpu::axis::AxisData::Buffer(x_ref.buffer.clone()),
                runmat_plot::gpu::axis::OwnedAxisData::Buffer(x_ref.buffer.clone()),
                bounds,
            )
        }
        NumericInput::Host(tensor) => {
            let values = vector_from_tensor(tensor)?;
            if values.len() != rows {
                return Err(area_error_with_detail(
                    &AREA_ERROR_INVALID_ARGUMENT,
                    "X length must match rows of Y",
                ));
            }
            let axis = match scalar {
                runmat_plot::gpu::ScalarType::F32 => {
                    let values_f32: Vec<f32> = values.iter().map(|v| *v as f32).collect();
                    let axis = runmat_plot::gpu::axis::AxisData::F32(&values_f32);
                    let buffer = runmat_plot::gpu::axis::axis_storage_buffer(
                        &context.device,
                        "area host x axis",
                        &axis,
                        scalar,
                    )
                    .map_err(|e| area_error_with_detail(&AREA_ERROR_INTERNAL, e))?;
                    runmat_plot::gpu::axis::AxisData::Buffer(buffer)
                }
                runmat_plot::gpu::ScalarType::F64 => {
                    let axis = runmat_plot::gpu::axis::AxisData::F64(&values);
                    let buffer = runmat_plot::gpu::axis::axis_storage_buffer(
                        &context.device,
                        "area host x axis",
                        &axis,
                        scalar,
                    )
                    .map_err(|e| area_error_with_detail(&AREA_ERROR_INTERNAL, e))?;
                    runmat_plot::gpu::axis::AxisData::Buffer(buffer)
                }
            };
            (
                axis,
                runmat_plot::gpu::axis::OwnedAxisData::F64(values.clone()),
                (
                    values.first().copied().unwrap_or(0.0) as f32,
                    values.last().copied().unwrap_or(0.0) as f32,
                ),
            )
        }
    };
    let mut plots = Vec::with_capacity(cols);
    let (min_cell, max_cell) =
        super::gpu_helpers::axis_bounds(y, BUILTIN_NAME).unwrap_or((0.0, 0.0));
    let min_stack = if min_cell < 0.0 {
        parsed.base_value as f32 + (min_cell * cols as f32)
    } else {
        parsed.base_value as f32
    };
    let max_stack = if max_cell > 0.0 {
        parsed.base_value as f32 + (max_cell * cols as f32)
    } else {
        parsed.base_value as f32
    };
    for idx in 0..cols {
        let inputs = runmat_plot::gpu::area::AreaGpuInputs {
            x_axis: x_axis.clone(),
            y_buffer: y_ref.buffer.clone(),
            rows: rows as u32,
            cols: cols as u32,
            target_col: idx as u32,
            scalar,
        };
        let gpu_source = runmat_plot::plots::AreaGpuSource {
            x_axis: x_source.clone(),
            y_buffer: inputs.y_buffer.clone(),
            rows,
            cols,
            target_col: idx,
            scalar,
        };
        let gpu_vertices = runmat_plot::gpu::area::pack_vertices(
            &context.device,
            &context.queue,
            &inputs,
            &runmat_plot::gpu::area::AreaGpuParams {
                color: parsed
                    .color
                    .unwrap_or(MATLAB_COLOR_ORDER[idx % MATLAB_COLOR_ORDER.len()]),
                baseline: parsed.base_value as f32,
            },
        )
        .map_err(|e| {
            area_error_with_detail(
                &AREA_ERROR_INTERNAL,
                format!("failed to build GPU vertices: {e}"),
            )
        })?;
        let mut plot = AreaPlot::from_gpu_buffer(
            parsed
                .color
                .unwrap_or(MATLAB_COLOR_ORDER[idx % MATLAB_COLOR_ORDER.len()]),
            parsed.base_value,
            None,
            gpu_vertices,
            (rows - 1) * 6,
            runmat_plot::core::BoundingBox::new(
                glam::Vec3::new(x_bounds.0, min_stack, 0.0),
                glam::Vec3::new(x_bounds.1, max_stack, 0.0),
            ),
        )
        .with_gpu_source(gpu_source);
        plot.label = Some(
            parsed
                .label
                .clone()
                .unwrap_or_else(|| format!("Series {}", idx + 1)),
        );
        plots.push(plot);
    }
    Ok(plots)
}

struct ParsedAreaStyle {
    color: Option<glam::Vec4>,
    label: Option<String>,
    base_value: f64,
}

fn parse_area_style_args(args: &[Value]) -> crate::BuiltinResult<ParsedAreaStyle> {
    let mut filtered = Vec::new();
    let mut base_value = 0.0;
    let mut idx = 0usize;
    if let Some(value) = args.first().filter(|value| is_area_numeric_scalar(value)) {
        base_value = area_scalar_f64(value)?;
        idx = 1;
    }
    while idx < args.len() {
        if let Some(key) = super::style::value_as_string(&args[idx]) {
            if key.trim().eq_ignore_ascii_case("BaseValue") && idx + 1 < args.len() {
                base_value = area_scalar_f64(&args[idx + 1])?;
                idx += 2;
                continue;
            }
        }
        filtered.push(args[idx].clone());
        idx += 1;
    }
    let parsed = parse_line_style_args(&filtered, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
    let explicit_color = area_color_was_explicit(&filtered);
    Ok(ParsedAreaStyle {
        color: explicit_color.then_some(parsed.appearance.color),
        label: parsed.label,
        base_value,
    })
}

fn begins_with_linespec(args: &[Value]) -> bool {
    let Some(token) = args.first().and_then(super::style::value_as_string) else {
        return false;
    };
    let token = token.trim();
    !token.is_empty()
        && token.chars().all(|ch| {
            matches!(
                ch,
                '-' | ':'
                    | '.'
                    | 'r'
                    | 'g'
                    | 'b'
                    | 'c'
                    | 'm'
                    | 'y'
                    | 'k'
                    | 'w'
                    | 'o'
                    | '+'
                    | '*'
                    | 'x'
                    | 's'
                    | 'd'
                    | '^'
                    | 'v'
                    | '>'
                    | '<'
                    | 'p'
                    | 'h'
            )
        })
}

fn area_color_was_explicit(args: &[Value]) -> bool {
    if let Some(token) = args.first().and_then(super::style::value_as_string) {
        let mut chars = token.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                'y' | 'm' | 'c' | 'r' | 'g' | 'b' | 'w' | 'k' => return true,
                '-' | '.' => {
                    if matches!(chars.peek(), Some('-' | '.')) {
                        chars.next();
                    }
                }
                _ => {}
            }
        }
    }
    let mut idx = 0usize;
    while idx + 1 < args.len() {
        if let Some(key) = super::style::value_as_string(&args[idx]) {
            if key.trim().eq_ignore_ascii_case("Color") {
                return true;
            }
        }
        idx += 2;
    }
    false
}

fn parse_area_args(
    args: Vec<Value>,
) -> crate::BuiltinResult<(Option<usize>, Value, Value, Vec<Value>)> {
    if args.is_empty() {
        return Err(area_error_with_detail(
            &AREA_ERROR_INVALID_ARGUMENT,
            "expected Y or X,Y inputs",
        ));
    }
    let mut it = args.into_iter();
    let mut target_axes = None;
    let first = it.next().unwrap();
    let first = if let Ok(crate::builtins::plotting::properties::PlotHandle::Axes(_, axes)) =
        crate::builtins::plotting::properties::resolve_plot_handle(&first, BUILTIN_NAME)
    {
        target_axes = Some(axes);
        it.next().ok_or_else(|| {
            area_error_with_detail(
                &AREA_ERROR_INVALID_ARGUMENT,
                "expected data after axes handle",
            )
        })?
    } else {
        first
    };
    let Some(second) = it.next() else {
        let x = implicit_area_x(&first)?;
        return Ok((target_axes, Value::Tensor(x), first, Vec::new()));
    };
    if matches!(second, Value::String(_) | Value::CharArray(_)) {
        let x = implicit_area_x(&first)?;
        let mut rest = vec![second];
        rest.extend(it);
        return Ok((target_axes, Value::Tensor(x), first, rest));
    }
    if is_area_numeric_scalar(&second) && area_value_len(&first).unwrap_or(1) > 1 {
        let x = implicit_area_x(&first)?;
        let mut rest = vec![second];
        rest.extend(it);
        return Ok((target_axes, Value::Tensor(x), first, rest));
    }
    Ok((target_axes, first, second, it.collect()))
}

fn implicit_area_x(y: &Value) -> crate::BuiltinResult<Tensor> {
    let rows = match y {
        Value::Num(_) | Value::Int(_) => 1,
        Value::Tensor(tensor) => area_shape_from_tensor(tensor).0,
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                return Err(area_error_with_detail(
                    &AREA_ERROR_INVALID_ARGUMENT,
                    "Y must be real numeric data",
                ));
            }
            let len = handle.shape.iter().product();
            area_shape_from_gpu_shape(&handle.shape, len).0
        }
        _ => {
            return Err(area_error_with_detail(
                &AREA_ERROR_INVALID_ARGUMENT,
                "Y must be real numeric data",
            ));
        }
    };
    Tensor::new((1..=rows).map(|index| index as f64).collect(), vec![rows])
        .map_err(|error| area_error_with_detail(&AREA_ERROR_INTERNAL, error))
}

fn area_value_len(value: &Value) -> Option<usize> {
    match value {
        Value::Num(_) | Value::Int(_) => Some(1),
        Value::Tensor(tensor) => Some(tensor.len()),
        Value::GpuTensor(handle) => Some(handle.shape.iter().product()),
        _ => None,
    }
}

fn is_area_numeric_scalar(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.len() == 1,
        Value::GpuTensor(handle) => handle.shape.iter().product::<usize>() == 1,
        _ => false,
    }
}

fn area_scalar_f64(value: &Value) -> crate::BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => Ok(NumericScalar::from(value.clone()).materialize_f64()),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor
            .numeric_value_at(0)
            .map(NumericScalar::materialize_f64)
            .ok_or_else(|| {
                area_error_with_detail(&AREA_ERROR_INVALID_ARGUMENT, "BaseValue must be numeric")
            }),
        Value::GpuTensor(handle) if handle.shape.iter().product::<usize>() == 1 => {
            if runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                return Err(area_error_with_detail(
                    &AREA_ERROR_INVALID_ARGUMENT,
                    "BaseValue must be a real numeric scalar",
                ));
            }
            let tensor = gather_tensor_from_gpu(handle.clone(), BUILTIN_NAME)?;
            area_scalar_f64(&Value::Tensor(tensor))
        }
        _ => Err(area_error_with_detail(
            &AREA_ERROR_INVALID_ARGUMENT,
            "BaseValue must be a real numeric scalar",
        )),
    }
}

fn area_numeric_input(value: Value, name: &str) -> crate::BuiltinResult<NumericInput> {
    match value {
        Value::Num(value) => NumericInput::from_value(Value::Num(value), BUILTIN_NAME),
        Value::Int(value) => {
            let tensor = Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|error| area_error_with_detail(&AREA_ERROR_INTERNAL, error))?;
            Ok(NumericInput::Host(tensor))
        }
        Value::Tensor(tensor) => Ok(NumericInput::Host(tensor)),
        Value::GpuTensor(handle)
            if !runmat_accelerate_api::handle_is_logical(&handle)
                && runmat_accelerate_api::handle_storage(&handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real =>
        {
            Ok(NumericInput::Gpu(handle))
        }
        _ => Err(area_error_with_detail(
            &AREA_ERROR_INVALID_ARGUMENT,
            format!("{name} must be real numeric data"),
        )),
    }
}

fn vector_from_tensor(tensor: &Tensor) -> crate::BuiltinResult<Vec<f64>> {
    if !(tensor.rows == 1 || tensor.cols == 1 || tensor.shape.len() <= 1) {
        return Err(area_error_with_detail(
            &AREA_ERROR_INVALID_ARGUMENT,
            "X input must be a vector matching the row count of Y",
        ));
    }
    Ok(tensor_utils::tensor_values_f64(tensor))
}

fn host_x_is_nondecreasing_vector(tensor: &Tensor) -> bool {
    if !(tensor.rows == 1 || tensor.cols == 1 || tensor.shape.len() <= 1) {
        return false;
    }
    (1..tensor.len()).all(|index| {
        let previous = tensor
            .numeric_value_at(index - 1)
            .expect("valid X tensor storage");
        let current = tensor
            .numeric_value_at(index)
            .expect("valid X tensor storage");
        compare_numeric_scalars(previous, current) != Ordering::Greater
    })
}

fn area_shape_from_tensor(tensor: &Tensor) -> (usize, usize) {
    if tensor.shape.len() <= 1 || tensor.rows == 1 || tensor.cols == 1 {
        (tensor_utils::tensor_element_len(tensor).max(1), 1)
    } else {
        (tensor.rows.max(1), tensor.cols.max(1))
    }
}

fn area_shape_from_gpu_shape(shape: &[usize], len: usize) -> (usize, usize) {
    let rows = shape.first().copied().unwrap_or(len).max(1);
    let cols = shape.get(1).copied().unwrap_or(1).max(1);
    if shape.len() <= 1 || rows == 1 || cols == 1 {
        (len.max(1), 1)
    } else {
        (rows, cols)
    }
}

fn area_series_from_tensors(x: &Tensor, y: &Tensor) -> crate::BuiltinResult<AreaSeries> {
    let (rows, cols) = area_shape_from_tensor(y);
    let x_columns = area_x_columns(x, rows, cols)?;
    let mut cumulative = vec![0.0; rows];
    let mut out: AreaSeries = Vec::with_capacity(cols);
    for (col, x_column) in x_columns.into_iter().enumerate() {
        let mut order = (0..rows).collect::<Vec<_>>();
        order.sort_by(|left, right| compare_numeric_scalars(x_column[*left], x_column[*right]));
        let x_values = order
            .iter()
            .map(|row| x_column[*row].materialize_f64())
            .collect::<Vec<_>>();
        let lower_values = cumulative.clone();
        for row in 0..rows {
            let index = if cols == 1 { row } else { col * rows + row };
            cumulative[row] += y
                .numeric_value_at(index)
                .map(NumericScalar::materialize_f64)
                .unwrap_or(0.0);
        }
        let upper = order.iter().map(|row| cumulative[*row]).collect();
        let lower = (col > 0).then(|| order.iter().map(|row| lower_values[*row]).collect());
        out.push((x_values, upper, lower));
    }
    Ok(out)
}

fn area_x_columns(
    x: &Tensor,
    rows: usize,
    cols: usize,
) -> crate::BuiltinResult<Vec<Vec<NumericScalar>>> {
    if x.rows == 1 || x.cols == 1 || x.shape.len() <= 1 {
        if x.len() != rows {
            return Err(area_error_with_detail(
                &AREA_ERROR_INVALID_ARGUMENT,
                "X length must match the number of rows in Y",
            ));
        }
        let column = (0..rows)
            .map(|index| x.numeric_value_at(index).expect("validated area X storage"))
            .collect::<Vec<_>>();
        return Ok(vec![column; cols]);
    }
    if x.shape.len() > 2 || x.rows != rows || x.cols != cols {
        return Err(area_error_with_detail(
            &AREA_ERROR_INVALID_ARGUMENT,
            "matrix X must have the same size as matrix Y",
        ));
    }
    Ok((0..cols)
        .map(|col| {
            (0..rows)
                .map(|row| {
                    x.numeric_value_at(col * rows + row)
                        .expect("validated area X matrix storage")
                })
                .collect()
        })
        .collect())
}

fn compare_numeric_scalars(left: NumericScalar, right: NumericScalar) -> Ordering {
    match (left, right) {
        (NumericScalar::F64(left), NumericScalar::F64(right)) => compare_floating(left, right),
        (NumericScalar::F32(left), NumericScalar::F32(right)) => compare_floating(left, right),
        (NumericScalar::I8(left), NumericScalar::I8(right)) => left.cmp(&right),
        (NumericScalar::I16(left), NumericScalar::I16(right)) => left.cmp(&right),
        (NumericScalar::I32(left), NumericScalar::I32(right)) => left.cmp(&right),
        (NumericScalar::I64(left), NumericScalar::I64(right)) => left.cmp(&right),
        (NumericScalar::U8(left), NumericScalar::U8(right)) => left.cmp(&right),
        (NumericScalar::U16(left), NumericScalar::U16(right)) => left.cmp(&right),
        (NumericScalar::U32(left), NumericScalar::U32(right)) => left.cmp(&right),
        (NumericScalar::U64(left), NumericScalar::U64(right)) => left.cmp(&right),
        (left, right) => compare_floating(left.materialize_f64(), right.materialize_f64()),
    }
}

fn compare_floating<T: PartialOrd>(left: T, right: T) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;

    #[cfg(feature = "wgpu")]
    fn register_wgpu_provider_available() -> bool {
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_ok()
            && runmat_accelerate_api::provider().is_some()
    }

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn matrix_tensor(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor::new(data, vec![rows, cols]).expect("area test matrix")
    }

    fn first_handle(value: &Value) -> Value {
        match value {
            Value::Num(_) => value.clone(),
            Value::Tensor(tensor) => Value::Num(
                tensor
                    .numeric_value_at(0)
                    .expect("area handle vector is nonempty")
                    .materialize_f64(),
            ),
            other => panic!("expected area handle output, got {other:?}"),
        }
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn integer_value(storage: IntegerStorage) -> Value {
        let len = storage.len();
        Value::Tensor(Tensor::new_integer(storage, vec![1, len]).expect("integer vector"))
    }

    fn all_integer_storages() -> [IntegerStorage; 8] {
        [
            IntegerStorage::I8(vec![1, 2]),
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I64(vec![1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ]
    }

    fn all_integer_scalar_storages(value: u8) -> [IntegerStorage; 8] {
        [
            IntegerStorage::I8(vec![value as i8]),
            IntegerStorage::I16(vec![value as i16]),
            IntegerStorage::I32(vec![value as i32]),
            IntegerStorage::I64(vec![value as i64]),
            IntegerStorage::U8(vec![value]),
            IntegerStorage::U16(vec![value as u16]),
            IntegerStorage::U32(vec![value as u32]),
            IntegerStorage::U64(vec![value as u64]),
        ]
    }

    #[test]
    fn area_vector_from_tensor_reads_typed_integer_storage_exactly() {
        let x = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![-1, 0, 1]),
            vec![1, 3],
        )
        .expect("typed area x vector");

        assert_eq!(
            vector_from_tensor(&x).expect("area vector"),
            vec![-1.0, 0.0, 1.0]
        );
        assert_eq!(area_shape_from_tensor(&x), (3, 1));
    }

    #[test]
    fn area_series_reads_typed_integer_storage_without_mirror() {
        let y = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![1, 2, 3, 4]),
            vec![2, 2],
        )
        .expect("typed area matrix");
        let x = Tensor::new(vec![1.0, 2.0], vec![2]).expect("area x");

        let series = area_series_from_tensors(&x, &y).expect("area series");
        assert_eq!(series[0], (vec![1.0, 2.0], vec![1.0, 2.0], None));
        assert_eq!(
            series[1],
            (vec![1.0, 2.0], vec![4.0, 6.0], Some(vec![1.0, 2.0]))
        );
    }

    #[test]
    fn area_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = AREA_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = area(Y)"));
        assert!(labels.contains(&"h = area(X, Y)"));
        assert!(labels.contains(&"h = area(Y, basevalue)"));
        assert!(labels.contains(&"h = area(ax, X, Y, basevalue)"));
        assert!(labels.contains(&"h = area(ax, X, Y, Name, Value, ...)"));
        assert_eq!(AREA_INTEGER_CAPABILITIES.len(), 4);
        assert_eq!(AREA_EXTENSIONS, [AREA_LINESPEC_EXTENSION]);
    }

    #[test]
    fn area_invalid_argument_uses_stable_identifier() {
        let err = area_builtin(vec![]).expect_err("missing args should fail");
        assert_eq!(err.identifier(), AREA_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn area_builds_stacked_series_from_matrix() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = area_builtin(vec![Value::Tensor(matrix_tensor(
            vec![1.0, 2.0, 0.5, 0.5],
            2,
            2,
        ))])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plots().count(), 2);
        let PlotElement::Area(first) = fig.plots().next().unwrap() else {
            panic!("expected area")
        };
        let PlotElement::Area(second) = fig.plots().nth(1).unwrap() else {
            panic!("expected area")
        };
        assert_eq!(first.y, vec![1.0, 2.0]);
        assert_eq!(second.lower_y, Some(vec![1.0, 2.0]));
        let Value::Tensor(handles) = &handle else {
            panic!("matrix Y should return a handle row vector")
        };
        assert_eq!(handles.shape, vec![1, 2]);
        assert_eq!(
            get_builtin(vec![first_handle(&handle), Value::String("Type".into())]).unwrap(),
            Value::String("area".into())
        );
    }

    #[test]
    fn area_accepts_explicit_x_with_matrix_series() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = area_builtin(vec![
            Value::Tensor(
                Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).expect("area x vector"),
            ),
            Value::Tensor(matrix_tensor(
                vec![
                    1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0,
                ],
                5,
                3,
            )),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plots().count(), 3);
        let colors = fig
            .plots()
            .take(3)
            .map(|plot| match plot {
                PlotElement::Area(area) => area.color,
                _ => panic!("expected area"),
            })
            .collect::<Vec<_>>();
        assert_eq!(colors[0], MATLAB_COLOR_ORDER[0]);
        assert_eq!(colors[1], MATLAB_COLOR_ORDER[1]);
        assert_eq!(colors[2], MATLAB_COLOR_ORDER[2]);
        let Value::Tensor(handles) = &handle else {
            panic!("matrix Y should return a handle row vector")
        };
        assert_eq!(handles.shape, vec![1, 3]);
        assert_eq!(
            get_builtin(vec![first_handle(&handle), Value::String("Type".into())]).unwrap(),
            Value::String("area".into())
        );
    }

    #[test]
    fn area_accepts_explicit_x_with_row_vector_y() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = area_builtin(vec![
            Value::Tensor(Tensor::new(vec![0.0, 0.2, 0.4, 0.6], vec![4]).expect("area x vector")),
            Value::Tensor(
                Tensor::new(vec![2.0, 2.2, 2.4, 2.6], vec![1, 4]).expect("area row vector"),
            ),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plots().count(), 1);
        let PlotElement::Area(plot) = fig.plots().next().unwrap() else {
            panic!("expected area")
        };
        assert_eq!(plot.x, vec![0.0, 0.2, 0.4, 0.6]);
        assert_eq!(plot.y, vec![2.0, 2.2, 2.4, 2.6]);
    }

    #[test]
    fn area_accepts_all_integer_x_and_y_classes() {
        let _guard = setup();
        for storage in all_integer_storages() {
            let implicit = area_builtin(vec![integer_value(storage.clone())])
                .expect("implicit X with documented integer Y");
            assert_eq!(
                tensor_data(
                    get_builtin(vec![first_handle(&implicit), Value::String("YData".into())])
                        .unwrap()
                ),
                vec![1.0, 2.0]
            );
            let explicit =
                area_builtin(vec![integer_value(storage.clone()), integer_value(storage)])
                    .expect("documented integer X and Y");
            assert_eq!(
                tensor_data(
                    get_builtin(vec![first_handle(&explicit), Value::String("XData".into())])
                        .unwrap()
                ),
                vec![1.0, 2.0]
            );
        }
    }

    #[test]
    fn area_accepts_all_integer_basevalue_and_linewidth_classes() {
        for storage in all_integer_scalar_storages(3) {
            let value = integer_value(storage);
            let positional = parse_area_style_args(std::slice::from_ref(&value))
                .expect("integer positional BaseValue");
            assert_eq!(positional.base_value, 3.0);
            let named = parse_area_style_args(&[Value::String("BaseValue".into()), value.clone()])
                .expect("integer BaseValue property");
            assert_eq!(named.base_value, 3.0);
            let width = parse_area_style_args(&[Value::String("LineWidth".into()), value])
                .expect("integer LineWidth property");
            assert_eq!(width.base_value, 0.0);
        }
    }

    #[test]
    fn area_positional_basevalue_applies_to_every_returned_series() {
        let _guard = setup();
        let handles = area_builtin(vec![
            Value::Tensor(matrix_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2)),
            Value::Int(runmat_value::IntValue::I16(-2)),
        ])
        .expect("positional integer BaseValue");
        let Value::Tensor(handles) = handles else {
            panic!("stacked matrix should return handles")
        };
        for index in 0..handles.len() {
            let handle = Value::Num(
                handles
                    .numeric_value_at(index)
                    .expect("handle")
                    .materialize_f64(),
            );
            assert_eq!(
                get_builtin(vec![handle, Value::String("BaseValue".into())]).unwrap(),
                Value::Num(-2.0)
            );
        }
    }

    #[test]
    fn area_sorts_wide_integer_x_before_graphics_conversion() {
        let x = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_992]),
            vec![2],
        )
        .expect("wide integer X");
        let y = Tensor::new_integer(IntegerStorage::I64(vec![10, 20]), vec![2]).expect("integer Y");
        let series = area_series_from_tensors(&x, &y).expect("area series");
        assert_eq!(series[0].1, vec![20.0, 10.0]);
    }

    #[test]
    fn area_supports_matrix_x_with_independent_exact_column_ordering() {
        let x = Tensor::new_integer(IntegerStorage::I16(vec![2, 1, 4, 3]), vec![2, 2])
            .expect("matrix X");
        let y = Tensor::new_integer(IntegerStorage::U16(vec![20, 10, 40, 30]), vec![2, 2])
            .expect("matrix Y");
        let series = area_series_from_tensors(&x, &y).expect("matrix X series");
        assert_eq!(series[0].0, vec![1.0, 2.0]);
        assert_eq!(series[0].1, vec![10.0, 20.0]);
        assert_eq!(series[1].0, vec![3.0, 4.0]);
        assert_eq!(series[1].1, vec![40.0, 60.0]);
        assert_eq!(series[1].2, Some(vec![10.0, 20.0]));
    }

    #[test]
    fn area_gathers_all_resident_integer_y_classes_on_the_client() {
        let _guard = setup();
        test_support::with_test_provider(|provider| {
            for (storage, baseline) in all_integer_storages()
                .into_iter()
                .zip(all_integer_scalar_storages(3))
            {
                let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("resident integer Y");
                let handle =
                    gpu_helpers::upload_tensor(provider, &tensor).expect("integer Y upload");
                let implicit = area_builtin(vec![Value::GpuTensor(handle.clone())])
                    .expect("resident documented integer Y");
                assert_eq!(
                    tensor_data(
                        get_builtin(vec![implicit, Value::String("YData".into())]).unwrap()
                    ),
                    vec![1.0, 2.0]
                );
                let baseline =
                    Tensor::new_integer(baseline, vec![1, 1]).expect("resident integer baseline");
                let baseline =
                    gpu_helpers::upload_tensor(provider, &baseline).expect("baseline upload");
                let explicit = area_builtin(vec![
                    Value::GpuTensor(handle.clone()),
                    Value::GpuTensor(handle),
                    Value::GpuTensor(baseline),
                ])
                .expect("resident documented integer X, Y, and BaseValue");
                assert_eq!(
                    tensor_data(
                        get_builtin(vec![explicit.clone(), Value::String("XData".into())]).unwrap()
                    ),
                    vec![1.0, 2.0]
                );
                assert_eq!(
                    get_builtin(vec![explicit, Value::String("BaseValue".into())]).unwrap(),
                    Value::Num(3.0)
                );
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn area_wgpu_gathers_all_resident_integer_y_classes_on_the_client() {
        let _plot_guard = setup();
        let _accel_guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("WGPU provider");
        for (storage, baseline) in all_integer_storages()
            .into_iter()
            .zip(all_integer_scalar_storages(3))
        {
            let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("resident integer Y");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer Y upload");
            let implicit = area_builtin(vec![Value::GpuTensor(handle.clone())])
                .expect("resident WGPU integer Y");
            assert_eq!(
                tensor_data(get_builtin(vec![implicit, Value::String("YData".into())]).unwrap()),
                vec![1.0, 2.0]
            );
            let baseline =
                Tensor::new_integer(baseline, vec![1, 1]).expect("resident integer baseline");
            let baseline =
                gpu_helpers::upload_tensor(provider, &baseline).expect("baseline upload");
            let explicit = area_builtin(vec![
                Value::GpuTensor(handle.clone()),
                Value::GpuTensor(handle),
                Value::GpuTensor(baseline),
            ])
            .expect("resident WGPU integer X, Y, and BaseValue");
            assert_eq!(
                tensor_data(
                    get_builtin(vec![explicit.clone(), Value::String("XData".into())]).unwrap()
                ),
                vec![1.0, 2.0]
            );
            assert_eq!(
                get_builtin(vec![explicit, Value::String("BaseValue".into())]).unwrap(),
                Value::Num(3.0)
            );
        }
    }

    #[test]
    fn area_rejects_logical_data() {
        let error = area_builtin(vec![Value::LogicalArray(
            runmat_value::LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical array"),
        )])
        .expect_err("logical Y is undocumented");
        assert_eq!(error.identifier(), AREA_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn area_linespec_is_an_explicit_runmat_extension() {
        let _guard = setup();
        let args = || {
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).expect("area Y")),
                Value::String("--r".into()),
            ]
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = area_builtin(args()).expect_err("LineSpec extension should be gated");
            assert_eq!(error.identifier(), AREA_LINESPEC_EXTENSION.error_identifier);
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            area_builtin(args()).expect("LineSpec extension enabled");
        }
    }
}
