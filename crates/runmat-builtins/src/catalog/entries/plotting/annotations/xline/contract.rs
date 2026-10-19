use crate::*;
const XLINE_INTEGER_COORDINATE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented for reference-line coordinates.",
    }];
const XLINE_INTEGER_COLOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Color",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer RGB components are accepted when every component is in the documented interval from zero through one.",
    }];
const XLINE_INTEGER_LINE_WIDTH_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "LineWidth",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer line widths are accepted when the scalar is finite and positive.",
    }];
const XLINE_INTEGER_VISIBLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Visible",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric visibility values must be exactly zero or one.",
    }];
pub const XLINE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = xline(integer_x, ...)", inputs: &XLINE_INTEGER_COORDINATE_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Coordinates cross the graphics floating-point boundary after numeric classification; returned graphics handles are double." },
    BuiltinIntegerCapabilityDescriptor { form: "h = xline(..., 'Color', integer_rgb)", inputs: &XLINE_INTEGER_COLOR_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Validated RGB components cross the renderer color boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = xline(..., 'LineWidth', integer_width)", inputs: &XLINE_INTEGER_LINE_WIDTH_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The validated positive width crosses the renderer scalar boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = xline(..., 'Visible', integer_visible)", inputs: &XLINE_INTEGER_VISIBLE_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The integer is classified exactly as zero or one before the graphics state is updated." },
];

const XLINE_OUTPUT_HANDLES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Reference line handle scalar or row vector when multiple coordinates are provided.",
}];

const XLINE_INPUTS_COORD: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X-coordinate scalar or numeric vector of vertical line positions.",
}];

const XLINE_INPUTS_COORD_STYLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-coordinate scalar or numeric vector of vertical line positions.",
    },
    BuiltinParamDescriptor {
        name: "style_or_label",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "LineSpec token (for example '--r') or label text.",
    },
];

const XLINE_INPUTS_COORD_STYLE_LABEL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-coordinate scalar or numeric vector of vertical line positions.",
    },
    BuiltinParamDescriptor {
        name: "linespec",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line style token (for example '--r').",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reference line label text.",
    },
];

const XLINE_INPUTS_COORD_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-coordinate scalar or numeric vector of vertical line positions.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Style arguments and Name/Value pairs (Color, LineWidth, LineStyle, LabelOrientation, Visible).",
    },
];

const XLINE_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = xline(x)",
        inputs: &XLINE_INPUTS_COORD,
        outputs: &XLINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = xline(x, styleOrLabel)",
        inputs: &XLINE_INPUTS_COORD_STYLE,
        outputs: &XLINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = xline(x, LineSpec, label)",
        inputs: &XLINE_INPUTS_COORD_STYLE_LABEL,
        outputs: &XLINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = xline(x, Name, Value, ...)",
        inputs: &XLINE_INPUTS_COORD_PROPS,
        outputs: &XLINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "xline(x)",
        inputs: &XLINE_INPUTS_COORD,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "xline(x, styleOrLabel)",
        inputs: &XLINE_INPUTS_COORD_STYLE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "xline(x, LineSpec, label)",
        inputs: &XLINE_INPUTS_COORD_STYLE_LABEL,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "xline(x, Name, Value, ...)",
        inputs: &XLINE_INPUTS_COORD_PROPS,
        outputs: &[],
    },
];

pub const XLINE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:xline:InvalidArgument"),
    when: "Coordinate input, style arguments, or Name/Value pairs are invalid.",
    message: "xline: invalid argument",
};

pub const XLINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLINE.INTERNAL",
    identifier: Some("RunMat:xline:Internal"),
    when: "Internal plotting state update fails.",
    message: "xline: internal operation failed",
};

const XLINE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    XLINE_ERROR_INVALID_ARGUMENT,
    XLINE_ERROR_INTERNAL,
    XLINE_ERROR_TOO_MANY_OUTPUTS,
];
pub const XLINE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLINE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:xline:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "xline: too many output arguments",
};

pub const XLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XLINE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XLINE_ERRORS,
};
