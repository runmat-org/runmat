use crate::*;
pub const ZLABEL_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "zlabel-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a typed-integer alias for an encoded axes handle",
    error_identifier: Some("RunMat:compatibility:ZlabelIntegerAxesHandleExtension"),
};
pub const ZLABEL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ZLABEL_INTEGER_AXES_EXTENSION];
const ZLABEL_INTEGER_TEXT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "txt",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "A scalar integer label is formatted from its exact signed or unsigned decimal value.",
    }];
const ZLABEL_INTEGER_FONT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FontSize",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented for positive FontSize values.",
    }];
const ZLABEL_INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for encoded axes handles are separately gated.",
    }];
pub const ZLABEL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = zlabel(integer_txt)",
        inputs: &ZLABEL_INTEGER_TEXT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact decimal spelling becomes the stored String property.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = zlabel(..., 'FontSize', integer_size)",
        inputs: &ZLABEL_INTEGER_FONT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The validated positive size crosses the graphics scalar boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = zlabel(integer_ax, txt, ...)",
        inputs: &ZLABEL_INTEGER_AXES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Strict mode rejects before graphics state access.",
    },
];

const ZLABEL_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated zlabel object.",
}];

const ZLABEL_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Label text (string/char/cellstr-like multiline forms).",
}];

const ZLABEL_INPUTS_AX_TEXT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Label text (string/char/cellstr-like multiline forms).",
    },
];

const ZLABEL_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Label text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const ZLABEL_INPUTS_AX_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Label text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const ZLABEL_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = zlabel(txt)",
        inputs: &ZLABEL_INPUTS_TEXT,
        outputs: &ZLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = zlabel(ax, txt)",
        inputs: &ZLABEL_INPUTS_AX_TEXT,
        outputs: &ZLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = zlabel(txt, Name, Value, ...)",
        inputs: &ZLABEL_INPUTS_TEXT_PROPS,
        outputs: &ZLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = zlabel(ax, txt, Name, Value, ...)",
        inputs: &ZLABEL_INPUTS_AX_TEXT_PROPS,
        outputs: &ZLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "zlabel(txt)",
        inputs: &ZLABEL_INPUTS_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "zlabel(ax, txt)",
        inputs: &ZLABEL_INPUTS_AX_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "zlabel(txt, Name, Value, ...)",
        inputs: &ZLABEL_INPUTS_TEXT_PROPS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "zlabel(ax, txt, Name, Value, ...)",
        inputs: &ZLABEL_INPUTS_AX_TEXT_PROPS,
        outputs: &[],
    },
];

pub const ZLABEL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZLABEL.INVALID_ARGUMENT",
    identifier: Some("RunMat:zlabel:InvalidArgument"),
    when: "Axes handle, text payload, or property/value arguments are invalid.",
    message: "zlabel: invalid argument",
};

pub const ZLABEL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZLABEL.INTERNAL",
    identifier: Some("RunMat:zlabel:Internal"),
    when: "Internal plotting state update fails.",
    message: "zlabel: internal operation failed",
};

const ZLABEL_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ZLABEL_ERROR_INVALID_ARGUMENT,
    ZLABEL_ERROR_INTERNAL,
    ZLABEL_ERROR_TOO_MANY_OUTPUTS,
];
pub const ZLABEL_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZLABEL.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:zlabel:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "zlabel: too many output arguments",
};

pub const ZLABEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZLABEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZLABEL_ERRORS,
};
