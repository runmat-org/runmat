use crate::*;
pub const YLABEL_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ylabel-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a typed-integer alias for an encoded axes handle",
    error_identifier: Some("RunMat:compatibility:YlabelIntegerAxesHandleExtension"),
};
pub const YLABEL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [YLABEL_INTEGER_AXES_EXTENSION];
const YLABEL_INTEGER_TEXT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "txt",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "A scalar integer label is formatted from its exact signed or unsigned decimal value.",
    }];
const YLABEL_INTEGER_FONT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FontSize",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented for positive FontSize values.",
    }];
const YLABEL_INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for encoded axes handles are separately gated.",
    }];
pub const YLABEL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = ylabel(integer_txt)",
        inputs: &YLABEL_INTEGER_TEXT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact decimal spelling becomes the stored String property.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = ylabel(..., 'FontSize', integer_size)",
        inputs: &YLABEL_INTEGER_FONT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The validated positive size crosses the graphics scalar boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = ylabel(integer_ax, txt, ...)",
        inputs: &YLABEL_INTEGER_AXES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Strict mode rejects before graphics state access.",
    },
];

const YLABEL_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated ylabel object.",
}];

const YLABEL_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Label text (string/char/cellstr-like multiline forms).",
}];

const YLABEL_INPUTS_AX_TEXT: [BuiltinParamDescriptor; 2] = [
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

const YLABEL_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
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

const YLABEL_INPUTS_AX_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
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

const YLABEL_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = ylabel(txt)",
        inputs: &YLABEL_INPUTS_TEXT,
        outputs: &YLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ylabel(ax, txt)",
        inputs: &YLABEL_INPUTS_AX_TEXT,
        outputs: &YLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ylabel(txt, Name, Value, ...)",
        inputs: &YLABEL_INPUTS_TEXT_PROPS,
        outputs: &YLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ylabel(ax, txt, Name, Value, ...)",
        inputs: &YLABEL_INPUTS_AX_TEXT_PROPS,
        outputs: &YLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ylabel(txt)",
        inputs: &YLABEL_INPUTS_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ylabel(ax, txt)",
        inputs: &YLABEL_INPUTS_AX_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ylabel(txt, Name, Value, ...)",
        inputs: &YLABEL_INPUTS_TEXT_PROPS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ylabel(ax, txt, Name, Value, ...)",
        inputs: &YLABEL_INPUTS_AX_TEXT_PROPS,
        outputs: &[],
    },
];

pub const YLABEL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLABEL.INVALID_ARGUMENT",
    identifier: Some("RunMat:ylabel:InvalidArgument"),
    when: "Axes handle, text payload, or property/value arguments are invalid.",
    message: "ylabel: invalid argument",
};

pub const YLABEL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLABEL.INTERNAL",
    identifier: Some("RunMat:ylabel:Internal"),
    when: "Internal plotting state update fails.",
    message: "ylabel: internal operation failed",
};

const YLABEL_ERRORS: [BuiltinErrorDescriptor; 3] = [
    YLABEL_ERROR_INVALID_ARGUMENT,
    YLABEL_ERROR_INTERNAL,
    YLABEL_ERROR_TOO_MANY_OUTPUTS,
];
pub const YLABEL_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLABEL.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ylabel:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "ylabel: too many output arguments",
};

pub const YLABEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &YLABEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &YLABEL_ERRORS,
};
