use crate::*;
pub const XLABEL_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "xlabel-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow typed-integer aliases for encoded axes handles",
    error_identifier: Some("RunMat:compatibility:XlabelIntegerAxesHandleExtension"),
};
pub const XLABEL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [XLABEL_INTEGER_AXES_EXTENSION];

const XLABEL_INTEGER_TEXT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "txt",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "A scalar integer label is formatted from its exact signed or unsigned decimal value.",
    }];
const XLABEL_INTEGER_FONT_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FontSize",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented for FontSize; accepted values are positive finite practical renderer sizes.",
    }];
const XLABEL_INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for RunMat's encoded axes handles are separately gated.",
    }];
pub const XLABEL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = xlabel(integer_txt)", inputs: &XLABEL_INTEGER_TEXT_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "The exact decimal spelling becomes the stored String property; the graphics handle is returned as a double scalar." },
    BuiltinIntegerCapabilityDescriptor { form: "h = xlabel(..., 'FontSize', integer_size)", inputs: &XLABEL_INTEGER_FONT_SIZE_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "FontSize crosses the graphics scalar boundary after shared positive finite size validation." },
    BuiltinIntegerCapabilityDescriptor { form: "h = xlabel(integer_ax, txt, ...)", inputs: &XLABEL_INTEGER_AXES_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat mode parses the gated scalar before resolving the encoded axes handle; strict mode rejects before graphics state access." },
];

const XLABEL_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated xlabel object.",
}];

const XLABEL_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Label text (string/char/cellstr-like multiline forms).",
}];

const XLABEL_INPUTS_AX_TEXT: [BuiltinParamDescriptor; 2] = [
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

const XLABEL_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
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

const XLABEL_INPUTS_AX_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
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

const XLABEL_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = xlabel(txt)",
        inputs: &XLABEL_INPUTS_TEXT,
        outputs: &XLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = xlabel(ax, txt)",
        inputs: &XLABEL_INPUTS_AX_TEXT,
        outputs: &XLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = xlabel(txt, Name, Value, ...)",
        inputs: &XLABEL_INPUTS_TEXT_PROPS,
        outputs: &XLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = xlabel(ax, txt, Name, Value, ...)",
        inputs: &XLABEL_INPUTS_AX_TEXT_PROPS,
        outputs: &XLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "xlabel(txt)",
        inputs: &XLABEL_INPUTS_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "xlabel(ax, txt)",
        inputs: &XLABEL_INPUTS_AX_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "xlabel(txt, Name, Value, ...)",
        inputs: &XLABEL_INPUTS_TEXT_PROPS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "xlabel(ax, txt, Name, Value, ...)",
        inputs: &XLABEL_INPUTS_AX_TEXT_PROPS,
        outputs: &[],
    },
];

pub const XLABEL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLABEL.INVALID_ARGUMENT",
    identifier: Some("RunMat:xlabel:InvalidArgument"),
    when: "Axes handle, text payload, or property/value arguments are invalid.",
    message: "xlabel: invalid argument",
};

pub const XLABEL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLABEL.INTERNAL",
    identifier: Some("RunMat:xlabel:Internal"),
    when: "Internal plotting state update fails.",
    message: "xlabel: internal operation failed",
};

const XLABEL_ERRORS: [BuiltinErrorDescriptor; 3] = [
    XLABEL_ERROR_INVALID_ARGUMENT,
    XLABEL_ERROR_INTERNAL,
    XLABEL_ERROR_TOO_MANY_OUTPUTS,
];
pub const XLABEL_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLABEL.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:xlabel:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "xlabel: too many output arguments",
};

pub const XLABEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XLABEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XLABEL_ERRORS,
};
