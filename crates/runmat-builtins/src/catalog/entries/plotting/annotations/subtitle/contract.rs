use crate::*;
pub const SUBTITLE_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "subtitle-integer-axes-handle",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Allow typed-integer aliases for encoded axes handles",
        error_identifier: Some("RunMat:compatibility:SubtitleIntegerAxesHandleExtension"),
    };
pub const SUBTITLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SUBTITLE_INTEGER_AXES_EXTENSION];
const SUBTITLE_INTEGER_FONT_SIZE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FontSize",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents every native integer class for the FontSize property.",
    }];
const SUBTITLE_INTEGER_AXES: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat graphics handles are internally encoded numeric scalars; typed-integer aliases are separately gated.",
    }];
pub const SUBTITLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = subtitle(..., 'FontSize', integer_size)",
        inputs: &SUBTITLE_INTEGER_FONT_SIZE,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented property crosses the graphics font-size boundary after exact scalar classification; the returned text handle is a host double scalar.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = subtitle(integer_ax, txt, ...)",
        inputs: &SUBTITLE_INTEGER_AXES,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat mode parses the gated scalar exactly before resolving the encoded axes handle; strict mode rejects before graphics state access.",
    },
];

const SUBTITLE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated subtitle object.",
}];

const SUBTITLE_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Subtitle text (string/char/cellstr-like multiline forms).",
}];

const SUBTITLE_INPUTS_AX_TEXT: [BuiltinParamDescriptor; 2] = [
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
        description: "Subtitle text (string/char/cellstr-like multiline forms).",
    },
];

const SUBTITLE_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Subtitle text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const SUBTITLE_INPUTS_AX_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
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
        description: "Subtitle text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const SUBTITLE_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = subtitle(txt)",
        inputs: &SUBTITLE_INPUTS_TEXT,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = subtitle(ax, txt)",
        inputs: &SUBTITLE_INPUTS_AX_TEXT,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = subtitle(txt, Name, Value, ...)",
        inputs: &SUBTITLE_INPUTS_TEXT_PROPS,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = subtitle(ax, txt, Name, Value, ...)",
        inputs: &SUBTITLE_INPUTS_AX_TEXT_PROPS,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "subtitle(txt)",
        inputs: &SUBTITLE_INPUTS_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "subtitle(ax, txt)",
        inputs: &SUBTITLE_INPUTS_AX_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "subtitle(txt, Name, Value, ...)",
        inputs: &SUBTITLE_INPUTS_TEXT_PROPS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "subtitle(ax, txt, Name, Value, ...)",
        inputs: &SUBTITLE_INPUTS_AX_TEXT_PROPS,
        outputs: &[],
    },
];

pub const SUBTITLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBTITLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:subtitle:InvalidArgument"),
    when: "Axes handle, text payload, or property/value arguments are invalid.",
    message: "subtitle: invalid argument",
};

pub const SUBTITLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBTITLE.INTERNAL",
    identifier: Some("RunMat:subtitle:Internal"),
    when: "Internal plotting state update fails.",
    message: "subtitle: internal operation failed",
};

const SUBTITLE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SUBTITLE_ERROR_INVALID_ARGUMENT,
    SUBTITLE_ERROR_INTERNAL,
    SUBTITLE_ERROR_TOO_MANY_OUTPUTS,
];
pub const SUBTITLE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBTITLE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:subtitle:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "subtitle: too many output arguments",
};

pub const SUBTITLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBTITLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUBTITLE_ERRORS,
};
