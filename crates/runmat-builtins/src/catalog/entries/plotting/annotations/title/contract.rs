use crate::*;
const TITLE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated title object.",
}];

const TITLE_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Title text (string/char/cellstr-like multiline forms).",
}];

const TITLE_INPUTS_AX_TEXT: [BuiltinParamDescriptor; 2] = [
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
        description: "Title text (string/char/cellstr-like multiline forms).",
    },
];

const TITLE_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Title text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const TITLE_INPUTS_AX_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
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
        description: "Title text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const TITLE_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = title(txt)",
        inputs: &TITLE_INPUTS_TEXT,
        outputs: &TITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = title(ax, txt)",
        inputs: &TITLE_INPUTS_AX_TEXT,
        outputs: &TITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = title(txt, Name, Value, ...)",
        inputs: &TITLE_INPUTS_TEXT_PROPS,
        outputs: &TITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = title(ax, txt, Name, Value, ...)",
        inputs: &TITLE_INPUTS_AX_TEXT_PROPS,
        outputs: &TITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "title(txt)",
        inputs: &TITLE_INPUTS_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "title(ax, txt)",
        inputs: &TITLE_INPUTS_AX_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "title(txt, Name, Value, ...)",
        inputs: &TITLE_INPUTS_TEXT_PROPS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "title(ax, txt, Name, Value, ...)",
        inputs: &TITLE_INPUTS_AX_TEXT_PROPS,
        outputs: &[],
    },
];

const TITLE_INTEGER_TEXT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "txt",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "A scalar integer title is formatted from its exact signed or unsigned decimal value.",
    }];
const TITLE_INTEGER_FONT_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FontSize",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented for FontSize; accepted values are positive finite practical renderer sizes.",
    }];
pub const TITLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = title(integer_txt)", inputs: &TITLE_INTEGER_TEXT_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "The exact decimal spelling becomes the stored String property; the graphics handle remains an ordinary double handle." },
    BuiltinIntegerCapabilityDescriptor { form: "h = title(..., 'FontSize', integer_size)", inputs: &TITLE_INTEGER_FONT_SIZE_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "FontSize crosses the client-renderer scalar boundary only after the shared positive finite size validation." },
];

pub const TITLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TITLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:title:InvalidArgument"),
    when: "Axes handle, text payload, or property/value arguments are invalid.",
    message: "title: invalid argument",
};

pub const TITLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TITLE.INTERNAL",
    identifier: Some("RunMat:title:Internal"),
    when: "Internal plotting state update fails.",
    message: "title: internal operation failed",
};

const TITLE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    TITLE_ERROR_INVALID_ARGUMENT,
    TITLE_ERROR_INTERNAL,
    TITLE_ERROR_TOO_MANY_OUTPUTS,
];
pub const TITLE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TITLE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:title:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "title: too many output arguments",
};

pub const TITLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TITLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TITLE_ERRORS,
};
