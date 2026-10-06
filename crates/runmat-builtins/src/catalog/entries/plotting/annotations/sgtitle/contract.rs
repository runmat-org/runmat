use crate::*;
const SGTITLE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated super-title object.",
}];

const SGTITLE_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Super-title text (string/char/cellstr-like multiline forms; numeric scalars also accepted).",
}];

const SGTITLE_INPUTS_FIG_TEXT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fig",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target figure handle.",
    },
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Super-title text (string/char/cellstr-like multiline forms; numeric scalars also accepted).",
    },
];

const SGTITLE_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Super-title text (string/char/cellstr-like multiline forms; numeric scalars also accepted).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const SGTITLE_INPUTS_FIG_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fig",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target figure handle.",
    },
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Super-title text (string/char/cellstr-like multiline forms; numeric scalars also accepted).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const SGTITLE_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = sgtitle(txt)",
        inputs: &SGTITLE_INPUTS_TEXT,
        outputs: &SGTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = sgtitle(fig, txt)",
        inputs: &SGTITLE_INPUTS_FIG_TEXT,
        outputs: &SGTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = sgtitle(txt, Name, Value, ...)",
        inputs: &SGTITLE_INPUTS_TEXT_PROPS,
        outputs: &SGTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = sgtitle(fig, txt, Name, Value, ...)",
        inputs: &SGTITLE_INPUTS_FIG_TEXT_PROPS,
        outputs: &SGTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "sgtitle(txt)",
        inputs: &SGTITLE_INPUTS_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "sgtitle(fig, txt)",
        inputs: &SGTITLE_INPUTS_FIG_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "sgtitle(txt, Name, Value, ...)",
        inputs: &SGTITLE_INPUTS_TEXT_PROPS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "sgtitle(fig, txt, Name, Value, ...)",
        inputs: &SGTITLE_INPUTS_FIG_TEXT_PROPS,
        outputs: &[],
    },
];

const SGTITLE_INTEGER_TEXT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "txt",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A scalar integer title is formatted directly from its native signed or unsigned value, including full-width uint64, without binary64 conversion.",
    }];
const SGTITLE_INTEGER_FONT_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FontSize",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented for FontSize; accepted positive finite values cross the explicit client-renderer scalar boundary.",
    }];
pub const SGTITLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = sgtitle(integer_txt)", inputs: &SGTITLE_INTEGER_TEXT_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "The exact decimal spelling becomes the stored text String property; the returned graphics handle remains RunMat's ordinary double handle." },
    BuiltinIntegerCapabilityDescriptor { form: "h = sgtitle(..., 'FontSize', integer_size)", inputs: &SGTITLE_INTEGER_FONT_SIZE_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "FontSize is a client-side graphics property, not an integer-preserving result; valid practical sizes are exactly representable at the renderer boundary." },
];

pub const SGTITLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SGTITLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:sgtitle:InvalidArgument"),
    when: "Figure handle, text payload, or property/value arguments are invalid.",
    message: "sgtitle: invalid argument",
};

pub const SGTITLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SGTITLE.INTERNAL",
    identifier: Some("RunMat:sgtitle:Internal"),
    when: "Internal plotting state update fails.",
    message: "sgtitle: internal operation failed",
};

const SGTITLE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SGTITLE_ERROR_INVALID_ARGUMENT,
    SGTITLE_ERROR_INTERNAL,
    SGTITLE_ERROR_TOO_MANY_OUTPUTS,
];
pub const SGTITLE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SGTITLE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:sgtitle:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "sgtitle: too many output arguments",
};

pub const SGTITLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SGTITLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SGTITLE_ERRORS,
};
