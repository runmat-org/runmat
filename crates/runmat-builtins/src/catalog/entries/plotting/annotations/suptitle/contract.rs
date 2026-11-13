use crate::*;
pub const SUPTITLE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::AliasOf,
    canonical_builtin: Some("sgtitle"),
    notes: "suptitle delegates directly to sgtitle, which owns the integer title, handle-alias, and FontSize semantics.",
};

const SUPTITLE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated super-title object.",
}];

const SUPTITLE_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Super-title text (string/char/cellstr-like multiline forms; numeric scalars also accepted).",
}];

const SUPTITLE_INPUTS_FIG_TEXT: [BuiltinParamDescriptor; 2] = [
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

const SUPTITLE_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
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

const SUPTITLE_INPUTS_FIG_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
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

const SUPTITLE_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = suptitle(txt)",
        inputs: &SUPTITLE_INPUTS_TEXT,
        outputs: &SUPTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = suptitle(fig, txt)",
        inputs: &SUPTITLE_INPUTS_FIG_TEXT,
        outputs: &SUPTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = suptitle(txt, Name, Value, ...)",
        inputs: &SUPTITLE_INPUTS_TEXT_PROPS,
        outputs: &SUPTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = suptitle(fig, txt, Name, Value, ...)",
        inputs: &SUPTITLE_INPUTS_FIG_TEXT_PROPS,
        outputs: &SUPTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "suptitle(txt)",
        inputs: &SUPTITLE_INPUTS_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "suptitle(fig, txt)",
        inputs: &SUPTITLE_INPUTS_FIG_TEXT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "suptitle(txt, Name, Value, ...)",
        inputs: &SUPTITLE_INPUTS_TEXT_PROPS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "suptitle(fig, txt, Name, Value, ...)",
        inputs: &SUPTITLE_INPUTS_FIG_TEXT_PROPS,
        outputs: &[],
    },
];

pub const SUPTITLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUPTITLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:suptitle:InvalidArgument"),
    when: "Figure handle, text payload, or property/value arguments are invalid.",
    message: "suptitle: invalid argument",
};

pub const SUPTITLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUPTITLE.INTERNAL",
    identifier: Some("RunMat:suptitle:Internal"),
    when: "Internal plotting state update fails.",
    message: "suptitle: internal operation failed",
};

const SUPTITLE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SUPTITLE_ERROR_INVALID_ARGUMENT,
    SUPTITLE_ERROR_INTERNAL,
    SUPTITLE_ERROR_TOO_MANY_OUTPUTS,
];
pub const SUPTITLE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUPTITLE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:suptitle:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "suptitle: too many output arguments",
};

pub const SUPTITLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUPTITLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUPTITLE_ERRORS,
};
