use crate::*;
pub const CAXIS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CLIM_INTEGER_LIMITS_EXTENSION,
    AXES_CONTROL_SETTER_OUTPUT_EXTENSION,
];

const CAXIS_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Displayed two-element color limit vector [cmin cmax], including automatically selected bounds.",
}];

const CAXIS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const CAXIS_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const CAXIS_INPUTS_LIMIT_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element numeric vector [cmin cmax].",
}];

const CAXIS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Color-limit mode string ('auto' or 'manual').",
}];

const CAXIS_TARGET: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "axes",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existing axes handle; does not change the current axes.",
};
const CAXIS_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [CAXIS_TARGET];
const CAXIS_INPUTS_TARGET_LIMITS: [BuiltinParamDescriptor; 2] = [
    CAXIS_TARGET,
    BuiltinParamDescriptor {
        name: "limits",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element numeric vector [cmin cmax].",
    },
];
const CAXIS_INPUTS_TARGET_MODE: [BuiltinParamDescriptor; 2] = [
    CAXIS_TARGET,
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Automatic or manual limit mode token.",
    },
];

const CAXIS_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "limits = caxis()",
        inputs: &CAXIS_INPUTS_NONE,
        outputs: &CAXIS_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "caxis([cmin cmax])",
        inputs: &CAXIS_INPUTS_LIMIT_VECTOR,
        outputs: &CAXIS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "caxis('auto'|'manual')",
        inputs: &CAXIS_INPUTS_MODE,
        outputs: &CAXIS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "limits = caxis(axes)",
        inputs: &CAXIS_INPUTS_TARGET,
        outputs: &CAXIS_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "caxis(axes,[cmin cmax])",
        inputs: &CAXIS_INPUTS_TARGET_LIMITS,
        outputs: &CAXIS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "caxis(axes,'auto'|'manual')",
        inputs: &CAXIS_INPUTS_TARGET_MODE,
        outputs: &CAXIS_OUTPUTS_NONE,
    },
];

const CAXIS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAXIS.INVALID_ARGUMENT",
    identifier: Some("RunMat:caxis:InvalidArgument"),
    when: "Argument count, mode, or limit vector is invalid.",
    message: "caxis: invalid argument",
};

const CAXIS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAXIS.INTERNAL",
    identifier: Some("RunMat:caxis:Internal"),
    when: "Internal plotting state update fails.",
    message: "caxis: internal operation failed",
};

const CAXIS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CAXIS_ERROR_INVALID_ARGUMENT,
    CAXIS_ERROR_INTERNAL,
    CAXIS_ERROR_TOO_MANY_OUTPUTS,
];

pub const CAXIS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CAXIS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CAXIS_ERRORS,
};

pub const CAXIS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::AliasOf,
    canonical_builtin: Some("clim"),
    notes: "Current public documentation identifies caxis as the legacy name for clim; both names share the same RunMat-only typed-integer limit extension and host-double graphics-state boundary.",
};
pub const CAXIS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAXIS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:caxis:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "caxis: at most one output is supported",
};
