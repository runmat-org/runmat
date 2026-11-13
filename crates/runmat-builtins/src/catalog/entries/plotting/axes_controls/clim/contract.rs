use crate::*;

const CLIM_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Displayed two-element color limit vector [cmin cmax], including automatically selected bounds.",
}];

const CLIM_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const CLIM_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const CLIM_INPUTS_LIMIT_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element numeric vector [cmin cmax].",
}];

const CLIM_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Color-limit mode string ('auto' or 'manual').",
}];

const CLIM_TARGET: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "axes",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existing axes handle; does not change the current axes.",
};
const CLIM_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [CLIM_TARGET];
const CLIM_INPUTS_TARGET_LIMITS: [BuiltinParamDescriptor; 2] = [
    CLIM_TARGET,
    BuiltinParamDescriptor {
        name: "limits",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element numeric vector [cmin cmax].",
    },
];
const CLIM_INPUTS_TARGET_MODE: [BuiltinParamDescriptor; 2] = [
    CLIM_TARGET,
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Automatic or manual limit mode token.",
    },
];

const CLIM_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "limits = clim()",
        inputs: &CLIM_INPUTS_NONE,
        outputs: &CLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "clim([cmin cmax])",
        inputs: &CLIM_INPUTS_LIMIT_VECTOR,
        outputs: &CLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "clim('auto'|'manual')",
        inputs: &CLIM_INPUTS_MODE,
        outputs: &CLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "limits = clim(axes)",
        inputs: &CLIM_INPUTS_TARGET,
        outputs: &CLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "clim(axes,[cmin cmax])",
        inputs: &CLIM_INPUTS_TARGET_LIMITS,
        outputs: &CLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "clim(axes,'auto'|'manual')",
        inputs: &CLIM_INPUTS_TARGET_MODE,
        outputs: &CLIM_OUTPUTS_NONE,
    },
];

const CLIM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLIM.INVALID_ARGUMENT",
    identifier: Some("RunMat:clim:InvalidArgument"),
    when: "Argument count, mode, or limit vector is invalid.",
    message: "clim: invalid argument",
};

const CLIM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLIM.INTERNAL",
    identifier: Some("RunMat:clim:Internal"),
    when: "Internal plotting state update fails.",
    message: "clim: internal operation failed",
};

const CLIM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CLIM_ERROR_INVALID_ARGUMENT,
    CLIM_ERROR_INTERNAL,
    CLIM_ERROR_TOO_MANY_OUTPUTS,
];

pub const CLIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLIM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLIM_ERRORS,
};

pub const CLIM_INTEGER_LIMITS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "clim-integer-limits",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "clim/caxis with typed integer limits is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ClimIntegerLimitsExtension"),
};

pub const CLIM_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CLIM_INTEGER_LIMITS_EXTENSION,
    AXES_CONTROL_SETTER_OUTPUT_EXTENSION,
];

const CLIM_INTEGER_LIMIT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "limits",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Current public limits accept only single or double; RunMat mode admits every integer class before one explicit host f64 graphics-state conversion.",
    }];

pub const CLIM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "clim(integer_limits)",
        inputs: &CLIM_INTEGER_LIMIT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The two-element vector is shape- and order-validated from authoritative integer storage, then stored and queried as double host graphics metadata.",
    }];
pub const CLIM_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLIM.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:clim:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "clim: at most one output is supported",
};
