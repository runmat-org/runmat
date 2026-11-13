use crate::*;

pub const INTEGER_ARRAY_FORMATTING_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "warning-integer-array-formatting",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "warning with a nonscalar typed-integer formatting argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WarningIntegerArrayFormattingExtension"),
    };
pub const WARNING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [INTEGER_ARRAY_FORMATTING_EXTENSION];

const INTEGER_SCALAR_FORMAT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric-scalar replacement value includes every built-in integer class; integer conversions format the authoritative value directly.",
    }];
const INTEGER_ARRAY_FORMAT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat can consume a host integer array across successive format conversions; this exceeds the warning page's numeric-scalar A contract and is independently gated.",
    }];
pub const WARNING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "warning(msg, integer_A, ...)",
        inputs: &INTEGER_SCALAR_FORMAT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Signed and unsigned decimal conversions preserve exact values, including uint64 values above flintmax; warning state and the success sentinel remain host metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "warning(msg, integer_array_A, ...)",
        inputs: &INTEGER_ARRAY_FORMAT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The extension gate runs before formatting or warning-state mutation; host values are flattened in column-major order and retain exact integer formatting.",
    },
];

const WARNING_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "state_or_status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric success sentinel or state/status struct/cell/string result.",
}];

const WARNING_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const WARNING_INPUTS_MESSAGE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "message",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Warning message text or command token.",
}];
const WARNING_INPUTS_MESSAGE_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Warning message template text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the warning message template.",
    },
];
const WARNING_INPUTS_IDENTIFIER_MESSAGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "message_id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"RunMat:warning\""),
        description: "Warning identifier.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Warning message text.",
    },
];
const WARNING_INPUTS_IDENTIFIER_MESSAGE_VARIADIC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "message_id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"RunMat:warning\""),
        description: "Warning identifier.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Warning message template text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the warning message template.",
    },
];
const WARNING_INPUTS_STATE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "state",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State struct/cell snapshot to restore.",
}];
const WARNING_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Mode token ('on','off','once','error','default','reset','query','status','backtrace').",
}];
const WARNING_INPUTS_MODE_TARGET: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mode token.",
    },
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"all\""),
        description: "Identifier/special target for mode updates or queries.",
    },
];
const WARNING_INPUTS_BACKTRACE_STATE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "command",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"backtrace\""),
        description: "Backtrace command token.",
    },
    BuiltinParamDescriptor {
        name: "state",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"off\""),
        description: "Backtrace state ('on' or 'off').",
    },
];

const WARNING_SIGNATURES: [BuiltinSignatureDescriptor; 17] = [
    BuiltinSignatureDescriptor {
        label: "state = warning()",
        inputs: &WARNING_INPUTS_NONE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(message)",
        inputs: &WARNING_INPUTS_MESSAGE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(message, A...)",
        inputs: &WARNING_INPUTS_MESSAGE_VARIADIC,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(message_id, message)",
        inputs: &WARNING_INPUTS_IDENTIFIER_MESSAGE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(message_id, message, A...)",
        inputs: &WARNING_INPUTS_IDENTIFIER_MESSAGE_VARIADIC,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(state)",
        inputs: &WARNING_INPUTS_STATE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(mode)",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(mode, target)",
        inputs: &WARNING_INPUTS_MODE_TARGET,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"default\")",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"default\", target)",
        inputs: &WARNING_INPUTS_MODE_TARGET,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"reset\")",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"query\")",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"query\", target)",
        inputs: &WARNING_INPUTS_MODE_TARGET,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"status\")",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"backtrace\")",
        inputs: &WARNING_INPUTS_MODE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(\"backtrace\", state)",
        inputs: &WARNING_INPUTS_BACKTRACE_STATE,
        outputs: &WARNING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "state = warning(mex)",
        inputs: &WARNING_INPUTS_STATE,
        outputs: &WARNING_OUTPUT,
    },
];

pub const WARNING_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WARNING.INVALID_INPUT",
    identifier: Some("RunMat:warning"),
    when: "Arguments are invalid for the warning parser branch or command contract.",
    message: "warning: invalid input arguments",
};

pub const WARNING_ERROR_PROMOTED_TO_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WARNING.PROMOTED_TO_ERROR",
    identifier: Some("RunMat:warning"),
    when: "Warning mode is configured to promote warnings to errors.",
    message: "warning: promoted to error",
};

const WARNING_ERRORS: [BuiltinErrorDescriptor; 2] =
    [WARNING_ERROR_INVALID_INPUT, WARNING_ERROR_PROMOTED_TO_ERROR];

pub const WARNING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WARNING_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WARNING_ERRORS,
};
