use crate::*;

pub const COMPOSE_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "compose-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "compose with resident array arguments is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ComposeResidentInputExtension"),
    };

pub const COMPOSE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [COMPOSE_RESIDENT_INPUT_EXTENSION];

const COMPOSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A...",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are formatted directly from native storage, including exact int64 and uint64 values above flintmax.",
}];

pub const COMPOSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "S = compose(formatSpec, integer_A...)",
    inputs: &COMPOSE_INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Integer substitution is textual and exact; the format-spec family selects a host string array or cell array of character vectors. Resident arguments are a separately gated RunMat extension and gather before formatting.",
}];

const COMPOSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Formatted string array or cell array of character vectors, selected by formatSpec.",
}];

const COMPOSE_INPUT_NO_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "formatSpec",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Format text or array returned in the corresponding string or cellstr family when no data arguments are supplied.",
}];

const COMPOSE_INPUT_WITH_ARGS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "formatSpec",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Format template text.",
    },
    BuiltinParamDescriptor {
        name: "A...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Values substituted into formatSpec placeholders.",
    },
];

const COMPOSE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "S = compose(formatSpec)",
        inputs: &COMPOSE_INPUT_NO_ARGS,
        outputs: &COMPOSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = compose(formatSpec, A...)",
        inputs: &COMPOSE_INPUT_WITH_ARGS,
        outputs: &COMPOSE_OUTPUT,
    },
];

pub const COMPOSE_ERROR_INVALID_FORMAT_SPEC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPOSE.INVALID_FORMAT_SPEC",
    identifier: Some("RunMat:compose:InvalidFormatSpec"),
    when: "formatSpec is not valid text input for compose formatting.",
    message: "compose: invalid formatSpec",
};

pub const COMPOSE_ERROR_ARGUMENT_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPOSE.ARGUMENT_MISMATCH",
    identifier: Some("RunMat:compose:ArgumentMismatch"),
    when: "Data arguments are not scalar or broadcast-compatible with formatSpec.",
    message: "compose: format data arguments must be scalars or match formatSpec size",
};

pub const COMPOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPOSE.INTERNAL",
    identifier: Some("RunMat:compose:InternalError"),
    when: "Internal string-array construction failed.",
    message: "compose: internal error",
};

pub const COMPOSE_ERROR_INVALID_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPOSE.INVALID_DATA",
    identifier: Some("RunMat:compose:InvalidData"),
    when: "A data argument is outside compose's documented numeric, logical, character, or string domain.",
    message: "compose: unsupported data argument",
};

const COMPOSE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    COMPOSE_ERROR_INVALID_FORMAT_SPEC,
    COMPOSE_ERROR_ARGUMENT_MISMATCH,
    COMPOSE_ERROR_INTERNAL,
    COMPOSE_ERROR_INVALID_DATA,
    COMPOSE_ERROR_TOO_MANY_OUTPUTS,
];

pub const COMPOSE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPOSE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:compose:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "compose: too many output arguments",
};

pub const COMPOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COMPOSE_ERRORS,
};
