use crate::*;

pub const WEBOPTIONS_STRUCT_COPY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "weboptions-struct-copy",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "copying and overriding an existing weboptions-compatible struct is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WeboptionsStructCopyExtension"),
    };
pub const WEBOPTIONS_QUERY_PARAMETERS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "weboptions-query-parameters",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "the QueryParameters property on weboptions is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WeboptionsQueryParametersExtension"),
    };
pub const WEBOPTIONS_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "weboptions-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "passing explicit gpuArray values to host-only weboptions is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WeboptionsExplicitGpuInputExtension"),
    };
pub const WEBOPTIONS_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    WEBOPTIONS_STRUCT_COPY_EXTENSION,
    WEBOPTIONS_QUERY_PARAMETERS_EXTENSION,
    WEBOPTIONS_EXPLICIT_GPU_EXTENSION,
];

const TIMEOUT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Timeout",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Every integer class is accepted as a positive scalar timeout when its value lies within the documented timeout range.",
}];
const KEY_VALUE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "KeyValue",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "A scalar key value retains its exact signedness, width, and value in the options object; webread/webwrite serialize native integers directly to HTTP header decimal text.",
}];
pub const WEBOPTIONS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "options = weboptions('Timeout', integer_seconds)",
        inputs: &TIMEOUT_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The documented positive numeric scalar crosses a bounded duration boundary. Explicit gpuArray values are separately gated before gather; automatic residency gathers transparently.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "options = weboptions('KeyName', name, 'KeyValue', integer_value)",
        inputs: &KEY_VALUE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The documented numeric key value remains exact in the host options object and is rendered directly from authoritative integer storage by request consumers.",
    },
];

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host HTTP options struct for webread, webwrite, and websave.",
}];
const NONE: [BuiltinParamDescriptor; 0] = [];
const STRUCT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "optionsStruct",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existing options struct to copy and override (RunMat extension).",
}];
const PAIRS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Case-insensitive option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option field value.",
    },
];
const STRUCT_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "optionsStruct",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing options struct to copy and override (RunMat extension).",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Case-insensitive option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option field value.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "options = weboptions()",
        inputs: &NONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = weboptions(optionsStruct)",
        inputs: &STRUCT,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = weboptions(name, value, ...)",
        inputs: &PAIRS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "options = weboptions(optionsStruct, name, value, ...)",
        inputs: &STRUCT_PAIRS,
        outputs: &OUTPUT,
    },
];

pub const WEBOPTIONS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.INVALID_ARGUMENT",
    identifier: Some("RunMat:weboptions:InvalidArgument"),
    when: "Argument shape/type does not match supported weboptions forms.",
    message: "weboptions: invalid argument",
};
pub const WEBOPTIONS_ERROR_INVALID_OPTION_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.INVALID_OPTION_NAME",
    identifier: Some("RunMat:weboptions:InvalidOptionName"),
    when: "Name-value option key is missing or not text scalar.",
    message: "weboptions: invalid option name",
};
pub const WEBOPTIONS_ERROR_MISSING_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.MISSING_OPTION_VALUE",
    identifier: Some("RunMat:weboptions:MissingOptionValue"),
    when: "A name-value key is not followed by a value.",
    message: "weboptions: missing option value",
};
pub const WEBOPTIONS_ERROR_INVALID_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.INVALID_OPTION_VALUE",
    identifier: Some("RunMat:weboptions:InvalidOptionValue"),
    when: "An option value fails type or domain validation.",
    message: "weboptions: invalid option value",
};
pub const WEBOPTIONS_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.UNKNOWN_OPTION",
    identifier: Some("RunMat:weboptions:UnknownOption"),
    when: "Name-value key does not map to a supported option.",
    message: "weboptions: unknown option",
};
pub const WEBOPTIONS_ERROR_INVALID_CREDENTIALS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.INVALID_CREDENTIALS",
    identifier: Some("RunMat:weboptions:InvalidCredentials"),
    when: "Password is provided without a username.",
    message: "weboptions: invalid credentials",
};
pub const WEBOPTIONS_ERROR_FLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.FLOW",
    identifier: Some("RunMat:weboptions:Flow"),
    when: "Nested flow fails while gathering input values.",
    message: "weboptions: flow failure",
};
pub const WEBOPTIONS_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBOPTIONS.OUTPUTS",
    identifier: Some("RunMat:weboptions:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "weboptions: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 8] = [
    WEBOPTIONS_ERROR_INVALID_ARGUMENT,
    WEBOPTIONS_ERROR_INVALID_OPTION_NAME,
    WEBOPTIONS_ERROR_MISSING_OPTION_VALUE,
    WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
    WEBOPTIONS_ERROR_UNKNOWN_OPTION,
    WEBOPTIONS_ERROR_INVALID_CREDENTIALS,
    WEBOPTIONS_ERROR_FLOW,
    WEBOPTIONS_ERROR_OUTPUTS,
];
pub const WEBOPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
