use crate::*;
pub const WEBREAD_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "webread-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "passing explicit gpuArray values to host-only webread is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WebreadExplicitGpuInputExtension"),
};
pub const WEBREAD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [WEBREAD_EXPLICIT_GPU_EXTENSION];

const QUERY_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "QueryValue",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Integer scalars and vectors are rendered directly from authoritative storage, including full-width int64 and uint64 values.",
}];
const TIMEOUT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "options.Timeout",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "A documented positive numeric timeout is bounded before conversion to the host HTTP duration.",
}];
pub const WEBREAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "data = webread(url, query_name, integer_query_value, ...)",
        inputs: &QUERY_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The documented numeric query value can be scalar or vector. Default comma-separated encoding preserves each integer exactly; response type is determined independently by the service and content options.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "data = webread(url, ..., options) with integer options.Timeout",
        inputs: &TIMEOUT_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The bounded timeout crosses the host duration boundary; explicit gpuArray arguments are separately gated before provider access while automatic residency gathers transparently.",
    },
];

const WEBREAD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Downloaded payload decoded as JSON, text, or binary tensor.",
}];
const WEBREAD_INPUTS_URL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "url",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HTTP/HTTPS URL to fetch.",
}];
const WEBREAD_INPUTS_URL_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "url",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HTTP/HTTPS URL to fetch.",
    },
    BuiltinParamDescriptor {
        name: "optionsStruct",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "weboptions struct or option struct literal.",
    },
];
const WEBREAD_INPUTS_URL_QUERY_CELL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "url",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HTTP/HTTPS URL to fetch.",
    },
    BuiltinParamDescriptor {
        name: "queryParameters",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-column cell array of query parameter names and values.",
    },
];
const WEBREAD_INPUTS_URL_NAME_VALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "url",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HTTP/HTTPS URL to fetch.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option or query parameter name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option or query parameter value.",
    },
];
const WEBREAD_INPUTS_URL_OPTIONS_NAME_VALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "url",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HTTP/HTTPS URL to fetch.",
    },
    BuiltinParamDescriptor {
        name: "optionsStruct",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "weboptions struct or option struct literal.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option or query parameter name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option or query parameter value.",
    },
];

const WEBREAD_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "data = webread(url)",
        inputs: &WEBREAD_INPUTS_URL,
        outputs: &WEBREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "data = webread(url, optionsStruct)",
        inputs: &WEBREAD_INPUTS_URL_OPTIONS,
        outputs: &WEBREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "data = webread(url, queryParameters)",
        inputs: &WEBREAD_INPUTS_URL_QUERY_CELL,
        outputs: &WEBREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "data = webread(url, name, value, ...)",
        inputs: &WEBREAD_INPUTS_URL_NAME_VALUE,
        outputs: &WEBREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "data = webread(url, optionsStruct, name, value, ...)",
        inputs: &WEBREAD_INPUTS_URL_OPTIONS_NAME_VALUE,
        outputs: &WEBREAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "data = webread(url, queryParameters, name, value, ...)",
        inputs: &WEBREAD_INPUTS_URL_OPTIONS_NAME_VALUE,
        outputs: &WEBREAD_OUTPUT,
    },
];

pub const WEBREAD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBREAD.INVALID_ARGUMENT",
    identifier: Some("RunMat:webread:InvalidArgument"),
    when: "Argument type/shape does not match webread call contract.",
    message: "webread: invalid argument",
};
pub const WEBREAD_ERROR_INVALID_URL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBREAD.INVALID_URL",
    identifier: Some("RunMat:webread:InvalidUrl"),
    when: "URL input is empty or invalid.",
    message: "webread: invalid URL",
};
pub const WEBREAD_ERROR_MISSING_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBREAD.MISSING_OPTION_VALUE",
    identifier: Some("RunMat:webread:MissingOptionValue"),
    when: "A name-value key has no corresponding value.",
    message: "webread: missing option value",
};
pub const WEBREAD_ERROR_INVALID_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBREAD.INVALID_OPTION_VALUE",
    identifier: Some("RunMat:webread:InvalidOptionValue"),
    when: "An option value fails validation.",
    message: "webread: invalid option value",
};
pub const WEBREAD_ERROR_INVALID_CREDENTIALS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBREAD.INVALID_CREDENTIALS",
    identifier: Some("RunMat:webread:InvalidCredentials"),
    when: "Password is provided without username.",
    message: "webread: invalid credentials",
};
pub const WEBREAD_ERROR_TRANSPORT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBREAD.TRANSPORT",
    identifier: Some("RunMat:webread:Transport"),
    when: "HTTP transport fails.",
    message: "webread: transport failure",
};
pub const WEBREAD_ERROR_RESPONSE_JSON: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBREAD.RESPONSE_JSON",
    identifier: Some("RunMat:webread:ResponseJson"),
    when: "Response body cannot be decoded as JSON.",
    message: "webread: failed to parse JSON response",
};
pub const WEBREAD_ERROR_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBREAD.OUTPUT",
    identifier: Some("RunMat:webread:Output"),
    when: "Output payload cannot be materialized.",
    message: "webread: output materialization failure",
};
pub const WEBREAD_ERROR_FLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBREAD.FLOW",
    identifier: Some("RunMat:webread:Flow"),
    when: "Nested flow fails while gathering inputs.",
    message: "webread: flow failure",
};

pub const WEBREAD_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBREAD.OUTPUTS",
    identifier: Some("RunMat:webread:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "webread: too many output arguments",
};

const WEBREAD_ERRORS: [BuiltinErrorDescriptor; 10] = [
    WEBREAD_ERROR_INVALID_ARGUMENT,
    WEBREAD_ERROR_INVALID_URL,
    WEBREAD_ERROR_MISSING_OPTION_VALUE,
    WEBREAD_ERROR_INVALID_OPTION_VALUE,
    WEBREAD_ERROR_INVALID_CREDENTIALS,
    WEBREAD_ERROR_TRANSPORT,
    WEBREAD_ERROR_RESPONSE_JSON,
    WEBREAD_ERROR_OUTPUT,
    WEBREAD_ERROR_FLOW,
    WEBREAD_ERROR_OUTPUTS,
];

pub const WEBREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WEBREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WEBREAD_ERRORS,
};
