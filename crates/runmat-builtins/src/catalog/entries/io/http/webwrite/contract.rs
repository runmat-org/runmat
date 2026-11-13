use crate::*;
pub const WEBWRITE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "webwrite-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "passing explicit gpuArray values to host-only webwrite is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WebwriteExplicitGpuInputExtension"),
    };
pub const WEBWRITE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [WEBWRITE_EXPLICIT_GPU_EXTENSION];

const POST_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "PostValue",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Integer scalars and vectors are form-encoded directly from authoritative storage, including full-width signed and unsigned values.",
}];
const DATA_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "data",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "A numeric scalar used as JSON data, or integer values nested in a documented structure or cell payload, are serialized as exact JSON integers.",
}];
const TIMEOUT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "options.Timeout",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "A positive integer timeout is bounded before conversion to the host HTTP duration.",
}];
pub const WEBWRITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "response = webwrite(url, post_name, integer_post_value, ...)",
        inputs: &POST_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Default comma-separated form encoding preserves every integer element exactly; response type is determined independently by the service and content options.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "response = webwrite(url, integer_data_or_nested_integer_data, options)",
        inputs: &DATA_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented JSON numeric data is serialized without binary64 materialization. Binary byte uploads remain a separate byte-range conversion contract.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "response = webwrite(url, data, options) with integer options.Timeout",
        inputs: &TIMEOUT_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The bounded timeout crosses the host duration boundary; explicit gpuArray arguments are separately gated before provider access while automatic residency gathers transparently.",
    },
];

const WEBWRITE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "response",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Decoded response payload from the remote endpoint.",
}];
const WEBWRITE_INPUTS_URL_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "url",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HTTP/HTTPS URL target.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Request payload value.",
    },
];
const WEBWRITE_INPUTS_URL_DATA_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "url",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HTTP/HTTPS URL target.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Request payload value.",
    },
    BuiltinParamDescriptor {
        name: "optionsStruct",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "weboptions struct or option struct literal.",
    },
];
const WEBWRITE_INPUTS_URL_DATA_NAME_VALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "url",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HTTP/HTTPS URL target.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Request payload value.",
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
const WEBWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "response = webwrite(url, data)",
        inputs: &WEBWRITE_INPUTS_URL_DATA,
        outputs: &WEBWRITE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "response = webwrite(url, data, optionsStruct)",
        inputs: &WEBWRITE_INPUTS_URL_DATA_OPTIONS,
        outputs: &WEBWRITE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "response = webwrite(url, data, name, value, ...)",
        inputs: &WEBWRITE_INPUTS_URL_DATA_NAME_VALUE,
        outputs: &WEBWRITE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "response = webwrite(url, data, optionsStruct, name, value, ...)",
        inputs: &WEBWRITE_INPUTS_URL_DATA_NAME_VALUE,
        outputs: &WEBWRITE_OUTPUT,
    },
];

pub const WEBWRITE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBWRITE.INVALID_ARGUMENT",
    identifier: Some("RunMat:webwrite:InvalidArgument"),
    when: "Argument type/shape does not match webwrite call contract.",
    message: "webwrite: invalid argument",
};
pub const WEBWRITE_ERROR_INVALID_URL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBWRITE.INVALID_URL",
    identifier: Some("RunMat:webwrite:InvalidUrl"),
    when: "URL is empty or malformed.",
    message: "webwrite: invalid URL",
};
pub const WEBWRITE_ERROR_MISSING_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBWRITE.MISSING_DATA",
    identifier: Some("RunMat:webwrite:MissingData"),
    when: "Required data argument is missing.",
    message: "webwrite: missing data argument",
};
pub const WEBWRITE_ERROR_MISSING_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBWRITE.MISSING_OPTION_VALUE",
    identifier: Some("RunMat:webwrite:MissingOptionValue"),
    when: "A name-value option key has no value.",
    message: "webwrite: missing option value",
};
pub const WEBWRITE_ERROR_INVALID_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBWRITE.INVALID_OPTION_VALUE",
    identifier: Some("RunMat:webwrite:InvalidOptionValue"),
    when: "An option value fails validation.",
    message: "webwrite: invalid option value",
};
pub const WEBWRITE_ERROR_INVALID_CREDENTIALS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBWRITE.INVALID_CREDENTIALS",
    identifier: Some("RunMat:webwrite:InvalidCredentials"),
    when: "Password is provided without username.",
    message: "webwrite: invalid credentials",
};
pub const WEBWRITE_ERROR_TRANSPORT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBWRITE.TRANSPORT",
    identifier: Some("RunMat:webwrite:Transport"),
    when: "HTTP transport fails.",
    message: "webwrite: transport failure",
};
pub const WEBWRITE_ERROR_RESPONSE_JSON: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBWRITE.RESPONSE_JSON",
    identifier: Some("RunMat:webwrite:ResponseJson"),
    when: "Response body cannot be decoded as JSON.",
    message: "webwrite: failed to parse JSON response",
};
pub const WEBWRITE_ERROR_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBWRITE.OUTPUT",
    identifier: Some("RunMat:webwrite:Output"),
    when: "Output payload cannot be materialized.",
    message: "webwrite: output materialization failure",
};
pub const WEBWRITE_ERROR_FLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBWRITE.FLOW",
    identifier: Some("RunMat:webwrite:Flow"),
    when: "Nested flow fails while gathering inputs or nested builtin calls.",
    message: "webwrite: flow failure",
};

pub const WEBWRITE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBWRITE.OUTPUTS",
    identifier: Some("RunMat:webwrite:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "webwrite: too many output arguments",
};

const WEBWRITE_ERRORS: [BuiltinErrorDescriptor; 11] = [
    WEBWRITE_ERROR_INVALID_ARGUMENT,
    WEBWRITE_ERROR_INVALID_URL,
    WEBWRITE_ERROR_MISSING_DATA,
    WEBWRITE_ERROR_MISSING_OPTION_VALUE,
    WEBWRITE_ERROR_INVALID_OPTION_VALUE,
    WEBWRITE_ERROR_INVALID_CREDENTIALS,
    WEBWRITE_ERROR_TRANSPORT,
    WEBWRITE_ERROR_RESPONSE_JSON,
    WEBWRITE_ERROR_OUTPUT,
    WEBWRITE_ERROR_FLOW,
    WEBWRITE_ERROR_OUTPUTS,
];

pub const WEBWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WEBWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WEBWRITE_ERRORS,
};
