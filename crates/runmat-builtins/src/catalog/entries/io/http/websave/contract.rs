use crate::*;
pub const WEBSAVE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WEBSAVE.OUTPUTS",
    identifier: Some("RunMat:websave:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "websave: too many output arguments",
};

pub const WEBSAVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "filename = websave(filename, url, query_name, query_value, ...)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "filename",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Output file path.",
            },
            BuiltinParamDescriptor {
                name: "url",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "HTTP or HTTPS URL.",
            },
            BuiltinParamDescriptor {
                name: "query_name",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Variadic,
                default: None,
                description: "Query or request-option name.",
            },
            BuiltinParamDescriptor {
                name: "query_value",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Variadic,
                default: None,
                description: "Query or request-option value.",
            },
        ],
        outputs: &[BuiltinParamDescriptor {
            name: "filename",
            ty: BuiltinParamType::StringScalar,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Saved file path.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[WEBSAVE_ERROR_OUTPUTS],
};
pub const WEBSAVE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "websave-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "passing explicit gpuArray values to host-only websave is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WebsaveExplicitGpuInputExtension"),
};
pub const WEBSAVE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [WEBSAVE_EXPLICIT_GPU_EXTENSION];
const WEBSAVE_QUERY_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "QueryValue",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer scalars and vectors are rendered directly from authoritative storage, including full-width signed and unsigned values.",
    }];
const WEBSAVE_TIMEOUT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "options.Timeout",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A positive integer timeout is bounded before conversion to the host HTTP duration.",
    }];
pub const WEBSAVE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "filename = websave(filename, url, query_name, integer_query_value, ...)",
        inputs: &WEBSAVE_QUERY_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Default comma-separated query encoding preserves every integer element exactly; the returned filename remains text.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "filename = websave(filename, url, ..., options) with integer options.Timeout",
        inputs: &WEBSAVE_TIMEOUT_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The bounded timeout crosses the host duration boundary; explicit gpuArray arguments are separately gated before provider access while automatic residency gathers transparently.",
    },
];
