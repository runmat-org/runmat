use crate::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or closure to invoke under an execution-owned tape.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Arguments forwarded without an implicit provider download.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Requested outputs returned by the callback.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "varargout = dlfeval(fun, args...)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const DLFEVAL_INVALID_FUNCTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "The first argument is absent or is not a function handle or closure.",
    message: "dlfeval requires a function handle",
};
pub const DLFEVAL_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.UNSUPPORTED",
    identifier: Some("RunMat:deepLearning:Unsupported"),
    when: "A provider-resident dlarray must be traced without resident tape kernels.",
    message: "provider-resident automatic differentiation is unavailable",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [DLFEVAL_INVALID_FUNCTION, DLFEVAL_UNSUPPORTED];
pub const DLFEVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_ARGS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "args",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer arguments are forwarded intact to the callback. A callback that constructs dlarray Data or differentiates integers applies its own narrower contract.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "dlfeval(fun, integer_args...)",
        inputs: &INTEGER_ARGS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "dlfeval does not itself compute over integer values; it preserves callback argument and output classes.",
    }];
