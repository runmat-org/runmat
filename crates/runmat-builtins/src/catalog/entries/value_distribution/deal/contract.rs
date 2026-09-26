use crate::*;

pub const DEAL_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "deal-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "deal with a resident input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DealResidentInputExtension"),
};

pub const DEAL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DEAL_RESIDENT_INPUT_EXTENSION];

const DEAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Distributed output values.",
}];

const DEAL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargin",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Input values to distribute.",
}];

const DEAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = deal(varargin)",
    inputs: &DEAL_INPUTS,
    outputs: &DEAL_OUTPUT,
}];

const DEAL_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.DEAL.INPUT_OUTPUT_COUNT_MISMATCH",
    identifier: Some("RunMat:deal:InputOutputCountMismatch"),
    when: "The call has multiple inputs and the requested output count differs from the input count, or has no input and a nonzero output count.",
    message: "deal: the number of outputs must match the number of inputs unless there is exactly one input",
}];

pub const DEAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DEAL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DEAL_ERRORS,
};

const DEAL_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "deal copies input arrays to outputs without numeric conversion; every integer class, shape, and value is preserved.",
}];

const DEAL_RESIDENT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident integer A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode passes resident integer handles through structurally without gather, allocation, provider dispatch, or floating conversion.",
    }];

pub const DEAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[B1,...,Bn] = deal(integer_A1,...,integer_An) or deal(integer_A)",
        inputs: &DEAL_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Matching inputs are distributed positionally; one input is replicated to every requested output. No integer arithmetic or floating materialization occurs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[B1,...,Bn] = deal(resident_integer_A1,...,resident_integer_An) or deal(resident_integer_A)",
        inputs: &DEAL_RESIDENT_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "This RunMat-only form is gated before provider access and retains each input's original resident handle and owner.",
    },
];
