use crate::*;

const WN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "wn",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Natural frequencies as an N-by-1 real double column vector.",
};
const ZETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "zeta",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Damping ratios as an N-by-1 real double column vector.",
};
const POLES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Model poles as an N-by-1 real or complex double column vector.",
};
const ONE_OUTPUT: [BuiltinParamDescriptor; 1] = [WN];
const TWO_OUTPUTS: [BuiltinParamDescriptor; 2] = [WN, ZETA];
const THREE_OUTPUTS: [BuiltinParamDescriptor; 3] = [WN, ZETA, POLES];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model or ss state-space model.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "wn = damp(sys)",
        inputs: &INPUTS,
        outputs: &ONE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[wn, zeta] = damp(sys)",
        inputs: &INPUTS,
        outputs: &TWO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[wn, zeta, p] = damp(sys)",
        inputs: &INPUTS,
        outputs: &THREE_OUTPUTS,
    },
];
pub const DAMP_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DAMP.INVALID_MODEL",
    identifier: Some("RunMat:damp:InvalidModel"),
    when: "Input system is malformed or missing model metadata.",
    message: "damp: invalid model",
};
pub const DAMP_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DAMP.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:damp:UnsupportedModel"),
    when: "Model class is unsupported.",
    message: "damp: unsupported model",
};
pub const DAMP_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DAMP.OUTPUT_COUNT",
    identifier: Some("RunMat:damp:OutputCount"),
    when: "More than three outputs are requested.",
    message: "damp: too many output arguments",
};
pub const DAMP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DAMP.INTERNAL",
    identifier: Some("RunMat:damp:Internal"),
    when: "Pole extraction or output construction failed.",
    message: "damp: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    DAMP_ERROR_INVALID_MODEL,
    DAMP_ERROR_UNSUPPORTED_MODEL,
    DAMP_ERROR_OUTPUT_COUNT,
    DAMP_ERROR_INTERNAL,
];

pub const DAMP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_SYS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sys",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "damp accepts a dynamic-system object. Integer coefficient support belongs to the tf/ss constructors and does not make an integer array a valid sys input.",
}];

pub const DAMP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[wn, zeta, p] = damp(integer_sys)",
        inputs: &INTEGER_SYS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "A direct integer sys is not a model and rejects without provider dispatch. This says nothing about typed integer properties of an existing tf/ss object: consumed properties are independently gated and checked at model admission.",
    }];
