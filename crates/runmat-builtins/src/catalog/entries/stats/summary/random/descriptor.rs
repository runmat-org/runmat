use crate::*;

const INPUT_DISTRIBUTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pd",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fitted ProbabilityDistribution object or named family.",
};
const INPUT_PARAMS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "params",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Scalar named-family parameters.",
};
const INPUT_SIZE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Requested output dimensions.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random samples.",
}];
const FITTED: [BuiltinParamDescriptor; 1] = [INPUT_DISTRIBUTION];
const FITTED_SIZE: [BuiltinParamDescriptor; 2] = [INPUT_DISTRIBUTION, INPUT_SIZE];
const NAMED: [BuiltinParamDescriptor; 3] = [INPUT_DISTRIBUTION, INPUT_PARAMS, INPUT_SIZE];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = random(pd)",
        inputs: &FITTED,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "r = random(pd, sz...)",
        inputs: &FITTED_SIZE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "r = random(distname, params..., sz...)",
        inputs: &NAMED,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.RANDOM.INVALID_ARGUMENT",
        identifier: Some("RunMat:random:InvalidArgument"),
        when: "Distribution, parameters, or size controls are invalid.",
        message: "random: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDOM.INTERNAL",
        identifier: Some("RunMat:random:Internal"),
        when: "Sample allocation or provider restoration fails.",
        message: "random: internal error",
    },
];
pub const RANDOM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
