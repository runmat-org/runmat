use crate::*;

const INPUT_DISTRIBUTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pd",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fitted ProbabilityDistribution object or named family.",
};
const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Evaluation points.",
};
const INPUT_PARAMS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A...D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Parameters for a named family.",
};
const INPUT_UPPER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "upper",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complementary upper-tail selector.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cumulative probability values.",
}];
const FITTED: [BuiltinParamDescriptor; 2] = [INPUT_DISTRIBUTION, INPUT_X];
const FITTED_UPPER: [BuiltinParamDescriptor; 3] = [INPUT_DISTRIBUTION, INPUT_X, INPUT_UPPER];
const NAMED: [BuiltinParamDescriptor; 3] = [INPUT_DISTRIBUTION, INPUT_X, INPUT_PARAMS];
const NAMED_UPPER: [BuiltinParamDescriptor; 4] =
    [INPUT_DISTRIBUTION, INPUT_X, INPUT_PARAMS, INPUT_UPPER];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "p = cdf(pd, x)",
        inputs: &FITTED,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "p = cdf(distname, x, params...)",
        inputs: &NAMED,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "p = cdf(pd, x, \"upper\")",
        inputs: &FITTED_UPPER,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "p = cdf(distname, x, params..., \"upper\")",
        inputs: &NAMED_UPPER,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.CDF.INVALID_ARGUMENT",
        identifier: Some("RunMat:cdf:InvalidArgument"),
        when:
            "Distribution, evaluation points, parameters, tail selector, or dimensions are invalid.",
        message: "cdf: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.CDF.INTERNAL",
        identifier: Some("RunMat:cdf:Internal"),
        when: "The numeric result cannot be constructed or restored.",
        message: "cdf: internal error",
    },
];
pub const CDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
