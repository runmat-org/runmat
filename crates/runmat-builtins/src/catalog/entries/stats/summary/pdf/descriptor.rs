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
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Density or probability-mass values.",
}];
const FITTED: [BuiltinParamDescriptor; 2] = [INPUT_DISTRIBUTION, INPUT_X];
const NAMED: [BuiltinParamDescriptor; 3] = [INPUT_DISTRIBUTION, INPUT_X, INPUT_PARAMS];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "y = pdf(pd, x)",
        inputs: &FITTED,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "y = pdf(distname, x, params...)",
        inputs: &NAMED,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.PDF.INVALID_ARGUMENT",
        identifier: Some("RunMat:pdf:InvalidArgument"),
        when: "Distribution, evaluation points, parameters, or broadcast dimensions are invalid.",
        message: "pdf: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.PDF.INTERNAL",
        identifier: Some("RunMat:pdf:Internal"),
        when: "The numeric result cannot be constructed or restored.",
        message: "pdf: internal error",
    },
];
pub const PDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
