use crate::*;

const INPUT_DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric vector of observations.",
};
const INPUT_FAMILY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "distname",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Supported distribution family.",
};
const INPUT_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional Frequency name-value pair.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pd",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fitted ProbabilityDistribution value object.",
}];
const BASIC: [BuiltinParamDescriptor; 2] = [INPUT_DATA, INPUT_FAMILY];
const WITH_OPTIONS: [BuiltinParamDescriptor; 3] = [INPUT_DATA, INPUT_FAMILY, INPUT_OPTIONS];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pd = fitdist(x, distname)",
        inputs: &BASIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "pd = fitdist(x, distname, \"Frequency\", w)",
        inputs: &WITH_OPTIONS,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.FITDIST.INVALID_ARGUMENT",
        identifier: Some("RunMat:fitdist:InvalidArgument"),
        when: "Sample data, family, options, or observation weights are malformed.",
        message: "fitdist: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.FITDIST.NUMERICAL",
        identifier: Some("RunMat:fitdist:Numerical"),
        when: "Parameter estimation fails to converge or is ill-conditioned.",
        message: "fitdist: numerical failure",
    },
    BuiltinErrorDescriptor {
        code: "RM.FITDIST.INTERNAL",
        identifier: Some("RunMat:fitdist:Internal"),
        when: "RunMat cannot construct the fitted distribution value.",
        message: "fitdist: internal error",
    },
];
pub const FITDIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
