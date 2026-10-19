use crate::*;

const MODEL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Mdl",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "ClassificationLinear object containing coefficients, class names, and fit metadata.",
};
const FIT_INFO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "FitInfo",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Training diagnostics for each Lambda value.",
};
const DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tblOrX",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input table or predictor matrix.",
};
const RESPONSE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "yOrResponse",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Response vector, response variable name, or table formula.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Variadic,
    default: None, description: "Name-value options including Learner, Lambda, Regularization, Solver, ObservationsIn, FitBias, ClassNames, PredictorNames, ResponseName, Weights, Prior, and ScoreTransform.",
};
const MODEL_ONLY: [BuiltinParamDescriptor; 1] = [MODEL];
const MODEL_INFO: [BuiltinParamDescriptor; 2] = [MODEL, FIT_INFO];
const BASIC: [BuiltinParamDescriptor; 2] = [DATA, RESPONSE];
const FULL: [BuiltinParamDescriptor; 3] = [DATA, RESPONSE, OPTIONS];
const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "Mdl = fitclinear(X, Y)",
        inputs: &BASIC,
        outputs: &MODEL_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitclinear(Tbl, ResponseVarName)",
        inputs: &BASIC,
        outputs: &MODEL_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitclinear(Tbl, formula)",
        inputs: &BASIC,
        outputs: &MODEL_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitclinear(Tbl, Y)",
        inputs: &BASIC,
        outputs: &MODEL_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitclinear(___, Name, Value)",
        inputs: &FULL,
        outputs: &MODEL_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "[Mdl, FitInfo] = fitclinear(___)",
        inputs: &FULL,
        outputs: &MODEL_INFO,
    },
];

pub const FITCLINEAR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITCLINEAR.INVALID_ARGUMENT",
    identifier: Some("RunMat:fitclinear:InvalidArgument"),
    when: "Inputs, binary response labels, dimensions, or name-value options are malformed or unsupported.",
    message: "fitclinear: invalid argument",
};
pub const FITCLINEAR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITCLINEAR.INTERNAL",
    identifier: Some("RunMat:fitclinear:Internal"),
    when: "RunMat cannot construct the ClassificationLinear result.",
    message: "fitclinear: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [FITCLINEAR_ERROR_INVALID_ARGUMENT, FITCLINEAR_ERROR_INTERNAL];
pub const FITCLINEAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FITCLINEAR_INTEGER_PREDICTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitclinear-integer-predictors",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitclinear with typed-integer predictors is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitclinearIntegerPredictorExtension"),
    };
pub const FITCLINEAR_INTEGER_NUMERIC_ARGUMENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitclinear-integer-numeric-arguments",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "fitclinear with typed-integer weights or numeric controls is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitclinearIntegerNumericArgumentExtension"),
    };
pub const FITCLINEAR_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitclinear-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitclinear host fitting from provider-resident input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitclinearResidentInputExtension"),
    };
pub const FITCLINEAR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FITCLINEAR_INTEGER_PREDICTOR_EXTENSION,
    FITCLINEAR_INTEGER_NUMERIC_ARGUMENT_EXTENSION,
    FITCLINEAR_RESIDENT_INPUT_EXTENSION,
];
