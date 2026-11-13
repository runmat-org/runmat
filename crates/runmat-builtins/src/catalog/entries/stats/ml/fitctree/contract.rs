use crate::*;

const MODEL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Mdl", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required,
    default: None, description: "ClassificationTree object containing class names, split rules, and posterior probabilities.",
};
const DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tblOrX",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input table or numeric predictor matrix.",
};
const RESPONSE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "yOrResponse",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "For table input, a response variable name, additive formula, or external label vector with one label per row; for matrix input, a response label vector. Labels may be numeric, logical, text, or fixed-width integer.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Variadic,
    default: None, description: "Name-value options including ClassNames, PredictorNames, ResponseName, MaxNumSplits, MinLeafSize, MinParentSize, SplitCriterion, and Weights.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [MODEL];
const BASIC: [BuiltinParamDescriptor; 2] = [DATA, RESPONSE];
const FULL: [BuiltinParamDescriptor; 3] = [DATA, RESPONSE, OPTIONS];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "Mdl = fitctree(Tbl, ResponseVarName)",
        inputs: &BASIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitctree(Tbl, formula)",
        inputs: &BASIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitctree(Tbl, Y)",
        inputs: &BASIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitctree(X, Y)",
        inputs: &BASIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitctree(___, Name, Value)",
        inputs: &FULL,
        outputs: &OUTPUT,
    },
];

pub const FITCTREE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITCTREE.INVALID_ARGUMENT",
    identifier: Some("RunMat:fitctree:InvalidArgument"),
    when:
        "Inputs, response labels, dimensions, or name-value options are malformed or unsupported.",
    message: "fitctree: invalid argument",
};
pub const FITCTREE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITCTREE.INTERNAL",
    identifier: Some("RunMat:fitctree:Internal"),
    when: "RunMat cannot construct the ClassificationTree result.",
    message: "fitctree: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [FITCTREE_ERROR_INVALID_ARGUMENT, FITCTREE_ERROR_INTERNAL];
pub const FITCTREE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FITCTREE_INTEGER_PREDICTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitctree-integer-predictors",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitctree with typed-integer predictors is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitctreeIntegerPredictorExtension"),
    };
pub const FITCTREE_INTEGER_NUMERIC_ARGUMENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitctree-integer-numeric-arguments",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "fitctree with typed-integer weights or numeric controls is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitctreeIntegerNumericArgumentExtension"),
    };
pub const FITCTREE_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitctree-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitctree host fitting from provider-resident input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitctreeResidentInputExtension"),
    };
pub const FITCTREE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FITCTREE_INTEGER_PREDICTOR_EXTENSION,
    FITCTREE_INTEGER_NUMERIC_ARGUMENT_EXTENSION,
    FITCTREE_RESIDENT_INPUT_EXTENSION,
];
