use crate::*;

const MODEL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mdl", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required,
    default: None, description: "LinearModel object containing coefficients, fitted values, residuals, and summary statistics.",
};
const DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tblOrX",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input table or predictor matrix.",
};
const RESPONSE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "yOrSpec",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Response vector, response variable name, model formula, or model specification.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Variadic,
    default: None, description: "Model specification and name-value options such as Intercept, ResponseVar, PredictorVars, VarNames, Exclude, and Weights.",
};
const MODEL_OUTPUT: [BuiltinParamDescriptor; 1] = [MODEL];
const TABLE: [BuiltinParamDescriptor; 1] = [DATA];
const DATA_RESPONSE: [BuiltinParamDescriptor; 2] = [DATA, RESPONSE];
const FULL: [BuiltinParamDescriptor; 3] = [DATA, RESPONSE, OPTIONS];
const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "mdl = fitlm(tbl)",
        inputs: &TABLE,
        outputs: &MODEL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "mdl = fitlm(tbl, ResponseVarName)",
        inputs: &DATA_RESPONSE,
        outputs: &MODEL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "mdl = fitlm(tbl, formula)",
        inputs: &DATA_RESPONSE,
        outputs: &MODEL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "mdl = fitlm(tbl, modelspec)",
        inputs: &DATA_RESPONSE,
        outputs: &MODEL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "mdl = fitlm(X, y)",
        inputs: &DATA_RESPONSE,
        outputs: &MODEL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "mdl = fitlm(___, Name, Value)",
        inputs: &FULL,
        outputs: &MODEL_OUTPUT,
    },
];

pub const FITLM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITLM.INVALID_ARGUMENT", identifier: Some("RunMat:fitlm:InvalidArgument"),
    when: "Inputs, model specifications, dimensions, or name-value options are malformed or unsupported.",
    message: "fitlm: invalid argument",
};
pub const FITLM_ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITLM.NUMERICAL",
    identifier: Some("RunMat:fitlm:Numerical"),
    when: "The linear model cannot be fit numerically.",
    message: "fitlm: numerical failure",
};
pub const FITLM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITLM.INTERNAL",
    identifier: Some("RunMat:fitlm:Internal"),
    when: "RunMat cannot construct a LinearModel result.",
    message: "fitlm: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    FITLM_ERROR_INVALID_ARGUMENT,
    FITLM_ERROR_NUMERICAL,
    FITLM_ERROR_INTERNAL,
];
pub const FITLM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FITLM_INTEGER_PREDICTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitlm-integer-predictor-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitlm with typed-integer predictor data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitlmIntegerPredictorExtension"),
    };
pub const FITLM_INTEGER_RESPONSE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitlm-integer-response-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitlm with typed-integer response data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitlmIntegerResponseExtension"),
    };
pub const FITLM_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitlm-integer-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitlm with typed-integer observation weights is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitlmIntegerWeightsExtension"),
    };
pub const FITLM_INTEGER_SELECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitlm-integer-selectors",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitlm with typed-integer Exclude or selector controls is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitlmIntegerSelectorExtension"),
    };
pub const FITLM_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitlm-integer-controls",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitlm with a typed-integer Intercept control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitlmIntegerControlExtension"),
    };
pub const FITLM_RESIDENT_FALLBACK_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitlm-resident-fallback",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitlm gathers resident inputs and returns a host linear-model object",
        error_identifier: Some("RunMat:compatibility:FitlmResidentFallbackExtension"),
    };
pub const FITLM_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    FITLM_INTEGER_PREDICTOR_EXTENSION,
    FITLM_INTEGER_RESPONSE_EXTENSION,
    FITLM_INTEGER_WEIGHTS_EXTENSION,
    FITLM_INTEGER_SELECTOR_EXTENSION,
    FITLM_INTEGER_CONTROL_EXTENSION,
    FITLM_RESIDENT_FALLBACK_EXTENSION,
];
