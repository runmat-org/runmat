use crate::*;

const MODEL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mdl",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Supported fitted statistical model or deep-learning network object.",
};
const PREDICTORS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Xnew",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "New predictor table, matrix, array, or dlarray.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Supported model-specific name-value options.",
};
const INPUTS: [BuiltinParamDescriptor; 2] = [MODEL, PREDICTORS];
const INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 3] = [MODEL, PREDICTORS, OPTIONS];

const RESPONSE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ypred",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predicted response values.",
};
const INTERVAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "yci",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pointwise confidence intervals for the predicted response.",
};
const LABEL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "label",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predicted class labels.",
};
const SCORE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "score",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Classification scores or posterior probabilities.",
};
const NODE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "node",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based terminal node index for each prediction.",
};
const CLASS_NUMBER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "cnum",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based class number for each predicted label.",
};
const NETWORK_RESULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predicted network scores in the selected output axis convention.",
};
const NETWORK_STATE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "state",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated dlnetwork state, when requested and supported.",
};

const RESPONSE_ONLY: [BuiltinParamDescriptor; 1] = [RESPONSE];
const RESPONSE_INTERVAL: [BuiltinParamDescriptor; 2] = [RESPONSE, INTERVAL];
const LABEL_ONLY: [BuiltinParamDescriptor; 1] = [LABEL];
const LABEL_SCORE: [BuiltinParamDescriptor; 2] = [LABEL, SCORE];
const TREE_OUTPUTS: [BuiltinParamDescriptor; 4] = [LABEL, SCORE, NODE, CLASS_NUMBER];
const NETWORK_ONLY: [BuiltinParamDescriptor; 1] = [NETWORK_RESULT];
const NETWORK_WITH_STATE: [BuiltinParamDescriptor; 2] = [NETWORK_RESULT, NETWORK_STATE];

const SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "ypred = predict(mdl, Xnew)",
        inputs: &INPUTS,
        outputs: &RESPONSE_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "[ypred, yci] = predict(mdl, Xnew)",
        inputs: &INPUTS,
        outputs: &RESPONSE_INTERVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[ypred, yci] = predict(mdl, Xnew, Name, Value)",
        inputs: &INPUTS_WITH_OPTIONS,
        outputs: &RESPONSE_INTERVAL,
    },
    BuiltinSignatureDescriptor {
        label: "label = predict(tree, Xnew)",
        inputs: &INPUTS,
        outputs: &LABEL_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "[label,score,node,cnum] = predict(tree, Xnew)",
        inputs: &INPUTS,
        outputs: &TREE_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[label,score] = predict(linearClassifier, Xnew)",
        inputs: &INPUTS_WITH_OPTIONS,
        outputs: &LABEL_SCORE,
    },
    BuiltinSignatureDescriptor {
        label: "Y = predict(net, X)",
        inputs: &INPUTS_WITH_OPTIONS,
        outputs: &NETWORK_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "[Y,state] = predict(dlnetwork, X)",
        inputs: &INPUTS_WITH_OPTIONS,
        outputs: &NETWORK_WITH_STATE,
    },
];

pub const PREDICT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PREDICT.INVALID_ARGUMENT",
    identifier: Some("RunMat:predict:InvalidArgument"),
    when: "The model, predictor data, or prediction options are malformed or unsupported.",
    message: "predict: invalid argument",
};
pub const PREDICT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PREDICT.INTERNAL",
    identifier: Some("RunMat:predict:Internal"),
    when: "RunMat cannot construct prediction outputs.",
    message: "predict: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [PREDICT_ERROR_INVALID_ARGUMENT, PREDICT_ERROR_INTERNAL];

pub const PREDICT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
