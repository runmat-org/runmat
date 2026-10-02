use crate::*;

const FUN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Objective callback invoked with one table row.",
};
const VARS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "vars",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "optimizableVariable object or collection.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optimization controls and optional feasibility callback.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "results",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host BayesianOptimization result object.",
}];
const BASE: [BuiltinParamDescriptor; 2] = [FUN, VARS];
const FULL: [BuiltinParamDescriptor; 3] = [FUN, VARS, OPTIONS];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "results = bayesopt(fun, vars)",
        inputs: &BASE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "results = bayesopt(fun, vars, Name, Value)",
        inputs: &FULL,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "bayesopt(fun, vars)",
        inputs: &BASE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "bayesopt(fun, vars, Name, Value)",
        inputs: &FULL,
        outputs: &[],
    },
];
pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAYESOPT.INVALID_ARGUMENT",
    identifier: Some("RunMat:bayesopt:InvalidArgument"),
    when: "Variables, options, callback outputs or constraints are malformed.",
    message: "bayesopt: invalid argument",
};
pub const ERROR_OBJECTIVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAYESOPT.OBJECTIVE",
    identifier: Some("RunMat:bayesopt:Objective"),
    when: "An objective callback fails or returns an invalid objective value.",
    message: "bayesopt: objective evaluation failed",
};
pub const ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAYESOPT.UNSUPPORTED",
    identifier: Some("RunMat:bayesopt:Unsupported"),
    when: "A requested option needs an unavailable optimization mode.",
    message: "bayesopt: unsupported option",
};
pub const ERROR_FLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAYESOPT.FLOW",
    identifier: Some("RunMat:bayesopt:Flow"),
    when: "Materializing an input or callback result fails.",
    message: "bayesopt: flow failure",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    ERROR_INVALID_ARGUMENT,
    ERROR_OBJECTIVE,
    ERROR_UNSUPPORTED,
    ERROR_FLOW,
];
pub const BAYESOPT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
