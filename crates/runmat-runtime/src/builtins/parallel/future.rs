use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

const FUTURE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "pool",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pool on which the function is scheduled.",
    },
    BuiltinParamDescriptor {
        name: "function",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle to execute.",
    },
    BuiltinParamDescriptor {
        name: "outputs",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of requested function outputs.",
    },
    BuiltinParamDescriptor {
        name: "arguments",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Arguments supplied to the scheduled function.",
    },
];
const AUTOMATIC_FUTURE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "function",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle to execute on the current or automatic pool.",
    },
    BuiltinParamDescriptor {
        name: "outputs",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of requested function outputs.",
    },
    BuiltinParamDescriptor {
        name: "arguments",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Arguments supplied to the scheduled function.",
    },
];
const FUTURE_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "future",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Future representing the scheduled invocation.",
}];
const FETCH_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "future",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Future whose outputs are requested.",
}];
const FETCH_OPTION_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "future",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Future array whose outputs are requested.",
    },
    BuiltinParamDescriptor {
        name: "UniformOutput",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"UniformOutput\""),
        description: "Select cell output for arrays of futures.",
    },
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "True to concatenate outputs; false to return cell arrays.",
    },
];
const FETCH_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "outputs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Outputs produced by the scheduled function.",
}];
const FETCH_NEXT_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "futures",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Future array from which to retrieve the next unread result.",
}];
const FETCH_NEXT_TIMEOUT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "futures",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Future array from which to retrieve the next unread result.",
    },
    BuiltinParamDescriptor {
        name: "timeout",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum number of seconds to wait for an unread result.",
    },
];
const FETCH_NEXT_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "index",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Linear index of the completed future.",
    },
    BuiltinParamDescriptor {
        name: "outputs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Outputs produced by the selected future.",
    },
];
const PARFEVAL_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "future = parfeval(pool, function, outputs, arguments...)",
        inputs: &FUTURE_INPUTS,
        outputs: &FUTURE_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "future = parfeval(function, outputs, arguments...)",
        inputs: &AUTOMATIC_FUTURE_INPUTS,
        outputs: &FUTURE_OUTPUTS,
    },
];
const PARFEVAL_ON_ALL_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "future = parfevalOnAll(pool, function, outputs, arguments...)",
        inputs: &FUTURE_INPUTS,
        outputs: &FUTURE_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "future = parfevalOnAll(function, outputs, arguments...)",
        inputs: &AUTOMATIC_FUTURE_INPUTS,
        outputs: &FUTURE_OUTPUTS,
    },
];
const FETCH_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[outputs...] = fetchOutputs(future)",
        inputs: &FETCH_INPUTS,
        outputs: &FETCH_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[outputs...] = fetchOutputs(futures, \"UniformOutput\", tf)",
        inputs: &FETCH_OPTION_INPUTS,
        outputs: &FETCH_OUTPUTS,
    },
];
const FETCH_NEXT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[index, outputs...] = fetchNext(futures)",
        inputs: &FETCH_NEXT_INPUT,
        outputs: &FETCH_NEXT_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[index, outputs...] = fetchNext(futures, timeout)",
        inputs: &FETCH_NEXT_TIMEOUT_INPUTS,
        outputs: &FETCH_NEXT_OUTPUTS,
    },
];
const LOWERING_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.PARALLEL.LOWERING_REQUIRED",
    identifier: Some("RunMat:parallel:LoweringRequired"),
    when: "The call bypasses the executor's semantic async lowering path.",
    message: "parallel call requires executor-aware lowering",
}];
pub const PARFEVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARFEVAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOWERING_ERRORS,
};
pub const PARFEVAL_ON_ALL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARFEVAL_ON_ALL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOWERING_ERRORS,
};
pub const FETCH_OUTPUTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FETCH_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOWERING_ERRORS,
};
pub const FETCH_NEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FETCH_NEXT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOWERING_ERRORS,
};

fn lowering_error(name: &str) -> crate::RuntimeError {
    crate::build_runtime_error(format!(
        "{name}: this call must be lowered by an execution-capable RunMat executor"
    ))
    .with_builtin(name)
    .with_identifier("RunMat:parallel:LoweringRequired")
    .build()
}

#[runtime_builtin(
    name = "parfeval",
    category = "parallel",
    summary = "Schedule a function for asynchronous execution on a pool.",
    keywords = "parallel,future,async,parfeval",
    descriptor(crate::builtins::parallel::future::PARFEVAL_DESCRIPTOR),
    builtin_path = "crate::builtins::parallel::future"
)]
async fn parfeval_builtin(_arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    Err(lowering_error("parfeval"))
}

#[runtime_builtin(
    name = "parfevalOnAll",
    category = "parallel",
    summary = "Schedule a function once on every worker in a pool.",
    keywords = "parallel,future,async,workers,parfevalOnAll",
    descriptor(crate::builtins::parallel::future::PARFEVAL_ON_ALL_DESCRIPTOR),
    builtin_path = "crate::builtins::parallel::future"
)]
async fn parfeval_on_all_builtin(_arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    Err(lowering_error("parfevalOnAll"))
}

#[runtime_builtin(
    name = "fetchOutputs",
    category = "parallel",
    summary = "Wait for a future and return its outputs.",
    keywords = "parallel,future,wait,fetchOutputs",
    descriptor(crate::builtins::parallel::future::FETCH_OUTPUTS_DESCRIPTOR),
    builtin_path = "crate::builtins::parallel::future"
)]
async fn fetch_outputs_builtin(_future: Value) -> crate::BuiltinResult<Value> {
    Err(lowering_error("fetchOutputs"))
}

#[runtime_builtin(
    name = "fetchNext",
    category = "parallel",
    summary = "Retrieve the next completed unread result from a future array.",
    keywords = "parallel,future,wait,fetchNext",
    descriptor(crate::builtins::parallel::future::FETCH_NEXT_DESCRIPTOR),
    builtin_path = "crate::builtins::parallel::future"
)]
async fn fetch_next_builtin(_futures: Value, _rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    Err(lowering_error("fetchNext"))
}
