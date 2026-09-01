use super::documentation::{
    FETCH_NEXT_DOCUMENTATION, FETCH_OUTPUTS_DOCUMENTATION, PARFEVAL_DOCUMENTATION,
    PARFEVAL_ON_ALL_DOCUMENTATION,
};
use super::*;

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
const FUTURE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
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
const FETCH_NEXT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
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
        ty: BuiltinParamType::NumericScalar,
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
        outputs: &FUTURE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "future = parfeval(function, outputs, arguments...)",
        inputs: &AUTOMATIC_FUTURE_INPUTS,
        outputs: &FUTURE_OUTPUT,
    },
];
const PARFEVAL_ON_ALL_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "future = parfevalOnAll(pool, function, outputs, arguments...)",
        inputs: &FUTURE_INPUTS,
        outputs: &FUTURE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "future = parfevalOnAll(function, outputs, arguments...)",
        inputs: &AUTOMATIC_FUTURE_INPUTS,
        outputs: &FUTURE_OUTPUT,
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
        inputs: &FETCH_NEXT_INPUTS,
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
parallel_entry!(
    PARFEVAL_CATALOG_ENTRY,
    "parfeval",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Parfeval),
    PARFEVAL_DOCUMENTATION,
    PARFEVAL_DESCRIPTOR,
    BuiltinContractMaturity::Complete,
    BuiltinAsyncBehavior::MaySuspend,
    BuiltinPurity::Impure,
    &SCHEDULE_EFFECTS
);
parallel_entry!(
    PARFEVAL_ON_ALL_CATALOG_ENTRY,
    "parfevalOnAll",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::ParfevalOnAll),
    PARFEVAL_ON_ALL_DOCUMENTATION,
    PARFEVAL_ON_ALL_DESCRIPTOR,
    BuiltinContractMaturity::Complete,
    BuiltinAsyncBehavior::MaySuspend,
    BuiltinPurity::Impure,
    &SCHEDULE_EFFECTS
);
parallel_entry!(
    FETCH_OUTPUTS_CATALOG_ENTRY,
    "fetchOutputs",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::FetchOutputs),
    FETCH_OUTPUTS_DOCUMENTATION,
    FETCH_OUTPUTS_DESCRIPTOR,
    BuiltinContractMaturity::Complete,
    BuiltinAsyncBehavior::RequiresAsyncRuntime,
    BuiltinPurity::Impure,
    &FETCH_EFFECTS
);
parallel_entry!(
    FETCH_NEXT_CATALOG_ENTRY,
    "fetchNext",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::FetchNext),
    FETCH_NEXT_DOCUMENTATION,
    FETCH_NEXT_DESCRIPTOR,
    BuiltinContractMaturity::Complete,
    BuiltinAsyncBehavior::RequiresAsyncRuntime,
    BuiltinPurity::Impure,
    &FETCH_EFFECTS
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &FETCH_NEXT_CATALOG_ENTRY,
    &FETCH_OUTPUTS_CATALOG_ENTRY,
    &PARFEVAL_CATALOG_ENTRY,
    &PARFEVAL_ON_ALL_CATALOG_ENTRY,
];
