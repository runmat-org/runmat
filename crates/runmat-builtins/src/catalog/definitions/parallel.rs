use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingAvailability,
    BuiltinBindingDeclaration, BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation, BuiltinErrorDescriptor,
    BuiltinFusionPolicy, BuiltinInferenceRuleId, BuiltinLinkContract, BuiltinLinkPolicy,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinPlacementContract, BuiltinPortability, BuiltinPurity, BuiltinReachability,
    BuiltinResidencyPolicy, BuiltinSemanticKind, BuiltinSignatureDescriptor,
};
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

const CURRENT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "context",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current parallel execution object, or an empty matrix when none is active.",
}];

const POOL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pool",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Execution pool owned by the current RunMat session.",
}];
const PARPOOL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional pool kind and worker count.",
}];
const GCP_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Use 'nocreate' to inspect the current pool without creating one.",
}];
const PARPOOL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pool = parpool(___)",
    inputs: &PARPOOL_INPUTS,
    outputs: &POOL_OUTPUT,
}];
const GCP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pool = gcp(option)",
    inputs: &GCP_INPUTS,
    outputs: &POOL_OUTPUT,
}];
const POOL_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.PARALLEL.POOL_INVALID_INPUT",
    identifier: Some("RunMat:parpool:InvalidInput"),
    when: "The requested pool kind, worker count, or option is invalid for the active host.",
    message: "parpool: invalid pool request",
}];
pub const PARPOOL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARPOOL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POOL_ERRORS,
};
pub const GCP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GCP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POOL_ERRORS,
};

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

const GET_CURRENT_TASK_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "task = getCurrentTask()",
    inputs: &[],
    outputs: &CURRENT_OUTPUT,
}];
const GET_CURRENT_WORKER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "worker = getCurrentWorker()",
        inputs: &[],
        outputs: &CURRENT_OUTPUT,
    }];
const GET_CURRENT_JOB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "job = getCurrentJob()",
    inputs: &[],
    outputs: &CURRENT_OUTPUT,
}];

pub const GET_CURRENT_TASK_DESCRIPTOR: BuiltinDescriptor =
    current_descriptor(&GET_CURRENT_TASK_SIGNATURES);
pub const GET_CURRENT_WORKER_DESCRIPTOR: BuiltinDescriptor =
    current_descriptor(&GET_CURRENT_WORKER_SIGNATURES);
pub const GET_CURRENT_JOB_DESCRIPTOR: BuiltinDescriptor =
    current_descriptor(&GET_CURRENT_JOB_SIGNATURES);

const fn current_descriptor(
    signatures: &'static [BuiltinSignatureDescriptor],
) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &[],
    }
}

const CURRENT_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
const POOL_READ_EFFECTS: [EffectKind; 2] = [EffectKind::EnvironmentRead, EffectKind::MayThrow];
const POOL_WRITE_EFFECTS: [EffectKind; 3] = [
    EffectKind::EnvironmentRead,
    EffectKind::EnvironmentWrite,
    EffectKind::MayThrow,
];
const SCHEDULE_EFFECTS: [EffectKind; 3] = [
    EffectKind::EnvironmentWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
const FETCH_EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
const PARALLEL_RUNTIME: [CapabilityRequirement; 1] = [CapabilityRequirement::ParallelRuntime];
const CURRENT_PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
};
const CURRENT_LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

macro_rules! parallel_entry {
    (
        $constant:ident,
        $name:literal,
        $rule:literal,
        $summary:literal,
        $keywords:expr,
        $descriptor:ident,
        $maturity:expr,
        $async_behavior:expr,
        $purity:expr,
        $effects:expr
    ) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: BuiltinDocumentation {
                summary: $summary,
                keywords: $keywords,
                related: &[],
                introduced: None,
                status: None,
                examples: &[],
            },
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: $maturity,
                inference_rule: BuiltinInferenceRuleId($rule),
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: $async_behavior,
                purity: $purity,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: $effects,
                capabilities: &PARALLEL_RUNTIME,
            },
            placement: CURRENT_PLACEMENT,
            link: CURRENT_LINK,
            bindings: &[BuiltinBindingDeclaration {
                identity: BuiltinBindingIdentity {
                    builtin: BuiltinCatalogIdentity { name: $name },
                    variant: "default",
                },
                availability: BuiltinBindingAvailability::Required,
            }],
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

parallel_entry!(
    PARPOOL_CATALOG_ENTRY,
    "parpool",
    "parallel.parpool",
    "Create or return the execution pool for the current session.",
    &["parallel", "pool", "workers", "parpool"],
    PARPOOL_DESCRIPTOR,
    BuiltinContractMaturity::Complete,
    BuiltinAsyncBehavior::MaySuspend,
    BuiltinPurity::Impure,
    &POOL_WRITE_EFFECTS
);
parallel_entry!(
    GCP_CATALOG_ENTRY,
    "gcp",
    "parallel.gcp",
    "Return the current execution pool.",
    &["parallel", "pool", "current", "gcp", "nocreate"],
    GCP_DESCRIPTOR,
    BuiltinContractMaturity::DynamicByDesign,
    BuiltinAsyncBehavior::MaySuspend,
    BuiltinPurity::DeterministicReadOnly,
    &POOL_READ_EFFECTS
);
parallel_entry!(
    PARFEVAL_CATALOG_ENTRY,
    "parfeval",
    "parallel.parfeval",
    "Schedule a function for asynchronous execution on a pool.",
    &["parallel", "future", "async", "parfeval"],
    PARFEVAL_DESCRIPTOR,
    BuiltinContractMaturity::Complete,
    BuiltinAsyncBehavior::MaySuspend,
    BuiltinPurity::Impure,
    &SCHEDULE_EFFECTS
);
parallel_entry!(
    PARFEVAL_ON_ALL_CATALOG_ENTRY,
    "parfevalOnAll",
    "parallel.parfeval-on-all",
    "Schedule a function once on every worker in a pool.",
    &["parallel", "future", "async", "workers"],
    PARFEVAL_ON_ALL_DESCRIPTOR,
    BuiltinContractMaturity::Complete,
    BuiltinAsyncBehavior::MaySuspend,
    BuiltinPurity::Impure,
    &SCHEDULE_EFFECTS
);
parallel_entry!(
    FETCH_OUTPUTS_CATALOG_ENTRY,
    "fetchOutputs",
    "parallel.fetch-outputs",
    "Wait for a future and return its outputs.",
    &["parallel", "future", "wait", "outputs"],
    FETCH_OUTPUTS_DESCRIPTOR,
    BuiltinContractMaturity::Complete,
    BuiltinAsyncBehavior::RequiresAsyncRuntime,
    BuiltinPurity::Impure,
    &FETCH_EFFECTS
);
parallel_entry!(
    FETCH_NEXT_CATALOG_ENTRY,
    "fetchNext",
    "parallel.fetch-next",
    "Retrieve the next completed unread result from a future array.",
    &["parallel", "future", "wait", "completion order"],
    FETCH_NEXT_DESCRIPTOR,
    BuiltinContractMaturity::Complete,
    BuiltinAsyncBehavior::RequiresAsyncRuntime,
    BuiltinPurity::Impure,
    &FETCH_EFFECTS
);

macro_rules! current_execution_entry {
    ($constant:ident, $name:literal, $rule:literal, $summary:literal, $keywords:expr, $descriptor:ident) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: BuiltinDocumentation {
                summary: $summary,
                keywords: $keywords,
                related: &[],
                introduced: None,
                status: None,
                examples: &[],
            },
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                // The result depends on the scoped execution identity: it is
                // empty outside that scope and an object inside it.
                maturity: BuiltinContractMaturity::DynamicByDesign,
                inference_rule: BuiltinInferenceRuleId($rule),
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::NeverSuspends,
                purity: BuiltinPurity::DeterministicReadOnly,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &CURRENT_EFFECTS,
                capabilities: &PARALLEL_RUNTIME,
            },
            placement: CURRENT_PLACEMENT,
            link: CURRENT_LINK,
            bindings: &[BuiltinBindingDeclaration {
                identity: BuiltinBindingIdentity {
                    builtin: BuiltinCatalogIdentity { name: $name },
                    variant: "default",
                },
                availability: BuiltinBindingAvailability::Required,
            }],
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

current_execution_entry!(
    GET_CURRENT_TASK_CATALOG_ENTRY,
    "getCurrentTask",
    "parallel.get-current-task",
    "Return the task executing the current function.",
    &["parallel", "current", "task", "worker"],
    GET_CURRENT_TASK_DESCRIPTOR
);
current_execution_entry!(
    GET_CURRENT_WORKER_CATALOG_ENTRY,
    "getCurrentWorker",
    "parallel.get-current-worker",
    "Return the worker executing the current function.",
    &["parallel", "current", "worker", "pool"],
    GET_CURRENT_WORKER_DESCRIPTOR
);
current_execution_entry!(
    GET_CURRENT_JOB_CATALOG_ENTRY,
    "getCurrentJob",
    "parallel.get-current-job",
    "Return the job executing the current function.",
    &["parallel", "current", "job", "worker"],
    GET_CURRENT_JOB_DESCRIPTOR
);
