use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingAvailability,
    BuiltinBindingDeclaration, BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation, BuiltinErrorDescriptor,
    BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinLinkContract, BuiltinLinkPolicy,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinPlacementContract, BuiltinPortability, BuiltinPurity, BuiltinReachability,
    BuiltinResidencyPolicy, BuiltinSemanticKind, BuiltinSignatureDescriptor, ParallelInferenceRule,
};
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

const ANY_REQUIRED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value operated on by the parallel runtime.",
};
const ANY_OPTIONAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional value supplied by the designated lab.",
};
const LAB_REQUIRED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "lab",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based lab index.",
};
const LAB_OPTIONAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "lab",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional one-based source lab index.",
};
const TAG_OPTIONAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tag",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional message tag.",
};
const DIMENSION_OPTIONAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dimension",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("2"),
    description: "One-based concatenation dimension.",
};
const REDUCER_REQUIRED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "reducer",
    ty: BuiltinParamType::Callable,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Associative binary reduction function.",
};
const ANY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result produced by the parallel operation.",
}];

const DISTRIBUTED_INPUTS: [BuiltinParamDescriptor; 1] = [ANY_REQUIRED];
const CODISTRIBUTED_INPUTS: [BuiltinParamDescriptor; 3] = [
    ANY_REQUIRED,
    BuiltinParamDescriptor {
        name: "codistributor_or_worker",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional codistributor or one-based designated worker.",
    },
    BuiltinParamDescriptor {
        name: "codistributor",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Codistributor used with a designated worker.",
    },
];
const CODISTRIBUTED_BUILD_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "local_part",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Partition contributed by the current worker.",
    },
    BuiltinParamDescriptor {
        name: "codistributor",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional codistributor describing the global array.",
    },
    BuiltinParamDescriptor {
        name: "validation",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional noCommunication validation policy.",
    },
];
const LOCAL_PART_INPUTS: [BuiltinParamDescriptor; 1] = [ANY_REQUIRED];
const GLOBAL_INDICES_INPUTS: [BuiltinParamDescriptor; 3] = [
    ANY_REQUIRED,
    BuiltinParamDescriptor {
        name: "dimension",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based distributed dimension.",
    },
    LAB_OPTIONAL,
];
const GLOBAL_INDICES_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "indices_or_first",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Global indices or first global index assigned to the worker.",
    },
    BuiltinParamDescriptor {
        name: "last",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Last global index assigned to the worker.",
    },
];
const IS_COMPLETE_INPUTS: [BuiltinParamDescriptor; 1] = [ANY_REQUIRED];
const REDISTRIBUTE_INPUTS: [BuiltinParamDescriptor; 2] = [ANY_REQUIRED, ANY_REQUIRED];
const CODISTRIBUTOR_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "scheme",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1d"),
        description: "Distribution scheme, 1d or 2dbc.",
    },
    BuiltinParamDescriptor {
        name: "first_parameter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional distribution dimension or worker grid.",
    },
    BuiltinParamDescriptor {
        name: "second_parameter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional partition vector or block size.",
    },
];
const CODISTRIBUTOR_1D_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dimension",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional one-based distribution dimension.",
    },
    BuiltinParamDescriptor {
        name: "partition",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional vector of per-lab partition lengths.",
    },
    BuiltinParamDescriptor {
        name: "global_size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional complete global array size.",
    },
];
const CODISTRIBUTOR_2DBC_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "worker_grid",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional two-element worker grid.",
    },
    BuiltinParamDescriptor {
        name: "block_size",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional positive block size.",
    },
    BuiltinParamDescriptor {
        name: "orientation",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("row"),
        description: "Worker-grid rank orientation, row or col.",
    },
    BuiltinParamDescriptor {
        name: "global_size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional complete matrix size.",
    },
];
const BROADCAST_INPUTS: [BuiltinParamDescriptor; 2] = [LAB_REQUIRED, ANY_OPTIONAL];
const SEND_INPUTS: [BuiltinParamDescriptor; 3] = [ANY_REQUIRED, LAB_REQUIRED, TAG_OPTIONAL];
const RECEIVE_INPUTS: [BuiltinParamDescriptor; 2] = [LAB_OPTIONAL, TAG_OPTIONAL];
const GPLUS_INPUTS: [BuiltinParamDescriptor; 2] = [ANY_REQUIRED, LAB_OPTIONAL];
const GCAT_INPUTS: [BuiltinParamDescriptor; 3] = [ANY_REQUIRED, DIMENSION_OPTIONAL, LAB_OPTIONAL];
const GOP_INPUTS: [BuiltinParamDescriptor; 3] = [REDUCER_REQUIRED, ANY_REQUIRED, LAB_OPTIONAL];
const SEND_RECEIVE_INPUTS: [BuiltinParamDescriptor; 4] =
    [LAB_REQUIRED, LAB_REQUIRED, ANY_REQUIRED, TAG_OPTIONAL];
const RECEIVE_OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Received value.",
    },
    BuiltinParamDescriptor {
        name: "source",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based rank of the sending lab.",
    },
    BuiltinParamDescriptor {
        name: "tag",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Tag attached to the received message.",
    },
];

macro_rules! signature {
    ($name:ident, $label:literal, $inputs:expr, $outputs:expr) => {
        const $name: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: $outputs,
        }];
    };
}

signature!(
    CODISTRIBUTOR_SIGNATURES,
    "codist = codistributor(scheme, first_parameter, second_parameter)",
    &CODISTRIBUTOR_INPUTS,
    &ANY_OUTPUT
);
signature!(
    DISTRIBUTED_SIGNATURES,
    "D = distributed(value)",
    &DISTRIBUTED_INPUTS,
    &ANY_OUTPUT
);
signature!(
    CODISTRIBUTED_SIGNATURES,
    "D = codistributed(value, codistributor_or_worker, codistributor)",
    &CODISTRIBUTED_INPUTS,
    &ANY_OUTPUT
);
signature!(
    CODISTRIBUTED_BUILD_SIGNATURES,
    "D = codistributed.build(local_part, codistributor, validation)",
    &CODISTRIBUTED_BUILD_INPUTS,
    &ANY_OUTPUT
);
signature!(
    REDISTRIBUTE_SIGNATURES,
    "D2 = redistribute(D1, codist)",
    &REDISTRIBUTE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    GET_CODISTRIBUTOR_SIGNATURES,
    "codist = getCodistributor(D)",
    &LOCAL_PART_INPUTS,
    &ANY_OUTPUT
);
signature!(
    GLOBAL_INDICES_SIGNATURES,
    "indices = globalIndices(value, dimension, lab)",
    &GLOBAL_INDICES_INPUTS,
    &GLOBAL_INDICES_OUTPUTS
);
signature!(
    CODISTRIBUTOR_1D_SIGNATURES,
    "codist = codistributor1d(dimension, partition, global_size)",
    &CODISTRIBUTOR_1D_INPUTS,
    &ANY_OUTPUT
);
signature!(
    CODISTRIBUTOR_2DBC_SIGNATURES,
    "codist = codistributor2dbc(worker_grid, block_size, orientation, global_size)",
    &CODISTRIBUTOR_2DBC_INPUTS,
    &ANY_OUTPUT
);
signature!(
    IS_COMPLETE_SIGNATURES,
    "tf = isComplete(codist)",
    &IS_COMPLETE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    IS_CODISTRIBUTED_SIGNATURES,
    "tf = iscodistributed(value)",
    &LOCAL_PART_INPUTS,
    &ANY_OUTPUT
);
signature!(
    SPMD_CAT_SIGNATURES,
    "value = spmdCat(value, dimension, destination)",
    &GCAT_INPUTS,
    &ANY_OUTPUT
);
signature!(
    SPMD_REDUCE_SIGNATURES,
    "value = spmdReduce(reducer, value, destination)",
    &GOP_INPUTS,
    &ANY_OUTPUT
);
signature!(SPMD_BARRIER_SIGNATURES, "spmdBarrier()", &[], &[]);
signature!(
    SPMD_BROADCAST_SIGNATURES,
    "value = spmdBroadcast(source, value)",
    &BROADCAST_INPUTS,
    &ANY_OUTPUT
);
signature!(
    SPMD_SEND_SIGNATURES,
    "spmdSend(value, destination, tag)",
    &SEND_INPUTS,
    &[]
);
signature!(
    SPMD_RECEIVE_SIGNATURES,
    "[value, source, tag] = spmdReceive(source, tag)",
    &RECEIVE_INPUTS,
    &RECEIVE_OUTPUTS
);
signature!(
    SPMD_PROBE_SIGNATURES,
    "ready = spmdProbe(source, tag)",
    &RECEIVE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    LAB_SEND_RECEIVE_SIGNATURES,
    "value = labSendReceive(destination, source, value, tag)",
    &SEND_RECEIVE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    SPMD_SEND_RECEIVE_SIGNATURES,
    "value = spmdSendReceive(destination, source, value, tag)",
    &SEND_RECEIVE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    SPMD_PLUS_SIGNATURES,
    "value = spmdPlus(value, destination)",
    &GPLUS_INPUTS,
    &ANY_OUTPUT
);
signature!(
    LOCAL_PART_SIGNATURES,
    "L = getLocalPart(D)",
    &LOCAL_PART_INPUTS,
    &ANY_OUTPUT
);
signature!(BARRIER_SIGNATURES, "labBarrier()", &[], &[]);
signature!(
    BROADCAST_SIGNATURES,
    "value = labBroadcast(source, value)",
    &BROADCAST_INPUTS,
    &ANY_OUTPUT
);
signature!(
    SEND_SIGNATURES,
    "labSend(value, destination, tag)",
    &SEND_INPUTS,
    &[]
);
signature!(
    RECEIVE_SIGNATURES,
    "value = labReceive(source, tag)",
    &RECEIVE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    PROBE_SIGNATURES,
    "ready = labProbe(source, tag)",
    &RECEIVE_INPUTS,
    &ANY_OUTPUT
);
signature!(
    GPLUS_SIGNATURES,
    "value = gplus(value, destination)",
    &GPLUS_INPUTS,
    &ANY_OUTPUT
);
signature!(
    GCAT_SIGNATURES,
    "value = gcat(value, dimension, destination)",
    &GCAT_INPUTS,
    &ANY_OUTPUT
);
signature!(
    GOP_SIGNATURES,
    "value = gop(reducer, value, destination)",
    &GOP_INPUTS,
    &ANY_OUTPUT
);

const LOWERING_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.PARALLEL.LOWERING_REQUIRED",
    identifier: Some("RunMat:parallel:LoweringRequired"),
    when: "The operation is invoked without an active compiler-owned SPMD or distributed execution context.",
    message: "parallel operation requires executor-aware lowering",
}];

macro_rules! descriptor {
    ($name:ident, $signatures:ident) => {
        pub const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &LOWERING_ERRORS,
        };
    };
}

descriptor!(DISTRIBUTED_DESCRIPTOR, DISTRIBUTED_SIGNATURES);
descriptor!(CODISTRIBUTED_DESCRIPTOR, CODISTRIBUTED_SIGNATURES);
descriptor!(
    CODISTRIBUTED_BUILD_DESCRIPTOR,
    CODISTRIBUTED_BUILD_SIGNATURES
);
descriptor!(GET_LOCAL_PART_DESCRIPTOR, LOCAL_PART_SIGNATURES);
descriptor!(REDISTRIBUTE_DESCRIPTOR, REDISTRIBUTE_SIGNATURES);
descriptor!(GET_CODISTRIBUTOR_DESCRIPTOR, GET_CODISTRIBUTOR_SIGNATURES);
pub const GLOBAL_INDICES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GLOBAL_INDICES_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOWERING_ERRORS,
};
descriptor!(CODISTRIBUTOR_DESCRIPTOR, CODISTRIBUTOR_SIGNATURES);
descriptor!(CODISTRIBUTOR_1D_DESCRIPTOR, CODISTRIBUTOR_1D_SIGNATURES);
descriptor!(CODISTRIBUTOR_2DBC_DESCRIPTOR, CODISTRIBUTOR_2DBC_SIGNATURES);
descriptor!(IS_COMPLETE_DESCRIPTOR, IS_COMPLETE_SIGNATURES);
descriptor!(IS_CODISTRIBUTED_DESCRIPTOR, IS_CODISTRIBUTED_SIGNATURES);
descriptor!(LAB_BARRIER_DESCRIPTOR, BARRIER_SIGNATURES);
descriptor!(LAB_BROADCAST_DESCRIPTOR, BROADCAST_SIGNATURES);
descriptor!(LAB_SEND_DESCRIPTOR, SEND_SIGNATURES);
descriptor!(LAB_RECEIVE_DESCRIPTOR, RECEIVE_SIGNATURES);
descriptor!(LAB_PROBE_DESCRIPTOR, PROBE_SIGNATURES);
descriptor!(GPLUS_DESCRIPTOR, GPLUS_SIGNATURES);
descriptor!(LAB_SEND_RECEIVE_DESCRIPTOR, LAB_SEND_RECEIVE_SIGNATURES);
descriptor!(SPMD_BARRIER_DESCRIPTOR, SPMD_BARRIER_SIGNATURES);
descriptor!(SPMD_BROADCAST_DESCRIPTOR, SPMD_BROADCAST_SIGNATURES);
descriptor!(SPMD_SEND_DESCRIPTOR, SPMD_SEND_SIGNATURES);
pub const SPMD_RECEIVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPMD_RECEIVE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOWERING_ERRORS,
};
descriptor!(SPMD_PROBE_DESCRIPTOR, SPMD_PROBE_SIGNATURES);
descriptor!(SPMD_SEND_RECEIVE_DESCRIPTOR, SPMD_SEND_RECEIVE_SIGNATURES);
descriptor!(SPMD_PLUS_DESCRIPTOR, SPMD_PLUS_SIGNATURES);
descriptor!(SPMD_CAT_DESCRIPTOR, SPMD_CAT_SIGNATURES);
descriptor!(SPMD_REDUCE_DESCRIPTOR, SPMD_REDUCE_SIGNATURES);
descriptor!(GCAT_DESCRIPTOR, GCAT_SIGNATURES);
descriptor!(GOP_DESCRIPTOR, GOP_SIGNATURES);

const PARALLEL_RUNTIME: [CapabilityRequirement; 1] = [CapabilityRequirement::ParallelRuntime];
const PARALLEL_EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
const PARALLEL_PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Dynamic,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: crate::BuiltinDistributedPolicy::Unsupported,
};
const DISTRIBUTED_INSPECTION_PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    distributed: crate::BuiltinDistributedPolicy::InspectHandles,
    ..PARALLEL_PLACEMENT
};
const PARALLEL_LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

macro_rules! parallel_data_entry {
    ($constant:ident, $name:literal, $rule:expr, $summary:literal, $descriptor:ident) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: BuiltinDocumentation {
                summary: $summary,
                keywords: &["parallel", "distributed", "spmd"],
                related: &[],
                introduced: None,
                status: None,
                examples: &[],
            },
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: $rule,
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                purity: BuiltinPurity::Impure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &PARALLEL_EFFECTS,
                capabilities: &PARALLEL_RUNTIME,
            },
            placement: PARALLEL_PLACEMENT,
            link: PARALLEL_LINK,
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

parallel_data_entry!(
    DISTRIBUTED_CATALOG_ENTRY,
    "distributed",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Distributed),
    "Create a distributed array.",
    DISTRIBUTED_DESCRIPTOR
);
parallel_data_entry!(
    CODISTRIBUTED_CATALOG_ENTRY,
    "codistributed",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Codistributed),
    "Create a codistributed array from a client value or designated worker.",
    CODISTRIBUTED_DESCRIPTOR
);
parallel_data_entry!(
    CODISTRIBUTED_BUILD_CATALOG_ENTRY,
    "codistributed.build",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::CodistributedBuild),
    "Build a codistributed array from worker-local partitions.",
    CODISTRIBUTED_BUILD_DESCRIPTOR
);
parallel_data_entry!(
    REDISTRIBUTE_CATALOG_ENTRY,
    "redistribute",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Redistribute),
    "Redistribute an array with another codistributor.",
    REDISTRIBUTE_DESCRIPTOR
);
parallel_data_entry!(
    GET_CODISTRIBUTOR_CATALOG_ENTRY,
    "getCodistributor",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::GetCodistributor),
    "Return the codistributor for a distributed array.",
    GET_CODISTRIBUTOR_DESCRIPTOR
);
parallel_data_entry!(
    GLOBAL_INDICES_CATALOG_ENTRY,
    "globalIndices",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::GlobalIndices),
    "Return the global indices assigned to a worker.",
    GLOBAL_INDICES_DESCRIPTOR
);

macro_rules! codistributor_entry {
    ($constant:ident, $name:literal, $rule:expr, $summary:literal, $descriptor:ident) => {
        codistributor_entry!(
            $constant,
            $name,
            $rule,
            $summary,
            $descriptor,
            PARALLEL_PLACEMENT
        );
    };
    ($constant:ident, $name:literal, $rule:expr, $summary:literal, $descriptor:ident, $placement:expr) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: BuiltinDocumentation {
                summary: $summary,
                keywords: &["parallel", "distributed", "codistributor"],
                related: &["redistribute"],
                introduced: None,
                status: None,
                examples: &[],
            },
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: $rule,
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::NeverSuspends,
                purity: BuiltinPurity::Pure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &[],
                capabilities: &[],
            },
            placement: $placement,
            link: PARALLEL_LINK,
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

codistributor_entry!(
    CODISTRIBUTOR_CATALOG_ENTRY,
    "codistributor",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Codistributor),
    "Create a one-dimensional or two-dimensional codistributor.",
    CODISTRIBUTOR_DESCRIPTOR
);
codistributor_entry!(
    CODISTRIBUTOR_1D_CATALOG_ENTRY,
    "codistributor1d",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Codistributor1d),
    "Create a one-dimensional codistributor.",
    CODISTRIBUTOR_1D_DESCRIPTOR
);
codistributor_entry!(
    CODISTRIBUTOR_2DBC_CATALOG_ENTRY,
    "codistributor2dbc",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Codistributor2dbc),
    "Create a two-dimensional block-cyclic codistributor.",
    CODISTRIBUTOR_2DBC_DESCRIPTOR
);
codistributor_entry!(
    IS_COMPLETE_CATALOG_ENTRY,
    "isComplete",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::CodistributorIsComplete),
    "Return whether a codistributor has a complete global size.",
    IS_COMPLETE_DESCRIPTOR
);
codistributor_entry!(
    IS_CODISTRIBUTED_CATALOG_ENTRY,
    "iscodistributed",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Iscodistributed),
    "Return whether a value is a codistributed array.",
    IS_CODISTRIBUTED_DESCRIPTOR,
    DISTRIBUTED_INSPECTION_PLACEMENT
);
parallel_data_entry!(
    GET_LOCAL_PART_CATALOG_ENTRY,
    "getLocalPart",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::LocalPart),
    "Return the partition local to the current lab.",
    GET_LOCAL_PART_DESCRIPTOR
);
parallel_data_entry!(
    LAB_BARRIER_CATALOG_ENTRY,
    "labBarrier",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Barrier),
    "Synchronize all labs in the current SPMD region.",
    LAB_BARRIER_DESCRIPTOR
);
parallel_data_entry!(
    LAB_BROADCAST_CATALOG_ENTRY,
    "labBroadcast",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Broadcast),
    "Broadcast a value from one lab to every lab.",
    LAB_BROADCAST_DESCRIPTOR
);
parallel_data_entry!(
    LAB_SEND_CATALOG_ENTRY,
    "labSend",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Send),
    "Send a value to another lab.",
    LAB_SEND_DESCRIPTOR
);
parallel_data_entry!(
    LAB_RECEIVE_CATALOG_ENTRY,
    "labReceive",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Receive),
    "Receive a value from another lab.",
    LAB_RECEIVE_DESCRIPTOR
);
parallel_data_entry!(
    LAB_PROBE_CATALOG_ENTRY,
    "labProbe",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Probe),
    "Test whether a matching lab message is available.",
    LAB_PROBE_DESCRIPTOR
);
parallel_data_entry!(
    GPLUS_CATALOG_ENTRY,
    "gplus",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Gplus),
    "Sum values across labs.",
    GPLUS_DESCRIPTOR
);
parallel_data_entry!(
    GCAT_CATALOG_ENTRY,
    "gcat",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Cat),
    "Concatenate values across labs in rank order.",
    GCAT_DESCRIPTOR
);
parallel_data_entry!(
    GOP_CATALOG_ENTRY,
    "gop",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::FunctionalReduce),
    "Reduce values across labs with an associative binary function.",
    GOP_DESCRIPTOR
);
parallel_data_entry!(
    LAB_SEND_RECEIVE_CATALOG_ENTRY,
    "labSendReceive",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::SendReceive),
    "Send and receive one value as an atomic point-to-point exchange.",
    LAB_SEND_RECEIVE_DESCRIPTOR
);
parallel_data_entry!(
    SPMD_BARRIER_CATALOG_ENTRY,
    "spmdBarrier",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Barrier),
    "Synchronize all labs in the current SPMD region.",
    SPMD_BARRIER_DESCRIPTOR
);
parallel_data_entry!(
    SPMD_BROADCAST_CATALOG_ENTRY,
    "spmdBroadcast",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Broadcast),
    "Broadcast a value from one lab to every lab.",
    SPMD_BROADCAST_DESCRIPTOR
);
parallel_data_entry!(
    SPMD_SEND_CATALOG_ENTRY,
    "spmdSend",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Send),
    "Send a value to another lab.",
    SPMD_SEND_DESCRIPTOR
);
parallel_data_entry!(
    SPMD_RECEIVE_CATALOG_ENTRY,
    "spmdReceive",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Receive),
    "Receive a value and optional sender metadata.",
    SPMD_RECEIVE_DESCRIPTOR
);
parallel_data_entry!(
    SPMD_PROBE_CATALOG_ENTRY,
    "spmdProbe",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Probe),
    "Test whether a matching lab message is available.",
    SPMD_PROBE_DESCRIPTOR
);
parallel_data_entry!(
    SPMD_SEND_RECEIVE_CATALOG_ENTRY,
    "spmdSendReceive",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::SendReceive),
    "Send and receive one value as an atomic point-to-point exchange.",
    SPMD_SEND_RECEIVE_DESCRIPTOR
);
parallel_data_entry!(
    SPMD_PLUS_CATALOG_ENTRY,
    "spmdPlus",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Gplus),
    "Sum values across labs.",
    SPMD_PLUS_DESCRIPTOR
);
parallel_data_entry!(
    SPMD_CAT_CATALOG_ENTRY,
    "spmdCat",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Cat),
    "Concatenate values across SPMD workers in rank order.",
    SPMD_CAT_DESCRIPTOR
);
parallel_data_entry!(
    SPMD_REDUCE_CATALOG_ENTRY,
    "spmdReduce",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::FunctionalReduce),
    "Reduce values across SPMD workers with an associative binary function.",
    SPMD_REDUCE_DESCRIPTOR
);
