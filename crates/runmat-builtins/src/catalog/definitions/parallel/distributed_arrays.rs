use super::*;

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
const REDISTRIBUTE_INPUTS: [BuiltinParamDescriptor; 2] = [ANY_REQUIRED, ANY_REQUIRED];

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
    LOCAL_PART_SIGNATURES,
    "L = getLocalPart(D)",
    &LOCAL_PART_INPUTS,
    &ANY_OUTPUT
);

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
parallel_data_entry!(
    GET_LOCAL_PART_CATALOG_ENTRY,
    "getLocalPart",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::LocalPart),
    "Return the partition local to the current lab.",
    GET_LOCAL_PART_DESCRIPTOR
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &CODISTRIBUTED_BUILD_CATALOG_ENTRY,
    &CODISTRIBUTED_CATALOG_ENTRY,
    &DISTRIBUTED_CATALOG_ENTRY,
    &GET_CODISTRIBUTOR_CATALOG_ENTRY,
    &GET_LOCAL_PART_CATALOG_ENTRY,
    &GLOBAL_INDICES_CATALOG_ENTRY,
    &REDISTRIBUTE_CATALOG_ENTRY,
];
