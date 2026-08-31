use super::*;

const LOCAL_PART_INPUTS: [BuiltinParamDescriptor; 1] = [ANY_REQUIRED];
const IS_COMPLETE_INPUTS: [BuiltinParamDescriptor; 1] = [ANY_REQUIRED];
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

signature!(
    CODISTRIBUTOR_SIGNATURES,
    "codist = codistributor(scheme, first_parameter, second_parameter)",
    &CODISTRIBUTOR_INPUTS,
    &ANY_OUTPUT
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

descriptor!(CODISTRIBUTOR_DESCRIPTOR, CODISTRIBUTOR_SIGNATURES);
descriptor!(CODISTRIBUTOR_1D_DESCRIPTOR, CODISTRIBUTOR_1D_SIGNATURES);
descriptor!(CODISTRIBUTOR_2DBC_DESCRIPTOR, CODISTRIBUTOR_2DBC_SIGNATURES);
descriptor!(IS_COMPLETE_DESCRIPTOR, IS_COMPLETE_SIGNATURES);
descriptor!(IS_CODISTRIBUTED_DESCRIPTOR, IS_CODISTRIBUTED_SIGNATURES);

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

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &CODISTRIBUTOR_1D_CATALOG_ENTRY,
    &CODISTRIBUTOR_2DBC_CATALOG_ENTRY,
    &CODISTRIBUTOR_CATALOG_ENTRY,
    &IS_CODISTRIBUTED_CATALOG_ENTRY,
    &IS_COMPLETE_CATALOG_ENTRY,
];
