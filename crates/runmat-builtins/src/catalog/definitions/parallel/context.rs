use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation, BuiltinFusionPolicy,
    BuiltinInferenceRule, BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract,
    BuiltinPortability, BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy,
    BuiltinSemanticKind, BuiltinSignatureDescriptor, ParallelInferenceRule,
};
use runmat_types::ExecutionStackRequirement;

const INDEX_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "index",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based index of the current SPMD worker.",
}];
const SIZE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of workers in the current SPMD group.",
}];
const INDEX_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "index = spmdIndex",
    inputs: &[],
    outputs: &INDEX_OUTPUT,
}];
const SIZE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "size = spmdSize",
    inputs: &[],
    outputs: &SIZE_OUTPUT,
}];
const INDEX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INDEX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const SIZE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIZE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: crate::BuiltinDistributedPolicy::Unsupported,
};
const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

macro_rules! context_entry {
    ($constant:ident, $name:literal, $rule:expr, $summary:literal, $descriptor:ident) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: BuiltinDocumentation {
                summary: $summary,
                keywords: &["parallel", "spmd", "worker"],
                related: &["spmdIndex", "spmdSize"],
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
                purity: BuiltinPurity::DeterministicReadOnly,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &[],
                capabilities: &[],
            },
            placement: PLACEMENT,
            link: LINK,
            bindings: &crate::REQUIRED_DEFAULT_BINDING,
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

context_entry!(
    SPMD_INDEX_CATALOG_ENTRY,
    "spmdIndex",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::SpmdIndex),
    "Return the index of the current SPMD worker.",
    INDEX_DESCRIPTOR
);
context_entry!(
    SPMD_SIZE_CATALOG_ENTRY,
    "spmdSize",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::SpmdSize),
    "Return the number of workers in the current SPMD group.",
    SIZE_DESCRIPTOR
);
context_entry!(
    LABINDEX_CATALOG_ENTRY,
    "labindex",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::SpmdIndex),
    "Return the index of the current SPMD worker.",
    INDEX_DESCRIPTOR
);
context_entry!(
    NUMLABS_CATALOG_ENTRY,
    "numlabs",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::SpmdSize),
    "Return the number of workers in the current SPMD group.",
    SIZE_DESCRIPTOR
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &SPMD_INDEX_CATALOG_ENTRY,
    &SPMD_SIZE_CATALOG_ENTRY,
    &LABINDEX_CATALOG_ENTRY,
    &NUMLABS_CATALOG_ENTRY,
];
