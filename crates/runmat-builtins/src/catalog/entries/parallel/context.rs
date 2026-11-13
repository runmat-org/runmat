use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinFusionPolicy,
    BuiltinInferenceRule, BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract,
    BuiltinPortability, BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy,
    BuiltinSemanticKind, BuiltinSignatureDescriptor, IdentityInferenceRule,
};
use runmat_types::ExecutionStackRequirement;

use super::documentation::{
    LABINDEX_DOCUMENTATION, NUMLABS_DOCUMENTATION, SPMD_INDEX_DOCUMENTATION,
    SPMD_SIZE_DOCUMENTATION,
};

mod inference;

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
macro_rules! context_descriptor {
    ($descriptor:ident, $signatures:ident, $error:ident, $label:literal, $output:ident, $code:literal, $identifier:literal, $message:literal) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &[],
            outputs: &$output,
        }];
        pub const $error: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($identifier),
            when: "More than one output is requested.",
            message: $message,
        };
        pub const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &[$error],
        };
    };
}

context_descriptor!(
    SPMD_INDEX_DESCRIPTOR,
    SPMD_INDEX_SIGNATURES,
    SPMD_INDEX_TOO_MANY_OUTPUTS,
    "index = spmdIndex()",
    INDEX_OUTPUT,
    "RM.SPMD_INDEX.TOO_MANY_OUTPUTS",
    "RunMat:spmdIndex:TooManyOutputs",
    "spmdIndex: too many output arguments"
);
context_descriptor!(
    SPMD_SIZE_DESCRIPTOR,
    SPMD_SIZE_SIGNATURES,
    SPMD_SIZE_TOO_MANY_OUTPUTS,
    "size = spmdSize()",
    SIZE_OUTPUT,
    "RM.SPMD_SIZE.TOO_MANY_OUTPUTS",
    "RunMat:spmdSize:TooManyOutputs",
    "spmdSize: too many output arguments"
);
context_descriptor!(
    LABINDEX_DESCRIPTOR,
    LABINDEX_SIGNATURES,
    LABINDEX_TOO_MANY_OUTPUTS,
    "index = labindex",
    INDEX_OUTPUT,
    "RM.LABINDEX.TOO_MANY_OUTPUTS",
    "RunMat:labindex:TooManyOutputs",
    "labindex: too many output arguments"
);
context_descriptor!(
    NUMLABS_DESCRIPTOR,
    NUMLABS_SIGNATURES,
    NUMLABS_TOO_MANY_OUTPUTS,
    "size = numlabs",
    SIZE_OUTPUT,
    "RM.NUMLABS.TOO_MANY_OUTPUTS",
    "RunMat:numlabs:TooManyOutputs",
    "numlabs: too many output arguments"
);
const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: crate::BuiltinOperandPreparation::RuntimeOwned,
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
    ($constant:ident, $name:literal, $rule:expr, $documentation:expr, $descriptor:ident) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: $documentation,
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
    BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference::infer)),
    SPMD_INDEX_DOCUMENTATION,
    SPMD_INDEX_DESCRIPTOR
);
context_entry!(
    SPMD_SIZE_CATALOG_ENTRY,
    "spmdSize",
    BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference::infer)),
    SPMD_SIZE_DOCUMENTATION,
    SPMD_SIZE_DESCRIPTOR
);
context_entry!(
    LABINDEX_CATALOG_ENTRY,
    "labindex",
    BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference::infer)),
    LABINDEX_DOCUMENTATION,
    LABINDEX_DESCRIPTOR
);
context_entry!(
    NUMLABS_CATALOG_ENTRY,
    "numlabs",
    BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference::infer)),
    NUMLABS_DOCUMENTATION,
    NUMLABS_DESCRIPTOR
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &SPMD_INDEX_CATALOG_ENTRY,
    &SPMD_SIZE_CATALOG_ENTRY,
    &LABINDEX_CATALOG_ENTRY,
    &NUMLABS_CATALOG_ENTRY,
];
