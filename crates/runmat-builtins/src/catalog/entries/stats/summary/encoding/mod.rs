//! Categorical and dummy-variable encoding contracts.
mod contract;
mod documentation;
mod inference;
mod options;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::{
    DUMMYVAR_DESCRIPTOR, DUMMYVAR_EXTENSIONS, DUMMYVAR_GPU_GROUP_EXTENSION,
    DUMMYVAR_INTEGER_CAPABILITIES, DUMMYVAR_INTEGER_GROUP_EXTENSION,
    ONEHOTDECODE_BROAD_ENCODED_EXTENSION, ONEHOTDECODE_DESCRIPTOR,
    ONEHOTDECODE_EXPLICIT_GPU_EXTENSION, ONEHOTDECODE_EXTENSIONS,
    ONEHOTDECODE_INTEGER_CAPABILITIES, ONEHOTDECODE_LOGICAL_CELL_OUTPUT_EXTENSION,
    ONEHOTDECODE_OUTPUT_TYPE_ALIAS_EXTENSION, ONEHOTENCODE_DESCRIPTOR,
    ONEHOTENCODE_EXPLICIT_GPU_EXTENSION, ONEHOTENCODE_EXTENSIONS,
    ONEHOTENCODE_INTEGER_CAPABILITIES, ONEHOTENCODE_OUTPUT_TYPE_ALIAS_EXTENSION,
};
pub use options::{EncodingOptionName, EncodingOutputKind};

const EFFECTS: &[EffectKind] = &[EffectKind::MaySuspend, EffectKind::MayThrow];
const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Optional,
    residency: BuiltinResidencyPolicy::Dynamic,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::MaterializeArguments,
};
const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

fn dummyvar_infer(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::dummyvar(request, entry)
}
fn encode_infer(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::encode(request, entry)
}
fn decode_infer(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::decode(request, entry)
}

pub const DUMMYVAR_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "dummyvar" },
    category: "stats/summary",
    documentation: documentation::DUMMYVAR,
    descriptor: &DUMMYVAR_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(dummyvar_infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &DUMMYVAR_EXTENSIONS,
    integer_capabilities: &DUMMYVAR_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub const ONEHOTENCODE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "onehotencode",
    },
    category: "stats/summary",
    documentation: documentation::ONEHOTENCODE,
    descriptor: &ONEHOTENCODE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(encode_infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &ONEHOTENCODE_EXTENSIONS,
    integer_capabilities: &ONEHOTENCODE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub const ONEHOTDECODE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "onehotdecode",
    },
    category: "stats/summary",
    documentation: documentation::ONEHOTDECODE,
    descriptor: &ONEHOTDECODE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(decode_infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &ONEHOTDECODE_EXTENSIONS,
    integer_capabilities: &ONEHOTDECODE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &DUMMYVAR_CATALOG_ENTRY,
    &ONEHOTENCODE_CATALOG_ENTRY,
    &ONEHOTDECODE_CATALOG_ENTRY,
];
