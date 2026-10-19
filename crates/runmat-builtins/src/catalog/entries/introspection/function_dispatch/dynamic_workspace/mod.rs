//! Catalog-owned contracts for source evaluation and workspace assignment.
//! Runtime execution remains with the selected VM workspace frame.

mod contract;
mod documentation;
mod inference;

pub use contract::*;

use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

const EVAL_EFFECTS: &[EffectKind] = &[
    EffectKind::WorkspaceRead,
    EffectKind::WorkspaceWrite,
    EffectKind::HostCallback,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
    EffectKind::Unknown,
];
const ASSIGNIN_EFFECTS: &[EffectKind] = &[
    EffectKind::WorkspaceWrite,
    EffectKind::HostCallback,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
const CAPABILITIES: &[CapabilityRequirement] = &[CapabilityRequirement::HostRuntime];
const BINDINGS: &[BuiltinBindingDeclaration] = &REQUIRED_DEFAULT_BINDING;
const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Dynamic,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};
const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Process,
    artifact_dependencies: &[],
};

pub const EVAL_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "eval" },
    category: "introspection",
    documentation: documentation::EVAL_DOCUMENTATION,
    descriptor: &EVAL_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer_eval,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Workspace,
        workspace_effect: Some(BuiltinWorkspaceEffect::DynamicEvalCurrentWorkspace),
        environment_effect: None,
        effects: EVAL_EFFECTS,
        capabilities: CAPABILITIES,
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: BINDINGS,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};

pub const EVALC_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "evalc" },
    category: "introspection",
    documentation: documentation::EVALC_DOCUMENTATION,
    descriptor: &EVALC_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer_evalc,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Workspace,
        workspace_effect: Some(BuiltinWorkspaceEffect::DynamicEvalCurrentWorkspace),
        environment_effect: None,
        effects: EVAL_EFFECTS,
        capabilities: CAPABILITIES,
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: BINDINGS,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};

pub const EVALIN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "evalin" },
    category: "introspection",
    documentation: documentation::EVALIN_DOCUMENTATION,
    descriptor: &EVALIN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer_evalin,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Workspace,
        workspace_effect: Some(BuiltinWorkspaceEffect::DynamicEval),
        environment_effect: None,
        effects: EVAL_EFFECTS,
        capabilities: CAPABILITIES,
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: BINDINGS,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};

pub const ASSIGNIN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "assignin" },
    category: "introspection",
    documentation: documentation::ASSIGNIN_DOCUMENTATION,
    descriptor: &ASSIGNIN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer_assignin,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Workspace,
        workspace_effect: Some(BuiltinWorkspaceEffect::CreatesBinding),
        environment_effect: None,
        effects: ASSIGNIN_EFFECTS,
        capabilities: CAPABILITIES,
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: BINDINGS,
    extensions: &[],
    integer_capabilities: &ASSIGNIN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
