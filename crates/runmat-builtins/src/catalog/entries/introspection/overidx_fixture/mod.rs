//! Internal executable indexing fixtures. These share the canonical runtime
//! binding registry but are not user-facing builtin documentation identities.

use crate::*;
use runmat_types::{CallContract, CallInference, CallRequest, DynamicReason};

mod contract;
pub use contract::*;
mod legacy_methods;
pub use legacy_methods::*;

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::DynamicDispatch);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    runmat_types::infer_call(&contract, request)
}

const SUBSREF_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.subsref"),
    slug: Some("overidx.subsref"),
    summary: "Exercise overloaded OverIdx indexing in runtime tests.",
    description: "Internal executable fixture for ordered standard substruct indexing. Parenthesis indexing returns a sentinel, brace and member indexing read test object properties, and unsupported payloads raise the fixture-specific descriptor error. The callback may return managed properties, so output ownership is preserved through the canonical rooted binding.",
    keywords: &["OverIdx", "subsref", "fixture"],
    example_exemption: Some("Internal runtime fixture; semantic and VM tests construct its object and substruct path directly."),
    ..BuiltinDocumentation::EMPTY
};

const SUBSASGN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("OverIdx.subsasgn"),
    slug: Some("overidx.subsasgn"),
    summary: "Exercise overloaded OverIdx indexed assignment in runtime tests.",
    description: "Internal executable fixture for ordered standard substruct assignment. The terminal callback writes a test object property and may carry managed right-hand-side values; unsupported payloads raise the fixture-specific descriptor error. The updated object remains rooted through canonical publication.",
    keywords: &["OverIdx", "subsasgn", "fixture"],
    example_exemption: Some("Internal runtime fixture; semantic and VM tests construct its object, path, and assigned value directly."),
    ..BuiltinDocumentation::EMPTY
};

const EFFECTS: &[runmat_types::EffectKind] = &[
    runmat_types::EffectKind::HostCallback,
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
    runmat_types::EffectKind::Unknown,
];

const CONTRACT: BuiltinContractDeclaration = BuiltinContractDeclaration {
    maturity: BuiltinContractMaturity::DynamicByDesign,
    inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
    compatibility: BuiltinCompatibility::Matlab,
    async_behavior: BuiltinAsyncBehavior::MaySuspend,
    purity: BuiltinPurity::Impure,
    semantic_kind: BuiltinSemanticKind::General,
    workspace_effect: None,
    environment_effect: None,
    effects: EFFECTS,
    capabilities: &[],
};

const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Optional,
    residency: BuiltinResidencyPolicy::Dynamic,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};

const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Dynamic,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: runmat_types::ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

pub const OVERIDX_SUBSREF_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Fixture,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "OverIdx.subsref",
    },
    category: "general",
    documentation: SUBSREF_DOCUMENTATION,
    descriptor: &OVERIDX_SUBSREF_DESCRIPTOR,
    contract: CONTRACT,
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};

pub const OVERIDX_SUBSASGN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Fixture,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "OverIdx.subsasgn",
    },
    category: "general",
    documentation: SUBSASGN_DOCUMENTATION,
    descriptor: &OVERIDX_SUBSASGN_DESCRIPTOR,
    contract: CONTRACT,
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};

const INDEXING_ENTRIES: &[&BuiltinCatalogEntry] = &[
    &OVERIDX_SUBSREF_CATALOG_ENTRY,
    &OVERIDX_SUBSASGN_CATALOG_ENTRY,
];

pub(super) fn extend_entries(values: &mut Vec<&'static BuiltinCatalogEntry>) {
    values.extend(INDEXING_ENTRIES.iter().copied());
    values.extend(legacy_methods::ENTRIES.iter().copied());
}
