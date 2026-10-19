//! Executable autodiff callback fixtures. These names are available to the
//! runtime test harness but are not public builtin documentation identities.

use crate::*;
use runmat_types::{CallContract, CallInference, CallRequest, DynamicReason, EffectKind};

mod contract;
pub use contract::{DEEP_LEARNING_NETWORK_GRAD_DESCRIPTOR, DEEP_LEARNING_SQUARE_GRAD_DESCRIPTOR};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::DynamicDispatch);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    runmat_types::infer_call(&contract, request)
}

const EFFECTS: &[EffectKind] = &[
    EffectKind::HostCallback,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
    EffectKind::Unknown,
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

const SQUARE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("__deep_learning_square_grad"),
    slug: Some("__deep_learning_square_grad"),
    summary: "Internal dlarray gradient callback fixture.",
    description: "The test callback squares its dlarray input, sums the loss, and returns the loss and gradient. Errors from the invoked arithmetic and autodiff operations retain their original causes.",
    keywords: &["dlarray", "dlgradient", "fixture"],
    example_exemption: Some("Internal runtime fixture exercised by dlfeval and gradient tests rather than the public example runner."),
    ..BuiltinDocumentation::EMPTY
};

const NETWORK_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("__deep_learning_network_grad"),
    slug: Some("__deep_learning_network_grad"),
    summary: "Internal dlnetwork gradient callback fixture.",
    description: "The test callback forwards a network, sums its output, and returns gradients for the network. Errors from forward execution and autodiff retain their original causes.",
    keywords: &["dlnetwork", "dlgradient", "fixture"],
    example_exemption: Some("Internal runtime fixture exercised by dlnetwork gradient tests rather than the public example runner."),
    ..BuiltinDocumentation::EMPTY
};

pub const DEEP_LEARNING_SQUARE_GRAD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Fixture,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "__deep_learning_square_grad",
    },
    category: "deep_learning/tests",
    documentation: SQUARE_DOCUMENTATION,
    descriptor: &DEEP_LEARNING_SQUARE_GRAD_DESCRIPTOR,
    contract: CONTRACT,
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};

pub const DEEP_LEARNING_NETWORK_GRAD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Fixture,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "__deep_learning_network_grad",
    },
    category: "deep_learning/tests",
    documentation: NETWORK_DOCUMENTATION,
    descriptor: &DEEP_LEARNING_NETWORK_GRAD_DESCRIPTOR,
    contract: CONTRACT,
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &DEEP_LEARNING_SQUARE_GRAD_CATALOG_ENTRY,
    &DEEP_LEARNING_NETWORK_GRAD_CATALOG_ENTRY,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_fixture_forms_keep_exact_private_output_contracts() {
        for entry in ENTRIES {
            assert_eq!(entry.audience, BuiltinCatalogAudience::Fixture);
            assert_eq!(
                entry.descriptor.completion_policy,
                BuiltinCompletionPolicy::HiddenInternal
            );
            assert!(entry.documentation.example_exemption.is_some());
            assert!(entry.contract.effects.contains(&EffectKind::HostCallback));
            assert!(entry.contract.effects.contains(&EffectKind::MaySuspend));
            assert!(entry.contract.effects.contains(&EffectKind::MayThrow));
        }
        assert_eq!(
            DEEP_LEARNING_SQUARE_GRAD_DESCRIPTOR.signatures[0]
                .inputs
                .len(),
            1
        );
        assert_eq!(
            DEEP_LEARNING_SQUARE_GRAD_DESCRIPTOR.signatures[0]
                .outputs
                .len(),
            2
        );
        assert_eq!(
            DEEP_LEARNING_NETWORK_GRAD_DESCRIPTOR.signatures[0]
                .inputs
                .len(),
            2
        );
        assert_eq!(
            DEEP_LEARNING_NETWORK_GRAD_DESCRIPTOR.signatures[0]
                .outputs
                .len(),
            1
        );
        assert_eq!(
            DEEP_LEARNING_SQUARE_GRAD_DESCRIPTOR.output_mode,
            BuiltinOutputMode::ByRequestedOutputCount
        );
        assert_eq!(
            DEEP_LEARNING_NETWORK_GRAD_DESCRIPTOR.output_mode,
            BuiltinOutputMode::Fixed
        );
    }
}
