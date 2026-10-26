use crate::*;
mod contract;
mod declaration;
mod documentation;
mod inference;
pub use contract::*;
pub use declaration::{
    is_valid_symbolic_identifier, parse_symbolic_declaration, symbolic_declaration_tokens,
    SymbolicDeclaration, SymbolicDeclarationError,
};
pub const SYMS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "syms" },
    category: "math/symbolic",
    documentation: documentation::DOCUMENTATION,
    descriptor: &SYMS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: Some(BuiltinWorkspaceEffect::CreatesBinding),
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::WorkspaceWrite,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        // Text-only declarations have no valid resident data operands; workspace
        // effects remain a fusion boundary, not an operand-gather operation.
        preparation: crate::BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::Host,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&SYMS_INTEGER_AUDIT),
    suppress_auto_output: true,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&SYMS_CATALOG_ENTRY];
