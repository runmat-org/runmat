mod contract;
mod documentation;
mod examples;
mod inference;

use crate::*;
pub use contract::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub const RUN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "run" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &RUN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference::infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: Some(BuiltinWorkspaceEffect::LoadsExternalBindings),
        environment_effect: None,
        // Source is resolved at runtime; its callbacks and effects are not
        // constrained by the path argument's type.
        effects: &[EffectKind::WorkspaceRead, EffectKind::WorkspaceWrite,
            EffectKind::EnvironmentRead, EffectKind::FilesystemRead,
            EffectKind::MaySuspend, EffectKind::MayThrow, EffectKind::Unknown],
        capabilities: &[CapabilityRequirement::HostRuntime, CapabilityRequirement::Filesystem],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::Host,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "The script path is scalar text. Numeric and resident numeric paths reject before provider access; values computed inside the script retain their own contracts.",
    }),
    suppress_auto_output: true,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&RUN_CATALOG_ENTRY];
