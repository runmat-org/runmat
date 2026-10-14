use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub const SAVE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "save" },
    category: "io/mat",
    documentation: super::documentation::SAVE,
    descriptor: &super::SAVE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(super::inference::save)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: Some(BuiltinWorkspaceEffect::ReadsWorkspace),
        environment_effect: None,
        effects: &[EffectKind::WorkspaceRead, EffectKind::EnvironmentRead, EffectKind::FilesystemRead, EffectKind::FilesystemWrite, EffectKind::HostCallback, EffectKind::MaySuspend, EffectKind::MayThrow, EffectKind::Unknown],
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
    extensions: &[super::SAVE_STATUS_EXTENSION],
    integer_capabilities: &[],
    integer_audit: Some(&BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Inputs select workspace names and modes. Persisted integer values retain exact signed/unsigned storage in the MAT encoder, including nested containers; no arithmetic is performed.",
    }),
    suppress_auto_output: false,
};
