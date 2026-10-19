//! Common receiver-call facts; each identity retains its own form and documentation.

use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub(super) const ACCESS_EFFECTS: &[EffectKind] = &[EffectKind::MayThrow];
pub(super) const READ_EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemRead,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(super) const WRITE_EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemRead,
    EffectKind::FilesystemWrite,
    EffectKind::Clock,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(super) const SNAPSHOT_EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemRead,
    EffectKind::FilesystemWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(super) const BEGIN_EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::EnvironmentWrite,
    EffectKind::FilesystemRead,
    EffectKind::Clock,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(super) const TRANSACTION_READ_EFFECTS: &[EffectKind] =
    &[EffectKind::EnvironmentRead, EffectKind::MayThrow];
pub(super) const TRANSACTION_QUEUE_EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::EnvironmentWrite,
    EffectKind::MayThrow,
];
pub(super) const TRANSACTION_COMMIT_EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::EnvironmentWrite,
    EffectKind::FilesystemRead,
    EffectKind::FilesystemWrite,
    EffectKind::Clock,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(super) const FILESYSTEM: &[CapabilityRequirement] = &[CapabilityRequirement::Filesystem];

pub(super) const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};
pub(super) const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

pub(super) struct Behavior {
    pub effects: &'static [EffectKind],
    pub capabilities: &'static [CapabilityRequirement],
    pub purity: BuiltinPurity,
    pub async_behavior: BuiltinAsyncBehavior,
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub integer_audit: Option<&'static BuiltinIntegerAuditDescriptor>,
}

pub(super) const fn entry(
    provenance: BuiltinCatalogProvenance,
    name: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    infer: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
    behavior: Behavior,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance,
        identity: BuiltinCatalogIdentity { name },
        category: "io/data",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::RunMat,
            async_behavior: behavior.async_behavior,
            purity: behavior.purity,
            semantic_kind: BuiltinSemanticKind::Filesystem,
            workspace_effect: None,
            environment_effect: None,
            effects: behavior.effects,
            capabilities: behavior.capabilities,
        },
        placement: PLACEMENT,
        link: LINK,
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: behavior.integer_capabilities,
        integer_audit: behavior.integer_audit,
        suppress_auto_output: false,
    }
}
