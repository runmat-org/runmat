//! Shared public facts for the ten path-level dataset operations.

use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub use super::super::data_family::DATASET_CLASS;

pub(super) const READ_EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemRead,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(super) const WRITE_EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(super) const COPY_EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemRead,
    EffectKind::FilesystemWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(super) const CREATE_EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemWrite,
    EffectKind::Clock,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(super) const FILESYSTEM_CAPABILITIES: &[CapabilityRequirement] =
    &[CapabilityRequirement::Filesystem];

pub(super) const PATH_PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};
pub(super) const PATH_LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

pub const DATA_NAMESPACE_LIFECYCLE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Path-level dataset movement and inspection accept text paths/formats, not integer controls. Opaque integer payload copying is governed by DataArray storage; data.create's nested shape/chunk integer form has its own capability descriptor.",
    };

pub(super) struct PathEntryBehavior {
    pub effects: &'static [EffectKind],
    pub purity: BuiltinPurity,
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub integer_audit: Option<&'static BuiltinIntegerAuditDescriptor>,
}

pub(super) const fn entry(
    provenance: BuiltinCatalogProvenance,
    name: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    infer: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
    behavior: PathEntryBehavior,
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
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: behavior.purity,
            semantic_kind: BuiltinSemanticKind::Filesystem,
            workspace_effect: None,
            environment_effect: None,
            effects: behavior.effects,
            capabilities: FILESYSTEM_CAPABILITIES,
        },
        placement: PATH_PLACEMENT,
        link: PATH_LINK,
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: behavior.integer_capabilities,
        integer_audit: behavior.integer_audit,
        suppress_auto_output: false,
    }
}
