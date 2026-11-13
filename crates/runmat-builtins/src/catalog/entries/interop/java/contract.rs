//! Shared Java foreign boundary facts; each public identity owns its forms.

use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub const JAVA_ENVIRONMENT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.javaclient.JavaEnvironment");

/// Public feature identity, independent of host availability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JavaFeature {
    Jvm,
    Desktop,
    Awt,
    Swing,
}

impl JavaFeature {
    pub fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("jvm") {
            Some(Self::Jvm)
        } else if name.eq_ignore_ascii_case("desktop") {
            Some(Self::Desktop)
        } else if name.eq_ignore_ascii_case("awt") {
            Some(Self::Awt)
        } else if name.eq_ignore_ascii_case("swing") {
            Some(Self::Swing)
        } else {
            None
        }
    }
}

/// Validated public `jenv` name-value control. Version values remain owned by
/// the adapter because their resolution depends on the selected host JVM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JavaEnvironmentControl {
    Version,
    ExecutionMode,
}

/// The currently supported Java execution mode, independent of host spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JavaExecutionMode {
    InProcess,
}

impl JavaExecutionMode {
    pub fn parse(name: &str) -> Option<Self> {
        name.eq_ignore_ascii_case("InProcess")
            .then_some(Self::InProcess)
    }
}

impl JavaEnvironmentControl {
    pub fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("Version") {
            Some(Self::Version)
        } else if name.eq_ignore_ascii_case("ExecutionMode") {
            Some(Self::ExecutionMode)
        } else {
            None
        }
    }
}

pub const JAVA_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Java bridge conversion preserves exact fixed-width signed integers and checks widening; Java array dimensions are checked as integer dimensions by the adapter.",
};

pub(super) const FOREIGN_PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeOnly,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};

pub(super) const FOREIGN_LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::ForeignRuntime(ForeignAdapterFamily::Java),
    execution_stack: ExecutionStackRequirement::Process,
    artifact_dependencies: &[],
};

pub(super) const FOREIGN_CAPABILITIES: &[CapabilityRequirement] =
    &[CapabilityRequirement::ForeignRuntime];
pub(super) const EDT_CAPABILITIES: &[CapabilityRequirement] = &[
    CapabilityRequirement::UserInterface,
    CapabilityRequirement::ForeignRuntime,
];
pub(super) const FOREIGN_EFFECTS: &[EffectKind] = &[
    EffectKind::HostCallback,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
    EffectKind::Unknown,
];
pub(super) const EDT_EFFECTS: &[EffectKind] = &[
    EffectKind::UserInterface,
    EffectKind::HostCallback,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
    EffectKind::Unknown,
];
pub(super) const CLASSPATH_EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentWrite,
    EffectKind::FilesystemRead,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub(super) struct JavaEntryBehavior {
    pub effects: &'static [EffectKind],
    pub capabilities: &'static [CapabilityRequirement],
    pub purity: BuiltinPurity,
    pub async_behavior: BuiltinAsyncBehavior,
    pub placement: BuiltinPlacementContract,
    pub link: BuiltinLinkContract,
    pub suppress_auto_output: bool,
}

pub(super) const fn entry(
    provenance: BuiltinCatalogProvenance,
    name: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    infer: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
    behavior: JavaEntryBehavior,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance,
        identity: BuiltinCatalogIdentity { name },
        category: "interop/java",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: behavior.async_behavior,
            purity: behavior.purity,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: behavior.effects,
            capabilities: behavior.capabilities,
        },
        placement: behavior.placement,
        link: behavior.link,
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: Some(&JAVA_INTEGER_AUDIT),
        suppress_auto_output: behavior.suppress_auto_output,
    }
}
