//! Catalog layout for FEA operations. Each identity supplies its own effect
//! profile, output inference, documentation, and integer boundary facts.

use crate::*;
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, EffectKind, ExecutionStackRequirement,
};

#[derive(Clone, Copy)]
pub enum OperationEffectProfile {
    /// Validation, planning, authoring, or solving persists evidence.
    EvidenceWrite,
    /// Queries existing run artifacts without changing them.
    ArtifactRead,
    /// Figure creation reads run artifacts and changes graphics state.
    Figure,
}

pub struct OperationEntryDefinition {
    pub provenance: BuiltinCatalogProvenance,
    pub name: &'static str,
    pub documentation: BuiltinDocumentation,
    pub descriptor: &'static BuiltinDescriptor,
    pub inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub integer_audit: Option<&'static BuiltinIntegerAuditDescriptor>,
    pub effects: OperationEffectProfile,
    pub accelerator: BuiltinAcceleratorPolicy,
    pub distributed: BuiltinDistributedPolicy,
}

pub const fn entry(definition: OperationEntryDefinition) -> BuiltinCatalogEntry {
    let (purity, semantic_kind, effects, capabilities) = match definition.effects {
        OperationEffectProfile::EvidenceWrite => (
            BuiltinPurity::Impure,
            BuiltinSemanticKind::Filesystem,
            &[
                EffectKind::EnvironmentRead,
                EffectKind::FilesystemRead,
                EffectKind::FilesystemWrite,
                EffectKind::MaySuspend,
                EffectKind::MayThrow,
            ] as &'static [EffectKind],
            &[CapabilityRequirement::Filesystem] as &'static [CapabilityRequirement],
        ),
        OperationEffectProfile::ArtifactRead => (
            BuiltinPurity::DeterministicReadOnly,
            BuiltinSemanticKind::Filesystem,
            &[
                EffectKind::EnvironmentRead,
                EffectKind::FilesystemRead,
                EffectKind::MaySuspend,
                EffectKind::MayThrow,
            ],
            &[CapabilityRequirement::Filesystem],
        ),
        OperationEffectProfile::Figure => (
            BuiltinPurity::Impure,
            BuiltinSemanticKind::Plotting,
            &[
                EffectKind::EnvironmentRead,
                EffectKind::FilesystemRead,
                EffectKind::UserInterface,
                EffectKind::MaySuspend,
                EffectKind::MayThrow,
            ],
            &[
                CapabilityRequirement::Filesystem,
                CapabilityRequirement::UserInterface,
            ],
        ),
    };
    BuiltinCatalogEntry {
        provenance: definition.provenance,
        identity: BuiltinCatalogIdentity {
            name: definition.name,
        },
        category: "fea",
        documentation: definition.documentation,
        descriptor: definition.descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                definition.inference,
            )),
            compatibility: BuiltinCompatibility::RunMat,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity,
            semantic_kind,
            workspace_effect: None,
            environment_effect: None,
            effects,
            capabilities,
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeOnly,
            accelerator: definition.accelerator,
            residency: BuiltinResidencyPolicy::Host,
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: definition.distributed,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: definition.integer_capabilities,
        integer_audit: definition.integer_audit,
        suppress_auto_output: false,
    }
}
