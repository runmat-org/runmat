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
    /// As `EvidenceWrite`, but the operation completes without suspending.
    SynchronousEvidenceWrite,
    /// As `ArtifactRead`, but the query completes without suspending.
    SynchronousArtifactRead,
    /// As `Figure`, but figure generation completes without suspending.
    SynchronousFigure,
}

impl OperationEffectProfile {
    const fn suspends(self) -> bool {
        matches!(
            self,
            Self::EvidenceWrite | Self::ArtifactRead | Self::Figure
        )
    }
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
    let (purity, semantic_kind, effects, capabilities): (
        BuiltinPurity,
        BuiltinSemanticKind,
        &'static [EffectKind],
        &'static [CapabilityRequirement],
    ) = match definition.effects {
        OperationEffectProfile::SynchronousEvidenceWrite => (
            BuiltinPurity::Impure,
            BuiltinSemanticKind::Filesystem,
            &[
                EffectKind::EnvironmentRead,
                EffectKind::FilesystemRead,
                EffectKind::FilesystemWrite,
                EffectKind::MayThrow,
            ] as &'static [EffectKind],
            // Evidence persists to the configured artifact store, which may
            // be in memory; filesystem access is not a universal prerequisite.
            &[] as &'static [CapabilityRequirement],
        ),
        OperationEffectProfile::SynchronousArtifactRead => (
            BuiltinPurity::DeterministicReadOnly,
            BuiltinSemanticKind::Filesystem,
            &[
                EffectKind::EnvironmentRead,
                EffectKind::FilesystemRead,
                EffectKind::MayThrow,
            ],
            &[],
        ),
        OperationEffectProfile::SynchronousFigure => (
            BuiltinPurity::Impure,
            BuiltinSemanticKind::Plotting,
            &[
                EffectKind::EnvironmentRead,
                EffectKind::FilesystemRead,
                EffectKind::UserInterface,
                EffectKind::MayThrow,
            ],
            &[CapabilityRequirement::UserInterface],
        ),
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
            // The configured artifact store may be in memory. FilesystemRead
            // records the default backend's possible effect, not a universal
            // capability prerequisite for this identity.
            &[],
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
            &[CapabilityRequirement::UserInterface],
        ),
    };
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
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
            async_behavior: if definition.effects.suspends() {
                BuiltinAsyncBehavior::MaySuspend
            } else {
                BuiltinAsyncBehavior::NeverSuspends
            },
            purity,
            semantic_kind,
            workspace_effect: None,
            environment_effect: None,
            effects,
            capabilities,
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            // FEA entrypoints are registered on wasm too. Browser calls need
            // an installed filesystem provider for paths/evidence and the
            // appropriate plotting host for figures; their absence is a
            // capability/service failure, not a blanket native-only identity.
            portability: BuiltinPortability::NativeAndWasm,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_readers_and_figures_admit_provider_suspension() {
        use crate::catalog::{
            FEA_COMPARE_CATALOG_ENTRY, FEA_FIELD_CATALOG_ENTRY, FEA_FIELD_PAGE_CATALOG_ENTRY,
            FEA_MESH_METHOD_CATALOG_ENTRY, FEA_PLOT_METHOD_CATALOG_ENTRY,
            FEA_RESULTS_CATALOG_ENTRY, FEA_TRENDS_CATALOG_ENTRY,
        };
        for (entry, purity, kind) in [
            (
                &FEA_RESULTS_CATALOG_ENTRY,
                BuiltinPurity::DeterministicReadOnly,
                BuiltinSemanticKind::Filesystem,
            ),
            (
                &FEA_FIELD_CATALOG_ENTRY,
                BuiltinPurity::DeterministicReadOnly,
                BuiltinSemanticKind::Filesystem,
            ),
            (
                &FEA_FIELD_PAGE_CATALOG_ENTRY,
                BuiltinPurity::DeterministicReadOnly,
                BuiltinSemanticKind::Filesystem,
            ),
            (
                &FEA_COMPARE_CATALOG_ENTRY,
                BuiltinPurity::DeterministicReadOnly,
                BuiltinSemanticKind::Filesystem,
            ),
            (
                &FEA_TRENDS_CATALOG_ENTRY,
                BuiltinPurity::DeterministicReadOnly,
                BuiltinSemanticKind::Filesystem,
            ),
            (
                &FEA_PLOT_METHOD_CATALOG_ENTRY,
                BuiltinPurity::Impure,
                BuiltinSemanticKind::Plotting,
            ),
            (
                &FEA_MESH_METHOD_CATALOG_ENTRY,
                BuiltinPurity::Impure,
                BuiltinSemanticKind::Plotting,
            ),
        ] {
            let contract = &entry.contract;
            assert_eq!(
                contract.async_behavior,
                BuiltinAsyncBehavior::MaySuspend,
                "{}",
                entry.identity.name
            );
            assert!(contract.effects.contains(&EffectKind::MaySuspend));
            assert!(contract.effects.contains(&EffectKind::FilesystemRead));
            assert_eq!(contract.purity, purity);
            assert_eq!(contract.semantic_kind, kind);
            // In-memory stores need no universal filesystem capability.
            assert!(!contract
                .capabilities
                .contains(&CapabilityRequirement::Filesystem));
        }
    }
}
