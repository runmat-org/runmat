//! Catalog entry layout shared by the meshing builtins.

use super::MESHING_INTEGER_AUDIT;
use crate::*;
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, EffectKind, ExecutionStackRequirement,
};

#[derive(Clone, Copy)]
pub(in crate::catalog::entries::meshing) enum MeshingEffects {
    /// Pure decoding of an in-memory meshing.Mesh.
    Inspection,
    /// Reads two artifact files.
    Load,
    /// Publishes two artifact files.
    Save,
    /// Runs the installed native meshing provider against the asset's source file.
    Generate,
    /// Draws a figure.
    Display,
}

pub(in crate::catalog::entries::meshing) struct MeshingEntryDefinition {
    pub provenance: BuiltinCatalogProvenance,
    pub name: &'static str,
    pub documentation: BuiltinDocumentation,
    pub descriptor: &'static BuiltinDescriptor,
    pub inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    pub effects: MeshingEffects,
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
}

pub(in crate::catalog::entries::meshing) const fn entry(
    definition: MeshingEntryDefinition,
) -> BuiltinCatalogEntry {
    let (purity, semantic_kind, effects, capabilities, portability): (
        BuiltinPurity,
        BuiltinSemanticKind,
        &'static [EffectKind],
        &'static [CapabilityRequirement],
        BuiltinPortability,
    ) = match definition.effects {
        MeshingEffects::Inspection => (
            BuiltinPurity::Pure,
            BuiltinSemanticKind::General,
            &[EffectKind::MayThrow],
            &[],
            BuiltinPortability::NativeAndWasm,
        ),
        MeshingEffects::Load => (
            BuiltinPurity::DeterministicReadOnly,
            BuiltinSemanticKind::Filesystem,
            &[EffectKind::FilesystemRead, EffectKind::MayThrow],
            &[CapabilityRequirement::Filesystem],
            BuiltinPortability::NativeAndWasm,
        ),
        MeshingEffects::Save => (
            BuiltinPurity::Impure,
            BuiltinSemanticKind::Filesystem,
            &[
                EffectKind::FilesystemRead,
                EffectKind::FilesystemWrite,
                EffectKind::MayThrow,
            ],
            &[CapabilityRequirement::Filesystem],
            BuiltinPortability::NativeAndWasm,
        ),
        MeshingEffects::Generate => (
            BuiltinPurity::Impure,
            BuiltinSemanticKind::General,
            &[
                EffectKind::FilesystemRead,
                EffectKind::Clock,
                EffectKind::HostCallback,
                EffectKind::MayThrow,
            ],
            &[
                CapabilityRequirement::HostRuntime,
                CapabilityRequirement::Filesystem,
                CapabilityRequirement::NativeCode,
                CapabilityRequirement::ParallelRuntime,
            ],
            BuiltinPortability::NativeOnly,
        ),
        MeshingEffects::Display => (
            BuiltinPurity::Impure,
            BuiltinSemanticKind::Plotting,
            &[EffectKind::UserInterface, EffectKind::MayThrow],
            &[CapabilityRequirement::UserInterface],
            BuiltinPortability::NativeAndWasm,
        ),
    };
    let audit = match definition.effects {
        MeshingEffects::Generate | MeshingEffects::Display => None,
        _ => Some(&MESHING_INTEGER_AUDIT),
    };
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: definition.provenance,
        identity: BuiltinCatalogIdentity {
            name: definition.name,
        },
        category: "meshing",
        documentation: definition.documentation,
        descriptor: definition.descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                definition.inference,
            )),
            compatibility: BuiltinCompatibility::RunMat,
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            purity,
            semantic_kind,
            workspace_effect: None,
            environment_effect: None,
            effects,
            capabilities,
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability,
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
        integer_capabilities: definition.integer_capabilities,
        integer_audit: audit,
        suppress_auto_output: matches!(definition.effects, MeshingEffects::Display),
    }
}
