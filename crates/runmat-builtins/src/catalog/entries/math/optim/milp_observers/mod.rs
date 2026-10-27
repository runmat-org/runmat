//! Canonical public observers for the mixed-integer event contract.
mod kind;
mod descriptor;
mod inference;
mod documentation;
#[cfg(test)]
mod tests;
use crate::*;
pub use descriptor::*;
pub use kind::MilpObserverKind;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};
const fn entry(
    kind: MilpObserverKind,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
) -> BuiltinCatalogEntry {
    let save = matches!(kind, MilpObserverKind::SaveSolutions);
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name: kind.name() },
        category: "math/optim",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Complete,
            inference_rule: match kind {
                MilpObserverKind::SaveSolutions => BuiltinInferenceRule::Identity(
                    IdentityInferenceRule::new(inference::infer_save),
                ),
                MilpObserverKind::PlotBounds => BuiltinInferenceRule::Identity(
                    IdentityInferenceRule::new(inference::infer_plot),
                ),
            },
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Impure,
            semantic_kind: if save {
                BuiltinSemanticKind::Workspace
            } else {
                BuiltinSemanticKind::General
            },
            workspace_effect: if save {
                Some(BuiltinWorkspaceEffect::CreatesBinding)
            } else {
                None
            },
            environment_effect: None,
            effects: if save {
                &[
                    EffectKind::WorkspaceRead,
                    EffectKind::WorkspaceWrite,
                    EffectKind::HostCallback,
                    EffectKind::MaySuspend,
                    EffectKind::MayThrow,
                ]
            } else {
                &[
                    EffectKind::UserInterface,
                    EffectKind::HostCallback,
                    EffectKind::MaySuspend,
                    EffectKind::MayThrow,
                ]
            },
            capabilities: if save {
                &[CapabilityRequirement::HostRuntime]
            } else {
                &[CapabilityRequirement::UserInterface]
            },
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Optional,
            residency: BuiltinResidencyPolicy::Dynamic,
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
        integer_audit: None,
        suppress_auto_output: true,
    }
}
pub const SAVEMILPSOLUTIONS_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    MilpObserverKind::SaveSolutions,
    &SAVEMILPSOLUTIONS_DESCRIPTOR,
    documentation::SAVE,
);
pub const OPTIMPLOTMILP_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    MilpObserverKind::PlotBounds,
    &OPTIMPLOTMILP_DESCRIPTOR,
    documentation::PLOT,
);
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &SAVEMILPSOLUTIONS_CATALOG_ENTRY,
    &OPTIMPLOTMILP_CATALOG_ENTRY,
];

/// Structural callback forms; a known string is still validated against the
/// selected predefined identity by the constructor/runtime owner.
pub(super) fn callback_metadata(value: &runmat_types::ValueFact) -> bool {
    use runmat_types::*;
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Callable(_) | ValueKindFact::String => {
            value.shape.element_count().is_none_or(|n| n == 1)
        }
        ValueKindFact::Character => value.shape.known_dims().is_none_or(|dims| {
            dims.len() <= 2 && dims.first().is_none_or(|n| n.is_none_or(|n| n <= 1))
        }),
        ValueKindFact::Cell(cell) => cell.elements.iter().all(|element| {
            !matches!(
                element.kind,
                ValueKindFact::Cell(_) | ValueKindFact::Numeric(_)
            ) && callback_metadata(element)
        }),
        ValueKindFact::Numeric(n) => {
            n.domain == NumericDomain::Real
                && value.shape.element_count() == Some(0)
                && value.storage != StorageFact::Sparse
        }
        _ => false,
    }
}
