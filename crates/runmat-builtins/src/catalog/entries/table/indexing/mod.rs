use crate::*;
use runmat_types::{CallInference, CallRequest};

mod descriptor;
mod dictionary_subsasgn;
mod dictionary_subsref;
mod inference;
mod integer;
mod table_subsasgn;
mod table_subsref;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use integer::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &dictionary_subsasgn::ENTRY,
    &dictionary_subsref::ENTRY,
    &table_subsasgn::ENTRY,
    &table_subsref::ENTRY,
];

const fn entry(
    provenance: BuiltinCatalogProvenance,
    name: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    assignment: bool,
    integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance,
        identity: BuiltinCatalogIdentity { name },
        category: "table/indexing",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Complete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: if assignment {
                BuiltinPurity::Impure
            } else {
                BuiltinPurity::DeterministicReadOnly
            },
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[
                runmat_types::EffectKind::MaySuspend,
                runmat_types::EffectKind::MayThrow,
            ],
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: BuiltinResidencyPolicy::Dynamic,
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
        integer_capabilities,
        integer_audit: None,
        suppress_auto_output: false,
    }
}
