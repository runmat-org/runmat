use crate::*;
use runmat_types::{CallInference, CallRequest};

mod array2table;
mod cell2table;
mod descriptor;
mod extensions;
mod inference;
mod integer;
mod struct2table;
mod table2array;
mod table2cell;
mod table2struct;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use extensions::*;
pub use integer::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &array2table::ENTRY,
    &cell2table::ENTRY,
    &struct2table::ENTRY,
    &table2array::ENTRY,
    &table2cell::ENTRY,
    &table2struct::ENTRY,
];

const fn entry(
    provenance: BuiltinCatalogProvenance,
    name: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    extensions: &'static [BuiltinExtensionDescriptor],
    integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance,
        identity: BuiltinCatalogIdentity { name },
        category: "table",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Complete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Pure,
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
            residency: BuiltinResidencyPolicy::GatherToHost,
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
        extensions,
        integer_capabilities,
        integer_audit: None,
        suppress_auto_output: false,
    }
}
