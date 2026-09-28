use crate::*;
use runmat_types::{CallInference, CallRequest};

mod array2timetable;
mod descriptor;
mod extensions;
mod inference;
mod integer;
mod readtimetable;
mod table2timetable;
mod timetable;
mod timetable2table;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use extensions::*;
pub use integer::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &timetable::ENTRY,
    &array2timetable::ENTRY,
    &table2timetable::ENTRY,
    &timetable2table::ENTRY,
    &readtimetable::ENTRY,
];

const fn entry(
    provenance: BuiltinCatalogProvenance,
    name: &'static str,
    category: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    extensions: &'static [BuiltinExtensionDescriptor],
    integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    filesystem_read: bool,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        provenance,
        identity: BuiltinCatalogIdentity { name },
        category,
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Complete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: if filesystem_read {
                BuiltinPurity::DeterministicReadOnly
            } else {
                BuiltinPurity::Pure
            },
            semantic_kind: if filesystem_read {
                BuiltinSemanticKind::Filesystem
            } else {
                BuiltinSemanticKind::General
            },
            workspace_effect: None,
            environment_effect: None,
            effects: if filesystem_read {
                &[
                    runmat_types::EffectKind::EnvironmentRead,
                    runmat_types::EffectKind::FilesystemRead,
                    runmat_types::EffectKind::MaySuspend,
                    runmat_types::EffectKind::MayThrow,
                ]
            } else {
                &[
                    runmat_types::EffectKind::MaySuspend,
                    runmat_types::EffectKind::MayThrow,
                ]
            },
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
