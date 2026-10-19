use super::{array2timetable, constructor, readtimetable, table2timetable, timetable2table};
use crate::*;
use runmat_types::{CallInference, CallRequest};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &constructor::ENTRY,
    &array2timetable::ENTRY,
    &table2timetable::ENTRY,
    &timetable2table::ENTRY,
    &readtimetable::ENTRY,
];

pub(super) struct EntryDefinition {
    pub(super) provenance: BuiltinCatalogProvenance,
    pub(super) name: &'static str,
    pub(super) category: &'static str,
    pub(super) documentation: BuiltinDocumentation,
    pub(super) descriptor: &'static BuiltinDescriptor,
    pub(super) inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    pub(super) extensions: &'static [BuiltinExtensionDescriptor],
    pub(super) integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub(super) filesystem_read: bool,
}

pub(super) const fn entry(definition: EntryDefinition) -> BuiltinCatalogEntry {
    let EntryDefinition {
        provenance,
        name,
        category,
        documentation,
        descriptor,
        inference,
        extensions,
        integer_capabilities,
        filesystem_read,
    } = definition;
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
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
            capabilities: if filesystem_read {
                &[runmat_types::CapabilityRequirement::Filesystem]
            } else {
                &[]
            },
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
