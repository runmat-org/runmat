use crate::*;
use runmat_types::{CallInference, CallRequest};

mod descriptor;
mod documentation;
mod inference;
mod integer;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &ISCATEGORICAL_ENTRY,
    &ISORDINAL_ENTRY,
    &ISTABLE_ENTRY,
    &ISTIMETABLE_ENTRY,
];

const fn entry(
    name: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    integer_audit: &'static BuiltinIntegerAuditDescriptor,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "table",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Complete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[],
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: BuiltinResidencyPolicy::Host,
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: BuiltinDistributedPolicy::InspectHandles,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: runmat_types::ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: Some(integer_audit),
        suppress_auto_output: false,
    }
}

pub const ISCATEGORICAL_ENTRY: BuiltinCatalogEntry = entry(
    "iscategorical",
    documentation::ISCATEGORICAL_DOCUMENTATION,
    &ISCATEGORICAL_DESCRIPTOR,
    inference::iscategorical,
    &integer::ISCATEGORICAL,
);
pub const ISORDINAL_ENTRY: BuiltinCatalogEntry = entry(
    "isordinal",
    documentation::ISORDINAL_DOCUMENTATION,
    &ISORDINAL_DESCRIPTOR,
    inference::isordinal,
    &integer::ISORDINAL,
);
pub const ISTABLE_ENTRY: BuiltinCatalogEntry = entry(
    "istable",
    documentation::ISTABLE_DOCUMENTATION,
    &ISTABLE_DESCRIPTOR,
    inference::istable,
    &integer::ISTABLE,
);
pub const ISTIMETABLE_ENTRY: BuiltinCatalogEntry = entry(
    "istimetable",
    documentation::ISTIMETABLE_DOCUMENTATION,
    &ISTIMETABLE_DESCRIPTOR,
    inference::istimetable,
    &integer::ISTIMETABLE,
);
