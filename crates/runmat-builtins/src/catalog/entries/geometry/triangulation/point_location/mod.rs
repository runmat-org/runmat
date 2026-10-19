mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::{
    DESCRIPTOR as POINT_LOCATION_DESCRIPTOR, EXTENSIONS as POINT_LOCATION_EXTENSIONS,
    INTEGER_CAPABILITIES as POINT_LOCATION_INTEGER_CAPABILITIES,
    METHOD_DESCRIPTOR as DELAUNAY_TRI_POINT_LOCATION_DESCRIPTOR,
    TOO_MANY_OUTPUTS as POINT_LOCATION_ERROR_TOO_MANY_OUTPUTS,
};

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const fn entry(
    name: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "geometry/triangulation",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                inference::infer,
            )),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Optional,
            residency: BuiltinResidencyPolicy::GatherToHost,
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: BuiltinDistributedPolicy::MaterializeArguments,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &POINT_LOCATION_EXTENSIONS,
        integer_capabilities: &POINT_LOCATION_INTEGER_CAPABILITIES,
        integer_audit: None,
        suppress_auto_output: false,
    }
}

pub const POINT_LOCATION_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "pointLocation",
    documentation::DOCUMENTATION,
    &POINT_LOCATION_DESCRIPTOR,
);
pub const DELAUNAY_TRI_POINT_LOCATION_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "DelaunayTri.pointLocation",
    documentation::METHOD_DOCUMENTATION,
    &DELAUNAY_TRI_POINT_LOCATION_DESCRIPTOR,
);
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &POINT_LOCATION_CATALOG_ENTRY,
    &DELAUNAY_TRI_POINT_LOCATION_CATALOG_ENTRY,
];
