mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::{
    DESCRIPTOR as NEAREST_NEIGHBOR_DESCRIPTOR,
    INTEGER_CAPABILITIES as NEAREST_NEIGHBOR_INTEGER_CAPABILITIES,
    METHOD_DESCRIPTOR as DELAUNAY_TRI_NEAREST_NEIGHBOR_DESCRIPTOR,
    TOO_MANY_OUTPUTS as NEAREST_NEIGHBOR_ERROR_TOO_MANY_OUTPUTS,
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
        extensions: &super::delaunay_tri::DELAUNAY_TRI_COORDINATE_EXTENSIONS,
        integer_capabilities: &NEAREST_NEIGHBOR_INTEGER_CAPABILITIES,
        integer_audit: None,
        suppress_auto_output: false,
    }
}

pub const NEAREST_NEIGHBOR_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "nearestNeighbor",
    documentation::DOCUMENTATION,
    &NEAREST_NEIGHBOR_DESCRIPTOR,
);
pub const DELAUNAY_TRI_NEAREST_NEIGHBOR_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "DelaunayTri.nearestNeighbor",
    documentation::METHOD_DOCUMENTATION,
    &DELAUNAY_TRI_NEAREST_NEIGHBOR_DESCRIPTOR,
);
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &NEAREST_NEIGHBOR_CATALOG_ENTRY,
    &DELAUNAY_TRI_NEAREST_NEIGHBOR_CATALOG_ENTRY,
];
