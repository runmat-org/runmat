mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::{
    DESCRIPTOR as FREE_BOUNDARY_DESCRIPTOR, EXTENSIONS as FREE_BOUNDARY_EXTENSIONS,
    INTEGER_CAPABILITIES as FREE_BOUNDARY_INTEGER_CAPABILITIES,
    INVALID_ARGUMENT as DELAUNAY_TRI_ERROR_INVALID_ARGUMENT,
    METHOD_DESCRIPTOR as DELAUNAY_TRI_FREE_BOUNDARY_DESCRIPTOR,
    TOO_MANY_OUTPUTS as FREE_BOUNDARY_ERROR_TOO_MANY_OUTPUTS,
    TOPOLOGY_EXTENSION as DELAUNAY_TRI_INTEGER_TOPOLOGY_EXTENSION,
    UNSUPPORTED as DELAUNAY_TRI_ERROR_UNSUPPORTED,
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
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[EffectKind::MayThrow],
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
        extensions: &FREE_BOUNDARY_EXTENSIONS,
        integer_capabilities: &FREE_BOUNDARY_INTEGER_CAPABILITIES,
        integer_audit: None,
        suppress_auto_output: false,
    }
}

pub const FREE_BOUNDARY_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "freeBoundary",
    documentation::DOCUMENTATION,
    &FREE_BOUNDARY_DESCRIPTOR,
);
pub const DELAUNAY_TRI_FREE_BOUNDARY_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "DelaunayTri.freeBoundary",
    documentation::METHOD_DOCUMENTATION,
    &DELAUNAY_TRI_FREE_BOUNDARY_DESCRIPTOR,
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &FREE_BOUNDARY_CATALOG_ENTRY,
    &DELAUNAY_TRI_FREE_BOUNDARY_CATALOG_ENTRY,
];
