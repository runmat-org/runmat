mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::{
    CLASS as DELAUNAY_TRI_CLASS, CONSTRUCTOR_EXTENSIONS as DELAUNAY_TRI_CONSTRUCTOR_EXTENSIONS,
    COORDINATES_EXTENSION as DELAUNAY_TRI_INTEGER_COORDINATES_EXTENSION,
    COORDINATE_EXTENSIONS as DELAUNAY_TRI_COORDINATE_EXTENSIONS,
    DESCRIPTOR as DELAUNAY_TRI_DESCRIPTOR,
    INTEGER_CAPABILITIES as DELAUNAY_TRI_CONSTRUCTOR_INTEGER_CAPABILITIES,
    TOO_MANY_OUTPUTS as DELAUNAY_TRI_ERROR_TOO_MANY_OUTPUTS,
};

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub const DELAUNAY_TRI_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "DelaunayTri",
    },
    category: "geometry/triangulation",
    documentation: documentation::DOCUMENTATION,
    descriptor: &DELAUNAY_TRI_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::EnvironmentWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
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
    extensions: &DELAUNAY_TRI_CONSTRUCTOR_EXTENSIONS,
    integer_capabilities: &DELAUNAY_TRI_CONSTRUCTOR_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&DELAUNAY_TRI_CATALOG_ENTRY];
