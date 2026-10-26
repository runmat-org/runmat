mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::{
    DESCRIPTOR as GEOMETRY_REGION_DESCRIPTOR, ERROR_INPUT as GEOMETRY_REGION_ERROR_INPUT,
    ERROR_INVALID as GEOMETRY_REGION_ERROR_INVALID, GEOMETRY_REGION_CLASS,
    INTEGER_AUDIT as GEOMETRY_REGION_INTEGER_AUDIT,
};
pub(in crate::catalog::entries::geometry) use inference::possible_asset;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub const GEOMETRY_REGION_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "geometry.region",
    },
    category: "geometry/assets",
    documentation: documentation::DOCUMENTATION,
    descriptor: &GEOMETRY_REGION_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::RunMat,
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
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::Host,
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
    integer_audit: Some(&GEOMETRY_REGION_INTEGER_AUDIT),
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GEOMETRY_REGION_CATALOG_ENTRY];
