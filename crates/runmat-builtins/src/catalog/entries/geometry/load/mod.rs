mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::{
    DESCRIPTOR as GEOMETRY_LOAD_DESCRIPTOR, ERROR_INPUT as GEOMETRY_LOAD_ERROR_INPUT,
    ERROR_INTERNAL as GEOMETRY_LOAD_ERROR_INTERNAL, ERROR_IO as GEOMETRY_LOAD_ERROR_IO,
    ERROR_OPERATION as GEOMETRY_LOAD_ERROR_OPERATION, ERROR_OUTPUTS as GEOMETRY_LOAD_ERROR_OUTPUTS,
    GEOMETRY_ASSET_CLASS, INTEGER_AUDIT as GEOMETRY_LOAD_INTEGER_AUDIT,
};

use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub const GEOMETRY_LOAD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "geometry.load",
    },
    category: "geometry/assets",
    documentation: documentation::DOCUMENTATION,
    descriptor: &GEOMETRY_LOAD_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::DeterministicReadOnly,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::FilesystemRead,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[CapabilityRequirement::Filesystem],
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
    integer_audit: Some(&GEOMETRY_LOAD_INTEGER_AUDIT),
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GEOMETRY_LOAD_CATALOG_ENTRY];
