mod descriptor;
mod documentation;
mod inference;

pub use descriptor::MEX_DESCRIPTOR;

use super::contract::{BUILD_INTEGER_AUDIT, BUILD_LINK, BUILD_PLACEMENT};
use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind};

pub const MEX_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "mex" },
    category: "interop/mex",
    documentation: documentation::DOCUMENTATION,
    descriptor: &MEX_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::EnvironmentRead,
            EffectKind::FilesystemRead,
            EffectKind::FilesystemWrite,
            EffectKind::HostCallback,
            EffectKind::MayThrow,
        ],
        capabilities: &[
            CapabilityRequirement::HostRuntime,
            CapabilityRequirement::Filesystem,
            CapabilityRequirement::NativeCode,
        ],
    },
    placement: BUILD_PLACEMENT,
    link: BUILD_LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&BUILD_INTEGER_AUDIT),
    suppress_auto_output: true,
};
