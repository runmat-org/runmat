//! Canonical `delete` contract across file, handle, listener, and pool forms.

mod contract;
mod documentation;
mod inference;

#[cfg(test)]
mod tests;

use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub use contract::{
    DELETE_DESCRIPTOR, DELETE_ERROR_EMPTY_FILENAME, DELETE_ERROR_FILE_NOT_FOUND,
    DELETE_ERROR_INVALID_HANDLE, DELETE_ERROR_INVALID_INPUT, DELETE_ERROR_INVALID_PATTERN,
    DELETE_ERROR_IS_DIRECTORY, DELETE_ERROR_OS_ERROR, DELETE_ERROR_RUNTIME_CONTEXT,
    DELETE_INTEGER_CAPABILITIES,
};

/// Parsed public option identity, retained through admission and execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeleteOption {
    ResolveSymbolicLinks,
}

impl DeleteOption {
    pub fn from_name(name: &str) -> Option<Self> {
        name.eq_ignore_ascii_case("ResolveSymbolicLinks")
            .then_some(Self::ResolveSymbolicLinks)
    }
}

const EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemRead,
    EffectKind::FilesystemWrite,
    EffectKind::HostCallback,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
    EffectKind::Unknown,
];

pub const DELETE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "delete" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &DELETE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[
            CapabilityRequirement::HostRuntime,
            CapabilityRequirement::Filesystem,
            CapabilityRequirement::ParallelRuntime,
        ],
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
    integer_capabilities: &DELETE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
