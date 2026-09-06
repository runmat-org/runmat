mod contract;
mod documentation;
mod examples;
mod faqs;
pub(super) mod inference;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::*;

const EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemRead,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];

pub const LS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "ls" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &LS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
            IoReplFsInferenceRule::directory(DirectoryInferenceRule::Listing(
                DirectoryListingInferenceRule::Names,
            )),
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::DeterministicReadOnly,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
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
    extensions: &[LS_PORTABLE_ROWS_EXTENSION],
    integer_capabilities: &[],
    integer_audit: Some(&LS_INTEGER_AUDIT),
    suppress_auto_output: true,
};
