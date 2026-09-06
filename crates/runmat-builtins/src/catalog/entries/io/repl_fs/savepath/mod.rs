mod contract;
mod documentation;
pub(in crate::catalog::entries::io::repl_fs) mod inference;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::{
    SAVEPATH_DESCRIPTOR, SAVEPATH_DIAGNOSTIC_OUTPUTS_EXTENSION,
    SAVEPATH_DIRECTORY_TARGET_EXTENSION, SAVEPATH_ERROR_ARGUMENT_TYPE,
    SAVEPATH_ERROR_CANNOT_RESOLVE, SAVEPATH_ERROR_CANNOT_WRITE, SAVEPATH_ERROR_EMPTY_FILENAME,
    SAVEPATH_ERROR_PROVIDER, SAVEPATH_ERROR_TOO_MANY_INPUTS, SAVEPATH_ERROR_TOO_MANY_OUTPUTS,
    SAVEPATH_NUMERIC_CHARACTER_CODES_EXTENSION,
};

const EFFECTS: [EffectKind; 5] = [
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemRead,
    EffectKind::FilesystemWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];

pub const SAVEPATH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "savepath" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &SAVEPATH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
            IoReplFsInferenceRule::SearchPath(SearchPathInferenceRule::Persist),
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::GatherToHost,
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
    extensions: contract::EXTENSIONS,
    integer_capabilities: contract::INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&SAVEPATH_CATALOG_ENTRY];
