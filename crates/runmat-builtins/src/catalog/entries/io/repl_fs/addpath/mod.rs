mod contract;
mod documentation;
pub(in crate::catalog::entries::io::repl_fs) mod inference;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::{
    ADDPATH_DESCRIPTOR, ADDPATH_ERROR_ARGUMENT_TYPE, ADDPATH_ERROR_CURRENT_FOLDER,
    ADDPATH_ERROR_FOLDER_NOT_FOUND, ADDPATH_ERROR_NOT_FOLDER, ADDPATH_ERROR_PATHDEF,
    ADDPATH_ERROR_POSITION, ADDPATH_ERROR_PROVIDER, ADDPATH_ERROR_TOO_FEW_ARGUMENTS,
    ADDPATH_NUMERIC_CHARACTER_CODES_EXTENSION,
};

const EFFECTS: [EffectKind; 5] = [
    EffectKind::EnvironmentRead,
    EffectKind::EnvironmentWrite,
    EffectKind::FilesystemRead,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];

pub const ADDPATH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "addpath" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &ADDPATH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
            IoReplFsInferenceRule::SearchPath(SearchPathInferenceRule::Add),
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Workspace,
        workspace_effect: None,
        environment_effect: Some(BuiltinEnvironmentEffect::PathMutation),
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

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ADDPATH_CATALOG_ENTRY];
