mod contract;
mod documentation;
pub(in crate::catalog::entries::io::repl_fs) mod inference;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::{
    GENPATH_DESCRIPTOR, GENPATH_ERROR_CURRENT_FOLDER, GENPATH_ERROR_EXCLUDES_TYPE,
    GENPATH_ERROR_FOLDER_NOT_FOUND, GENPATH_ERROR_FOLDER_TYPE, GENPATH_ERROR_NOT_FOLDER,
    GENPATH_ERROR_PROVIDER, GENPATH_ERROR_TOO_MANY_INPUTS, GENPATH_EXCLUDES_EXTENSION,
    GENPATH_NUMERIC_CHARACTER_CODES_EXTENSION,
};

const EFFECTS: [EffectKind; 4] = [
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemRead,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];

pub const GENPATH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "genpath" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &GENPATH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
            IoReplFsInferenceRule::path(PathInferenceRule::Search(
                SearchPathInferenceRule::Generate,
            )),
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
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GENPATH_CATALOG_ENTRY];
