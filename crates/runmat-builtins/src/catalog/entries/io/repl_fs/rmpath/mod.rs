mod contract;
mod documentation;
pub(in crate::catalog::entries::io::repl_fs) mod inference;

use crate::*;
use runmat_types::ExecutionStackRequirement;

pub use contract::{
    RMPATH_DESCRIPTOR, RMPATH_ERROR_ARGUMENT_TYPE, RMPATH_ERROR_CURRENT_FOLDER,
    RMPATH_ERROR_FOLDER_NOT_FOUND, RMPATH_ERROR_NOT_FOLDER, RMPATH_ERROR_NOT_ON_PATH,
    RMPATH_ERROR_TOO_FEW_ARGUMENTS, RMPATH_ERROR_TOO_MANY_OUTPUTS,
};

pub const RMPATH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "rmpath" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &RMPATH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
            IoReplFsInferenceRule::path(PathInferenceRule::Search(SearchPathInferenceRule::Remove)),
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Workspace,
        workspace_effect: None,
        environment_effect: Some(BuiltinEnvironmentEffect::PathMutation),
        effects: &contract::EFFECTS,
        capabilities: &[runmat_types::CapabilityRequirement::Filesystem],
    },
    placement: BuiltinPlacementContract {
        preparation: crate::BuiltinOperandPreparation::RuntimeOwned,
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
    integer_audit: Some(&contract::INTEGER_AUDIT),
    suppress_auto_output: true,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&RMPATH_CATALOG_ENTRY];
