mod contract;
mod documentation;
pub(super) mod inference;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::*;

pub const FULLFILE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "fullfile" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &FULLFILE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
            IoReplFsInferenceRule::path(PathInferenceRule::Syntax(PathSyntaxInferenceRule::Join)),
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
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
    extensions: &[FULLFILE_NUMERIC_CHARACTER_CODES_EXTENSION],
    integer_capabilities: contract::INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
