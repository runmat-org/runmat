use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemRead,
    EffectKind::FilesystemWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];

pub const SAVEPATH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "savepath" },
    category: "io/repl_fs",
    documentation: super::documentation::DOCUMENTATION,
    descriptor: &super::SAVEPATH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
            IoReplFsInferenceRule::path(PathInferenceRule::Search(
                SearchPathInferenceRule::Persist,
            )),
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
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
    extensions: super::contract::EXTENSIONS,
    integer_capabilities: super::contract::INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
