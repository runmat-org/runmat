use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::EnvironmentWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];

pub const GETPREF_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "getpref" },
    category: "io/repl_fs",
    documentation: super::documentation::DOCUMENTATION,
    descriptor: &super::super::contract::GETPREF_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
            IoReplFsInferenceRule::Preference(PreferenceInferenceRule::Read),
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
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
    extensions: super::super::contract::GET_EXTENSIONS,
    integer_capabilities: &super::super::contract::GETPREF_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
