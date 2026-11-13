use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

const EFFECTS: &[EffectKind] = &[
    EffectKind::EnvironmentRead,
    EffectKind::FilesystemRead,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
const CAPABILITIES: &[CapabilityRequirement] = &[CapabilityRequirement::Filesystem];

pub const MEMMAPFILE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "memmapfile" },
    category: "io/repl_fs",
    documentation: super::documentation::DOCUMENTATION,
    descriptor: &super::contract::MEMMAPFILE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            super::inference::infer,
        )),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: CAPABILITIES,
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
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
    extensions: &super::contract::MEMMAPFILE_EXTENSIONS,
    integer_capabilities: &super::contract::MEMMAPFILE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
