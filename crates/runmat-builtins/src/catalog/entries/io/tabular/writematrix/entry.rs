use super::*;
use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub const WRITEMATRIX_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "writematrix",
    },
    category: "io/tabular",
    documentation: documentation::DOCUMENTATION,
    descriptor: &WRITEMATRIX_DESCRIPTOR,
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
        effects: &[
            EffectKind::EnvironmentRead,
            EffectKind::FilesystemWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[CapabilityRequirement::Filesystem],
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
    extensions: &WRITEMATRIX_EXTENSIONS,
    integer_capabilities: &WRITEMATRIX_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
