mod contract;
mod documentation;
mod examples;
mod inference;
use crate::*;
pub use contract::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};
pub const OPENTOLINE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "opentoline" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &OPENTOLINE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
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
            EffectKind::FilesystemRead,
            EffectKind::UserInterface,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[
            CapabilityRequirement::HostRuntime,
            CapabilityRequirement::Filesystem,
            CapabilityRequirement::UserInterface,
        ],
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
    extensions: &OPENTOLINE_EXTENSIONS,
    integer_capabilities: &OPENTOLINE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&OPENTOLINE_CATALOG_ENTRY];
