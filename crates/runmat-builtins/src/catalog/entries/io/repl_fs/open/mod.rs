mod contract;
mod documentation;
mod examples;
mod inference;
use crate::*;
pub use contract::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};
pub const OPEN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance:BuiltinCatalogProvenance::new(file!(),
    module_path!()),
    identity:BuiltinCatalogIdentity{
        name:"open"
    },
    category:"io/repl_fs",
    documentation:documentation::DOCUMENTATION,
    descriptor:&OPEN_DESCRIPTOR,
    contract:BuiltinContractDeclaration {
        maturity:BuiltinContractMaturity::DynamicByDesign,
        inference_rule:BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference::infer)),
        compatibility:BuiltinCompatibility::Matlab,
        async_behavior:BuiltinAsyncBehavior::MaySuspend,
        purity:BuiltinPurity::Impure,
        semantic_kind:BuiltinSemanticKind::Filesystem,
        workspace_effect:Some(BuiltinWorkspaceEffect::ReadsWorkspace),
        environment_effect:None,
        effects:&[
            EffectKind::WorkspaceRead,
            EffectKind::EnvironmentRead,
            EffectKind::FilesystemRead,
            EffectKind::UserInterface,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
            EffectKind::Unknown
        ],
        capabilities:&[
            CapabilityRequirement::HostRuntime,
            CapabilityRequirement::Filesystem
        ],
    },
    placement:BuiltinPlacementContract {
        preparation:BuiltinOperandPreparation::RuntimeOwned,
        portability:BuiltinPortability::NativeAndWasm,
        accelerator:BuiltinAcceleratorPolicy::Forbidden,
        residency:BuiltinResidencyPolicy::Dynamic,
        fusion:BuiltinFusionPolicy::Boundary,
        distributed:BuiltinDistributedPolicy::Unsupported,
    },
    link:BuiltinLinkContract{
        reachability:BuiltinReachability::Always,
        policy:BuiltinLinkPolicy::PortableRuntime,
        execution_stack:ExecutionStackRequirement::Any,
        artifact_dependencies:&[
        ]
    },
    bindings:&REQUIRED_DEFAULT_BINDING,
    extensions:&[
    ],
    integer_capabilities:&[
    ],
    integer_audit:Some(&BuiltinIntegerAuditDescriptor{
        kind:BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin:None,
        notes:"The target name must be host scalar text. Numeric inputs reject before provider access; loaded payloads and extension handler outputs retain their own classes and residency."
    }),
    suppress_auto_output:true,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&OPEN_CATALOG_ENTRY];
