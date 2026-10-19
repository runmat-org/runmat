mod contract;
mod documentation;
mod examples;
mod inference;
use crate::*;
pub use contract::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};
pub const PCODE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(),
    module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "pcode"
    },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &PCODE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference::infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::EnvironmentRead,
            EffectKind::FilesystemRead,
            EffectKind::FilesystemWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow
        ],
        capabilities: &[
            CapabilityRequirement::Filesystem
        ],
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
        artifact_dependencies: &[
        ]
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[
    ],
    integer_capabilities: &[
    ],
    integer_audit: Some(&BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Source names, patterns, folders, and options are scalar text; numeric and resident numeric inputs reject before provider or filesystem access."
    }),
    suppress_auto_output: true,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&PCODE_CATALOG_ENTRY];
