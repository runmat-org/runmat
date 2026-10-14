mod contract;
mod documentation;
mod inference;
mod int8;
mod int16;
mod int32;
mod int64;
mod uint8;
mod uint16;
mod uint32;
mod uint64;
use crate::*;
pub use contract::*;
pub use int16::*;
pub use int32::*;
pub use int64::*;
pub use int8::*;
pub use uint16::*;
pub use uint32::*;
pub use uint64::*;
pub use uint8::*;
pub const EMPTY_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "empty" },
    category: "array/creation",
    documentation: documentation::DOCUMENTATION,
    descriptor: &EMPTY_GLOBAL_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer_global,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: crate::BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &EMPTY_GLOBAL_EXTENSIONS,
    integer_capabilities: &EMPTY_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &EMPTY_CATALOG_ENTRY,
    &INT8_EMPTY_CATALOG_ENTRY,
    &INT16_EMPTY_CATALOG_ENTRY,
    &INT32_EMPTY_CATALOG_ENTRY,
    &INT64_EMPTY_CATALOG_ENTRY,
    &UINT8_EMPTY_CATALOG_ENTRY,
    &UINT16_EMPTY_CATALOG_ENTRY,
    &UINT32_EMPTY_CATALOG_ENTRY,
    &UINT64_EMPTY_CATALOG_ENTRY,
];
