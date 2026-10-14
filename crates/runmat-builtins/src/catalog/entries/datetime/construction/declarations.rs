use super::{documentation, inference};
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

pub use super::contracts::{
    DATETIME_DESCRIPTOR, DATETIME_ERRORS, DATETIME_ERROR_INVALID_ARGUMENT,
    DATETIME_ERROR_INVALID_INPUT, DATETIME_INTEGER_CAPABILITIES,
};

pub const DATETIME_RAW_DATENUM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "datetime-implicit-datenum",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "A one-argument numeric value outside the date-vector forms is interpreted as a serial date number only in RunMat mode",
    error_identifier: Some("RunMat:compatibility:DatetimeImplicitDatenumExtension"),
};
pub const DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datetime-four-five-components",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Four- and five-component constructor forms are RunMat extensions",
        error_identifier: Some("RunMat:compatibility:DatetimeLegacyComponentArityExtension"),
    };
pub const DATETIME_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datetime-logical-numeric-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Logical values in numeric date positions are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatetimeLogicalInputExtension"),
    };
pub const DATETIME_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "datetime-resident-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Resident numeric input is gathered into a host datetime only in RunMat mode",
    error_identifier: Some("RunMat:compatibility:DatetimeGpuInputExtension"),
};
pub const DATETIME_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    DATETIME_RAW_DATENUM_EXTENSION,
    DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&DATETIME_CATALOG_ENTRY];

pub const DATETIME_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "datetime" },
    category: "datetime/construction",
    documentation: documentation::DOCUMENTATION,
    descriptor: &DATETIME_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        // The no-argument and relative-text forms read the local clock.
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::Clock,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[],
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
    extensions: &DATETIME_EXTENSIONS,
    integer_capabilities: &DATETIME_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
