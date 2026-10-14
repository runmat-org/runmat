//! Detection and replacement of outlying numeric observations.
mod contract;
mod documentation;
mod inference;
mod policy;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::{
    FILLOUTLIERS_DESCRIPTOR, FILLOUTLIERS_EXTENSIONS, FILLOUTLIERS_INTEGER_CAPABILITIES,
    FILLOUTLIERS_INTEGER_DATA_EXTENSION, FILLOUTLIERS_INTEGER_FILL_EXTENSION,
    FILLOUTLIERS_NUMERIC_MASK_EXTENSION, FILLOUTLIERS_RESIDENT_INPUT_EXTENSION,
    ISOUTLIER_DESCRIPTOR, ISOUTLIER_EXTENSIONS, ISOUTLIER_GPU_DATA_VARIABLES_EXTENSION,
    ISOUTLIER_GPU_MOVMEDIAN_EXTENSION, ISOUTLIER_GPU_SAMPLE_POINTS_EXTENSION,
    ISOUTLIER_INTEGER_CAPABILITIES, ISOUTLIER_INTEGER_DATA_EXTENSION,
    ISOUTLIER_INTEGER_DIMENSION_EXTENSION, ISOUTLIER_INTEGER_THRESHOLD_EXTENSION,
    ISOUTLIER_INTEGER_WINDOW_EXTENSION, OUTLIER_ERROR_INTERNAL, OUTLIER_ERROR_INVALID_ARGUMENT,
    OUTLIER_ERROR_TOO_MANY_OUTPUTS, OUTLIER_ERROR_UNSUPPORTED,
};
pub use policy::OutlierOption;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OutlierKind {
    Detect,
    Fill,
}

fn infer_detect(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::infer(request, entry, OutlierKind::Detect)
}

fn infer_fill(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::infer(request, entry, OutlierKind::Fill)
}

const EFFECTS: &[EffectKind] = &[EffectKind::MaySuspend, EffectKind::MayThrow];

const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Optional,
    residency: BuiltinResidencyPolicy::Dynamic,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::MaterializeArguments,
};

const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

pub const ISOUTLIER_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "isoutlier" },
    category: "stats/summary",
    documentation: documentation::ISOUTLIER_DOCUMENTATION,
    descriptor: &ISOUTLIER_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer_detect)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &ISOUTLIER_EXTENSIONS,
    integer_capabilities: &ISOUTLIER_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub const FILLOUTLIERS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "filloutliers",
    },
    category: "stats/summary",
    documentation: documentation::FILLOUTLIERS_DOCUMENTATION,
    descriptor: &FILLOUTLIERS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer_fill)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &FILLOUTLIERS_EXTENSIONS,
    integer_capabilities: &FILLOUTLIERS_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&ISOUTLIER_CATALOG_ENTRY, &FILLOUTLIERS_CATALOG_ENTRY];
