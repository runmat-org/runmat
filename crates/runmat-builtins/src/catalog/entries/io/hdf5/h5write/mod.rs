use super::common::*;
use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

mod documentation;
mod inference;
#[cfg(test)]
mod tests;

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "h5write(filename, dataset, data)",
        inputs: &IN_FILE_DATASET_VALUE,
        outputs: &OUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "h5write(filename, dataset, data, start, count[, stride])",
        inputs: &IN_FILE_DATASET_VALUE_SELECTION,
        outputs: &OUT_NONE,
    },
];

pub const H5WRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "native integer dataset",
    inputs: &INTEGER_DATA_INPUT,
    computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Writes preserve the exact signed/unsigned storage class; invalid selection indices reject before native file access.",
}];

pub const H5WRITE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "h5write" },
    category: "io/hdf5",
    documentation: documentation::DOCUMENTATION,
    descriptor: &H5WRITE_DESCRIPTOR,
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
            EffectKind::FilesystemRead,
            EffectKind::FilesystemWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[
            CapabilityRequirement::Filesystem,
            CapabilityRequirement::NativeCode,
        ],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeOnly,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::NativeSymbol,
        execution_stack: ExecutionStackRequirement::Process,
        artifact_dependencies: &["hdf5-metno"],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
