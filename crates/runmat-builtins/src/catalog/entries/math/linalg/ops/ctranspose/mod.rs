mod documentation;
mod inference;
#[cfg(test)]
mod tests;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub const CTRANSPOSE_INTEGER_SPARSE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ctranspose-integer-sparse",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ctranspose of integer-valued sparse storage is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CtransposeIntegerSparseExtension"),
    };
pub const CTRANSPOSE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [CTRANSPOSE_INTEGER_SPARSE_EXTENSION];
const CTRANSPOSE_DENSE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes:
            "All eight real dense integer classes are documented and transpose without conversion.",
    }];
const CTRANSPOSE_SPARSE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer-valued sparse storage is a gated RunMat extension; MATLAB sparse value classes are floating or logical.",
    }];
pub const CTRANSPOSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "B = ctranspose(integer_A)", inputs: &CTRANSPOSE_DENSE_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Dense real integer storage, shape, and GPU residency are preserved exactly. Complex integer storage is rejected." },
    BuiltinIntegerCapabilityDescriptor { form: "B = ctranspose(sparse(integer_A))", inputs: &CTRANSPOSE_SPARSE_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "The sparse CSC structure and authoritative integer values are transposed exactly after the RunMat extension gate." },
];

const CTRANSPOSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value with first two dimensions swapped and complex conjugated.",
}];
const CTRANSPOSE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar/array value.",
}];
const CTRANSPOSE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = ctranspose(A)",
    inputs: &CTRANSPOSE_INPUTS,
    outputs: &CTRANSPOSE_OUTPUT,
}];
pub const CTRANSPOSE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CTRANSPOSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:ctranspose:InvalidArgument"),
    when: "Call does not provide exactly one input argument.",
    message: "ctranspose: invalid argument",
};
pub const CTRANSPOSE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CTRANSPOSE.INVALID_INPUT",
    identifier: Some("RunMat:ctranspose:InvalidInput"),
    when: "Input type is unsupported for conjugate transpose, the input has a nonsingleton dimension above two, or the input uses complex integer storage.",
    message: "ctranspose: unsupported input type",
};
pub const CTRANSPOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CTRANSPOSE.INTERNAL",
    identifier: Some("RunMat:ctranspose:Internal"),
    when: "Runtime cannot materialize conjugate-transpose output.",
    message: "ctranspose: internal runtime failure",
};
const CTRANSPOSE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CTRANSPOSE_ERROR_INVALID_ARGUMENT,
    CTRANSPOSE_ERROR_INVALID_INPUT,
    CTRANSPOSE_ERROR_INTERNAL,
];
pub const CTRANSPOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CTRANSPOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CTRANSPOSE_ERRORS,
};

pub const CTRANSPOSE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "ctranspose" },
    category: "math/linalg/ops",
    documentation: documentation::DOCUMENTATION,
    descriptor: &CTRANSPOSE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
            EffectKind::Unknown,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
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
    extensions: &CTRANSPOSE_EXTENSIONS,
    integer_capabilities: &CTRANSPOSE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
