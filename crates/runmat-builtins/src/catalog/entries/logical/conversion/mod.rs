mod documentation;
mod inference;
use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use documentation::LOGICAL_DOCUMENTATION;
pub const LOGICAL_STRING_ARRAY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "logical-string-array-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "logical with string-array input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LogicalStringArrayInputExtension"),
};
pub const LOGICAL_SYMBOLIC_CONSTANT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "logical-symbolic-constant-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "logical with a symbolic numeric constant is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LogicalSymbolicConstantInputExtension"),
    };
pub const LOGICAL_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    LOGICAL_STRING_ARRAY_EXTENSION,
    LOGICAL_SYMBOLIC_CONSTANT_EXTENSION,
];

const LOGICAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight real integer classes convert elementwise without floating materialization; zero becomes false and every nonzero value becomes true.",
    }];
pub const LOGICAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = logical(integer_A)",
        inputs: &LOGICAL_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host conversion reads authoritative integer storage exactly; resident conversion uses a validated owning-provider path or exact gather and class-preserving restoration.",
    }];

const LOGICAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical-converted result.",
}];

const LOGICAL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to convert.",
}];

const LOGICAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = logical(A)",
    inputs: &LOGICAL_INPUTS,
    outputs: &LOGICAL_OUTPUT,
}];

pub const LOGICAL_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGICAL.TOO_MANY_INPUTS",
    identifier: Some("RunMat:logical:TooManyInputs"),
    when: "More than one input argument is provided.",
    message: "logical: too many input arguments",
};

pub const LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGICAL.CONVERSION_NOT_POSSIBLE",
    identifier: Some("RunMat:logical:ConversionNotPossible"),
    when: "Input type cannot be converted to logical.",
    message: "logical: conversion to logical is not possible for this input type",
};

pub const LOGICAL_ERROR_GPU_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGICAL.GPU_GATHER_FAILED",
    identifier: Some("RunMat:logical:GpuGatherFailed"),
    when: "GPU input gather fails during host fallback.",
    message: "logical: failed to gather gpuArray input",
};

pub const LOGICAL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGICAL.INTERNAL",
    identifier: Some("RunMat:logical:InternalError"),
    when: "Internal logical buffer materialization fails.",
    message: "logical: internal conversion error",
};

pub const LOGICAL_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGICAL.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:logical:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "logical: too many output arguments",
};

const LOGICAL_ERRORS: [BuiltinErrorDescriptor; 5] = [
    LOGICAL_ERROR_TOO_MANY_OUTPUTS,
    LOGICAL_ERROR_TOO_MANY_INPUTS,
    LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE,
    LOGICAL_ERROR_GPU_GATHER_FAILED,
    LOGICAL_ERROR_INTERNAL,
];

pub const LOGICAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOGICAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOGICAL_ERRORS,
};

const BINDINGS: [crate::BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [runmat_types::EffectKind; 2] = [
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
];
pub const LOGICAL_CATALOG_ENTRY: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: crate::BuiltinCatalogIdentity { name: "logical" },
    category: "logical",
    documentation: LOGICAL_DOCUMENTATION,
    descriptor: &LOGICAL_DESCRIPTOR,
    contract: crate::BuiltinContractDeclaration {
        maturity: crate::BuiltinContractMaturity::Complete,
        inference_rule: crate::BuiltinInferenceRule::Identity(crate::IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: crate::BuiltinCompatibility::Matlab,
        async_behavior: crate::BuiltinAsyncBehavior::MaySuspend,
        purity: crate::BuiltinPurity::Pure,
        semantic_kind: crate::BuiltinSemanticKind::Elementwise,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: crate::BuiltinPlacementContract {
        preparation: crate::BuiltinOperandPreparation::Forms(&[crate::BuiltinPreparationForm {
            id: crate::BuiltinPreparationFormId(0),
            binding_variant: "default",
            signatures: &[0],
            data_arguments: &[0],
            operation: crate::BuiltinPreparationOperation::Unary,
            provider_requirement: crate::BuiltinPreparationProviderRequirement::InputSupported,
        }]),
        portability: crate::BuiltinPortability::NativeAndWasm,
        accelerator: crate::BuiltinAcceleratorPolicy::Optional,
        residency: crate::BuiltinResidencyPolicy::PreserveInputs,
        fusion: crate::BuiltinFusionPolicy::Boundary,
        distributed: crate::BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: crate::BuiltinLinkContract {
        reachability: crate::BuiltinReachability::Always,
        policy: crate::BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &LOGICAL_EXTENSIONS,
    integer_capabilities: &LOGICAL_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LOGICAL_CATALOG_ENTRY];
