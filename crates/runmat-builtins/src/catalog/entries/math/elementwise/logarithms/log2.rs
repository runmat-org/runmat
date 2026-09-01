use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

use super::documentation::LOG2_DOCUMENTATION;

const LOG2_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input or a supported table; integer, logical, and character forms are RunMat-only extensions.",
}];
const LOG2_VALUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise base-2 logarithm, promoted to complex for negative real input.",
}];
const LOG2_DISSECTION_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Real floating-point fraction with the same class and shape as the supported input.",
    },
    BuiltinParamDescriptor {
        name: "E",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real floating-point exponent satisfying X = F .* 2.^E.",
    },
];
const LOG2_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = log2(X)",
        inputs: &LOG2_INPUTS,
        outputs: &LOG2_VALUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[F,E] = log2(X)",
        inputs: &LOG2_INPUTS,
        outputs: &LOG2_DISSECTION_OUTPUTS,
    },
];

pub const LOG2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.INVALID_INPUT",
    identifier: Some("RunMat:log2:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric, logical, character, or table data.",
    message: "log2: invalid input",
};
pub const LOG2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.INTERNAL",
    identifier: Some("RunMat:log2:Internal"),
    when: "Internal tensor construction, table mapping, or provider interaction fails.",
    message: "log2: internal error",
};
pub const LOG2_ERROR_COMPLEX_DISSECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.COMPLEX_DISSECTION",
    identifier: Some("RunMat:log2:ComplexDissection"),
    when: "Complex input is supplied to the two-output floating-point dissection form.",
    message: "log2: two-output dissection requires real input",
};
pub const LOG2_ERROR_GPU_DISSECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.GPU_DISSECTION",
    identifier: Some("RunMat:log2:GpuDissectionUnsupported"),
    when: "A GPU-resident input is supplied to the two-output floating-point dissection form.",
    message: "log2: two-output dissection does not support gpuArray input",
};
pub const LOG2_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:log2:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "log2: at most two outputs are available",
};
pub const LOG2_ERROR_PROVIDER_OWNERSHIP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.PROVIDER_OWNERSHIP_MISMATCH",
    identifier: Some("RunMat:gpu:ProviderOwnershipMismatch"),
    when: "A resident input has no exact owning provider.",
    message: "log2: resident input has no exact owning provider",
};
pub const LOG2_ERROR_GPU_COMPLEX_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.GPU_COMPLEX_INPUT_REQUIRED",
    identifier: Some("RunMat:log2:GpuComplexInputRequired"),
    when: "Explicitly resident real input would require a complex result.",
    message: "log2: real gpuArray input must be explicitly complex when the result can be complex",
};
const LOG2_ERRORS: [BuiltinErrorDescriptor; 7] = [
    LOG2_ERROR_INVALID_INPUT,
    LOG2_ERROR_INTERNAL,
    LOG2_ERROR_COMPLEX_DISSECTION,
    LOG2_ERROR_GPU_DISSECTION,
    LOG2_ERROR_TOO_MANY_OUTPUTS,
    LOG2_ERROR_PROVIDER_OWNERSHIP,
    LOG2_ERROR_GPU_COMPLEX_INPUT,
];
pub const LOG2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOG2_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOG2_ERRORS,
};

pub const LOG2_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log2-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log2 with integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log2IntegerInputExtension"),
};
pub const LOG2_LOGICAL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log2-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log2 with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log2LogicalInputExtension"),
};
pub const LOG2_CHARACTER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log2-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log2 with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log2CharacterInputExtension"),
};
pub const LOG2_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    LOG2_INTEGER_EXTENSION,
    LOG2_LOGICAL_EXTENSION,
    LOG2_CHARACTER_EXTENSION,
];

const LOG2_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are accepted only in RunMat mode and only when every value lies in the inclusive exact binary64 interval [-2^53, 2^53].",
    }];
pub const LOG2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = log2(integer_X)",
        inputs: &LOG2_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The RunMat-only overload validates exact binary64 conversion before computation. Negative real values produce complex double output; resident values gather through their exact owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[F,E] = log2(integer_X)",
        inputs: &LOG2_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The RunMat-only dissection validates exact binary64 conversion and returns host double fraction and exponent arrays.",
    },
];

const LOG2_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const MAY_THROW: [EffectKind; 1] = [EffectKind::MayThrow];

pub const LOG2_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "log2" },
    category: "math/elementwise",
    documentation: LOG2_DOCUMENTATION,
    descriptor: &LOG2_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Log2),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &MAY_THROW,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &LOG2_BINDINGS,
    extensions: &LOG2_EXTENSIONS,
    integer_capabilities: &LOG2_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
