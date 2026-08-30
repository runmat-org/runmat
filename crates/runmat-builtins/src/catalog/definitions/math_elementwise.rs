use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingAvailability,
    BuiltinBindingDeclaration, BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

const EXP_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise exponential result.",
}];
const EXP_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const EXP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = exp(X)",
    inputs: &EXP_INPUTS,
    outputs: &EXP_OUTPUT,
}];
pub const EXP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXP.INVALID_INPUT",
    identifier: Some("RunMat:exp:InvalidInput"),
    when: "Input cannot be interpreted as numeric, logical, char, or complex data.",
    message: "exp: invalid input",
};
pub const EXP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXP.INTERNAL",
    identifier: Some("RunMat:exp:Internal"),
    when: "Internal tensor construction or provider interaction failed.",
    message: "exp: internal error",
};
const EXP_ERRORS: [BuiltinErrorDescriptor; 2] = [EXP_ERROR_INVALID_INPUT, EXP_ERROR_INTERNAL];

pub const EXP_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exp-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exp with integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExpIntegerInputExtension"),
};
pub const EXP_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exp-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exp with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExpLogicalInputExtension"),
};
pub const EXP_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exp-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exp with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExpCharacterInputExtension"),
};
pub const EXP_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    EXP_INTEGER_INPUT_EXTENSION,
    EXP_LOGICAL_INPUT_EXTENSION,
    EXP_CHARACTER_INPUT_EXTENSION,
];

const EXP_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are accepted only in RunMat extension mode and only when every value lies in the inclusive exact binary64 interval [-2^53, 2^53].",
}];
pub const EXP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = exp(integer_X)",
        inputs: &EXP_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The RunMat-only overload validates exact binary64 conversion before exponentiation. Resident integer input gathers exactly through its owning provider; the double result is restored only when that provider physically supports binary64, otherwise it remains a host double.",
    }];
pub const EXP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EXP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EXP_ERRORS,
};

const EXP_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "exp" },
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const EXP_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];

pub const EXP_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "exp" },
    category: "math/elementwise",
    documentation: BuiltinDocumentation {
        summary: "Compute element-wise exponential values.",
        keywords: &["exp", "exponential", "elementwise", "gpu"],
        related: &["log", "expm1"],
        introduced: None,
        status: None,
        examples: &[],
    },
    descriptor: &EXP_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Exp),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EXP_EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Candidate,
        distributed: crate::BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &EXP_BINDINGS,
    extensions: &EXP_EXTENSIONS,
    integer_capabilities: &EXP_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

const UINT16_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "uint16-converted output value.",
}];
const UINT16_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar or array value to convert.",
}];
const UINT16_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = uint16(X)",
    inputs: &UINT16_INPUTS,
    outputs: &UINT16_OUTPUT,
}];
pub const UINT16_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT16.INVALID_ARGUMENT",
    identifier: Some("RunMat:uint16:InvalidArgument"),
    when: "The call has an unsupported number of arguments.",
    message: "uint16: invalid argument",
};
pub const UINT16_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT16.INVALID_INPUT",
    identifier: Some("RunMat:uint16:InvalidInput"),
    when: "Input cannot be converted to uint16.",
    message: "uint16: invalid input",
};
pub const UINT16_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT16.INTERNAL",
    identifier: Some("RunMat:uint16:Internal"),
    when: "Internal conversion, gather, or provider upload fails.",
    message: "uint16: internal error",
};
const UINT16_ERRORS: [BuiltinErrorDescriptor; 3] = [
    UINT16_ERROR_INVALID_ARGUMENT,
    UINT16_ERROR_INVALID_INPUT,
    UINT16_ERROR_INTERNAL,
];
pub const UINT16_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UINT16_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UINT16_ERRORS,
};
const UINT16_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every native integer class converts directly to authoritative uint16 storage without a floating intermediate.",
    }];
pub const UINT16_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = uint16(integer_X)",
        inputs: &UINT16_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host and resident conversion is exact and saturating. Real and paired-complex gpuArray inputs preserve native uint16 device storage, owner, and residency.",
    }];
const UINT16_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "uint16" },
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const UINT16_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];

pub const UINT16_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "uint16" },
    category: "math/elementwise",
    documentation: BuiltinDocumentation {
        summary: "Convert values to unsigned 16-bit integer storage.",
        keywords: &["uint16", "cast", "integer", "conversion", "gpuArray"],
        related: &["double", "single", "int16", "uint8", "uint32"],
        introduced: None,
        status: None,
        examples: &[],
    },
    descriptor: &UINT16_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::NumericConversion(
            runmat_types::NumericClass::UInt16,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &UINT16_EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Candidate,
        distributed: crate::BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &UINT16_BINDINGS,
    extensions: &[],
    integer_capabilities: &UINT16_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
