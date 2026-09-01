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
    BuiltinSignatureDescriptor, InverseHyperbolicFunction, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;

use documentation::ACOSH_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise principal inverse-hyperbolic-cosine result.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = acosh(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ACOSH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACOSH.INVALID_INPUT",
    identifier: Some("RunMat:acosh:InvalidInput"),
    when: "Input is not a supported numeric, logical, or character value.",
    message: "acosh: invalid input",
};
pub const ACOSH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACOSH.INTERNAL",
    identifier: Some("RunMat:acosh:Internal"),
    when:
        "Internal gather, domain inspection, conversion, allocation, or provider restoration fails.",
    message: "acosh: internal error",
};
pub const ACOSH_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACOSH.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:acosh:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "acosh: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ACOSH_ERROR_INVALID_INPUT,
    ACOSH_ERROR_INTERNAL,
    ACOSH_ERROR_TOO_MANY_OUTPUTS,
];
pub const ACOSH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ACOSH_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "acosh-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "acosh with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AcoshIntegerInputExtension"),
};
pub const ACOSH_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "acosh-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "acosh with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AcoshLogicalInputExtension"),
};
pub const ACOSH_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "acosh-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "acosh with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AcoshCharacterInputExtension"),
    };
pub const ACOSH_GPU_REAL_COMPLEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "acosh-gpu-real-complex-promotion",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "acosh resident real input that requires complex output is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AcoshGpuRealComplexPromotionExtension"),
    };
pub const ACOSH_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    ACOSH_INTEGER_INPUT_EXTENSION,
    ACOSH_LOGICAL_INPUT_EXTENSION,
    ACOSH_CHARACTER_INPUT_EXTENSION,
    ACOSH_GPU_REAL_COMPLEX_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes enter the declared binary64 inverse-hyperbolic-cosine computation boundary.",
}];
pub const ACOSH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = acosh(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative integer storage converts explicitly at the binary64 algorithm boundary. Values below one produce complex double output; resident values gather and restore through their owning provider.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const ACOSH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "acosh" },
    category: "math/trigonometry",
    documentation: ACOSH_DOCUMENTATION,
    descriptor: &ACOSH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::InverseHyperbolic(
            InverseHyperbolicFunction::Cosine,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &ACOSH_EXTENSIONS,
    integer_capabilities: &ACOSH_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
