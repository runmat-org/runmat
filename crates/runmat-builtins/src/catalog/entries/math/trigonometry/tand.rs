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
    BuiltinSignatureDescriptor, DegreeTrigonometricFunction, MathInferenceRule,
    ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

use super::documentation::TAND_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise tangent result for degree-valued input.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = tand(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const TAND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAND.INVALID_INPUT",
    identifier: Some("RunMat:tand:InvalidInput"),
    when: "Input is not a supported numeric, logical, or character value.",
    message: "tand: invalid input",
};
pub const TAND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAND.INTERNAL",
    identifier: Some("RunMat:tand:Internal"),
    when: "Internal gather, conversion, allocation, or provider restoration failed.",
    message: "tand: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [TAND_ERROR_INVALID_INPUT, TAND_ERROR_INTERNAL];
pub const TAND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TAND_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tand-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tand with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TandIntegerInputExtension"),
};
pub const TAND_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tand-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tand with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TandLogicalInputExtension"),
};
pub const TAND_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tand-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tand with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TandCharacterInputExtension"),
};
pub const TAND_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TAND_INTEGER_INPUT_EXTENSION,
    TAND_LOGICAL_INPUT_EXTENSION,
    TAND_CHARACTER_INPUT_EXTENSION,
];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Native integer values are reduced exactly modulo 360 before floating evaluation.",
}];
pub const TAND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = tand(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Exact native modular reduction preserves wide int64/uint64 canonical values and poles; resident output returns to its owner.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const TAND_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "tand" },
    category: "math/trigonometry",
    documentation: TAND_DOCUMENTATION,
    descriptor: &TAND_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::DegreeTrigonometric(
            DegreeTrigonometricFunction::Tan,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
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
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &TAND_EXTENSIONS,
    integer_capabilities: &TAND_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
