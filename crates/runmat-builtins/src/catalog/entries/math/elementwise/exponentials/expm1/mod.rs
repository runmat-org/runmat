use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, ExponentialKind, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;

use documentation::EXPM1_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise exp(X)-1 result with the input floating-point class and shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = expm1(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const EXPM1_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXPM1.INVALID_INPUT",
    identifier: Some("RunMat:expm1:InvalidInput"),
    when: "Input cannot be interpreted as numeric, logical, character, or complex data.",
    message: "expm1: invalid input",
};
pub const EXPM1_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXPM1.INTERNAL",
    identifier: Some("RunMat:expm1:Internal"),
    when: "Internal tensor construction or provider interaction fails.",
    message: "expm1: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [EXPM1_ERROR_INVALID_INPUT, EXPM1_ERROR_INTERNAL];
pub const EXPM1_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const EXPM1_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "expm1-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "expm1 with integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Expm1IntegerInputExtension"),
};
pub const EXPM1_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "expm1-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "expm1 with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Expm1LogicalInputExtension"),
};
pub const EXPM1_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "expm1-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "expm1 with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Expm1CharacterInputExtension"),
    };
const EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    EXPM1_INTEGER_INPUT_EXTENSION,
    EXPM1_LOGICAL_INPUT_EXTENSION,
    EXPM1_CHARACTER_INPUT_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are accepted only in RunMat extension mode and only when every value lies in the inclusive exact binary64 interval [-2^53, 2^53].",
}];
pub const EXPM1_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = expm1(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The RunMat-only overload validates exact binary64 conversion before accurate exponentiation. Resident integer input gathers exactly through its owning provider; the double result is restored only when that provider physically supports binary64, otherwise it remains a host double.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];

pub const EXPM1_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "expm1" },
    category: "math/elementwise",
    documentation: EXPM1_DOCUMENTATION,
    descriptor: &EXPM1_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Exponential(
            ExponentialKind::MinusOne,
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
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Never,
        distributed: crate::BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &EXTENSIONS,
    integer_capabilities: &EXPM1_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
