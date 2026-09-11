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

use documentation::EXP_DOCUMENTATION;

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

const EXP_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EXP_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];

pub const EXP_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "exp" },
    category: "math/elementwise",
    documentation: EXP_DOCUMENTATION,
    descriptor: &EXP_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Exponential(
            ExponentialKind::Natural,
        )),
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
