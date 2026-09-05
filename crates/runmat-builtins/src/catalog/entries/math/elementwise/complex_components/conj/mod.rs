mod documentation;

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
    BuiltinSignatureDescriptor, MathInferenceRule, NumericComponentRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

use documentation::DOCUMENTATION;

const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex conjugate of X.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, character, or complex input.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = conj(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONJ.INVALID_INPUT",
    identifier: Some("RunMat:conj:InvalidInput"),
    when: "Input cannot be interpreted as numeric, logical, character, or complex data.",
    message: "conj: invalid input",
};
pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONJ.INTERNAL",
    identifier: Some("RunMat:conj:Internal"),
    when: "Internal tensor conversion, allocation, or provider interaction fails.",
    message: "conj: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];
pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "conj-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "conj with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ConjCharacterInputExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [CHARACTER_INPUT_EXTENSION];

const REAL_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes are exact identity inputs.",
}];
const COMPLEX_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed complex integer storage retains its class; imaginary-component negation saturates while signed-minimum and unsigned endpoint compatibility remains evidence-open.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = conj(real_integer_X)",
        inputs: &REAL_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host inputs are returned unchanged; supported resident real integer inputs preserve the same exact handle and metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = conj(complex_integer_X)",
        inputs: &COMPLEX_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::EvidenceOpen,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Typed-complex integer storage uses saturating imaginary negation, preserves paired native storage on the host and owning provider, and retains the evidence qualification for signed-minimum and unsigned-imaginary endpoints.",
    },
];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
pub const CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "conj" },
    category: "math/elementwise",
    documentation: DOCUMENTATION,
    descriptor: &DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::NumericComponent(
            NumericComponentRule::Conjugate,
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
        fusion: BuiltinFusionPolicy::Candidate,
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
    integer_capabilities: &INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub use CATALOG_ENTRY as CONJ_CATALOG_ENTRY;
pub use CHARACTER_INPUT_EXTENSION as CONJ_CHARACTER_INPUT_EXTENSION;
pub use DESCRIPTOR as CONJ_DESCRIPTOR;
pub use ERROR_INTERNAL as CONJ_ERROR_INTERNAL;
pub use ERROR_INVALID_INPUT as CONJ_ERROR_INVALID_INPUT;
pub use EXTENSIONS as CONJ_EXTENSIONS;
pub use INTEGER_CAPABILITIES as CONJ_INTEGER_CAPABILITIES;
