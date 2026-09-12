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
    BuiltinSignatureDescriptor, HyperbolicFunction, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::COSH_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise hyperbolic-cosine result.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = cosh(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const COSH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSH.INVALID_INPUT",
    identifier: Some("RunMat:cosh:InvalidInput"),
    when: "Input is not a supported numeric, logical, or character value.",
    message: "cosh: invalid input",
};
pub const COSH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSH.INTERNAL",
    identifier: Some("RunMat:cosh:Internal"),
    when: "Internal gather, conversion, allocation, provider execution, or restoration fails.",
    message: "cosh: internal error",
};
pub const COSH_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSH.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:cosh:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "cosh: too many output arguments",
};
pub const COSH_ERROR_INEXACT_INTEGER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSH.INEXACT_INTEGER",
    identifier: None,
    when: "An integer input cannot be represented exactly at the binary64 computation boundary.",
    message: "cosh: integer X values must be exactly representable as double",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    COSH_ERROR_INVALID_INPUT,
    COSH_ERROR_INTERNAL,
    COSH_ERROR_TOO_MANY_OUTPUTS,
    COSH_ERROR_INEXACT_INTEGER,
];
pub const COSH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const COSH_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cosh-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cosh with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CoshIntegerInputExtension"),
};
pub const COSH_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cosh-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cosh with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CoshLogicalInputExtension"),
};
pub const COSH_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cosh-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cosh with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CoshCharacterInputExtension"),
};
pub const COSH_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    COSH_INTEGER_INPUT_EXTENSION,
    COSH_LOGICAL_INPUT_EXTENSION,
    COSH_CHARACTER_INPUT_EXTENSION,
];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes require exact binary64 representability at the hyperbolic-cosine computation boundary.",
}];
pub const COSH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = cosh(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative integer storage converts explicitly at the binary64 algorithm boundary. Large finite values may overflow naturally to infinity.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const COSH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "cosh" },
    category: "math/trigonometry",
    documentation: COSH_DOCUMENTATION,
    descriptor: &COSH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Hyperbolic(
            HyperbolicFunction::Cosine,
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
        fusion: BuiltinFusionPolicy::Candidate,
        distributed: BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &COSH_EXTENSIONS,
    integer_capabilities: &COSH_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&COSH_CATALOG_ENTRY];
