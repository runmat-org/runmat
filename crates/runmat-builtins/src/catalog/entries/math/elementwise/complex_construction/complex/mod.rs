mod documentation;

use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinErrorDescriptor, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

use documentation::DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex result with the selected numeric class and compatible input shape.",
}];
const UNARY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric input to lift, or an existing complex value to preserve.",
}];
const BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real component.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Imaginary component.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Z = complex(A)",
        inputs: &UNARY_INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Z = complex(A, B)",
        inputs: &BINARY_INPUTS,
        outputs: &OUTPUTS,
    },
];

pub const COMPLEX_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.INVALID_ARGUMENT",
    identifier: Some("RunMat:complex:InvalidArgument"),
    when: "Argument arity is invalid.",
    message: "complex: invalid argument",
};
pub const COMPLEX_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.INVALID_INPUT",
    identifier: Some("RunMat:complex:InvalidInput"),
    when:
        "An input is not a supported real numeric component, or a binary input is already complex.",
    message: "complex: invalid input",
};
pub const COMPLEX_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.SIZE_MISMATCH",
    identifier: Some("RunMat:complex:SizeMismatch"),
    when: "Two non-scalar components do not have the same shape.",
    message: "complex: size mismatch",
};
pub const COMPLEX_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.INTERNAL",
    identifier: Some("RunMat:complex:Internal"),
    when: "Internal conversion, allocation, provider execution, or residency restoration fails.",
    message: "complex: internal error",
};
pub const COMPLEX_ERROR_INTEGER_CLASS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.INTEGER_CLASS",
    identifier: Some("RunMat:complex:IntegerClass"),
    when: "An integer component is paired with an unsupported unlike class.",
    message: "complex: integer inputs require matching integer classes or a scalar double",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    COMPLEX_ERROR_INVALID_ARGUMENT,
    COMPLEX_ERROR_INVALID_INPUT,
    COMPLEX_ERROR_SIZE_MISMATCH,
    COMPLEX_ERROR_INTERNAL,
    COMPLEX_ERROR_INTEGER_CLASS,
];
pub const COMPLEX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const BINARY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "If either component is integer, its peer must use the same integer class or be a full scalar double." },
    BuiltinIntegerInputCapability { name: "B", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The complex result retains the integer class; a scalar double uses that class's conversion semantics." },
];
const UNARY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Unary complex preserves the integer class and adds an exact same-class zero imaginary component." }];
pub const COMPLEX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "Z = complex(integer_A)", inputs: &UNARY_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "All eight integer classes retain exact real storage and receive a same-class zero imaginary lane." },
    BuiltinIntegerCapabilityDescriptor { form: "Z = complex(integer_A, integer_B_or_scalar_double)", inputs: &BINARY_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "Host composition is exact for all eight classes. Resident typed integers gather exactly and restore paired native storage to their owner." },
];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const COMPLEX_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "complex" },
    category: "math/elementwise",
    documentation: DOCUMENTATION,
    descriptor: &COMPLEX_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::ComplexConstruction),
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
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &[],
    integer_capabilities: &COMPLEX_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
