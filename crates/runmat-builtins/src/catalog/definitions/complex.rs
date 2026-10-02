use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingAvailability,
    BuiltinBindingDeclaration, BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation, BuiltinErrorDescriptor,
    BuiltinFusionPolicy, BuiltinInferenceRuleId, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, ALL_INTEGER_CLASSES,
};
use runmat_types::EffectKind;

const COMPLEX_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "If either component is integer, the other component must use the same integer class or be a full scalar double." },
    BuiltinIntegerInputCapability { name: "B", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The integer class is retained by the complex result; scalar doubles use integer conversion semantics." },
];

const COMPLEX_UNARY_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Unary complex preserves the input integer class and adds an exact zero imaginary component.",
    }];

pub const COMPLEX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Z = complex(integer_A)",
        inputs: &COMPLEX_UNARY_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "All eight integer classes retain exact real storage and receive a same-class zero imaginary lane.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Z = complex(integer_A, integer_B_or_scalar_double)",
        inputs: &COMPLEX_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Host complex integer storage is exact for all eight classes. Resident typed integers gather exactly for composition and restore paired native storage to the owning provider.",
    },
];

const COMPLEX_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex result.",
}];

const COMPLEX_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric input to lift into complex storage.",
}];

const COMPLEX_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real part operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Imaginary part operand.",
    },
];

const COMPLEX_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Z = complex(A)",
        inputs: &COMPLEX_INPUTS_A,
        outputs: &COMPLEX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Z = complex(A, B)",
        inputs: &COMPLEX_INPUTS_A_B,
        outputs: &COMPLEX_OUTPUT,
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
    when: "Input value cannot be converted into real numeric tensor inputs.",
    message: "complex: invalid input",
};

pub const COMPLEX_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.SIZE_MISMATCH",
    identifier: Some("RunMat:complex:SizeMismatch"),
    when: "Real and imaginary parts are not compatible for scalar expansion.",
    message: "complex: size mismatch",
};

pub const COMPLEX_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.INTERNAL",
    identifier: Some("RunMat:complex:Internal"),
    when: "Internal complex tensor construction failed.",
    message: "complex: internal error",
};

pub const COMPLEX_ERROR_INTEGER_CLASS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COMPLEX.INTEGER_CLASS",
    identifier: Some("RunMat:complex:IntegerClass"),
    when: "An integer input is paired with an unsupported unlike input class.",
    message: "complex: integer inputs require matching integer classes or a scalar double",
};

const COMPLEX_ERRORS: [BuiltinErrorDescriptor; 5] = [
    COMPLEX_ERROR_INVALID_ARGUMENT,
    COMPLEX_ERROR_INVALID_INPUT,
    COMPLEX_ERROR_SIZE_MISMATCH,
    COMPLEX_ERROR_INTERNAL,
    COMPLEX_ERROR_INTEGER_CLASS,
];

pub const COMPLEX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPLEX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COMPLEX_ERRORS,
};

const COMPLEX_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "complex" },
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const COMPLEX_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const COMPLEX_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "complex" },
    category: "math/elementwise",
    documentation: BuiltinDocumentation {
        summary: "Construct complex values from real and imaginary parts.",
        keywords: &["complex", "construct", "imaginary", "real", "elementwise"],
        related: &[],
        introduced: None,
        status: None,
        examples: &[],
    },
    descriptor: &COMPLEX_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRuleId("math.complex"),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &COMPLEX_EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Never,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        artifact_dependencies: &[],
    },
    bindings: &COMPLEX_BINDINGS,
    extensions: &[],
    integer_capabilities: &COMPLEX_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
