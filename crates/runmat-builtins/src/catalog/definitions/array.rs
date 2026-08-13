use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingAvailability,
    BuiltinBindingDeclaration, BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRuleId,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, ALL_INTEGER_CLASSES,
};
use runmat_types::EffectKind;

pub const FULL_INTEGER_SPARSE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "full-integer-sparse",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sparse matrices with integer value storage are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FullIntegerSparseExtension"),
};

pub const FULL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [FULL_INTEGER_SPARSE_EXTENSION];

const FULL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Full matrix or already-full input.",
}];
const FULL_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse or full matrix.",
}];
const FULL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "A = full(S)",
    inputs: &FULL_INPUT,
    outputs: &FULL_OUTPUT,
}];
const FULL_DENSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "S",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An already-full integer matrix is returned identically, preserving exact class, shape, values, storage, and residency.",
    }];
const FULL_SPARSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "S",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat sparse integer CSC storage densifies exactly to the same integer class; public sparse value storage is limited to double, single, and logical.",
    }];
pub const FULL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "A = full(S) with already-full integer S",
        inputs: &FULL_DENSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host values pass through unchanged. Resident values retain the original handle and owning provider without dispatch through the ambient provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "A = full(S) with RunMat sparse integer S",
        inputs: &FULL_SPARSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The host CSC payload is expanded into exact dense integer storage after the independent full-integer-sparse compatibility gate.",
    },
];
pub const FULL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FULL.INVALID_INPUT",
    identifier: Some("RunMat:full:InvalidInput"),
    when: "Input is not a sparse matrix or already-full matrix value.",
    message: "full: invalid input",
};
pub const FULL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FULL.INTERNAL",
    identifier: Some("RunMat:full:Internal"),
    when: "Sparse-to-full materialisation fails internally.",
    message: "full: internal error",
};
const FULL_ERRORS: [BuiltinErrorDescriptor; 2] = [FULL_ERROR_INVALID_INPUT, FULL_ERROR_INTERNAL];
pub const FULL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FULL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FULL_ERRORS,
};

const FULL_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "full" },
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const FULL_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];

pub const FULL_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "full" },
    category: "array/creation",
    documentation: BuiltinDocumentation {
        summary: "Convert sparse matrix storage to full storage.",
        keywords: &["dense", "full", "matrix", "sparse", "storage"],
        related: &[],
        introduced: None,
        status: None,
        examples: &[],
    },
    descriptor: &FULL_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRuleId("array.full"),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &FULL_EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Boundary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        artifact_dependencies: &[],
    },
    bindings: &FULL_BINDINGS,
    extensions: &FULL_EXTENSIONS,
    integer_capabilities: &FULL_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

const ZEROS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output array.",
}];
const ZEROS_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];
const ZEROS_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];
const ZEROS_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];
const ZEROS_SIG_PROTOTYPE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prototype",
    ty: BuiltinParamType::LikePrototype,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Prototype value when no numeric dimension arguments are provided.",
}];
const ZEROS_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];
const ZEROS_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Class name override.",
    },
];
const ZEROS_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype array used for class/device.",
    },
];
const ZEROS_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "A = zeros()",
        inputs: &ZEROS_SIG_EMPTY_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(n)",
        inputs: &ZEROS_SIG_N_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(size_vector)",
        inputs: &ZEROS_SIG_SIZE_VECTOR_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(m, n, ...)",
        inputs: &ZEROS_SIG_DIMS_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(prototype)",
        inputs: &ZEROS_SIG_PROTOTYPE_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(..., typename)",
        inputs: &ZEROS_SIG_CLASS_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(..., \"like\", prototype)",
        inputs: &ZEROS_SIG_LIKE_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
];
pub const ZEROS_ERROR_LIKE_EXPECTED_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZEROS.LIKE_EXPECTED_PROTOTYPE",
    identifier: None,
    when: "The 'like' keyword is provided without a prototype argument.",
    message: "zeros: expected prototype after 'like'",
};
pub const ZEROS_ERROR_CLASS_CONFLICT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZEROS.CLASS_CONFLICT",
    identifier: None,
    when: "A class keyword and a 'like' prototype are both provided.",
    message: "zeros: cannot combine 'like' with other class specifiers",
};
pub const ZEROS_ERROR_UNRECOGNIZED_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZEROS.UNRECOGNIZED_OPTION",
    identifier: None,
    when: "A trailing option string is not a supported class keyword.",
    message: "zeros: unrecognised option",
};
pub const ZEROS_ERROR_LIKE_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZEROS.LIKE_DUPLICATE",
    identifier: None,
    when: "The 'like' keyword is specified more than once.",
    message: "zeros: multiple 'like' specifications are not supported",
};
const ZEROS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ZEROS_ERROR_LIKE_EXPECTED_PROTOTYPE,
    ZEROS_ERROR_CLASS_CONFLICT,
    ZEROS_ERROR_UNRECOGNIZED_OPTION,
    ZEROS_ERROR_LIKE_DUPLICATE,
];
pub const ZEROS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZEROS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZEROS_ERRORS,
};
const ZEROS_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "zeros" },
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const ZEROS_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const ZEROS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "zeros" },
    category: "array/creation",
    documentation: BuiltinDocumentation {
        summary: "Create arrays filled with zero values.",
        keywords: &["array", "gpu", "like", "logical", "zeros"],
        related: &[],
        introduced: None,
        status: None,
        examples: &[],
    },
    descriptor: &ZEROS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRuleId("array.zeros"),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &ZEROS_EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Candidate,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        artifact_dependencies: &[],
    },
    bindings: &ZEROS_BINDINGS,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};
