mod documentation;
mod inference;
use crate::*;
pub const TRUE_IMPLICIT_PROTOTYPE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "true-implicit-prototype",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "true(A) implicit size-prototype syntax is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TrueImplicitPrototypeExtension"),
    };

pub const TRUE_LOGICAL_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "true-logical-class-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "true(...,'logical') is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrueLogicalOptionExtension"),
};

pub const TRUE_SINGLE_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "true-single-size-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision true size controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrueSingleSizeExtension"),
};

pub const TRUE_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "true-resident-size-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "resident true size controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrueResidentSizeExtension"),
};

pub const TRUE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    TRUE_IMPLICIT_PROTOTYPE_EXTENSION,
    TRUE_LOGICAL_OPTION_EXTENSION,
    TRUE_SINGLE_SIZE_EXTENSION,
    TRUE_RESIDENT_SIZE_EXTENSION,
];

const TRUE_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz/szN",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are exact structural size controls. Negative signed values clamp to zero.",
    }];

const TRUE_INTEGER_LIKE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An integer prototype selects applicable sparsity and residency; true output remains logical and the prototype never supplies shape.",
    }];

pub const TRUE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "T = true(integer_n or integer_sz1,...,integer_szN)",
        inputs: &TRUE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Exact integer dimensions determine shape only; output is always logical.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = true(integer_sz)",
        inputs: &TRUE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented row size vector is decoded exactly from every supported integer class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = true(..., like=integer_p)",
        inputs: &TRUE_INTEGER_LIKE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The logical output preserves applicable sparse or owning-provider residency but not the integer prototype class.",
    },
];

const TRUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical output array.",
}];

const TRUE_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const TRUE_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const TRUE_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const TRUE_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const TRUE_SIG_PROTOTYPE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prototype",
    ty: BuiltinParamType::LikePrototype,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Prototype value when no numeric dimension arguments are provided.",
}];

const TRUE_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
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
        default: Some("\"logical\""),
        description: "Class override keyword (logical).",
    },
];

const TRUE_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
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

const TRUE_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "L = true()",
        inputs: &TRUE_SIG_EMPTY_INPUTS,
        outputs: &TRUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(n)",
        inputs: &TRUE_SIG_N_INPUTS,
        outputs: &TRUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(size_vector)",
        inputs: &TRUE_SIG_SIZE_VECTOR_INPUTS,
        outputs: &TRUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(m, n, ...)",
        inputs: &TRUE_SIG_DIMS_INPUTS,
        outputs: &TRUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(prototype)",
        inputs: &TRUE_SIG_PROTOTYPE_INPUTS,
        outputs: &TRUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(..., \"logical\")",
        inputs: &TRUE_SIG_CLASS_INPUTS,
        outputs: &TRUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = true(..., \"like\", prototype)",
        inputs: &TRUE_SIG_LIKE_INPUTS,
        outputs: &TRUE_OUTPUT,
    },
];

pub const TRUE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRUE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:true:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "true: too many output arguments",
};
const TRUE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    TRUE_ERROR_TOO_MANY_OUTPUTS,
    BuiltinErrorDescriptor {
        code: "RM.TRUE.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "true: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.TRUE.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "true: multiple 'like' specifications are not supported",
    },
    BuiltinErrorDescriptor {
        code: "RM.TRUE.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not supported.",
        message: "true: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.TRUE.INVALID_DIMS",
        identifier: None,
        when: "A numeric dimension is fractional or nonfinite; negative dimensions instead clamp to zero.",
        message: "true: dimensions must be finite integer values",
    },
];

pub const TRUE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRUE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRUE_ERRORS,
};

pub const TRUE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "true" },
    category: "array/creation",
    documentation: documentation::DOCUMENTATION,
    descriptor: &TRUE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: crate::BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &TRUE_EXTENSIONS,
    integer_capabilities: &TRUE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&TRUE_CATALOG_ENTRY];
pub(super) const CONSTANTS: &[BuiltinConstantCatalogEntry] = &[BuiltinConstantCatalogEntry::new(
    "true",
    BuiltinConstantKind::Logical,
    BuiltinCatalogProvenance::new(file!(), module_path!()),
)
.with_literal(crate::BuiltinConstantLiteral::Logical(true))];
