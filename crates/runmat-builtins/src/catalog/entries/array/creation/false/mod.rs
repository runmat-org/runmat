mod documentation;
mod inference;
use crate::*;
pub const FALSE_IMPLICIT_PROTOTYPE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "false-implicit-prototype",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "false(A) implicit size-prototype syntax is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FalseImplicitPrototypeExtension"),
    };

pub const FALSE_LOGICAL_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "false-logical-class-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "false(...,'logical') is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FalseLogicalOptionExtension"),
};

pub const FALSE_SINGLE_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "false-single-size-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision false size controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FalseSingleSizeExtension"),
};

pub const FALSE_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "false-resident-size-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "resident false size controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FalseResidentSizeExtension"),
};

pub const FALSE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FALSE_IMPLICIT_PROTOTYPE_EXTENSION,
    FALSE_LOGICAL_OPTION_EXTENSION,
    FALSE_SINGLE_SIZE_EXTENSION,
    FALSE_RESIDENT_SIZE_EXTENSION,
];

const FALSE_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz/szN",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are exact structural size controls. Negative signed values clamp to zero.",
    }];

const FALSE_INTEGER_LIKE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An integer prototype selects applicable sparsity and residency; false output remains logical and the prototype never supplies shape.",
    }];

pub const FALSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "F = false(integer_n or integer_sz1,...,integer_szN)",
        inputs: &FALSE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Exact integer dimensions determine shape only; output is always logical.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "F = false(integer_sz)",
        inputs: &FALSE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented row size vector is decoded exactly from every supported integer class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "F = false(..., like=integer_p)",
        inputs: &FALSE_INTEGER_LIKE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The logical output preserves applicable sparse or owning-provider residency but not the integer prototype class.",
    },
];

const FALSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical output array.",
}];

const FALSE_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const FALSE_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const FALSE_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const FALSE_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const FALSE_SIG_PROTOTYPE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prototype",
    ty: BuiltinParamType::LikePrototype,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Prototype value when no numeric dimension arguments are provided.",
}];

const FALSE_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const FALSE_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
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

const FALSE_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "L = false()",
        inputs: &FALSE_SIG_EMPTY_INPUTS,
        outputs: &FALSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(n)",
        inputs: &FALSE_SIG_N_INPUTS,
        outputs: &FALSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(size_vector)",
        inputs: &FALSE_SIG_SIZE_VECTOR_INPUTS,
        outputs: &FALSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(m, n, ...)",
        inputs: &FALSE_SIG_DIMS_INPUTS,
        outputs: &FALSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(prototype)",
        inputs: &FALSE_SIG_PROTOTYPE_INPUTS,
        outputs: &FALSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(..., \"logical\")",
        inputs: &FALSE_SIG_CLASS_INPUTS,
        outputs: &FALSE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = false(..., \"like\", prototype)",
        inputs: &FALSE_SIG_LIKE_INPUTS,
        outputs: &FALSE_OUTPUT,
    },
];

pub const FALSE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FALSE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:false:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "false: too many output arguments",
};
const FALSE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FALSE_ERROR_TOO_MANY_OUTPUTS,
    BuiltinErrorDescriptor {
        code: "RM.FALSE.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "false: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.FALSE.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "false: multiple 'like' specifications are not supported",
    },
    BuiltinErrorDescriptor {
        code: "RM.FALSE.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not supported.",
        message: "false: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.FALSE.INVALID_DIMS",
        identifier: None,
        when: "A numeric dimension is fractional or nonfinite; negative dimensions instead clamp to zero.",
        message: "false: dimensions must be finite integer values",
    },
];

pub const FALSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FALSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FALSE_ERRORS,
};

pub const FALSE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "false" },
    category: "array/creation",
    documentation: documentation::DOCUMENTATION,
    descriptor: &FALSE_DESCRIPTOR,
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
    extensions: &FALSE_EXTENSIONS,
    integer_capabilities: &FALSE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FALSE_CATALOG_ENTRY];
pub(super) const CONSTANTS: &[BuiltinConstantCatalogEntry] = &[BuiltinConstantCatalogEntry::new(
    "false",
    BuiltinConstantKind::Logical,
    BuiltinCatalogProvenance::new(file!(), module_path!()),
)
.with_literal(crate::BuiltinConstantLiteral::Logical(false))];
