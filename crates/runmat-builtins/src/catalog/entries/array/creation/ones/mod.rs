mod documentation;
mod inference;
use crate::*;
use runmat_types::EffectKind;

const ONES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output array.",
}];

const ONES_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const ONES_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const ONES_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const ONES_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const ONES_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
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
        description:
            "Class name override (double|single|logical|int8|int16|int32|int64|uint8|uint16|uint32|uint64).",
    },
];

const ONES_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
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

const ONES_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "A = ones()",
        inputs: &ONES_SIG_EMPTY_INPUTS,
        outputs: &ONES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = ones(n)",
        inputs: &ONES_SIG_N_INPUTS,
        outputs: &ONES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = ones(size_vector)",
        inputs: &ONES_SIG_SIZE_VECTOR_INPUTS,
        outputs: &ONES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = ones(m, n, ...)",
        inputs: &ONES_SIG_DIMS_INPUTS,
        outputs: &ONES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = ones(..., typename)",
        inputs: &ONES_SIG_CLASS_INPUTS,
        outputs: &ONES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = ones(..., \"like\", prototype)",
        inputs: &ONES_SIG_LIKE_INPUTS,
        outputs: &ONES_OUTPUT,
    },
];

pub const ONES_COLUMN_SIZE_VECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ones-column-size-vector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ones with a column size vector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OnesColumnSizeVectorExtension"),
    };
pub const ONES_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ones-resident-size-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ones with a resident size control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:OnesResidentSizeControlExtension"),
};
pub const ONES_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ONES_COLUMN_SIZE_VECTOR_EXTENSION,
    ONES_RESIDENT_SIZE_EXTENSION,
];
const ONES_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz1...szN/sz",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are exact structural size controls; negative signed values clamp to zero and trailing singleton dimensions normalize away.",
    }];
const ONES_INTEGER_LIKE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An integer prototype selects exact output class, sparsity, complexity, and applicable residency; without dimensions the output is scalar.",
    }];
pub const ONES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "X = ones(integer_n[, integer_sz2, ...])",
        inputs: &ONES_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The default output is double; typename or like can select logical, single, or an exact integer class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = ones(integer_sz)",
        inputs: &ONES_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented size vector is a row vector of exact integer values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = ones(..., integer_typename)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Every integer typename creates exact native one storage in the selected class, including explicit gpuArray construction.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = ones(..., like=integer_p)",
        inputs: &ONES_INTEGER_LIKE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer prototypes preserve exact class; resident prototypes use typed owning-provider upload when available.",
    },
];

pub const ONES_ERROR_LIKE_EXPECTED_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONES.LIKE_EXPECTED_PROTOTYPE",
    identifier: None,
    when: "The 'like' keyword is provided without a prototype argument.",
    message: "ones: expected prototype after 'like'",
};

pub const ONES_ERROR_CLASS_CONFLICT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONES.CLASS_CONFLICT",
    identifier: None,
    when: "A class keyword and a 'like' prototype are both provided.",
    message: "ones: cannot combine 'like' with other class specifiers",
};

pub const ONES_ERROR_UNRECOGNIZED_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONES.UNRECOGNIZED_OPTION",
    identifier: None,
    when: "A trailing option string is not a supported class keyword.",
    message: "ones: unrecognised option",
};

pub const ONES_ERROR_LIKE_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONES.LIKE_DUPLICATE",
    identifier: None,
    when: "The 'like' keyword is specified more than once.",
    message: "ones: multiple 'like' specifications are not supported",
};

const ONES_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ONES_ERROR_LIKE_EXPECTED_PROTOTYPE,
    ONES_ERROR_CLASS_CONFLICT,
    ONES_ERROR_UNRECOGNIZED_OPTION,
    ONES_ERROR_LIKE_DUPLICATE,
];

pub const ONES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ONES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ONES_ERRORS,
};

pub const ONES_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "ones" },
    category: "array/creation",
    documentation: documentation::DOCUMENTATION,
    descriptor: &ONES_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Candidate,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &ONES_EXTENSIONS,
    integer_capabilities: &ONES_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ONES_CATALOG_ENTRY];
