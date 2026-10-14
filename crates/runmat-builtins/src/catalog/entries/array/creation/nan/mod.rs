mod documentation;
mod inference;
use crate::*;
const NAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "NaN-filled output array.",
}];

const NAN_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const NAN_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const NAN_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const NAN_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const NAN_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
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
        description: "Class name override (double|single|gpuArray).",
    },
];

const NAN_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
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

const NAN_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "A = nan()",
        inputs: &NAN_SIG_EMPTY_INPUTS,
        outputs: &NAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = nan(n)",
        inputs: &NAN_SIG_N_INPUTS,
        outputs: &NAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = nan(size_vector)",
        inputs: &NAN_SIG_SIZE_VECTOR_INPUTS,
        outputs: &NAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = nan(m, n, ...)",
        inputs: &NAN_SIG_DIMS_INPUTS,
        outputs: &NAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = nan(..., typename)",
        inputs: &NAN_SIG_CLASS_INPUTS,
        outputs: &NAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = nan(..., \"like\", prototype)",
        inputs: &NAN_SIG_LIKE_INPUTS,
        outputs: &NAN_OUTPUT,
    },
];

pub const NAN_COLUMN_SIZE_VECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "nan-column-size-vector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "nan with a column size vector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:NanColumnSizeVectorExtension"),
    };
pub const NAN_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nan-resident-size-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "nan with a resident size control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NanResidentSizeControlExtension"),
};
pub const NAN_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    NAN_COLUMN_SIZE_VECTOR_EXTENSION,
    NAN_RESIDENT_SIZE_EXTENSION,
];
const NAN_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz1...szN/sz",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are exact structural size controls; negative signed values clamp to zero and trailing singleton dimensions normalize away.",
    }];
pub const NAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "X = nan(integer_n[, integer_sz2, ...])",
        inputs: &NAN_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The default output is host double; typename can select single and explicit gpuArray syntax selects residency.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = nan(integer_sz)",
        inputs: &NAN_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented size vector is a row vector of exact integer values.",
    },
];

pub const NAN_ERROR_LIKE_EXPECTED_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAN.LIKE_EXPECTED_PROTOTYPE",
    identifier: None,
    when: "The 'like' keyword is provided without a prototype argument.",
    message: "nan: expected prototype after 'like'",
};

pub const NAN_ERROR_CLASS_CONFLICT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAN.CLASS_CONFLICT",
    identifier: None,
    when: "A class keyword and a 'like' prototype are both provided.",
    message: "nan: cannot combine 'like' with other class specifiers",
};

pub const NAN_ERROR_UNRECOGNIZED_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAN.UNRECOGNIZED_OPTION",
    identifier: None,
    when: "A trailing option string is not a supported class keyword.",
    message: "nan: unrecognised option",
};

pub const NAN_ERROR_LIKE_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAN.LIKE_DUPLICATE",
    identifier: None,
    when: "The 'like' keyword is specified more than once.",
    message: "nan: multiple 'like' specifications are not supported",
};

pub const NAN_ERROR_INTEGER_LIKE_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAN.INTEGER_LIKE_PROTOTYPE",
    identifier: None,
    when: "The 'like' prototype has an integer data type that cannot represent NaN.",
    message: "nan: integer 'like' prototypes are not supported",
};

pub const NAN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:nan:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "nan: too many output arguments",
};
const NAN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    NAN_ERROR_TOO_MANY_OUTPUTS,
    NAN_ERROR_LIKE_EXPECTED_PROTOTYPE,
    NAN_ERROR_CLASS_CONFLICT,
    NAN_ERROR_UNRECOGNIZED_OPTION,
    NAN_ERROR_LIKE_DUPLICATE,
    NAN_ERROR_INTEGER_LIKE_PROTOTYPE,
];

pub const NAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NAN_ERRORS,
};

pub const NAN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "nan" },
    category: "array/creation",
    documentation: documentation::DOCUMENTATION,
    descriptor: &NAN_DESCRIPTOR,
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
    extensions: &NAN_EXTENSIONS,
    integer_capabilities: &NAN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&NAN_CATALOG_ENTRY];
pub(super) const CONSTANTS: &[BuiltinConstantCatalogEntry] = &[
    BuiltinConstantCatalogEntry::new(
        "nan",
        BuiltinConstantKind::RealDouble,
        BuiltinCatalogProvenance::new(file!(), module_path!()),
    ),
    BuiltinConstantCatalogEntry::new(
        "NaN",
        BuiltinConstantKind::RealDouble,
        BuiltinCatalogProvenance::new(file!(), module_path!()),
    ),
];
