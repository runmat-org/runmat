use crate::{
    ArrayCreationInferenceRule, ArrayInferenceRule, BuiltinAcceleratorPolicy, BuiltinAsyncBehavior,
    BuiltinBindingDeclaration, BuiltinCatalogEntry, BuiltinCatalogIdentity, BuiltinCompatibility,
    BuiltinCompletionPolicy, BuiltinContractDeclaration, BuiltinContractMaturity,
    BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor, BuiltinExtensionMode,
    BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, ALL_INTEGER_CLASSES,
};
use runmat_types::EffectKind;

use super::super::documentation::ZEROS_DOCUMENTATION;

pub const ZEROS_COLUMN_SIZE_VECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "zeros-column-size-vector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "Allow a column vector where the public size-vector form requires a row vector",
        error_identifier: Some("RunMat:compatibility:ZerosColumnSizeVectorExtension"),
    };
pub const ZEROS_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "zeros-resident-size-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a resident numeric value as a size control",
    error_identifier: Some("RunMat:compatibility:ZerosResidentSizeControlExtension"),
};
pub const ZEROS_IMPLICIT_PROTOTYPE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "zeros-implicit-prototype",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Allow a prototype array without the like keyword",
        error_identifier: Some("RunMat:compatibility:ZerosImplicitPrototypeExtension"),
    };
pub const ZEROS_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ZEROS_COLUMN_SIZE_VECTOR_EXTENSION,
    ZEROS_RESIDENT_SIZE_EXTENSION,
    ZEROS_IMPLICIT_PROTOTYPE_EXTENSION,
];

const ZEROS_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz1...szN/sz",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are exact structural size controls; negative signed values clamp to zero and trailing singleton dimensions normalize away.",
    }];
const ZEROS_INTEGER_LIKE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An integer prototype selects exact output class, sparsity, complexity, and applicable residency.",
    }];
pub const ZEROS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "X = zeros(integer_n[, integer_sz2, ...])",
        inputs: &ZEROS_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The default output is double; typename or like can select logical, single, or an exact integer class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = zeros(integer_sz)",
        inputs: &ZEROS_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented size vector is a row vector of exact integer values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = zeros(..., integer_typename)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Every integer typename creates exact native zero storage in the selected class, including explicit gpuArray construction.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = zeros(..., like=integer_p)",
        inputs: &ZEROS_INTEGER_LIKE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer prototypes preserve exact class; resident prototypes use typed owning-provider allocation or upload when available.",
    },
];
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
const ZEROS_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const ZEROS_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const ZEROS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "zeros" },
    category: "array/creation",
    documentation: ZEROS_DOCUMENTATION,
    descriptor: &ZEROS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Creation(
            ArrayCreationInferenceRule::Zeros,
        )),
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
        distributed: crate::BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &ZEROS_BINDINGS,
    extensions: &ZEROS_EXTENSIONS,
    integer_capabilities: &ZEROS_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ZEROS_CATALOG_ENTRY];
