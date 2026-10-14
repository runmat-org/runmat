use crate::*;

pub const FIND_DIRECTION_ONLY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "find-direction-only",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "find(X,direction) is a RunMat convenience extension",
    error_identifier: Some("RunMat:compatibility:FindDirectionOnlyExtension"),
};
pub const FIND_INTEGER_SPARSE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "find-integer-sparse-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "find on typed-integer sparse storage is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FindIntegerSparseExtension"),
};
pub const FIND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [FIND_DIRECTION_ONLY_EXTENSION, FIND_INTEGER_SPARSE_EXTENSION];

const FIND_INTEGER_X_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes use authoritative storage for the exact nonzero predicate.",
}];
const FIND_INTEGER_K_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "K",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "K is an exact positive scalar count; zero, negative, and out-of-platform-range values reject.",
    }];
const FIND_INTEGER_SPARSE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB sparse values are single, double, or logical; typed-integer sparse storage is RunMat-only.",
    }];
pub const FIND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "k = find(integer_X,___)",
        inputs: &FIND_INTEGER_X_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Linear indices are exact binary64 indices; resident integer values gather exactly through their owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[row,col,v] = find(integer_X,___)",
        inputs: &FIND_INTEGER_X_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "row and col are exact doubles; v preserves the authoritative integer class and value.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "find(X,integer_K[,direction])",
        inputs: &FIND_INTEGER_K_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The positive count is converted exactly to a platform index before any input traversal.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[k|row,col,v] = find(integer_sparse_X,___)",
        inputs: &FIND_INTEGER_SPARSE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Strict compatibility gates this RunMat-only form before CSC traversal; v preserves integer storage.",
    },
];

const FIND_OUTPUT_LINEAR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "idx",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Linear indices of non-zero elements.",
}];

const FIND_OUTPUT_ROW_COL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "row",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row subscripts of non-zero elements.",
    },
    BuiltinParamDescriptor {
        name: "col",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column subscripts of non-zero elements.",
    },
];

const FIND_OUTPUT_ROW_COL_VAL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "row",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row subscripts of non-zero elements.",
    },
    BuiltinParamDescriptor {
        name: "col",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column subscripts of non-zero elements.",
    },
    BuiltinParamDescriptor {
        name: "v",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Values at the reported row/column locations.",
    },
];

const FIND_INPUTS_BASE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array to search.",
}];

const FIND_INPUTS_LIMIT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array to search.",
    },
    BuiltinParamDescriptor {
        name: "K",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum number of indices to return.",
    },
];

const FIND_INPUTS_DIRECTION_ONLY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array to search.",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "RunMat-only direction selector: first or last.",
    },
];

const FIND_INPUTS_LIMIT_DIR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array to search.",
    },
    BuiltinParamDescriptor {
        name: "K",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum number of indices to return.",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"first\""),
        description: "Direction selector: `\"first\"` or `\"last\"`.",
    },
];

const FIND_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "idx = find(X)",
        inputs: &FIND_INPUTS_BASE,
        outputs: &FIND_OUTPUT_LINEAR,
    },
    BuiltinSignatureDescriptor {
        label: "idx = find(X, K)",
        inputs: &FIND_INPUTS_LIMIT,
        outputs: &FIND_OUTPUT_LINEAR,
    },
    BuiltinSignatureDescriptor {
        label: "idx = find(X, direction)",
        inputs: &FIND_INPUTS_DIRECTION_ONLY,
        outputs: &FIND_OUTPUT_LINEAR,
    },
    BuiltinSignatureDescriptor {
        label: "idx = find(X, K, direction)",
        inputs: &FIND_INPUTS_LIMIT_DIR,
        outputs: &FIND_OUTPUT_LINEAR,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col] = find(X)",
        inputs: &FIND_INPUTS_BASE,
        outputs: &FIND_OUTPUT_ROW_COL,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col] = find(X, K)",
        inputs: &FIND_INPUTS_LIMIT,
        outputs: &FIND_OUTPUT_ROW_COL,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col] = find(X, direction)",
        inputs: &FIND_INPUTS_DIRECTION_ONLY,
        outputs: &FIND_OUTPUT_ROW_COL,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col] = find(X, K, direction)",
        inputs: &FIND_INPUTS_LIMIT_DIR,
        outputs: &FIND_OUTPUT_ROW_COL,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col, v] = find(X)",
        inputs: &FIND_INPUTS_BASE,
        outputs: &FIND_OUTPUT_ROW_COL_VAL,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col, v] = find(X, K)",
        inputs: &FIND_INPUTS_LIMIT,
        outputs: &FIND_OUTPUT_ROW_COL_VAL,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col, v] = find(X, direction)",
        inputs: &FIND_INPUTS_DIRECTION_ONLY,
        outputs: &FIND_OUTPUT_ROW_COL_VAL,
    },
    BuiltinSignatureDescriptor {
        label: "[row, col, v] = find(X, K, direction)",
        inputs: &FIND_INPUTS_LIMIT_DIR,
        outputs: &FIND_OUTPUT_ROW_COL_VAL,
    },
];

pub const FIND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIND.INVALID_INPUT",
    identifier: Some("RunMat:find:InvalidInput"),
    when: "Input type or option arguments are not valid for find.",
    message: "find: invalid input arguments",
};

pub const FIND_ERROR_PROVIDER_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIND.PROVIDER_OUTPUT",
    identifier: Some("RunMat:find:ProviderOutput"),
    when: "GPU provider does not return expected output buffers for requested nargout.",
    message: "find: provider output buffer mismatch",
};

pub const FIND_ERROR_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIND.PROVIDER",
    identifier: Some("RunMat:find:ProviderError"),
    when: "The owning provider fails to transfer or produce an admitted find result.",
    message: "find: provider execution failed",
};

pub const FIND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIND.INTERNAL",
    identifier: Some("RunMat:find:InternalError"),
    when: "Internal tensor conversion/materialization fails while building outputs.",
    message: "find: internal error",
};

const FIND_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FIND_ERROR_INVALID_INPUT,
    FIND_ERROR_PROVIDER_OUTPUT,
    FIND_ERROR_PROVIDER,
    FIND_ERROR_INTERNAL,
];

pub const FIND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIND_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FIND_ERRORS,
};
mod documentation;
use documentation::DOCUMENTATION;

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [runmat_types::EffectKind; 2] = [
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
];
pub const FIND_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "find" },
    category: "array/indexing",
    documentation: DOCUMENTATION,
    descriptor: &FIND_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Indexing(
            IndexingInferenceRule::Find,
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
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &FIND_EXTENSIONS,
    integer_capabilities: &FIND_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
