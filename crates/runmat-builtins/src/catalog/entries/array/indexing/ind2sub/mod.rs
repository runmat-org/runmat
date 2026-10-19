use crate::*;

const IND2SUB_SIZE_CLASSES: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "size vector",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The size vector accepts single, double, and every integer class, must contain at least two positive integral elements, and is read exactly from authoritative storage.",
    },
    BuiltinIntegerInputCapability {
        name: "linear indices",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Linear indices accept single, double, and every integer class; outputs are double and preserve the index array shape.",
    },
];

pub const IND2SUB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[I1,...,In] = ind2sub(integer_size, integer_indices)",
        inputs: &IND2SUB_SIZE_CLASSES,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Requested output count controls trailing-dimension collapse; indices beyond prod(sz) expand the final effective dimension. Resident inputs always produce double output and return to the exact source owner only when that owner can physically preserve binary64.",
    }];

const IND2SUB_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I1,...,IN",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Requested double subscript arrays; the final requested output folds trailing dimensions.",
}];

const IND2SUB_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size vector describing source array dimensions.",
    },
    BuiltinParamDescriptor {
        name: "ind",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Linear indices to convert into per-dimension subscripts.",
    },
];

const IND2SUB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[I1,...,IN] = ind2sub(sz, ind)",
    inputs: &IND2SUB_INPUTS,
    outputs: &IND2SUB_OUTPUTS,
}];

pub const IND2SUB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2SUB.INVALID_INPUT",
    identifier: Some("RunMat:ind2sub:InvalidInput"),
    when: "Size vector or linear index inputs are malformed or unsupported.",
    message: "ind2sub: invalid input arguments",
};

pub const IND2SUB_ERROR_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2SUB.PROVIDER",
    identifier: Some("RunMat:ind2sub:ProviderError"),
    when: "Provider-side ind2sub execution fails or returns malformed outputs.",
    message: "ind2sub: provider execution failed",
};

pub const IND2SUB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2SUB.INTERNAL",
    identifier: Some("RunMat:ind2sub:InternalError"),
    when: "Internal tensor/materialization logic fails while building outputs.",
    message: "ind2sub: internal error",
};

const IND2SUB_ERRORS: [BuiltinErrorDescriptor; 3] = [
    IND2SUB_ERROR_INVALID_INPUT,
    IND2SUB_ERROR_PROVIDER,
    IND2SUB_ERROR_INTERNAL,
];

pub const IND2SUB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IND2SUB_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IND2SUB_ERRORS,
};
mod documentation;
use documentation::DOCUMENTATION;

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [runmat_types::EffectKind; 2] = [
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
];
pub const IND2SUB_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "ind2sub" },
    category: "array/indexing",
    documentation: DOCUMENTATION,
    descriptor: &IND2SUB_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Indexing(
            IndexingInferenceRule::Ind2Sub,
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
    extensions: &[],
    integer_capabilities: &IND2SUB_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
