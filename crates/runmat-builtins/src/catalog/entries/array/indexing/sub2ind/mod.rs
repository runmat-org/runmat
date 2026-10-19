use crate::*;

const SUB2IND_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "sz",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The size vector accepts every native integer class and is parsed directly as positive platform dimensions.",
    },
    BuiltinIntegerInputCapability {
        name: "I1...In",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every subscript role accepts all eight integer classes, including scalar expansion and full-width exact bounds checks.",
    },
];
pub const SUB2IND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "ind = sub2ind(integer_sz, integer_I1, integer_In...)",
        inputs: &SUB2IND_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Dimensions and subscripts are read from authoritative integer storage, combined with checked column-major index arithmetic, and returned as documented double indices. Resident integer inputs use exact gather fallback and return resident double output when the owner supports upload.",
    }];

const SUB2IND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ind",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Column-major linear indices corresponding to provided subscripts.",
}];

const SUB2IND_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size vector describing source array dimensions.",
    },
    BuiltinParamDescriptor {
        name: "I1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First-dimension subscript values.",
    },
    BuiltinParamDescriptor {
        name: "In",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Remaining per-dimension subscript arrays/scalars.",
    },
];

const SUB2IND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ind = sub2ind(sz, I1, In...)",
    inputs: &SUB2IND_INPUTS,
    outputs: &SUB2IND_OUTPUT,
}];

pub const SUB2IND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUB2IND.INVALID_INPUT",
    identifier: Some("RunMat:sub2ind:InvalidInput"),
    when: "Size vector, subscript count, or subscript types are invalid.",
    message: "sub2ind: invalid input arguments",
};

pub const SUB2IND_ERROR_INDEX_BOUNDS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUB2IND.INDEX_BOUNDS",
    identifier: Some("RunMat:sub2ind:IndexBounds"),
    when: "At least one subscript lies outside bounds for its dimension.",
    message: "sub2ind: subscript index exceeds dimension bounds",
};

pub const SUB2IND_ERROR_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUB2IND.PROVIDER",
    identifier: Some("RunMat:sub2ind:ProviderError"),
    when: "GPU provider sub2ind hook fails.",
    message: "sub2ind: provider execution failed",
};

pub const SUB2IND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUB2IND.INTERNAL",
    identifier: Some("RunMat:sub2ind:InternalError"),
    when: "Internal tensor conversion/output construction fails.",
    message: "sub2ind: internal error",
};

const SUB2IND_ERRORS: [BuiltinErrorDescriptor; 4] = [
    SUB2IND_ERROR_INVALID_INPUT,
    SUB2IND_ERROR_INDEX_BOUNDS,
    SUB2IND_ERROR_PROVIDER,
    SUB2IND_ERROR_INTERNAL,
];

pub const SUB2IND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUB2IND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUB2IND_ERRORS,
};
mod documentation;
use documentation::DOCUMENTATION;

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [runmat_types::EffectKind; 2] = [
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
];
pub const SUB2IND_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "sub2ind" },
    category: "array/indexing",
    documentation: DOCUMENTATION,
    descriptor: &SUB2IND_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Indexing(
            IndexingInferenceRule::Sub2Ind,
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
    integer_capabilities: &SUB2IND_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
