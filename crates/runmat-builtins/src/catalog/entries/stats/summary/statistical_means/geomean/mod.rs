use crate::*;

mod documentation;

fn infer(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::inference::infer(request, entry, super::StatisticalMeanKind::Geometric)
}

const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonnegative single- or double-precision real data array; integer data is a RunMat extension.",
};
const INPUT_SELECTOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim_or_vecdim_or_all",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Operating dimension, dimension vector, or all.",
};
const INPUT_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"includenan\""),
    description: "Include or omit NaN observations.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Geometric mean for each selected slice.",
}];
const INPUTS_X: [BuiltinParamDescriptor; 1] = [INPUT_X];
const INPUTS_X_SELECTOR: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_SELECTOR];
const INPUTS_X_NANFLAG: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_NANFLAG];
const INPUTS_X_SELECTOR_NANFLAG: [BuiltinParamDescriptor; 3] =
    [INPUT_X, INPUT_SELECTOR, INPUT_NANFLAG];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "M = geomean(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = geomean(X, dim_or_vecdim_or_all)",
        inputs: &INPUTS_X_SELECTOR,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = geomean(X, nanflag)",
        inputs: &INPUTS_X_NANFLAG,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = geomean(X, dim_or_vecdim_or_all, nanflag)",
        inputs: &INPUTS_X_SELECTOR_NANFLAG,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.geomean.INVALID_ARGUMENT",
        identifier: Some("RunMat:geomean:InvalidArgument"),
        when: "Input, dimension, or missing-value control is invalid.",
        message: "geomean: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.geomean.INTERNAL",
        identifier: Some("RunMat:geomean:Internal"),
        when: "Storage, provider, or result admission fails.",
        message: "geomean: internal operation failed",
    },
];
pub const GEOMEAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const GEOMEAN_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "geomean-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "geomean with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GeomeanIntegerDataExtension"),
};
pub const GEOMEAN_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "geomean-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "geomean with a typed-integer dimension control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GeomeanTypedIntegerControlExtension"),
    };
const EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    GEOMEAN_INTEGER_DATA_EXTENSION,
    GEOMEAN_INTEGER_CONTROL_EXTENSION,
];
const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Nonnegative integer data enters a floating statistical computation at the reduction boundary.",
}];
const INTEGER_CONTROL: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "dim_or_vecdim",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Exact positive integer axes are decoded without a binary64 detour.",
}];
pub const GEOMEAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "M = geomean(integer_X, dim_or_vecdim_or_all, nanflag)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat integer-data extension returns double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "M = geomean(X, integer_dim_or_vecdim, nanflag)",
        inputs: &INTEGER_CONTROL,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The control is structural and does not choose numerical result precision.",
    },
];

pub const GEOMEAN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "geomean" },
    category: "stats/summary",
    documentation: documentation::DOCUMENTATION,
    descriptor: &GEOMEAN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
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
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &EXTENSIONS,
    integer_capabilities: &GEOMEAN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
