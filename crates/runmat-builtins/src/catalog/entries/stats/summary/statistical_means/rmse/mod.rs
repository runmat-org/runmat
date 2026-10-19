use crate::*;
mod documentation;

fn infer(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::inference::infer(
        request,
        entry,
        super::StatisticalMeanKind::RootMeanSquaredError,
    )
}

const INPUT_F: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "F",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Forecast array, broadcast-compatible with A.",
};
const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Actual array, broadcast-compatible with F.",
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
    default: Some("\"includemissing\""),
    description: "Include or omit NaN/missing forecast, actual, or weight values.",
};
const INPUT_WEIGHT_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Weights",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Name of the optional nonnegative weighting scheme.",
};
const INPUT_W: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "W",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Nonnegative weights compatible with the selected operating dimension.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "E",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real root-mean-squared-error result.",
}];
const F_A: [BuiltinParamDescriptor; 2] = [INPUT_F, INPUT_A];
const F_A_SELECTOR: [BuiltinParamDescriptor; 3] = [INPUT_F, INPUT_A, INPUT_SELECTOR];
const F_A_NANFLAG: [BuiltinParamDescriptor; 3] = [INPUT_F, INPUT_A, INPUT_NANFLAG];
const F_A_SELECTOR_NANFLAG: [BuiltinParamDescriptor; 4] =
    [INPUT_F, INPUT_A, INPUT_SELECTOR, INPUT_NANFLAG];
const F_A_WEIGHTS: [BuiltinParamDescriptor; 4] = [INPUT_F, INPUT_A, INPUT_WEIGHT_NAME, INPUT_W];
const F_A_SELECTOR_WEIGHTS: [BuiltinParamDescriptor; 5] =
    [INPUT_F, INPUT_A, INPUT_SELECTOR, INPUT_WEIGHT_NAME, INPUT_W];
const F_A_SELECTOR_NANFLAG_WEIGHTS: [BuiltinParamDescriptor; 6] = [
    INPUT_F,
    INPUT_A,
    INPUT_SELECTOR,
    INPUT_NANFLAG,
    INPUT_WEIGHT_NAME,
    INPUT_W,
];
const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "E = rmse(F, A)",
        inputs: &F_A,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "E = rmse(F, A, dim_or_vecdim_or_all)",
        inputs: &F_A_SELECTOR,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "E = rmse(F, A, nanflag)",
        inputs: &F_A_NANFLAG,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "E = rmse(F, A, dim_or_vecdim_or_all, nanflag)",
        inputs: &F_A_SELECTOR_NANFLAG,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "E = rmse(F, A, \"Weights\", W)",
        inputs: &F_A_WEIGHTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "E = rmse(F, A, dim, \"Weights\", W)",
        inputs: &F_A_SELECTOR_WEIGHTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "E = rmse(F, A, dim, nanflag, \"Weights\", W)",
        inputs: &F_A_SELECTOR_NANFLAG_WEIGHTS,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.rmse.INVALID_ARGUMENT",
        identifier: Some("RunMat:rmse:InvalidArgument"),
        when: "Inputs, broadcast geometry, dimension, missing-value mode, or weights are invalid.",
        message: "rmse: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.rmse.INTERNAL",
        identifier: Some("RunMat:rmse:Internal"),
        when: "Storage, provider, or result admission fails.",
        message: "rmse: internal operation failed",
    },
];
pub const RMSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const RMSE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rmse-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "rmse with typed real or complex integer forecast/actual data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RmseIntegerDataExtension"),
};
pub const RMSE_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rmse-integer-weights",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rmse with typed-integer weights is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RmseIntegerWeightsExtension"),
};
const EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [RMSE_INTEGER_DATA_EXTENSION, RMSE_INTEGER_WEIGHTS_EXTENSION];
const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "F_or_A", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Real or paired-complex integer observations enter a floating difference-and-magnitude computation.",
}];
const INTEGER_WEIGHTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Weights",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed integer weights are validated exactly for nonnegativity.",
}];
pub const RMSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "E = rmse(integer_F_or_A, dim_or_vecdim_or_all, nanflag)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat integer-data extension returns double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "E = rmse(F, A, dim, \"Weights\", integer_W)",
        inputs: &INTEGER_WEIGHTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer weights do not independently determine result class.",
    },
];

pub const RMSE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "rmse" },
    category: "stats/summary",
    documentation: documentation::DOCUMENTATION,
    descriptor: &RMSE_DESCRIPTOR,
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
    integer_capabilities: &RMSE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
