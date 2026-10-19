use crate::*;
mod documentation;

fn infer(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::inference::infer(request, entry, super::StatisticalMeanKind::RootMeanSquare)
}

const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real or complex numeric, logical, or character data array.",
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
    description: "Include or omit NaN or missing observations.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real root-mean-square result.",
}];
const X: [BuiltinParamDescriptor; 1] = [INPUT_X];
const X_SELECTOR: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_SELECTOR];
const X_NANFLAG: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_NANFLAG];
const X_SELECTOR_NANFLAG: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_SELECTOR, INPUT_NANFLAG];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Y = rms(X)",
        inputs: &X,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = rms(X, dim_or_vecdim_or_all)",
        inputs: &X_SELECTOR,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = rms(X, nanflag)",
        inputs: &X_NANFLAG,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = rms(X, dim_or_vecdim_or_all, nanflag)",
        inputs: &X_SELECTOR_NANFLAG,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.rms.INVALID_ARGUMENT",
        identifier: Some("RunMat:rms:InvalidArgument"),
        when: "Input, dimension, or missing-value control is invalid.",
        message: "rms: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.rms.INTERNAL",
        identifier: Some("RunMat:rms:Internal"),
        when: "Storage, provider, or result admission fails.",
        message: "rms: internal operation failed",
    },
];
pub const RMS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const RMS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rms-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rms with typed real or complex integer data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RmsIntegerDataExtension"),
};
const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [RMS_INTEGER_DATA_EXTENSION];
const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer magnitudes enter a floating reduction and return double.",
}];
pub const RMS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = rms(integer_X, dim_or_vecdim_or_all, nanflag)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat accepts real and paired-complex integer storage and returns double RMS.",
    }];

pub const RMS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "rms" },
    category: "math/reduction",
    documentation: documentation::DOCUMENTATION,
    descriptor: &RMS_DESCRIPTOR,
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
    integer_capabilities: &RMS_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
