use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod documentation;

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::inference::infer(request, entry, super::policy::MomentStatisticKind::Kurtosis)
}

const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real single or double observations; typed integers in RunMat mode.",
};
const INPUT_FLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flag",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "1 uses the biased moment; 0 applies sample bias correction.",
};
const INPUT_SELECTOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim_or_vecdim_or_all",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Operating dimension, dimension vector, or all elements.",
};
const INPUT_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"omitnan\""),
    description: "Include or omit NaN observations.",
};
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fourth standardized central moment, without subtracting three.",
}];
const X: [BuiltinParamDescriptor; 1] = [INPUT_X];
const X_FLAG: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_FLAG];
const X_SELECTOR: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_SELECTOR];
const X_FLAG_SELECTOR: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_FLAG, INPUT_SELECTOR];
const X_FLAG_SELECTOR_NAN: [BuiltinParamDescriptor; 4] =
    [INPUT_X, INPUT_FLAG, INPUT_SELECTOR, INPUT_NANFLAG];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "Y = kurtosis(X)",
        inputs: &X,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = kurtosis(X, flag)",
        inputs: &X_FLAG,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = kurtosis(X, dim_or_vecdim_or_all)",
        inputs: &X_SELECTOR,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = kurtosis(X, flag, dim_or_vecdim_or_all)",
        inputs: &X_FLAG_SELECTOR,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = kurtosis(X, flag, dim_or_vecdim_or_all, nanflag)",
        inputs: &X_FLAG_SELECTOR_NAN,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.kurtosis.INVALID_ARGUMENT",
        identifier: Some("RunMat:kurtosis:InvalidArgument"),
        when: "Data, flag, dimension, or missing-value option is invalid.",
        message: "kurtosis: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.kurtosis.INTERNAL",
        identifier: Some("RunMat:kurtosis:Internal"),
        when: "Tensor conversion or output allocation fails.",
        message: "kurtosis: internal error",
    },
];
pub const KURTOSIS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const KURTOSIS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "kurtosis-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "kurtosis with typed-integer input data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KurtosisIntegerDataExtension"),
    };
pub const KURTOSIS_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "kurtosis-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "kurtosis with a typed-integer flag or dimension control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KurtosisTypedIntegerControlExtension"),
    };
pub const KURTOSIS_GPU_MULTI_AXIS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "kurtosis-gpu-all-or-vecdim",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "kurtosis with all or vecdim on a GPU input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KurtosisGpuMultiAxisExtension"),
    };
const EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    KURTOSIS_INTEGER_DATA_EXTENSION,
    KURTOSIS_INTEGER_CONTROL_EXTENSION,
    KURTOSIS_GPU_MULTI_AXIS_EXTENSION,
];
const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer observations enter a double statistical calculation.",
}];
const INTEGER_CONTROL: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "flag_or_dimension",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Integer controls are independently mode-gated and decoded exactly.",
}];
pub const KURTOSIS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = kurtosis(integer_X, flag, dim_or_vecdim_or_all)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer data enters an explicit double output boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = kurtosis(X, integer_flag_or_dimension, ...) with floating X",
        inputs: &INTEGER_CONTROL,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer controls do not change the floating input's output class.",
    },
];

pub const KURTOSIS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "kurtosis" },
    category: "stats/summary",
    documentation: documentation::DOCUMENTATION,
    descriptor: &KURTOSIS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
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
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &EXTENSIONS,
    integer_capabilities: &KURTOSIS_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
