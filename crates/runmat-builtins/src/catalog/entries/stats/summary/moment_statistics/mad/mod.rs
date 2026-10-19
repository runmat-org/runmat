use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod documentation;

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::inference::infer(
        request,
        entry,
        super::policy::MomentStatisticKind::MeanAbsoluteDeviation,
    )
}

const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real single or double data; typed integers in RunMat mode.",
};
const INPUT_FLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flag",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "0 computes mean deviation; 1 computes median deviation.",
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
    description: "Mean or median absolute deviation.",
}];
const X: [BuiltinParamDescriptor; 1] = [INPUT_X];
const X_FLAG: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_FLAG];
const X_SELECTOR: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_SELECTOR];
const X_FLAG_SELECTOR: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_FLAG, INPUT_SELECTOR];
const X_FLAG_SELECTOR_NAN: [BuiltinParamDescriptor; 4] =
    [INPUT_X, INPUT_FLAG, INPUT_SELECTOR, INPUT_NANFLAG];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "Y = mad(X)",
        inputs: &X,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = mad(X, flag)",
        inputs: &X_FLAG,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = mad(X, dim_or_vecdim_or_all)",
        inputs: &X_SELECTOR,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = mad(X, flag, dim_or_vecdim_or_all)",
        inputs: &X_FLAG_SELECTOR,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = mad(X, flag, dim_or_vecdim_or_all, nanflag)",
        inputs: &X_FLAG_SELECTOR_NAN,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.mad.INVALID_ARGUMENT",
        identifier: Some("RunMat:mad:InvalidArgument"),
        when: "Data, flag, dimension, or missing-value option is invalid.",
        message: "mad: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.mad.INTERNAL",
        identifier: Some("RunMat:mad:Internal"),
        when: "Tensor conversion or output allocation fails.",
        message: "mad: internal error",
    },
];
pub const MAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const MAD_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mad-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mad with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MadIntegerDataExtension"),
};
pub const MAD_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "mad-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "mad with a typed-integer flag or dimension control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:MadTypedIntegerControlExtension"),
    };
const EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    MAD_INTEGER_DATA_EXTENSION,
    MAD_TYPED_INTEGER_CONTROL_EXTENSION,
];
const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Real integer observations enter the floating deviation calculation and produce double.",
}];
const INTEGER_CONTROL: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "flag_or_dimension",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Exact integer controls are independently mode-gated and range-checked.",
}];
pub const MAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = mad(integer_X, flag, dim_or_vecdim_or_all)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer data enters an explicit double computation/output boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = mad(X, integer_flag_or_dimension, ...) with floating X",
        inputs: &INTEGER_CONTROL,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer controls do not change the floating input's result precision.",
    },
];

pub const MAD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "mad" },
    category: "stats/summary",
    documentation: documentation::DOCUMENTATION,
    descriptor: &MAD_DESCRIPTOR,
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
    integer_capabilities: &MAD_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
