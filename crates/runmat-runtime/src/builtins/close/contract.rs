use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

const CLOSE_OUTPUT_RESULT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar double status: 1 when the requested close operation completes, 0 when it is refused.",
}];

const CLOSE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const CLOSE_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "target",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Figure target, tcp resource handle, option token, or target container.",
}];
const CLOSE_INPUTS_TARGETS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "targets",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One or more close targets.",
}];

const CLOSE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "result = close()",
        inputs: &CLOSE_INPUTS_NONE,
        outputs: &CLOSE_OUTPUT_RESULT,
    },
    BuiltinSignatureDescriptor {
        label: "result = close(target)",
        inputs: &CLOSE_INPUTS_TARGET,
        outputs: &CLOSE_OUTPUT_RESULT,
    },
    BuiltinSignatureDescriptor {
        label: "result = close(targets...)",
        inputs: &CLOSE_INPUTS_TARGETS,
        outputs: &CLOSE_OUTPUT_RESULT,
    },
];

pub(super) const CLOSE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:close:InvalidArgument"),
    when: "Close target values are invalid or unsupported.",
    message: "close: invalid argument",
};
const CLOSE_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.INVALID_HANDLE",
    identifier: Some("RunMat:close:InvalidHandle"),
    when: "A structure target is not a valid networking resource handle.",
    message: "close: invalid handle",
};
const CLOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.INTERNAL",
    identifier: None,
    when: "Internal networking, gather, or plotting close processing fails.",
    message: "close: internal error",
};
const CLOSE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CLOSE_ERROR_INVALID_ARGUMENT,
    CLOSE_ERROR_INVALID_HANDLE,
    CLOSE_ERROR_INTERNAL,
];

pub const CLOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLOSE_ERRORS,
};

pub(crate) const CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "close-integer-figure-number",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "close with a typed integer figure number is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CloseIntegerFigureNumberExtension"),
    };

pub(crate) const CLOSE_VARIADIC_TARGETS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "close-variadic-targets",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "close with separate variadic targets is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CloseVariadicTargetsExtension"),
    };

pub const CLOSE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION,
    CLOSE_VARIADIC_TARGETS_EXTENSION,
];

const CLOSE_INTEGER_TARGET_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fig",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode interprets a scalar or array of any built-in integer class as figure numbers. The public compatibility contract documents figure numbers but does not advertise typed integer classes.",
    }];

pub const CLOSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "status = close(integer_fig)",
        inputs: &CLOSE_INTEGER_TARGET_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Figure numbers are read from authoritative host integer storage and must be positive and representable as RunMat's u32 figure identifier. Resident numeric targets are rejected before networking/provider gather. Successful and no-op plotting closures return scalar double 1; callback-driven refusal remains unavailable until CloseRequestFcn is implemented.",
    }];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::close")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "close",
    op_kind: GpuOpKind::Custom("host-resource"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Figure and networking resources are host-owned; close rejects unsupported resident targets or gathers admitted values before host dispatch.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::close")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "close",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Resource closure executes eagerly on the host and does not participate in expression fusion.",
};
