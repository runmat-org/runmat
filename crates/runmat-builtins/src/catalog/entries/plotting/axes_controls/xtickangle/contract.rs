use crate::*;

pub const XTICKANGLE_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "xtickangle-integer-axes-handle",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Allow a typed-integer alias for an encoded axes handle",
        error_identifier: Some("RunMat:compatibility:XtickangleIntegerAxesHandleExtension"),
    };
pub const XTICKANGLE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    XTICKANGLE_INTEGER_AXES_EXTENSION,
    AXES_CONTROL_SETTER_OUTPUT_EXTENSION,
];
const XTICKANGLE_INTEGER_ANGLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "angle",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public numeric-scalar rotation contract accepts integer degree values.",
    }];
const XTICKANGLE_INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for encoded axes handles are separately gated.",
    }];
pub const XTICKANGLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "xtickangle(integer_angle)",
        inputs: &XTICKANGLE_INTEGER_ANGLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The finite angle crosses the graphics rotation boundary; queries return double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "xtickangle(integer_ax, angle)",
        inputs: &XTICKANGLE_INTEGER_AXES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Strict mode rejects before graphics state access.",
    },
];

const OUTPUT_ANGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "angle",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis tick label rotation in degrees.",
}];
const NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const INPUTS_ANGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "angle",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X-axis tick label rotation in degrees.",
}];

const INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar target axes handle.",
}];

const INPUTS_AX_ANGLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle or array of axes handles.",
    },
    BuiltinParamDescriptor {
        name: "angle",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis tick label rotation in degrees.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "angle = xtickangle()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_ANGLE,
    },
    BuiltinSignatureDescriptor {
        label: "xtickangle(angle)",
        inputs: &INPUTS_ANGLE,
        outputs: &NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "angle = xtickangle(ax)",
        inputs: &INPUTS_AX,
        outputs: &OUTPUT_ANGLE,
    },
    BuiltinSignatureDescriptor {
        label: "xtickangle(ax, angle)",
        inputs: &INPUTS_AX_ANGLE,
        outputs: &NO_OUTPUTS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKANGLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:xtickangle:InvalidArgument"),
    when: "Argument count, angle value, or axes handle is invalid.",
    message: "xtickangle: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKANGLE.INTERNAL",
    identifier: Some("RunMat:xtickangle:Internal"),
    when: "Internal plotting state update fails.",
    message: "xtickangle: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ERROR_INVALID_ARGUMENT,
    ERROR_INTERNAL,
    XTICKANGLE_ERROR_TOO_MANY_OUTPUTS,
];

pub const XTICKANGLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const XTICKANGLE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKANGLE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:xtickangle:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "xtickangle: at most one output is supported",
};
