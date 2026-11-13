use crate::*;
pub const VIEW_INTEGER_ANGLES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "view-integer-angles",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "view with typed-integer angles or preset storage is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ViewIntegerAnglesExtension"),
};
pub const VIEW_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "view-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "view with a typed-integer numeric axes alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ViewIntegerAxesHandleExtension"),
};
pub const VIEW_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    VIEW_INTEGER_ANGLES_EXTENSION,
    VIEW_INTEGER_AXES_EXTENSION,
    VIEW_VECTOR_OUTPUT_EXTENSION,
];

const INTEGER_ANGLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "az, el, v, or dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public surface describes numbers and line-of-sight arrays without enumerating native integer storage; RunMat exposes typed integer angles and 2/3 presets behind one explicit extension gate.",
    }];
const INTEGER_AXES_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented target is an Axes object; typed-integer numeric aliases are a gated RunMat graphics representation extension.",
    }];
pub const VIEW_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "angles = view(integer_angles_or_preset)",
        inputs: &INTEGER_ANGLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host integer storage remains exact until each angle crosses the renderer's binary32 camera-state boundary; values that cannot be represented exactly reject before plot mutation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "angles = view(integer_ax, ...)",
        inputs: &INTEGER_AXES_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility gate runs before numeric handle decoding or axes-state access; resident integer handles are not gathered.",
    },
];

const VIEW_OUTPUT_ANGLES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "angles",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current or assigned view angles as [az el].",
}];

const VIEW_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const VIEW_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axes handle to query.",
}];

const VIEW_INPUTS_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "angles",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element [az el] vector or preset value 2/3.",
}];

const VIEW_INPUTS_AZ_EL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "az",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Azimuth in degrees.",
    },
    BuiltinParamDescriptor {
        name: "el",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Elevation in degrees.",
    },
];

const VIEW_INPUTS_AX_VECTOR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Axes handle to set.",
    },
    BuiltinParamDescriptor {
        name: "angles",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element [az el] vector or preset value 2/3.",
    },
];

const VIEW_INPUTS_AX_AZ_EL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Axes handle to set.",
    },
    BuiltinParamDescriptor {
        name: "az",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Azimuth in degrees.",
    },
    BuiltinParamDescriptor {
        name: "el",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Elevation in degrees.",
    },
];

const VIEW_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "[az, el] = view(...)",
        inputs: &VIEW_INPUTS_VECTOR,
        outputs: &VIEW_OUTPUT_PAIR,
    },
    BuiltinSignatureDescriptor {
        label: "angles = view()",
        inputs: &VIEW_INPUTS_NONE,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
    BuiltinSignatureDescriptor {
        label: "angles = view(ax)",
        inputs: &VIEW_INPUTS_AX,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
    BuiltinSignatureDescriptor {
        label: "angles = view([az el] | preset)",
        inputs: &VIEW_INPUTS_VECTOR,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
    BuiltinSignatureDescriptor {
        label: "angles = view(az, el)",
        inputs: &VIEW_INPUTS_AZ_EL,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
    BuiltinSignatureDescriptor {
        label: "angles = view(ax, [az el] | preset)",
        inputs: &VIEW_INPUTS_AX_VECTOR,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
    BuiltinSignatureDescriptor {
        label: "angles = view(ax, az, el)",
        inputs: &VIEW_INPUTS_AX_AZ_EL,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
];

pub const VIEW_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VIEW.INVALID_ARGUMENT",
    identifier: Some("RunMat:view:InvalidArgument"),
    when: "Arguments are malformed, wrong arity, wrong shape, or unsupported preset.",
    message: "view: invalid argument",
};

pub const VIEW_ERROR_INVALID_AXES: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VIEW.INVALID_AXES",
    identifier: Some("RunMat:view:InvalidAxes"),
    when: "Axes target is invalid or no longer exists.",
    message: "view: invalid axes target",
};

pub const VIEW_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VIEW.INTERNAL",
    identifier: Some("RunMat:view:Internal"),
    when: "Internal plotting state operation fails.",
    message: "view: internal operation failed",
};

pub const VIEW_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VIEW.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:view:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "view: too many output arguments",
};

const VIEW_ERRORS: [BuiltinErrorDescriptor; 4] = [
    VIEW_ERROR_INVALID_ARGUMENT,
    VIEW_ERROR_INVALID_AXES,
    VIEW_ERROR_INTERNAL,
    VIEW_ERROR_TOO_MANY_OUTPUTS,
];

pub const VIEW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VIEW_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VIEW_ERRORS,
};

pub const VIEW_VECTOR_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "view-vector-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "A single output returns the RunMat [az el] vector, not a MATLAB view matrix",
    error_identifier: Some("RunMat:compatibility:ViewVectorOutputExtension"),
};
const VIEW_OUTPUT_PAIR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "az",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Azimuth angle in degrees.",
    },
    BuiltinParamDescriptor {
        name: "el",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Elevation angle in degrees.",
    },
];
