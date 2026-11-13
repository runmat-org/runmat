use crate::*;
pub const AXIS_STATE_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor { id: "axis-state-output", mode: BuiltinExtensionMode::RunMatOnly, description: "An assigned logical success value for axis setters is a RunMat extension; limit queries return MATLAB-compatible double vectors.", error_identifier: Some("RunMat:compatibility:AxisStateOutputExtension") };
pub const AXIS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [AXIS_STATE_OUTPUT_EXTENSION];
pub const AXIS_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "lim",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current x/y limits, with z limits included for a configured 3-D view.",
}];
pub const AXIS_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
pub const AXIS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
pub const AXIS_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existing axes handle whose limits are queried.",
}];
pub const AXIS_INPUTS_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Four-, six-, or eight-element x/y, x/y/z, or x/y/z/color limits vector.",
}];
pub const AXIS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mode token: 'equal'|'image'|'auto'|'tight'|'manual'|'ij'|'xy'|'on'|'off'.",
}];
pub const AXIS_INPUTS_VISIBILITY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "visibility",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axes visibility as on/off, logical true/false, or numeric 1/0.",
}];
pub const AXIS_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "lim = axis()",
        inputs: &AXIS_INPUTS_NONE,
        outputs: &AXIS_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "lim = axis(ax)",
        inputs: &AXIS_INPUTS_TARGET,
        outputs: &AXIS_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "axis([xmin xmax ymin ymax | ... cmin cmax])",
        inputs: &AXIS_INPUTS_LIMITS,
        outputs: &AXIS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "axis(mode)",
        inputs: &AXIS_INPUTS_MODE,
        outputs: &AXIS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "axis(visibility)",
        inputs: &AXIS_INPUTS_VISIBILITY,
        outputs: &AXIS_OUTPUTS_NONE,
    },
];
pub const AXIS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AXIS.INVALID_ARGUMENT",
    identifier: Some("RunMat:axis:InvalidArgument"),
    when: "Axis argument is unsupported, malformed, non-finite, or has invalid bounds ordering.",
    message: "axis: invalid argument",
};
pub const AXIS_ERRORS: [BuiltinErrorDescriptor; 2] =
    [AXIS_ERROR_INVALID_ARGUMENT, AXIS_ERROR_OUTPUTS];
pub const AXIS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AXIS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &AXIS_ERRORS,
};

pub const AXIS_LIMIT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "limits",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Four-, six-, and eight-element limit vectors accept every built-in integer class.",
    }];
pub const AXIS_VISIBILITY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "visibility",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Current numeric visibility syntax accepts scalar zero or one; logical false/true is the noninteger counterpart.",
    }];
pub const AXIS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "axis(integer_limits)",
        inputs: &AXIS_LIMIT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Authoritative integer storage is shape- and order-validated before one conversion into host f64 graphics limits; the separate query form returns double limits.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "axis(integer_visibility)",
        inputs: &AXIS_VISIBILITY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Only exact scalar zero and one are accepted, without provider dispatch or integer output.",
    },
];
pub const AXIS_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AXIS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:axis:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "axis: at most one output is supported",
};
