use crate::*;
const ADDPOINTS_INPUTS_2D: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "an",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Animated line handle.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
];

const ADDPOINTS_INPUTS_3D: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "an",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Animated line handle.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates for 3-D animated lines.",
    },
];

const ADDPOINTS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "addpoints(an, x, y)",
        inputs: &ADDPOINTS_INPUTS_2D,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "addpoints(an, x, y, z)",
        inputs: &ADDPOINTS_INPUTS_3D,
        outputs: &[],
    },
];

pub const ADDPOINTS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPOINTS.INVALID_ARGUMENT",
    identifier: Some("RunMat:addpoints:InvalidArgument"),
    when: "The handle is not an animated line, coordinates are invalid, or vector lengths differ.",
    message: "addpoints: invalid argument",
};

pub const ADDPOINTS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPOINTS.INTERNAL",
    identifier: Some("RunMat:addpoints:Internal"),
    when: "Internal plotting state update fails.",
    message: "addpoints: internal operation failed",
};

const ADDPOINTS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ADDPOINTS_ERROR_INVALID_ARGUMENT,
    ADDPOINTS_ERROR_INTERNAL,
    ADDPOINTS_ERROR_TOO_MANY_OUTPUTS,
];

pub const ADDPOINTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADDPOINTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ADDPOINTS_ERRORS,
};

const ADDPOINTS_2D_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "X coordinates independently accept every integer class or documented floating coordinate data.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Y coordinates independently accept every integer class or documented floating coordinate data.",
    },
];

const ADDPOINTS_3D_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "X coordinates independently accept every integer class or documented floating coordinate data.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Y coordinates independently accept every integer class or documented floating coordinate data.",
    },
    BuiltinIntegerInputCapability {
        name: "z",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Z coordinates independently accept every integer class or documented floating coordinate data.",
    },
];

pub const ADDPOINTS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "addpoints(an, integer_x, integer_y)",
        inputs: &ADDPOINTS_2D_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Coordinate vectors must have equal element counts. Host integer storage is read authoritatively before values enter the f64 graphics domain; GPU arrays gather because graphics state and rendering are client-side.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "addpoints(an, integer_x, integer_y, integer_z)",
        inputs: &ADDPOINTS_3D_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "All three coordinate vectors must have equal element counts. Host integer storage is read authoritatively before values enter the f64 graphics domain; GPU arrays gather because graphics state and rendering are client-side.",
    },
];

pub const ADDPOINTS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPOINTS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:addpoints:TooManyOutputs"),
    when: "More than 0 output arguments are requested.",
    message: "addpoints: too many output arguments",
};
