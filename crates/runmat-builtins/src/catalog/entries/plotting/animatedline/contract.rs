use crate::*;
const ANIMATEDLINE_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "an",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Animated line graphics handle.",
}];

const ANIMATEDLINE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional X/Y[/Z] vectors, axes target, and Name/Value properties.",
}];

const ANIMATEDLINE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "an = animatedline()",
        inputs: &[],
        outputs: &ANIMATEDLINE_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "an = animatedline(x, y)",
        inputs: &ANIMATEDLINE_INPUTS,
        outputs: &ANIMATEDLINE_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "an = animatedline(x, y, z)",
        inputs: &ANIMATEDLINE_INPUTS,
        outputs: &ANIMATEDLINE_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "an = animatedline(..., Name, Value)",
        inputs: &ANIMATEDLINE_INPUTS,
        outputs: &ANIMATEDLINE_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "an = animatedline(ax, ...)",
        inputs: &ANIMATEDLINE_INPUTS,
        outputs: &ANIMATEDLINE_OUTPUTS,
    },
];

const ANIMATEDLINE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANIMATEDLINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:animatedline:InvalidArgument"),
    when: "Coordinates, axes target, or Name/Value pairs are invalid.",
    message: "animatedline: invalid argument",
};

const ANIMATEDLINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANIMATEDLINE.INTERNAL",
    identifier: Some("RunMat:animatedline:Internal"),
    when: "Internal plotting state update fails.",
    message: "animatedline: internal operation failed",
};

const ANIMATEDLINE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ANIMATEDLINE_ERROR_INVALID_ARGUMENT,
    ANIMATEDLINE_ERROR_INTERNAL,
    ANIMATEDLINE_ERROR_TOO_MANY_OUTPUTS,
];

const ANIMATEDLINE_2D_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Starting X coordinates independently accept every integer class or other documented coordinate data.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Starting Y coordinates independently accept every integer class or other documented coordinate data.",
    },
];

const ANIMATEDLINE_3D_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Starting X coordinates independently accept every integer class or other documented coordinate data.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Starting Y coordinates independently accept every integer class or other documented coordinate data.",
    },
    BuiltinIntegerInputCapability {
        name: "z",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Starting Z coordinates independently accept every integer class or other documented coordinate data.",
    },
];

const ANIMATEDLINE_MAXIMUM_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "MaximumNumPoints",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The point limit accepts every integer class as a positive scalar; positive floating infinity selects an unbounded limit.",
    }];

pub const ANIMATEDLINE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "an = animatedline(integer_x, integer_y)",
        inputs: &ANIMATEDLINE_2D_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Coordinate vectors must have equal element counts. Host integer storage is read authoritatively before values enter the f64 graphics domain; GPU arrays gather because animated graphics state and rendering are client-side.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "an = animatedline(integer_x, integer_y, integer_z)",
        inputs: &ANIMATEDLINE_3D_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "All three coordinate vectors must have equal element counts. Host integer storage is read authoritatively before values enter the f64 graphics domain; GPU arrays gather because animated graphics state and rendering are client-side.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "an = animatedline(..., \"MaximumNumPoints\", integer_n)",
        inputs: &ANIMATEDLINE_MAXIMUM_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The limit is parsed from authoritative scalar integer storage, must be positive and representable as a host point count, and may gather from a resident scalar because the retained-point state is client-side.",
    },
];

pub const ANIMATEDLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ANIMATEDLINE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ANIMATEDLINE_ERRORS,
};

pub const ANIMATEDLINE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANIMATEDLINE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:animatedline:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "animatedline: too many output arguments",
};
