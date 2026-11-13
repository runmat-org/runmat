use crate::*;
const GOBJECTS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Graphics-placeholder array; RunMat currently uses NaN numeric handle slots as its bounded representation.",
}];

const GOBJECTS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const GOBJECTS_INPUTS_DIMS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output dimensions, either as separate scalar sizes or a size vector.",
}];

const GOBJECTS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "h = gobjects()",
        inputs: &GOBJECTS_INPUTS_NONE,
        outputs: &GOBJECTS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = gobjects(sz...)",
        inputs: &GOBJECTS_INPUTS_DIMS,
        outputs: &GOBJECTS_OUTPUT,
    },
];

pub const GOBJECTS_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GOBJECTS.INVALID_SIZE",
    identifier: Some("RunMat:gobjects:InvalidSize"),
    when: "A size is nonnumeric, nonintegral, logical, resident, or has invalid vector geometry.",
    message: "gobjects: invalid size argument",
};
pub const GOBJECTS_ERROR_ALLOCATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GOBJECTS.ALLOCATION",
    identifier: Some("RunMat:gobjects:AllocationLimit"),
    when: "The requested shape exceeds addressable or allocatable placeholder storage.",
    message: "gobjects: requested placeholder array is too large",
};
const GOBJECTS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    GOBJECTS_ERROR_INVALID_SIZE,
    GOBJECTS_ERROR_ALLOCATION,
    GOBJECTS_ERROR_TOO_MANY_OUTPUTS,
];

const GOBJECTS_INTEGER_SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "size",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents all eight integer classes plus integral single and double sizes. Negative signed dimensions are treated as zero.",
    }];
pub const GOBJECTS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "H = gobjects(integer_sizes...)",
        inputs: &GOBJECTS_INTEGER_SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Scalar input creates a square shape; two or more scalar inputs create those dimensions; one row-vector input supplies dimensions exactly. RunMat retains a documented representation gap: NaN numeric slots stand in for GraphicsPlaceholder objects.",
    }];

pub const GOBJECTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GOBJECTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GOBJECTS_ERRORS,
};

pub const GOBJECTS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GOBJECTS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:gobjects:TooManyOutputs"),
    when: "More than 1 output arguments are requested.",
    message: "gobjects: too many output arguments",
};
