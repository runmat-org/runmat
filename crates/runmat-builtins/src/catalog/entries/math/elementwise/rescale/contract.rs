use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ALL_INTEGER_CLASSES,
};

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scaled output array.",
}];
const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric or logical input array.",
};
const INPUT_L: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "l",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Lower output bound.",
};
const INPUT_U: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "u",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Upper output bound.",
};
const INPUT_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "InputMin or InputMax option name.",
};
const INPUT_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Input range bound.",
};
const INPUTS_A: [BuiltinParamDescriptor; 1] = [INPUT_A];
const INPUTS_INTERVAL: [BuiltinParamDescriptor; 3] = [INPUT_A, INPUT_L, INPUT_U];
const INPUTS_OPTION: [BuiltinParamDescriptor; 3] = [INPUT_A, INPUT_NAME, INPUT_VALUE];
const INPUTS_INTERVAL_OPTION: [BuiltinParamDescriptor; 5] =
    [INPUT_A, INPUT_L, INPUT_U, INPUT_NAME, INPUT_VALUE];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "R = rescale(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = rescale(A, l, u)",
        inputs: &INPUTS_INTERVAL,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = rescale(A, Name, Value)",
        inputs: &INPUTS_OPTION,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = rescale(A, l, u, Name, Value)",
        inputs: &INPUTS_INTERVAL_OPTION,
        outputs: &OUTPUT,
    },
];

pub const RESCALE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESCALE.INVALID_ARGUMENT",
    identifier: Some("RunMat:rescale:InvalidArgument"),
    when: "Arguments, name-value pairs, interval bounds, or input range bounds are invalid.",
    message: "rescale: invalid argument",
};
pub const RESCALE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESCALE.INVALID_INPUT",
    identifier: Some("RunMat:rescale:InvalidInput"),
    when: "Input values cannot be converted to real numeric or logical arrays.",
    message: "rescale: invalid input",
};
pub const RESCALE_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESCALE.SIZE_MISMATCH",
    identifier: Some("RunMat:rescale:SizeMismatch"),
    when: "A bound array cannot be implicitly expanded with the input array.",
    message: "rescale: size mismatch",
};
pub const RESCALE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESCALE.INTERNAL",
    identifier: Some("RunMat:rescale:Internal"),
    when: "Tensor construction, allocation, or provider transfer fails internally.",
    message: "rescale: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    RESCALE_ERROR_INVALID_ARGUMENT,
    RESCALE_ERROR_INVALID_INPUT,
    RESCALE_ERROR_SIZE_MISMATCH,
    RESCALE_ERROR_INTERNAL,
];

pub const RESCALE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const DATA_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Integer samples are checked for exact binary64 representation at the explicit floating range-scaling boundary." }];
const BOUND_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "l/u/InputMin/InputMax",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Integer bounds are checked independently before the floating calculation.",
}];
pub const RESCALE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "R = rescale(integer_A, ...)",
        inputs: &DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes:
            "Integer data produces double output after a checked binary64 normalization boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "R = rescale(A, integer_l, integer_u, integer_InputMin, integer_InputMax)",
        inputs: &BOUND_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes:
            "The class of A controls single-versus-double output; bound classes do not change it.",
    },
];
