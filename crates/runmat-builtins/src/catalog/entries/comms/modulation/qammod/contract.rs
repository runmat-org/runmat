use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex rectangular-QAM baseband samples.",
}];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Integer symbols or grouped bit input.",
};
const PARAM_M: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Power-of-two modulation order greater than one.",
};
const BASE_INPUTS: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_M];

const EXTENDED_INPUTS: [BuiltinParamDescriptor; 3] = [
    PARAM_X,
    PARAM_M,
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional symbol order and name-value options.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = qammod(X, M)",
        inputs: &BASE_INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Y = qammod(X, M, symorder, Name, Value)",
        inputs: &EXTENDED_INPUTS,
        outputs: &OUTPUTS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QAMMOD.INVALID_ARGUMENT",
    identifier: None,
    when: "Symbols, modulation order, mapping, or options are invalid.",
    message: "qammod: invalid argument",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_ARGUMENT];

pub const QAMMOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::INTEGER_CLASSES_THROUGH_16_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer symbol/bit input accepts int8, int16, uint8, and uint16 on host and GPU; wider integer symbol classes reject.",
    },
    BuiltinIntegerInputCapability {
        name: "M",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Modulation order accepts every exact integer class or an integer-valued floating scalar.",
    },
    BuiltinIntegerInputCapability {
        name: "symorder",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Custom symbol-order vectors accept all eight exact integer classes.",
    },
];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = qammod(X, M, symorder, Name, Value)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Built-in integer symbol input defaults to complex double output; OutputDataType can select single, and provider/fallback paths preserve the selected resident precision.",
    }];
