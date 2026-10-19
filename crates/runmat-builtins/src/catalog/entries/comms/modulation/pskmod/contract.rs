use crate::*;

pub const PSKMOD_INTEGER_ORDER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pskmod-integer-modulation-order",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pskmod with typed-integer modulation order M is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PskmodIntegerModulationOrderExtension"),
};

pub const PSKMOD_INTEGER_PHASE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pskmod-integer-phase-offset",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pskmod with a typed-integer phase offset is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PskmodIntegerPhaseOffsetExtension"),
};

pub const PSKMOD_INTEGER_CUSTOM_ORDER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pskmod-integer-custom-order",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pskmod with a typed-integer custom symbol order is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PskmodIntegerCustomOrderExtension"),
    };

pub const PSKMOD_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    PSKMOD_INTEGER_ORDER_EXTENSION,
    PSKMOD_INTEGER_PHASE_EXTENSION,
    PSKMOD_INTEGER_CUSTOM_ORDER_EXTENSION,
];

const PSKMOD_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex M-PSK baseband samples.",
}];

const PSKMOD_BASE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer symbols or grouped bit input.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Modulation order greater than one.",
    },
];

const PSKMOD_EXTENDED_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer symbols or grouped bit input.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Modulation order greater than one.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional phase offset, symbol order, and name-value options.",
    },
];

const PSKMOD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = pskmod(X, M)",
        inputs: &PSKMOD_BASE_INPUTS,
        outputs: &PSKMOD_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Y = pskmod(X, M, phaseoffset, symorder, Name, Value)",
        inputs: &PSKMOD_EXTENDED_INPUTS,
        outputs: &PSKMOD_OUTPUTS,
    },
];

const PSKMOD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PSKMOD.INVALID_ARGUMENT",
    identifier: None,
    when: "Symbols, modulation order, phase, mapping, or options are invalid.",
    message: "pskmod: invalid argument",
};

const PSKMOD_ERRORS: [BuiltinErrorDescriptor; 1] = [PSKMOD_ERROR_INVALID_ARGUMENT];

pub const PSKMOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PSKMOD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PSKMOD_ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer symbol/bit input accepts signed and unsigned classes through 32 bits on host and GPU; int64/uint64 symbols reject.",
    },
    BuiltinIntegerInputCapability {
        name: "M",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer modulation order is gated by pskmod-integer-modulation-order; documented floating scalar order remains ordinary.",
    },
    BuiltinIntegerInputCapability {
        name: "phaseoffset",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer phase offset is gated by pskmod-integer-phase-offset.",
    },
    BuiltinIntegerInputCapability {
        name: "symorder",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer custom symbol order is gated by pskmod-integer-custom-order.",
    },
];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = pskmod(X, M, phaseoffset, symorder, Name, Value)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Default output is complex double for every integer input; OutputDataType can select single, and provider/fallback paths preserve selected resident precision.",
    }];
