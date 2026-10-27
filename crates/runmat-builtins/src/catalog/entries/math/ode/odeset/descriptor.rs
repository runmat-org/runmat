use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar ODE options structure; unspecified fields are empty.",
}];
const PAIRS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name_value_pairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Complete supported option name/value pairs.",
}];
const OLD_PAIRS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "oldopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing scalar options structure or numeric [].",
    },
    BuiltinParamDescriptor {
        name: "name_value_pairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Complete supported option name/value pairs.",
    },
];
const MERGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "oldopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing scalar options structure or numeric [].",
    },
    BuiltinParamDescriptor {
        name: "newopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "New scalar options structure; nonempty fields replace old values.",
    },
];
pub const ODESET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODESET.INVALID_ARGUMENT",
    identifier: Some("RunMat:odeset:InvalidArgument"),
    when: "Output count or options/name-value argument grammar is invalid.",
    message: "odeset: invalid argument",
};
pub const ODESET_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODESET.INVALID_INPUT",
    identifier: Some("RunMat:odeset:InvalidInput"),
    when: "An option name is not a text scalar.",
    message: "odeset: invalid option name",
};
pub const ODESET_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODESET.UNKNOWN_OPTION",
    identifier: Some("RunMat:odeset:UnknownOption"),
    when: "An option is unknown, unsupported, or an ambiguous abbreviation.",
    message: "odeset: unsupported or ambiguous option",
};
pub const ODESET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "options = odeset()",
            inputs: &[],
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "options = odeset(name, value, ...)",
            inputs: &PAIRS,
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "options = odeset(oldopts, name, value, ...)",
            inputs: &OLD_PAIRS,
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "options = odeset(oldopts, newopts)",
            inputs: &MERGE,
            outputs: &OUTPUT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        ODESET_ERROR_INVALID_ARGUMENT,
        ODESET_ERROR_INVALID_INPUT,
        ODESET_ERROR_UNKNOWN_OPTION,
    ],
};
pub const ODESET_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "odeset-integer-option", mode: BuiltinExtensionMode::RunMatOnly,
    description: "Preserving native integer option payloads is a RunMat extension; solver admission validates their use.",
    error_identifier: Some("RunMat:compatibility:OdesetIntegerOptionExtension"),
};
pub const ODESET_RESIDENT_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "odeset-resident-option", mode: BuiltinExtensionMode::RunMatOnly,
    description: "Preserving explicit gpuArray option payloads is a RunMat extension; the constructor does not gather them.",
    error_identifier: Some("RunMat:compatibility:OdesetResidentOptionExtension"),
};
pub const ODESET_MAX_STEPS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "odeset-max-steps",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "MaxSteps is a RunMat solver resource control, not a MATLAB ODE option.",
    error_identifier: Some("RunMat:compatibility:OdesetMaxStepsExtension"),
};
pub const ODESET_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ODESET_INTEGER_OPTION_EXTENSION,
    ODESET_RESIDENT_OPTION_EXTENSION,
    ODESET_MAX_STEPS_EXTENSION,
];
pub const ODESET_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "options = odeset(___, name, integer_value, ___)",
        inputs: &[BuiltinIntegerInputCapability {
            name: "option value", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "No numeric conversion; the original native-class value is retained for solver admission.",
        }],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Option storage preserves exact values and classes; ODE solver rules independently validate each numeric use.",
    },
];
