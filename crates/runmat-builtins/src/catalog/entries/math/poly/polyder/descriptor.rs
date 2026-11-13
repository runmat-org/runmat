use crate::*;

const OUTPUT_D: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Derivative coefficient vector.",
}];
const OUTPUT_QUOTIENT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "num",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Quotient-rule numerator coefficients.",
    },
    BuiltinParamDescriptor {
        name: "den",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Quotient-rule denominator coefficients.",
    },
];
const INPUT_SINGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Polynomial coefficient vector.",
}];
const INPUT_BINARY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First polynomial coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second polynomial coefficient vector.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "d = polyder(p)",
        inputs: &INPUT_SINGLE,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = polyder(a, b)",
        inputs: &INPUT_BINARY,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "[num, den] = polyder(u, v)",
        inputs: &INPUT_BINARY,
        outputs: &OUTPUT_QUOTIENT,
    },
];

pub const POLYDER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYDER.INVALID_ARGUMENT",
    identifier: Some("RunMat:polyder:InvalidArgument"),
    when: "Input arity/output mode combination is invalid.",
    message: "polyder: invalid argument",
};
pub const POLYDER_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYDER.INVALID_INPUT",
    identifier: Some("RunMat:polyder:InvalidInput"),
    when: "Inputs cannot be interpreted as numeric coefficient vectors.",
    message: "polyder: invalid input",
};
pub const POLYDER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYDER.INTERNAL",
    identifier: Some("RunMat:polyder:Internal"),
    when: "Runtime fails while building derivative outputs or provider fallback paths.",
    message: "polyder: internal runtime failure",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    POLYDER_ERROR_INVALID_ARGUMENT,
    POLYDER_ERROR_INVALID_INPUT,
    POLYDER_ERROR_INTERNAL,
];
pub const POLYDER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const POLYDER_INTEGER_COEFFICIENTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "polyder-integer-coefficients",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "polyder accepts typed-integer coefficient vectors as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PolyderIntegerCoefficientsExtension"),
    };
pub const POLYDER_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [POLYDER_INTEGER_COEFFICIENTS_EXTENSION];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "p, a, or b",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integers cross only after exact-binary64 validation.",
}];
pub const POLYDER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "d = polyder(integer_p) or d = polyder(integer_a,b) or [q,d] = polyder(integer_a,b)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Every integer coefficient operand is checked before provider dispatch; derivative, product, and quotient arithmetic use floating precision.",
    }];
