use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Polynomial roots as a column vector.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Descending-power polynomial coefficient vector.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "r = roots(c)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

pub const ROOTS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROOTS.INVALID_INPUT",
    identifier: Some("RunMat:roots:InvalidInput"),
    when: "Input cannot be interpreted as a numeric coefficient vector.",
    message: "roots: invalid input",
};
pub const ROOTS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROOTS.INTERNAL",
    identifier: Some("RunMat:roots:Internal"),
    when: "Runtime fails while building companion-matrix output or solving eigenvalues.",
    message: "roots: internal runtime failure",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [ROOTS_ERROR_INVALID_INPUT, ROOTS_ERROR_INTERNAL];
pub const ROOTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ROOTS_INTEGER_COEFFICIENTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "roots-integer-coefficients",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "roots accepts typed-integer polynomial coefficients as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RootsIntegerCoefficientsExtension"),
    };
pub const ROOTS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ROOTS_INTEGER_COEFFICIENTS_EXTENSION];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "c",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "RunMat admits typed integers only when every coefficient is exactly representable in binary64.",
}];
pub const ROOTS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "r = roots(integer_c)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The checked extension crosses into the host double companion-matrix eigenvalue algorithm. Resident input gathers through its owning provider.",
    }];
