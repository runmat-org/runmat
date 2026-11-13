use crate::*;

pub const KRON_ND_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "kron-nd-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "kron with an N-D operand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:KronNdInputExtension"),
};
pub const KRON_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [KRON_ND_EXTENSION];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Native integer storage remains authoritative; paired integer operands must use the same class and complex integer arithmetic is rejected." },
    BuiltinIntegerInputCapability { name: "B", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "A scalar double may scale an integer operand; other mixed numeric arrays are rejected by the settled integer arithmetic rule." },
];
pub const KRON_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "K = kron(A, B)", inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer products preserve the integer class and saturate. Resident fallback gathers authoritative storage without consuming source handles and restores eligible output to the owning provider." }];

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Kronecker product of A and B.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left numeric, logical, character, or sparse input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right numeric, logical, character, or sparse input.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = kron(A, B)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];
pub const KRON_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KRON.TOO_MANY_INPUTS",
    identifier: Some("RunMat:kron:TooManyInputs"),
    when: "Extra arguments were supplied after A and B.",
    message: "kron: too many input arguments",
};
pub const KRON_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KRON.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:kron:UnsupportedInput"),
    when: "Input values are not supported numeric, logical, character, or sparse arrays.",
    message: "kron: unsupported input type",
};
pub const KRON_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KRON.INTERNAL",
    identifier: Some("RunMat:kron:Internal"),
    when: "Internal conversion, allocation, or provider path fails.",
    message: "kron: internal operation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    KRON_ERROR_TOO_MANY_INPUTS,
    KRON_ERROR_UNSUPPORTED_INPUT,
    KRON_ERROR_INTERNAL,
];
pub const KRON_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
