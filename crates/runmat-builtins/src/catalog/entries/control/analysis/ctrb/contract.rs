use crate::*;

const A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonempty square real or complex single/double state matrix.",
};
const B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Same-class input matrix with one or more columns and size(A,1) rows.",
};
const SYS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One ordinary state-space ss model supplying A and B.",
};
const CO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Co",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Controllability matrix [B, A*B, ..., A^(n-1)*B].",
};
const MATRIX_INPUTS: [BuiltinParamDescriptor; 2] = [A, B];
const MODEL_INPUTS: [BuiltinParamDescriptor; 1] = [SYS];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [CO];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Co = ctrb(A,B)",
        inputs: &MATRIX_INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Co = ctrb(sys)",
        inputs: &MODEL_INPUTS,
        outputs: &OUTPUTS,
    },
];

pub const CTRB_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CTRB.INVALID_ARGUMENT",
    identifier: Some("RunMat:ctrb:InvalidArgument"),
    when: "The number or kind of arguments, or the requested output count, is invalid.",
    message: "ctrb: invalid argument",
};
pub const CTRB_ERROR_DIMENSIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CTRB.DIMENSIONS",
    identifier: Some("RunMat:ctrb:InvalidDimensions"),
    when: "A is not nonempty square, B has incompatible rows or no columns, or the output extent overflows.",
    message: "ctrb: invalid matrix dimensions",
};
pub const CTRB_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CTRB.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:ctrb:UnsupportedInput"),
    when: "The operands do not have matching finite real/complex single/double dense storage, or a product overflows that class.",
    message: "ctrb: unsupported matrix input",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    CTRB_ERROR_INVALID_ARGUMENT,
    CTRB_ERROR_DIMENSIONS,
    CTRB_ERROR_UNSUPPORTED_INPUT,
];

pub const CTRB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_A: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "A",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Direct integer storage is not converted to a floating controllability operator.",
};
const INTEGER_B: BuiltinIntegerInputCapability = BuiltinIntegerInputCapability {
    name: "B",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Direct integer storage is not converted to the class of A.",
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [INTEGER_A, INTEGER_B];
pub const CTRB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Co = ctrb(integer_A, integer_B)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Direct typed integer matrices are rejected. Existing ss model properties are read according to their stored native class.",
    }];
