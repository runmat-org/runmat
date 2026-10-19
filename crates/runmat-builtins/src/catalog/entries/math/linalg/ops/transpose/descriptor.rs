use crate::*;

pub const TRANSPOSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value with first two dimensions swapped.",
}];

pub const TRANSPOSE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar/array value.",
}];

pub const TRANSPOSE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = transpose(A)",
    inputs: &TRANSPOSE_INPUTS,
    outputs: &TRANSPOSE_OUTPUT,
}];

pub const TRANSPOSE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRANSPOSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:transpose:InvalidArgument"),
    when: "Call does not provide exactly one input argument.",
    message: "transpose: invalid argument",
};

pub const TRANSPOSE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRANSPOSE.INVALID_INPUT",
    identifier: Some("RunMat:transpose:InvalidInput"),
    when: "Input type is unsupported for transpose or the input has a nonsingleton dimension above two.",
    message: "transpose: unsupported input type",
};

pub const TRANSPOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRANSPOSE.INTERNAL",
    identifier: Some("RunMat:transpose:Internal"),
    when: "Runtime cannot materialize transpose output.",
    message: "transpose: internal runtime failure",
};

pub const TRANSPOSE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    TRANSPOSE_ERROR_INVALID_ARGUMENT,
    TRANSPOSE_ERROR_INVALID_INPUT,
    TRANSPOSE_ERROR_INTERNAL,
];

pub const TRANSPOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRANSPOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRANSPOSE_ERRORS,
};

pub const TRANSPOSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented matrix inputs. Transpose reorders elements without arithmetic or conversion.",
    }];

pub const TRANSPOSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = transpose(integer_A)",
        inputs: &TRANSPOSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Dense, sparse, empty, host, and supported resident inputs preserve exact integer class, shape, and values; provider fallback gathers and restores through the input handle's owner.",
    }];
