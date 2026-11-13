use crate::*;

pub const PAGETRANSPOSE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public input-class table lists all eight integer classes; real and paired complex-integer storage is transposed without arithmetic.",
    }];

pub const PAGETRANSPOSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = pagetranspose(integer_X)",
        inputs: &PAGETRANSPOSE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The first two dimensions are permuted while exact native real or paired-complex integer payloads remain in their original class; providers may keep supported layouts resident and otherwise use transparent gather/restore.",
    }];

pub const PAGETRANSPOSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value with the first two dimensions swapped on every page.",
}];

pub const PAGETRANSPOSE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar or array value.",
}];

pub const PAGETRANSPOSE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "Y = pagetranspose(X)",
        inputs: &PAGETRANSPOSE_INPUTS,
        outputs: &PAGETRANSPOSE_OUTPUT,
    }];

pub const PAGETRANSPOSE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGETRANSPOSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:pagetranspose:InvalidArgument"),
    when: "Call does not provide exactly one input argument.",
    message: "pagetranspose: expected exactly one input argument",
};

pub const PAGETRANSPOSE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGETRANSPOSE.INVALID_INPUT",
    identifier: Some("RunMat:pagetranspose:InvalidInput"),
    when: "Input type is unsupported for page-wise transpose.",
    message: "pagetranspose: unsupported input type",
};

pub const PAGETRANSPOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGETRANSPOSE.INTERNAL",
    identifier: Some("RunMat:pagetranspose:Internal"),
    when: "Runtime cannot materialize the page-wise transpose output.",
    message: "pagetranspose: internal runtime failure",
};

pub const PAGETRANSPOSE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    PAGETRANSPOSE_ERROR_INVALID_ARGUMENT,
    PAGETRANSPOSE_ERROR_INVALID_INPUT,
    PAGETRANSPOSE_ERROR_INTERNAL,
];

pub const PAGETRANSPOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PAGETRANSPOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PAGETRANSPOSE_ERRORS,
};
