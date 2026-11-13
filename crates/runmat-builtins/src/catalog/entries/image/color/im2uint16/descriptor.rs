use crate::*;

const IM2UINT16_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "J",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Converted image data in uint16 class.",
}];

const IM2UINT16_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Image data to convert.",
}];

const IM2UINT16_INDEXED_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image data to convert.",
    },
    BuiltinParamDescriptor {
        name: "indexed",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "The literal \"indexed\", selecting indexed-image offset semantics.",
    },
];

const IM2UINT16_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "J = im2uint16(I)",
        inputs: &IM2UINT16_INPUTS,
        outputs: &IM2UINT16_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "J = im2uint16(I, \"indexed\")",
        inputs: &IM2UINT16_INDEXED_INPUTS,
        outputs: &IM2UINT16_OUTPUT,
    },
];

pub const IM2UINT16_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT16.TOO_MANY_INPUTS",
    identifier: Some("RunMat:im2uint16:TooManyInputs"),
    when: "More than two input arguments are supplied.",
    message: "im2uint16: too many input arguments",
};

pub const IM2UINT16_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT16.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:im2uint16:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "im2uint16: too many output arguments",
};

pub const IM2UINT16_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT16.INVALID_INPUT",
    identifier: Some("RunMat:im2uint16:InvalidInput"),
    when: "Input cannot be gathered or interpreted for image conversion.",
    message: "im2uint16: invalid input",
};

pub const IM2UINT16_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT16.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:im2uint16:UnsupportedInputType"),
    when: "Input type is outside supported numeric/logical image classes.",
    message: "im2uint16: unsupported input type",
};

pub const IM2UINT16_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT16.INTERNAL",
    identifier: Some("RunMat:im2uint16:Internal"),
    when: "Internal conversion step fails while building output tensor.",
    message: "im2uint16: internal conversion failure",
};

const IM2UINT16_ERRORS: [BuiltinErrorDescriptor; 5] = [
    IM2UINT16_ERROR_TOO_MANY_INPUTS,
    IM2UINT16_ERROR_TOO_MANY_OUTPUTS,
    IM2UINT16_ERROR_INVALID_INPUT,
    IM2UINT16_ERROR_UNSUPPORTED_INPUT_TYPE,
    IM2UINT16_ERROR_INTERNAL,
];

pub const IM2UINT16_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IM2UINT16_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IM2UINT16_ERRORS,
};

const IM2UINT16_DOCUMENTED_INTEGER_CLASSES: [BuiltinIntegerClass; 3] = [
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
];
const IM2UINT16_REJECTED_INTEGER_CLASSES: [BuiltinIntegerClass; 5] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];
const IM2UINT16_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &IM2UINT16_DOCUMENTED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Intensity and truecolor uint8, uint16, and int16 inputs map exactly into the uint16 image range.",
    }];
const IM2UINT16_INDEXED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &[BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16],
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Indexed uint8 and uint16 values are already zero-based and widen or copy without intensity scaling.",
    }];
const IM2UINT16_REJECTED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &IM2UINT16_REJECTED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "These integer image classes are outside the documented im2uint16 input surface and reject before conversion.",
    }];
pub const IM2UINT16_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "J = im2uint16(integer_I)",
        inputs: &IM2UINT16_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "uint8 expands by exact byte replication, int16 is offset exactly, and uint16 is retained in exact uint16 storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "J = im2uint16(integer_I, \"indexed\")",
        inputs: &IM2UINT16_INDEXED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Indexed zero-based integer values are copied or widened exactly into uint16 storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "im2uint16(unsupported_integer_I, ...)",
        inputs: &IM2UINT16_REJECTED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Unsupported integer dtype metadata is rejected consistently for host and resident inputs.",
    },
];
