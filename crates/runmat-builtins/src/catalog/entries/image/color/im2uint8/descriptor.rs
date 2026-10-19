use crate::*;

const IM2UINT8_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "J",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Converted image data in uint8 class.",
}];

const IM2UINT8_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Image data to convert.",
}];

const IM2UINT8_INDEXED_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const IM2UINT8_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "J = im2uint8(I)",
        inputs: &IM2UINT8_INPUTS,
        outputs: &IM2UINT8_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "J = im2uint8(I, \"indexed\")",
        inputs: &IM2UINT8_INDEXED_INPUTS,
        outputs: &IM2UINT8_OUTPUT,
    },
];

pub const IM2UINT8_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT8.TOO_MANY_INPUTS",
    identifier: Some("RunMat:im2uint8:TooManyInputs"),
    when: "More than two input arguments are supplied.",
    message: "im2uint8: too many input arguments",
};

pub const IM2UINT8_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT8.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:im2uint8:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "im2uint8: too many output arguments",
};

pub const IM2UINT8_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT8.INVALID_INPUT",
    identifier: Some("RunMat:im2uint8:InvalidInput"),
    when: "Input cannot be gathered or interpreted for image conversion.",
    message: "im2uint8: invalid input",
};

pub const IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT8.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:im2uint8:UnsupportedInputType"),
    when: "Input type is outside supported numeric/logical image classes.",
    message: "im2uint8: unsupported input type",
};

pub const IM2UINT8_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2UINT8.INTERNAL",
    identifier: Some("RunMat:im2uint8:Internal"),
    when: "Internal conversion step fails while building output tensor.",
    message: "im2uint8: internal conversion failure",
};

const IM2UINT8_ERRORS: [BuiltinErrorDescriptor; 5] = [
    IM2UINT8_ERROR_TOO_MANY_INPUTS,
    IM2UINT8_ERROR_TOO_MANY_OUTPUTS,
    IM2UINT8_ERROR_INVALID_INPUT,
    IM2UINT8_ERROR_UNSUPPORTED_INPUT_TYPE,
    IM2UINT8_ERROR_INTERNAL,
];

pub const IM2UINT8_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IM2UINT8_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IM2UINT8_ERRORS,
};

const IM2UINT8_DOCUMENTED_INTEGER_CLASSES: [BuiltinIntegerClass; 3] = [
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
];
const IM2UINT8_REJECTED_INTEGER_CLASSES: [BuiltinIntegerClass; 5] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];
const IM2UINT8_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &IM2UINT8_DOCUMENTED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Intensity and truecolor uint8, uint16, and int16 inputs quantize into the uint8 image range.",
    }];
const IM2UINT8_INDEXED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &[BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16],
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Indexed uint8 values are retained; indexed uint16 values must not exceed 255 and narrow without intensity scaling.",
    }];
const IM2UINT8_REJECTED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &IM2UINT8_REJECTED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "These integer image classes are outside the documented im2uint8 input surface and reject before conversion.",
    }];
pub const IM2UINT8_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "J = im2uint8(integer_I)",
        inputs: &IM2UINT8_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "uint16 and offset int16 inputs quantize at the documented half-byte boundaries; output is exact uint8 storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "J = im2uint8(integer_I, \"indexed\")",
        inputs: &IM2UINT8_INDEXED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Indexed conversion preserves zero-based integer indices and errors when a uint16 index cannot fit in uint8.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "im2uint8(unsupported_integer_I, ...)",
        inputs: &IM2UINT8_REJECTED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Unsupported integer dtype metadata is rejected consistently for host and resident inputs.",
    },
];
