use crate::*;

const IM2DOUBLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "J",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Converted image data in double precision.",
}];

const IM2DOUBLE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Image data to convert.",
}];

const IM2DOUBLE_INDEXED_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const IM2DOUBLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "J = im2double(I)",
        inputs: &IM2DOUBLE_INPUTS,
        outputs: &IM2DOUBLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "J = im2double(I, \"indexed\")",
        inputs: &IM2DOUBLE_INDEXED_INPUTS,
        outputs: &IM2DOUBLE_OUTPUT,
    },
];

pub const IM2DOUBLE_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2DOUBLE.TOO_MANY_INPUTS",
    identifier: Some("RunMat:im2double:TooManyInputs"),
    when: "More than two input arguments are supplied.",
    message: "im2double: too many input arguments",
};

pub const IM2DOUBLE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2DOUBLE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:im2double:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "im2double: too many output arguments",
};

pub const IM2DOUBLE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2DOUBLE.INVALID_INPUT",
    identifier: Some("RunMat:im2double:InvalidInput"),
    when: "Input cannot be gathered or interpreted for image conversion.",
    message: "im2double: invalid input",
};

pub const IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2DOUBLE.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:im2double:UnsupportedInputType"),
    when: "Input type is outside supported numeric/logical image classes.",
    message: "im2double: unsupported input type",
};

pub const IM2DOUBLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IM2DOUBLE.INTERNAL",
    identifier: Some("RunMat:im2double:Internal"),
    when: "Internal conversion step fails while building output tensor.",
    message: "im2double: internal conversion failure",
};

const IM2DOUBLE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    IM2DOUBLE_ERROR_TOO_MANY_INPUTS,
    IM2DOUBLE_ERROR_TOO_MANY_OUTPUTS,
    IM2DOUBLE_ERROR_INVALID_INPUT,
    IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE,
    IM2DOUBLE_ERROR_INTERNAL,
];

pub const IM2DOUBLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IM2DOUBLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IM2DOUBLE_ERRORS,
};

const IM2DOUBLE_DOCUMENTED_INTEGER_CLASSES: [BuiltinIntegerClass; 3] = [
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
];
const IM2DOUBLE_REJECTED_INTEGER_CLASSES: [BuiltinIntegerClass; 5] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];
const IM2DOUBLE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &IM2DOUBLE_DOCUMENTED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Intensity and truecolor uint8, uint16, and int16 inputs are rescaled to double in [0,1] from authoritative storage.",
    }];
const IM2DOUBLE_INDEXED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &[
            BuiltinIntegerClass::Uint8,
            BuiltinIntegerClass::Uint16,
        ],
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Indexed integer values are zero-based and convert to one-based double indices by adding exactly one.",
    }];
const IM2DOUBLE_REJECTED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &IM2DOUBLE_REJECTED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "These integer image classes are outside the documented im2double input surface and reject before conversion.",
    }];
pub const IM2DOUBLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "J = im2double(integer_I)",
        inputs: &IM2DOUBLE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The host reference path preserves shape and produces double output; documented resident input gathers through its owner before conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "J = im2double(integer_I, \"indexed\")",
        inputs: &IM2DOUBLE_INDEXED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The zero-based integer index is read exactly, incremented once, and emitted as one-based double image data.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "im2double(unsupported_integer_I, ...)",
        inputs: &IM2DOUBLE_REJECTED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Unsupported integer dtype metadata is rejected consistently for host and resident inputs.",
    },
];
