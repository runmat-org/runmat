use crate::*;

const DATA_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host or provider-resident double, single, complex, or logical data; native integers and non-array objects reject.",
};
const DATA: [BuiltinParamDescriptor; 1] = [DATA_PARAM];
const FORMATTED_DATA: [BuiltinParamDescriptor; 2] = [
    DATA_PARAM,
    BuiltinParamDescriptor {
        name: "format",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Optional dimension-label text stored as Format and Labels.",
    },
];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host dlarray object retaining Data's numeric class, shape, and residency.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = dlarray(data)",
        inputs: &DATA,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "X = dlarray(data, format)",
        inputs: &FORMATTED_DATA,
        outputs: &OUTPUT,
    },
];

pub const DLARRAY_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "Data is not an admitted numeric/logical payload, has a native integer class, the format is not host text, the argument count is invalid, or more than one output is requested.",
    message: "dlarray: invalid data or dimension-label format",
};
pub const DLARRAY_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.UNSUPPORTED",
    identifier: Some("RunMat:deepLearning:Unsupported"),
    when: "A dlfeval autodiff tape is active and Data requires a provider-resident tape kernel that this implementation does not have.",
    message: "dlarray: provider-resident automatic differentiation is not supported",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [DLARRAY_INVALID_INPUT, DLARRAY_UNSUPPORTED];
pub const DLARRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "data",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "This current dlarray representation rejects native integer data on host and provider before wrapping it; explicit double or single conversion remains available.",
}];
pub const DLARRAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "dlarray(integer_data, ...)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "Integer payloads reject without provider download; no silent double widening occurs.",
    }];
