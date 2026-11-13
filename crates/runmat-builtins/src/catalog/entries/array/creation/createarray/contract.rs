//! Canonical createarray signatures, errors, extensions, and integer forms.
use crate::*;

const FILL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Constant-valued output array.",
}];

const FILL_ERRORS: [BuiltinErrorDescriptor; 8] = [
    BuiltinErrorDescriptor {
        code: "RM.CREATE_ARRAY.NON_SCALAR_VALUE",
        identifier: None,
        when: "The fill value is not scalar.",
        message: "createArray: FillValue must be a scalar",
    },
    BuiltinErrorDescriptor {
        code: "RM.CREATE_ARRAY.INVALID_FILL_VALUE_TYPE",
        identifier: None,
        when: "The fill value type is unsupported.",
        message: "createArray: unsupported FillValue type",
    },
    BuiltinErrorDescriptor {
        code: "RM.CREATE_ARRAY.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "createArray: Like requires a prototype",
    },
    BuiltinErrorDescriptor {
        code: "RM.CREATE_ARRAY.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "createArray: Like may be specified only once",
    },
    BuiltinErrorDescriptor {
        code: "RM.CREATE_ARRAY.CLASS_CONFLICT",
        identifier: None,
        when: "A class keyword and a 'like' prototype are both provided.",
        message: "createArray: classname and Like cannot be combined",
    },
    BuiltinErrorDescriptor {
        code: "RM.CREATE_ARRAY.UNSUPPORTED_CLASS",
        identifier: None,
        when: "An unsupported output class is requested.",
        message: "createArray: unsupported class",
    },
    BuiltinErrorDescriptor {
        code: "RM.CREATE_ARRAY.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not recognized.",
        message: "createArray: unrecognized option",
    },
    BuiltinErrorDescriptor {
        code: "RM.CREATE_ARRAY.INVALID_DIMS",
        identifier: None,
        when: "Dimension arguments fail numeric/shape parsing.",
        message: "createArray: dimensions must be integer values",
    },
];

const CREATE_ARRAY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimensions, optional class name, and Like/FillValue name-value arguments.",
}];

const CREATE_ARRAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = createArray(sz1, ..., szN, classname?, Like=prototype?, FillValue=value?)",
    inputs: &CREATE_ARRAY_INPUTS,
    outputs: &FILL_OUTPUT,
}];

pub const CREATE_ARRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CREATE_ARRAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FILL_ERRORS,
};

const INTEGER_DIMS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "dimensions",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Every built-in integer class is accepted exactly as a structural size; negative dimensions normalize to zero.",
}];

const INTEGER_FILL_VALUE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "FillValue",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "An integer FillValue preserves its class unless classname or Like requests documented assignment conversion.",
}];

const INTEGER_LIKE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Like",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "An integer prototype preserves exact class and device residency.",
}];

pub const CREATE_ARRAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "X = createArray(integer_dimensions, ...)",
        inputs: &INTEGER_DIMS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Dimensions are consumed exactly before allocation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = createArray(..., FillValue=integer_value)",
        inputs: &INTEGER_FILL_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Without an override, output preserves FillValue class; documented assignment conversion applies for classname or Like. Integer GPU FillValue forms preserve native real or paired-complex storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = createArray(..., Like=integer_prototype)",
        inputs: &INTEGER_LIKE,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer GPU prototypes preserve exact class, complexity, and residency.",
    },
];
