use crate::*;

const BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Host dlarray or admitted numeric/logical left operand.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Host dlarray or admitted numeric/logical right operand.",
    },
];
const SUM_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Host dlarray with double, single, or logical data; logical data convert to double for arithmetic.",
    },
    BuiltinParamDescriptor {
        name: "all",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional 'all' reduction selector; dimension-specific reductions are not implemented by this method.",
    },
];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host dlarray retaining the admitted result class, shape, Format, and Labels.",
}];

pub const INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "An operand has unsupported class/storage, its shapes are incompatible, a sum selector is unsupported, or more than one output is requested.",
    message: "dlarray method received invalid input",
};
pub const UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.UNSUPPORTED",
    identifier: Some("RunMat:deepLearning:Unsupported"),
    when: "The method is asked to operate on provider-resident dlarray Data without resident tape kernels.",
    message: "provider-resident dlarray arithmetic is not available in this host tape",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [INVALID_INPUT, UNSUPPORTED];

macro_rules! method_descriptor {
    ($constant:ident, $label:literal, $inputs:expr) => {
        pub const $constant: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: $inputs,
                outputs: &OUTPUT,
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::MethodOnly,
            errors: &ERRORS,
        };
    };
}
method_descriptor!(
    DLARRAY_PLUS_DESCRIPTOR,
    "Y = dlarray.plus(lhs, rhs)",
    &BINARY_INPUTS
);
method_descriptor!(
    DLARRAY_MINUS_DESCRIPTOR,
    "Y = dlarray.minus(lhs, rhs)",
    &BINARY_INPUTS
);
method_descriptor!(
    DLARRAY_TIMES_DESCRIPTOR,
    "Y = dlarray.times(lhs, rhs)",
    &BINARY_INPUTS
);
method_descriptor!(
    DLARRAY_RDIVIDE_DESCRIPTOR,
    "Y = dlarray.rdivide(lhs, rhs)",
    &BINARY_INPUTS
);
method_descriptor!(
    DLARRAY_MTIMES_DESCRIPTOR,
    "Y = dlarray.mtimes(lhs, rhs)",
    &BINARY_INPUTS
);
method_descriptor!(
    DLARRAY_SUM_DESCRIPTOR,
    "Y = dlarray.sum(X, 'all')",
    &SUM_INPUTS
);

const INTEGER_OPERAND: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "operand",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Native integer payloads are rejected before any f64 arithmetic conversion; explicit double or single conversion is required. Logical payloads are admitted separately as 0/1 double data.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "dlarray.method(integer_operand)",
        inputs: &INTEGER_OPERAND,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "All six current host autodiff methods reject native integer operands; no exact integer array is silently widened.",
    }];
