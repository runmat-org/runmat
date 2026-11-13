use crate::*;

const ASSERT_OUTPUTS: [BuiltinParamDescriptor; 0] = [];

pub const ASSERT_COMPLEX_CONDITION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "assert-complex-condition",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "assert with a complex condition is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AssertComplexConditionExtension"),
    };

pub const ASSERT_UNQUALIFIED_IDENTIFIER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "assert-unqualified-identifier",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "assert with an unqualified custom error identifier is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AssertUnqualifiedIdentifierExtension"),
    };

pub const ASSERT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ASSERT_COMPLEX_CONDITION_EXTENSION,
    ASSERT_UNQUALIFIED_IDENTIFIER_EXTENSION,
];

const ASSERT_INTEGER_CONDITION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "cond",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every real integer scalar or array is convertible to logical; the condition passes only when it is nonempty and every element is nonzero.",
    }];

const ASSERT_COMPLEX_INTEGER_CONDITION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "cond",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode additionally accepts paired complex-integer storage and tests each element for a nonzero real or imaginary component.",
    }];

const ASSERT_INTEGER_FORMAT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Formatting replacement values accept numeric scalars and preserve exact integer values through integer and string conversion specifiers.",
    }];

pub const ASSERT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "assert(integer_cond, ...)",
        inputs: &ASSERT_INTEGER_CONDITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Truth testing reads exact authoritative integer storage. The public builtin has no output; resident conditions gather to the host because assert accepts gpuArray input but does not execute on the GPU.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "assert(complex_integer_cond, ...)",
        inputs: &ASSERT_COMPLEX_INTEGER_CONDITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Public logical conversion rejects complex numeric values. RunMat mode retains the pre-existing nonzero-component predicate without floating materialization.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "assert(cond, msg, integer_A...)",
        inputs: &ASSERT_INTEGER_FORMAT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Each documented replacement A is a character vector, string scalar, or numeric scalar. Integer scalars remain exact during host formatting, including after resident gather.",
    },
];

const ASSERT_INPUTS_CONDITION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "condition",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical or logically convertible condition that must evaluate to true.",
}];

const ASSERT_INPUTS_MESSAGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "condition",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical or logically convertible condition that must evaluate to true.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Assertion failed.\""),
        description: "Failure message text.",
    },
];

const ASSERT_INPUTS_MESSAGE_VARIADIC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "condition",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical or logically convertible condition that must evaluate to true.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Assertion failed.\""),
        description: "Failure message template text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the message template.",
    },
];

const ASSERT_INPUTS_IDENTIFIER_MESSAGE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "condition",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical or logically convertible condition that must evaluate to true.",
    },
    BuiltinParamDescriptor {
        name: "message_id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"RunMat:assertion:failed\""),
        description: "Message identifier.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Assertion failed.\""),
        description: "Failure message text.",
    },
];

const ASSERT_INPUTS_IDENTIFIER_MESSAGE_VARIADIC: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "condition",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical or logically convertible condition that must evaluate to true.",
    },
    BuiltinParamDescriptor {
        name: "message_id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"RunMat:assertion:failed\""),
        description: "Message identifier.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Assertion failed.\""),
        description: "Failure message template text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the message template.",
    },
];

const ASSERT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "assert(condition)",
        inputs: &ASSERT_INPUTS_CONDITION,
        outputs: &ASSERT_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "assert(condition, message)",
        inputs: &ASSERT_INPUTS_MESSAGE,
        outputs: &ASSERT_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "assert(condition, message, A...)",
        inputs: &ASSERT_INPUTS_MESSAGE_VARIADIC,
        outputs: &ASSERT_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "assert(condition, message_id, message)",
        inputs: &ASSERT_INPUTS_IDENTIFIER_MESSAGE,
        outputs: &ASSERT_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "assert(condition, message_id, message, A...)",
        inputs: &ASSERT_INPUTS_IDENTIFIER_MESSAGE_VARIADIC,
        outputs: &ASSERT_OUTPUTS,
    },
];

pub const ASSERT_ERROR_ASSERTION_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.ASSERTION_FAILED",
    identifier: Some("RunMat:assertion:failed"),
    when: "Condition evaluates to false and no custom identifier/message override is provided.",
    message: "Assertion failed.",
};

pub const ASSERT_ERROR_INVALID_CONDITION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.INVALID_CONDITION",
    identifier: Some("RunMat:assertion:invalidCondition"),
    when: "First argument is not logical or convertible to a logical condition.",
    message: "assert: first input must be logical or convertible to logical.",
};

pub const ASSERT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.INVALID_INPUT",
    identifier: Some("RunMat:assertion:invalidInput"),
    when: "Message identifier/message text or formatting payload is invalid.",
    message: "assert: invalid input argument",
};

pub const ASSERT_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:minrhs"),
    when: "No condition argument is provided.",
    message: "Not enough input arguments.",
};

pub const ASSERT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSERT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:assertion:TooManyOutputs"),
    when: "One or more public outputs are requested from assert.",
    message: "assert: too many output arguments",
};

const ASSERT_ERRORS: [BuiltinErrorDescriptor; 5] = [
    ASSERT_ERROR_ASSERTION_FAILED,
    ASSERT_ERROR_INVALID_CONDITION,
    ASSERT_ERROR_INVALID_INPUT,
    ASSERT_ERROR_NOT_ENOUGH_INPUTS,
    ASSERT_ERROR_TOO_MANY_OUTPUTS,
];

pub const ASSERT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ASSERT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ASSERT_ERRORS,
};
