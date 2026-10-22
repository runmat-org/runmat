use crate::*;

pub const INT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "F",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar symbolic indefinite or definite integral result.",
}];

pub const INT_INPUTS_EXPR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "expr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar symbolic or numeric expression to integrate.",
}];

pub const INT_INPUTS_EXPR_VAR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "expr",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar symbolic or numeric expression to integrate.",
    },
    BuiltinParamDescriptor {
        name: "var",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("symvar(expr, 1)"),
        description: "Integration variable as a symbolic variable or text scalar.",
    },
];

pub const INT_INPUTS_EXPR_A_B: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "expr",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar symbolic or numeric expression to integrate.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Lower integration bound.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Upper integration bound.",
    },
];

pub const INT_INPUTS_EXPR_VAR_A_B: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "expr",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar symbolic or numeric expression to integrate.",
    },
    BuiltinParamDescriptor {
        name: "var",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("symvar(expr, 1)"),
        description: "Integration variable as a symbolic variable or text scalar.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Lower integration bound.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Upper integration bound.",
    },
];

pub const INT_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "F = int(expr)",
        inputs: &INT_INPUTS_EXPR,
        outputs: &INT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = int(expr, var)",
        inputs: &INT_INPUTS_EXPR_VAR,
        outputs: &INT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = int(expr, a, b)",
        inputs: &INT_INPUTS_EXPR_A_B,
        outputs: &INT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = int(expr, var, a, b)",
        inputs: &INT_INPUTS_EXPR_VAR_A_B,
        outputs: &INT_OUTPUT,
    },
];

pub const INT_ERRORS: [BuiltinErrorDescriptor; 6] = [
    BuiltinErrorDescriptor {
        code: "RM.INT.INVALID_EXPR",
        identifier: Some("RunMat:int:InvalidExpression"),
        when: "The expression cannot be represented as a scalar symbolic expression.",
        message: "int: expected a scalar symbolic or numeric expression",
    },
    BuiltinErrorDescriptor {
        code: "RM.INT.INVALID_VARIABLE",
        identifier: Some("RunMat:int:InvalidVariable"),
        when: "The integration variable cannot be inferred or is not a symbolic variable/text scalar.",
        message: "int: invalid symbolic variable",
    },
    BuiltinErrorDescriptor {
        code: "RM.INT.INVALID_BOUND",
        identifier: Some("RunMat:int:InvalidBound"),
        when: "A definite integration bound cannot be represented as a scalar symbolic expression.",
        message: "int: invalid integration bound",
    },
    BuiltinErrorDescriptor {
        code: "RM.INT.INVALID_OPTION",
        identifier: Some("RunMat:int:InvalidOption"),
        when: "A recognized IgnoreAnalyticConstraints, PrincipalValue, IgnoreSpecialCases or Hold option is not implemented, or its value has an unsupported type.",
        message: "int: invalid option",
    },
    BuiltinErrorDescriptor {
        code: "RM.INT.ARG_COUNT",
        identifier: Some("RunMat:int:ArgCount"),
        when: "The argument grammar does not match int(expr), int(expr,var), int(expr,a,b), or int(expr,var,a,b).",
        message: "int: invalid number of input arguments",
    },
    INT_ERROR_TOO_MANY_OUTPUTS,
];

pub const INT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INT_ERRORS,
};

pub const INT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "expr, a, or b",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat's current integration boundary requires numeric integer constants to convert exactly to binary64 before symbolic processing; this restriction is separate from the exact BigInt rational representation used by other symbolic APIs.",
    }];

pub const INT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "F = int(integer_expr_or_bound, ...)",
        inputs: &INT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "This is symbolic integration, not integer datatype conversion. Numeric integers that cannot convert exactly to binary64 reject under the current int boundary policy instead of rounding; symbolic rational expressions remain a distinct supported input form.",
    }];

pub const INT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:int:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "int: too many output arguments",
};
