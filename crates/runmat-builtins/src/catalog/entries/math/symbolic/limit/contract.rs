use crate::*;

pub const LIMIT_NUMERIC_EXPRESSION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "limit-numeric-expression",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "limit with a nonsymbolic numeric expression is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LimitNumericExpressionExtension"),
    };

pub const LIMIT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [LIMIT_NUMERIC_EXPRESSION_EXTENSION];

pub const LIMIT_INTEGER_EXPRESSION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "expr",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public input is symbolic; RunMat mode also admits a scalar native integer when it is exactly representable by the current binary64 symbolic core.",
    }];

pub const LIMIT_INTEGER_POINT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "point",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A scalar integer limit point enters the current binary64 symbolic core only after an exact-representability check.",
    }];

pub const LIMIT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "L = limit(integer_expr, ...)",
        inputs: &LIMIT_INTEGER_EXPRESSION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "This independently gated RunMat extension rejects integers that would round before constructing a symbolic constant; the result follows the symbolic result contract.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "L = limit(expr, var, integer_point, direction)",
        inputs: &LIMIT_INTEGER_POINT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The point is read from authoritative integer storage and must convert exactly before substitution into RunMat's current binary64 symbolic representation. Wider integer points reject explicitly instead of rounding.",
    },
];

pub const LIMIT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar symbolic limit expression, including symbolic real numeric constants.",
}];

const LIMIT_PARAM_EXPR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "expr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar symbolic expression.",
};
const LIMIT_PARAM_VAR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "var",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Limit variable. Inferred when omitted and the expression has one variable.",
};
const LIMIT_PARAM_POINT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "point",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Numeric point approached by the limit variable.",
};
const LIMIT_PARAM_DIRECTION:BuiltinParamDescriptor=BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Text direction: left/- or right/+; empty text, both, two-sided and twosided select the default two-sided limit. Trimmed and case-insensitive.",
    };
pub const LIMIT_INPUTS: [BuiltinParamDescriptor; 4] = [
    LIMIT_PARAM_EXPR,
    LIMIT_PARAM_VAR,
    LIMIT_PARAM_POINT,
    LIMIT_PARAM_DIRECTION,
];

pub const LIMIT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "L = limit(expr)",
        inputs: &[LIMIT_PARAM_EXPR],
        outputs: &LIMIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = limit(expr, var)",
        inputs: &[LIMIT_PARAM_EXPR, LIMIT_PARAM_VAR],
        outputs: &LIMIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = limit(expr, point)",
        inputs: &[LIMIT_PARAM_EXPR, LIMIT_PARAM_POINT],
        outputs: &LIMIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = limit(expr, var, point)",
        inputs: &[LIMIT_PARAM_EXPR, LIMIT_PARAM_VAR, LIMIT_PARAM_POINT],
        outputs: &LIMIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "L = limit(expr, var, point, direction)",
        inputs: &LIMIT_INPUTS,
        outputs: &LIMIT_OUTPUT,
    },
];

pub const LIMIT_ERRORS: [BuiltinErrorDescriptor; 7] = [
    BuiltinErrorDescriptor {
        code: "RM.LIMIT.INVALID_EXPR",
        identifier: Some("RunMat:limit:InvalidExpression"),
        when: "The expression cannot be represented as a scalar symbolic expression.",
        message: "limit: expected a scalar symbolic or numeric expression",
    },
    BuiltinErrorDescriptor {
        code: "RM.LIMIT.INVALID_VARIABLE",
        identifier: Some("RunMat:limit:InvalidVariable"),
        when: "The limit variable cannot be inferred or is not a symbolic variable/text scalar.",
        message: "limit: invalid or ambiguous symbolic variable",
    },
    BuiltinErrorDescriptor {
        code: "RM.LIMIT.INVALID_POINT",
        identifier: Some("RunMat:limit:InvalidPoint"),
        when: "The limit point is not a finite scalar numeric value.",
        message: "limit: expected a finite scalar numeric point",
    },
    BuiltinErrorDescriptor {
        code: "RM.LIMIT.INVALID_DIRECTION",
        identifier: Some("RunMat:limit:InvalidDirection"),
        when: "The optional direction is not supported scalar text (left/-, right/+, empty/both/two-sided/twosided).",
        message: "limit: invalid direction",
    },
    BuiltinErrorDescriptor {
        code: "RM.LIMIT.UNSUPPORTED",
        identifier: Some("RunMat:limit:Unsupported"),
        when:
            "The scalar symbolic limit cannot be resolved by substitution or L'Hopital reduction.",
        message: "limit: unable to resolve symbolic limit",
    },
    BuiltinErrorDescriptor {
        code: "RM.LIMIT.ARG_COUNT",
        identifier: Some("RunMat:limit:ArgCount"),
        when: "Too many arguments were supplied.",
        message: "limit: too many input arguments",
    },
    LIMIT_ERROR_TOO_MANY_OUTPUTS,
];

pub const LIMIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LIMIT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LIMIT_ERRORS,
};

pub const LIMIT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LIMIT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:limit:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "limit: too many output arguments",
};
