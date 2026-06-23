//! MATLAB-compatible scalar symbolic `int` builtin.

use num_traits::Zero;
use runmat_builtins::{
    symbolic::SymbolicFunction, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, SymbolicExpr, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::{
    is_valid_identifier, symbolic_expr_to_value, symbolic_variable_name_from_value, text_scalar,
    value_to_symbolic_scalar,
};

const BUILTIN_NAME: &str = "int";
const ZERO_EPSILON: f64 = 1.0e-12;

const INT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "F",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar symbolic indefinite or definite integral result.",
}];

const INT_INPUTS_EXPR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "expr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar symbolic or numeric expression to integrate.",
}];

const INT_INPUTS_EXPR_VAR: [BuiltinParamDescriptor; 2] = [
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

const INT_INPUTS_EXPR_A_B: [BuiltinParamDescriptor; 3] = [
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

const INT_INPUTS_EXPR_VAR_A_B: [BuiltinParamDescriptor; 4] = [
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

const INT_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
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

const INT_ERRORS: [BuiltinErrorDescriptor; 5] = [
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
        when: "A name-value option is unknown or its value is not scalar.",
        message: "int: invalid option",
    },
    BuiltinErrorDescriptor {
        code: "RM.INT.ARG_COUNT",
        identifier: Some("RunMat:int:ArgCount"),
        when: "The argument grammar does not match int(expr), int(expr,var), int(expr,a,b), or int(expr,var,a,b).",
        message: "int: invalid number of input arguments",
    },
];

pub const INT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INT_ERRORS,
};

#[derive(Debug)]
struct IntArgs {
    variable: String,
    lower: Option<SymbolicExpr>,
    upper: Option<SymbolicExpr>,
}

#[runtime_builtin(
    name = "int",
    category = "math/symbolic",
    summary = "Evaluate scalar symbolic indefinite and definite integrals.",
    keywords = "int,symbolic,integration,calculus,definite integral",
    descriptor(crate::builtins::math::symbolic::int::INT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::symbolic::int"
)]
async fn int_builtin(expr: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let expr = value_to_symbolic_scalar(&expr).ok_or_else(|| int_error(&INT_ERRORS[0]))?;
    let args = parse_int_args(&expr, &rest)?;
    let result = if let Some(integral) = integrate_expr(&expr, &args.variable) {
        match (args.lower, args.upper) {
            (Some(lower), Some(upper)) => {
                let upper_value = integral.substitute(&args.variable, &upper);
                let lower_value = integral.substitute(&args.variable, &lower);
                SymbolicExpr::sub_expr(upper_value, lower_value)
            }
            _ => integral,
        }
    } else {
        formal_int(expr, &args.variable, args.lower, args.upper)
    };

    Ok(symbolic_expr_to_value(result.simplify()))
}

fn parse_int_args(expr: &SymbolicExpr, rest: &[Value]) -> BuiltinResult<IntArgs> {
    let primary_len = strip_name_value_options(rest)?;
    let primary = &rest[..primary_len];
    match primary.len() {
        0 => Ok(IntArgs {
            variable: infer_default_variable(expr),
            lower: None,
            upper: None,
        }),
        1 => Ok(IntArgs {
            variable: symbolic_variable_name_from_value(&primary[0])
                .ok_or_else(|| int_error(&INT_ERRORS[1]))?,
            lower: None,
            upper: None,
        }),
        2 => Ok(IntArgs {
            variable: infer_default_variable(expr),
            lower: Some(bound_expr(&primary[0])?),
            upper: Some(bound_expr(&primary[1])?),
        }),
        3 => Ok(IntArgs {
            variable: symbolic_variable_name_from_value(&primary[0])
                .ok_or_else(|| int_error(&INT_ERRORS[1]))?,
            lower: Some(bound_expr(&primary[1])?),
            upper: Some(bound_expr(&primary[2])?),
        }),
        _ => Err(int_error(&INT_ERRORS[4])),
    }
}

fn strip_name_value_options(rest: &[Value]) -> BuiltinResult<usize> {
    let len = rest.len();
    if len >= 2 {
        let Some(name) = text_scalar(&rest[len - 2]) else {
            return validate_remaining_int_args(rest, len);
        };
        if !is_known_option(name.trim()) {
            return validate_remaining_int_args(rest, len);
        }
        validate_option_value(&rest[len - 1])?;
        return Err(int_error_with_detail(
            &INT_ERRORS[3],
            format!("option '{}' is not yet supported", name.trim()),
        ));
    }
    validate_remaining_int_args(rest, len)
}

fn validate_remaining_int_args(rest: &[Value], len: usize) -> BuiltinResult<usize> {
    if len > 3 {
        return Err(int_error(&INT_ERRORS[4]));
    }
    if rest[len..].chunks_exact(2).any(|pair| {
        text_scalar(&pair[0])
            .map(|name| !is_known_option(name.trim()))
            .unwrap_or(true)
    }) {
        return Err(int_error(&INT_ERRORS[3]));
    }
    Ok(len)
}

fn is_known_option(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "ignoreanalyticconstraints" | "principalvalue" | "ignorespecialcases" | "hold"
    )
}

fn validate_option_value(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Bool(_) | Value::Num(_) | Value::Int(_) | Value::String(_) => Ok(()),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(()),
        _ if text_scalar(value).is_some() => Ok(()),
        _ => Err(int_error(&INT_ERRORS[3])),
    }
}

fn infer_default_variable(expr: &SymbolicExpr) -> String {
    let variables = expr.variables();
    if variables.contains("x") {
        return "x".to_string();
    }
    variables
        .into_iter()
        .find(|name| is_valid_identifier(name))
        .unwrap_or_else(|| "x".to_string())
}

fn bound_expr(value: &Value) -> BuiltinResult<SymbolicExpr> {
    value_to_symbolic_scalar(value).ok_or_else(|| int_error(&INT_ERRORS[2]))
}

fn integrate_expr(expr: &SymbolicExpr, variable: &str) -> Option<SymbolicExpr> {
    if variable.is_empty() || !expr.contains_variable(variable) {
        return Some(independent_integral(expr, variable));
    }

    match expr {
        SymbolicExpr::Constant(_)
        | SymbolicExpr::Rational { .. }
        | SymbolicExpr::DecimalLiteral { .. } => Some(independent_integral(expr, variable)),
        SymbolicExpr::Variable(name) if name == variable => Some(power_integral(
            SymbolicExpr::variable(variable),
            SymbolicExpr::constant(1.0),
        )),
        SymbolicExpr::Variable(_) => Some(independent_integral(expr, variable)),
        SymbolicExpr::Neg(inner) => Some(SymbolicExpr::neg_expr(integrate_expr(inner, variable)?)),
        SymbolicExpr::Add(lhs, rhs) => Some(SymbolicExpr::add_expr(
            integrate_expr(lhs, variable)?,
            integrate_expr(rhs, variable)?,
        )),
        SymbolicExpr::Sub(lhs, rhs) => Some(SymbolicExpr::sub_expr(
            integrate_expr(lhs, variable)?,
            integrate_expr(rhs, variable)?,
        )),
        SymbolicExpr::Mul(lhs, rhs) if !lhs.contains_variable(variable) => Some(
            SymbolicExpr::mul_expr((**lhs).clone(), integrate_expr(rhs, variable)?),
        ),
        SymbolicExpr::Mul(lhs, rhs) if !rhs.contains_variable(variable) => Some(
            SymbolicExpr::mul_expr((**rhs).clone(), integrate_expr(lhs, variable)?),
        ),
        SymbolicExpr::Mul(_, _) => None,
        SymbolicExpr::Div(lhs, rhs) if !rhs.contains_variable(variable) => Some(
            SymbolicExpr::div_expr(integrate_expr(lhs, variable)?, (**rhs).clone()),
        ),
        SymbolicExpr::Div(lhs, rhs)
            if !lhs.contains_variable(variable) && is_variable_expr(rhs, variable) =>
        {
            Some(SymbolicExpr::mul_expr(
                (**lhs).clone(),
                SymbolicExpr::function(SymbolicFunction::Log, SymbolicExpr::variable(variable)),
            ))
        }
        SymbolicExpr::Div(_, _) => None,
        SymbolicExpr::Pow(base, exponent)
            if is_variable_expr(base, variable) && !exponent.contains_variable(variable) =>
        {
            Some(power_integral((**base).clone(), (**exponent).clone()))
        }
        SymbolicExpr::Pow(_, _) => None,
        SymbolicExpr::Function(function, inner) => integrate_function(*function, inner, variable),
        SymbolicExpr::FunctionReference(_, _)
        | SymbolicExpr::FunctionCall(_, _)
        | SymbolicExpr::Equation(_, _)
        | SymbolicExpr::Derivative { .. } => {
            if expr.contains_variable(variable) {
                None
            } else {
                Some(independent_integral(expr, variable))
            }
        }
    }
}

fn independent_integral(expr: &SymbolicExpr, variable: &str) -> SymbolicExpr {
    let variable = SymbolicExpr::variable(variable);
    if matches!(
        expr,
        SymbolicExpr::Constant(_)
            | SymbolicExpr::Rational { .. }
            | SymbolicExpr::DecimalLiteral { .. }
    ) {
        SymbolicExpr::mul_expr(expr.clone(), variable)
    } else {
        SymbolicExpr::mul_expr(variable, expr.clone())
    }
}

fn integrate_function(
    function: SymbolicFunction,
    inner: &SymbolicExpr,
    variable: &str,
) -> Option<SymbolicExpr> {
    if !inner.contains_variable(variable) {
        return Some(SymbolicExpr::mul_expr(
            SymbolicExpr::function(function, inner.clone()),
            SymbolicExpr::variable(variable),
        ));
    }

    let scale = linear_inner_scale(inner, variable)?;
    let primitive = match function {
        SymbolicFunction::Sin => {
            SymbolicExpr::neg_expr(SymbolicExpr::function(SymbolicFunction::Cos, inner.clone()))
        }
        SymbolicFunction::Cos => SymbolicExpr::function(SymbolicFunction::Sin, inner.clone()),
        SymbolicFunction::Exp => SymbolicExpr::function(SymbolicFunction::Exp, inner.clone()),
        SymbolicFunction::Log if is_variable_expr(inner, variable) => {
            let x = SymbolicExpr::variable(variable);
            SymbolicExpr::sub_expr(
                SymbolicExpr::mul_expr(
                    x.clone(),
                    SymbolicExpr::function(SymbolicFunction::Log, x.clone()),
                ),
                x,
            )
        }
        SymbolicFunction::Sqrt if is_variable_expr(inner, variable) => {
            let x = SymbolicExpr::variable(variable);
            SymbolicExpr::mul_expr(rational(2, 3), SymbolicExpr::pow_expr(x, rational(3, 2)))
        }
        _ => return None,
    };

    if is_one_value(&scale) {
        Some(primitive)
    } else {
        Some(SymbolicExpr::div_expr(primitive, scale))
    }
}

fn linear_inner_scale(inner: &SymbolicExpr, variable: &str) -> Option<SymbolicExpr> {
    let derivative = inner.derivative(variable).simplify();
    if derivative.contains_variable(variable) {
        None
    } else {
        Some(derivative)
    }
}

fn power_integral(base: SymbolicExpr, exponent: SymbolicExpr) -> SymbolicExpr {
    if exact_rational_is_minus_one(&exponent) {
        return SymbolicExpr::function(SymbolicFunction::Log, base);
    }
    if let Some(power) = exponent.numeric_constant_value() {
        if !matches!(exponent, SymbolicExpr::Rational { .. }) && (power + 1.0).abs() <= ZERO_EPSILON
        {
            return SymbolicExpr::function(SymbolicFunction::Log, base);
        }
        let next_power = exact_or_float_power(power + 1.0);
        return SymbolicExpr::div_expr(
            SymbolicExpr::pow_expr(base, next_power.clone()),
            next_power,
        );
    }

    SymbolicExpr::div_expr(
        SymbolicExpr::pow_expr(
            base.clone(),
            SymbolicExpr::add_expr(exponent.clone(), SymbolicExpr::constant(1.0)),
        ),
        SymbolicExpr::add_expr(exponent, SymbolicExpr::constant(1.0)),
    )
}

fn exact_rational_is_minus_one(expr: &SymbolicExpr) -> bool {
    match expr {
        SymbolicExpr::Rational {
            numerator,
            denominator,
        } => !denominator.is_zero() && numerator == &(-denominator.clone()),
        _ => false,
    }
}

fn exact_or_float_power(value: f64) -> SymbolicExpr {
    let rounded = value.round();
    if (rounded - value).abs() <= ZERO_EPSILON
        && rounded >= i64::MIN as f64
        && rounded <= i64::MAX as f64
    {
        return rational(rounded as i64, 1);
    }
    SymbolicExpr::constant(value)
}

fn rational(numerator: i64, denominator: i64) -> SymbolicExpr {
    SymbolicExpr::rational(numerator, denominator)
        .expect("non-zero denominator for symbolic rational")
}

fn is_variable_expr(expr: &SymbolicExpr, variable: &str) -> bool {
    matches!(expr, SymbolicExpr::Variable(name) if name == variable)
}

fn is_one_value(expr: &SymbolicExpr) -> bool {
    matches!(expr.constant_value(), Some(value) if (value - 1.0).abs() <= ZERO_EPSILON)
}

fn formal_int(
    expr: SymbolicExpr,
    variable: &str,
    lower: Option<SymbolicExpr>,
    upper: Option<SymbolicExpr>,
) -> SymbolicExpr {
    let mut args = vec![expr, SymbolicExpr::variable(variable)];
    if let (Some(lower), Some(upper)) = (lower, upper) {
        args.push(lower);
        args.push(upper);
    }
    SymbolicExpr::function_call("int", args)
}

fn int_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    int_error_with_detail(error, error.message)
}

fn int_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(detail).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x() -> SymbolicExpr {
        SymbolicExpr::variable("x")
    }

    fn value_text(value: Value) -> String {
        match value {
            Value::Symbolic(expr) => expr.to_string(),
            other => panic!("expected symbolic result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn integrates_polynomial_with_default_variable() {
        let expr = SymbolicExpr::pow_expr(x(), SymbolicExpr::constant(2.0));
        let result = int_builtin(Value::Symbolic(expr), vec![]).await.unwrap();
        assert_eq!(value_text(result), "x^3/3");
    }

    #[tokio::test]
    async fn integrates_trig_definite_with_inferred_variable() {
        let expr = SymbolicExpr::function(SymbolicFunction::Sin, x());
        let result = int_builtin(
            Value::Symbolic(expr),
            vec![Value::Num(0.0), Value::Num(std::f64::consts::PI)],
        )
        .await
        .unwrap();
        assert_eq!(value_text(result), "2");
    }

    #[tokio::test]
    async fn integrates_exp_with_explicit_variable() {
        let expr = SymbolicExpr::function(SymbolicFunction::Exp, x());
        let result = int_builtin(
            Value::Symbolic(expr),
            vec![Value::Symbolic(SymbolicExpr::variable("x"))],
        )
        .await
        .unwrap();
        assert_eq!(value_text(result), "exp(x)");
    }

    #[tokio::test]
    async fn integrates_constant_definite() {
        let result = int_builtin(
            Value::Num(3.0),
            vec![
                Value::Symbolic(SymbolicExpr::variable("x")),
                Value::Num(1.0),
                Value::Num(4.0),
            ],
        )
        .await
        .unwrap();
        assert_eq!(value_text(result), "9");
    }

    #[tokio::test]
    async fn integrates_independent_symbolic_variable_in_display_order() {
        let result = int_builtin(
            Value::Symbolic(SymbolicExpr::variable("y")),
            vec![Value::Symbolic(SymbolicExpr::variable("x"))],
        )
        .await
        .unwrap();
        assert_eq!(value_text(result), "x*y");
    }

    #[tokio::test]
    async fn default_variable_prefers_x_in_multi_variable_expression() {
        let expr = SymbolicExpr::mul_expr(SymbolicExpr::variable("a"), x());
        let result = int_builtin(Value::Symbolic(expr), vec![]).await.unwrap();
        assert_eq!(value_text(result), "a*(x^2/2)");
    }

    #[tokio::test]
    async fn rejects_unsupported_name_value_options() {
        let expr = SymbolicExpr::function(SymbolicFunction::Cos, x());
        let err = int_builtin(
            Value::Symbolic(expr),
            vec![Value::String("PrincipalValue".into()), Value::Bool(true)],
        )
        .await
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:int:InvalidOption"));
    }

    #[tokio::test]
    async fn returns_formal_integral_for_unsupported_symbolic_form() {
        let expr = SymbolicExpr::function(SymbolicFunction::Tan, x());
        let result = int_builtin(Value::Symbolic(expr), vec![]).await.unwrap();
        assert_eq!(value_text(result), "int(tan(x), x)");
    }

    #[tokio::test]
    async fn returns_formal_integral_for_variable_dependent_power_exponent() {
        let expr = SymbolicExpr::pow_expr(x(), x());
        let result = int_builtin(Value::Symbolic(expr), vec![]).await.unwrap();
        assert_eq!(value_text(result), "int(x^x, x)");
    }

    #[tokio::test]
    async fn returns_formal_integral_for_exponential_power() {
        let expr = SymbolicExpr::pow_expr(SymbolicExpr::constant(2.0), x());
        let result = int_builtin(Value::Symbolic(expr), vec![]).await.unwrap();
        assert_eq!(value_text(result), "int(2^x, x)");
    }

    #[tokio::test]
    async fn returns_formal_definite_integral_for_unsupported_symbolic_form() {
        let expr = SymbolicExpr::function(SymbolicFunction::Tan, x());
        let result = int_builtin(
            Value::Symbolic(expr),
            vec![Value::Num(0.0), Value::Num(1.0)],
        )
        .await
        .unwrap();
        assert_eq!(value_text(result), "int(tan(x), x, 0, 1)");
    }
}
