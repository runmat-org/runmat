//! MATLAB-compatible scalar symbolic `limit` builtin.

use crate::{
    call::{BuiltinArguments, FunctionArgumentView},
    context::RuntimeContext,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, SymbolicLimitDirection as LimitDirection, SymbolicScalarOperation,
    LIMIT_DESCRIPTOR, LIMIT_ERRORS, LIMIT_ERROR_TOO_MANY_OUTPUTS,
    LIMIT_NUMERIC_EXPRESSION_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{SymbolicExpr, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::{is_valid_identifier, symbolic_expr_to_value, text_scalar, value_to_symbolic_scalar};

const BUILTIN_NAME: &str = "limit";
const MAX_LOPITAL_DEPTH: usize = 8;
const ZERO_EPSILON: f64 = 1.0e-12;
const SAMPLE_EPSILON: f64 = 1.0e-7;

#[runtime_builtin(
    name = "limit",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::symbolic::limit",
    context = true
)]
async fn limit_builtin(
    runtime: Option<RuntimeContext>,
    expr: Value,
    rest: BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &LIMIT_DESCRIPTOR,
        &LIMIT_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )
    .map_err(super::invocation::terminal)?;
    super::invocation::validate(SymbolicScalarOperation::Limit, 0, rest.len() + 1, &expr)?;
    if rest.len() > 3 {
        return Err(limit_error(&LIMIT_ERRORS[5]));
    }
    for (index, value) in rest.iter().enumerate() {
        super::invocation::validate(
            SymbolicScalarOperation::Limit,
            index + 1,
            rest.len() + 1,
            value,
        )?;
    }
    if !matches!(&expr, Value::Symbolic(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LIMIT_NUMERIC_EXPRESSION_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    ensure_exact_integer_symbolic_boundary(&expr, &LIMIT_ERRORS[0], "expression")?;
    super::invocation::admit(SymbolicScalarOperation::Limit, runtime.as_ref(), true)?;
    let expr = value_to_symbolic_scalar(&expr).ok_or_else(|| limit_error(&LIMIT_ERRORS[0]))?;
    let args = parse_limit_args(&expr, rest.view())?;
    let result = evaluate_limit(&expr, &args.variable, args.point, args.direction, 0)?;
    Ok(symbolic_expr_to_value(result.simplify()))
}

#[derive(Debug)]
struct LimitArgs {
    variable: String,
    point: f64,
    direction: LimitDirection,
}

fn parse_limit_args(
    expr: &SymbolicExpr,
    rest: FunctionArgumentView<'_>,
) -> BuiltinResult<LimitArgs> {
    if rest.len() > 3 {
        return Err(limit_error(&LIMIT_ERRORS[5]));
    }
    let (variable, point, direction) = match rest.len() {
        0 => (infer_single_variable(expr)?, 0.0, LimitDirection::TwoSided),
        1 => {
            if let Some(variable) = variable_name_from_value(&rest[0])? {
                (variable, 0.0, LimitDirection::TwoSided)
            } else {
                (
                    infer_single_variable(expr)?,
                    finite_scalar_point(&rest[0])?,
                    LimitDirection::TwoSided,
                )
            }
        }
        2 => (
            variable_name_from_value(&rest[0])?.ok_or_else(|| limit_error(&LIMIT_ERRORS[1]))?,
            finite_scalar_point(&rest[1])?,
            LimitDirection::TwoSided,
        ),
        3 => (
            variable_name_from_value(&rest[0])?.ok_or_else(|| limit_error(&LIMIT_ERRORS[1]))?,
            finite_scalar_point(&rest[1])?,
            parse_direction(&rest[2])?,
        ),
        _ => unreachable!(),
    };
    Ok(LimitArgs {
        variable,
        point,
        direction,
    })
}

fn infer_single_variable(expr: &SymbolicExpr) -> BuiltinResult<String> {
    let variables = expr.variables();
    if variables.len() == 1 {
        Ok(variables.into_iter().next().unwrap_or_default())
    } else {
        Err(limit_error(&LIMIT_ERRORS[1]))
    }
}

fn variable_name_from_value(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::Symbolic(expr) => expr.variable_name().map(ToOwned::to_owned),
        _ => text_scalar(value)
            .map_err(|source| {
                let mut error = limit_error(&LIMIT_ERRORS[1]);
                error.source = Some(Box::new(source));
                error
            })?
            .map(|text| text.trim().to_string()),
    }
    .filter(|name| is_valid_identifier(name)))
}

fn finite_scalar_point(value: &Value) -> BuiltinResult<f64> {
    ensure_exact_integer_symbolic_boundary(value, &LIMIT_ERRORS[2], "point")?;
    let Some(expr) = value_to_symbolic_scalar(value) else {
        return Err(limit_error(&LIMIT_ERRORS[2]));
    };
    let Some(point) = expr.constant_value() else {
        return Err(limit_error(&LIMIT_ERRORS[2]));
    };
    if point.is_finite() {
        Ok(point)
    } else {
        Err(limit_error(&LIMIT_ERRORS[2]))
    }
}

fn ensure_exact_integer_symbolic_boundary(
    value: &Value,
    error: &'static BuiltinErrorDescriptor,
    role: &str,
) -> BuiltinResult<()> {
    if crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
        Ok(())
    } else {
        Err(limit_error_with_message(
            error,
            format!("limit: integer {role} must be exactly representable as double"),
        ))
    }
}

fn parse_direction(value: &Value) -> BuiltinResult<LimitDirection> {
    let Some(text) = text_scalar(value).map_err(|source| {
        let mut error = limit_error(&LIMIT_ERRORS[1]);
        error.source = Some(Box::new(source));
        error
    })?
    else {
        return Err(limit_error(&LIMIT_ERRORS[3]));
    };
    LimitDirection::from_public_option(&text).ok_or_else(|| limit_error(&LIMIT_ERRORS[3]))
}

fn evaluate_limit(
    expr: &SymbolicExpr,
    variable: &str,
    point: f64,
    direction: LimitDirection,
    depth: usize,
) -> BuiltinResult<SymbolicExpr> {
    if !expr.contains_variable(variable) {
        return Ok(expr.clone().simplify());
    }

    let point_expr = SymbolicExpr::constant(point);
    if let SymbolicExpr::Derivative {
        expr: derivative_expr,
        variable: derivative_variable,
        ..
    } = expr
    {
        if derivative_variable == variable && contains_formal_symbolic_function(derivative_expr) {
            return Ok(expr.clone());
        }
    }

    let substituted = expr.substitute(variable, &point_expr).simplify();
    if let Some(value) = substituted.constant_value() {
        return Ok(SymbolicExpr::constant(value));
    }

    match expr {
        SymbolicExpr::Constant(_)
        | SymbolicExpr::Rational { .. }
        | SymbolicExpr::DecimalLiteral { .. } => Ok(expr.clone()),
        SymbolicExpr::Variable(name) if name == variable => Ok(point_expr),
        SymbolicExpr::Variable(_) => Ok(expr.clone()),
        SymbolicExpr::FunctionReference(name, parameters) => {
            let mut substituted = false;
            let args = parameters
                .iter()
                .map(|parameter| {
                    if parameter == variable {
                        substituted = true;
                        Ok(point_expr.clone())
                    } else {
                        Ok(SymbolicExpr::variable(parameter))
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            if substituted {
                Ok(SymbolicExpr::function_call(name.clone(), args))
            } else {
                Ok(SymbolicExpr::FunctionReference(name.clone(), parameters.clone()).simplify())
            }
        }
        SymbolicExpr::FunctionCall(name, args) => Ok(SymbolicExpr::function_call(
            name.clone(),
            args.iter()
                .map(|arg| evaluate_limit(arg, variable, point, direction, depth))
                .collect::<BuiltinResult<Vec<_>>>()?,
        )),
        SymbolicExpr::Equation(lhs, rhs) => Ok(SymbolicExpr::equation(
            evaluate_limit(lhs, variable, point, direction, depth)?,
            evaluate_limit(rhs, variable, point, direction, depth)?,
        )),
        SymbolicExpr::Derivative {
            expr,
            variable: derivative_variable,
            order,
        } => {
            if derivative_variable == variable && contains_formal_symbolic_function(expr) {
                return Ok(SymbolicExpr::Derivative {
                    expr: expr.clone(),
                    variable: derivative_variable.clone(),
                    order: *order,
                });
            }
            Ok(SymbolicExpr::Derivative {
                expr: runmat_value::symbolic::SymbolicOwner::new_unbounded(evaluate_limit(
                    expr, variable, point, direction, depth,
                )?),
                variable: derivative_variable.clone(),
                order: *order,
            }
            .simplify())
        }
        SymbolicExpr::Neg(inner) => Ok(SymbolicExpr::neg_expr(evaluate_limit(
            inner, variable, point, direction, depth,
        )?)),
        SymbolicExpr::Add(lhs, rhs) => Ok(SymbolicExpr::add_expr(
            evaluate_limit(lhs, variable, point, direction, depth)?,
            evaluate_limit(rhs, variable, point, direction, depth)?,
        )),
        SymbolicExpr::Sub(lhs, rhs) => Ok(SymbolicExpr::sub_expr(
            evaluate_limit(lhs, variable, point, direction, depth)?,
            evaluate_limit(rhs, variable, point, direction, depth)?,
        )),
        SymbolicExpr::Mul(lhs, rhs) => {
            evaluate_product_limit(lhs, rhs, expr, variable, point, direction, depth)
        }
        SymbolicExpr::Div(numerator, denominator) => evaluate_division_limit(
            numerator,
            denominator,
            expr,
            variable,
            point,
            direction,
            depth,
        ),
        SymbolicExpr::Pow(base, exponent) => {
            evaluate_power_limit(base, exponent, expr, variable, point, direction, depth)
        }
        SymbolicExpr::Function(function, inner) => Ok(SymbolicExpr::function(
            *function,
            evaluate_limit(inner, variable, point, direction, depth)?,
        )),
    }
}

fn contains_formal_symbolic_function(expr: &SymbolicExpr) -> bool {
    match expr {
        SymbolicExpr::FunctionReference(_, _) | SymbolicExpr::FunctionCall(_, _) => true,
        SymbolicExpr::Equation(lhs, rhs)
        | SymbolicExpr::Add(lhs, rhs)
        | SymbolicExpr::Sub(lhs, rhs)
        | SymbolicExpr::Mul(lhs, rhs)
        | SymbolicExpr::Div(lhs, rhs)
        | SymbolicExpr::Pow(lhs, rhs) => {
            contains_formal_symbolic_function(lhs) || contains_formal_symbolic_function(rhs)
        }
        SymbolicExpr::Derivative { expr, .. } | SymbolicExpr::Neg(expr) => {
            contains_formal_symbolic_function(expr)
        }
        SymbolicExpr::Function(_, expr) => contains_formal_symbolic_function(expr),
        SymbolicExpr::Constant(_)
        | SymbolicExpr::Rational { .. }
        | SymbolicExpr::DecimalLiteral { .. }
        | SymbolicExpr::Variable(_) => false,
    }
}

fn evaluate_product_limit(
    lhs: &SymbolicExpr,
    rhs: &SymbolicExpr,
    original: &SymbolicExpr,
    variable: &str,
    point: f64,
    direction: LimitDirection,
    depth: usize,
) -> BuiltinResult<SymbolicExpr> {
    let lhs_fraction = fraction_parts(lhs);
    let rhs_fraction = fraction_parts(rhs);
    if lhs_fraction.has_denominator || rhs_fraction.has_denominator {
        let numerator = SymbolicExpr::mul_expr(lhs_fraction.numerator, rhs_fraction.numerator);
        let denominator =
            SymbolicExpr::mul_expr(lhs_fraction.denominator, rhs_fraction.denominator);
        let quotient = SymbolicExpr::Div(
            runmat_value::symbolic::SymbolicOwner::new_unbounded(numerator.clone()),
            runmat_value::symbolic::SymbolicOwner::new_unbounded(denominator.clone()),
        );
        return evaluate_division_limit(
            &numerator,
            &denominator,
            &quotient,
            variable,
            point,
            direction,
            depth,
        );
    }

    let lhs_limit = evaluate_limit(lhs, variable, point, direction, depth)?;
    let rhs_limit = evaluate_limit(rhs, variable, point, direction, depth)?;
    let product = SymbolicExpr::mul_expr(lhs_limit, rhs_limit);
    if product.has_undefined_constant_subexpression() {
        if let Some(value) = sample_simple_pole(original, variable, point, direction) {
            return Ok(SymbolicExpr::constant(value));
        }
        return Err(limit_error(&LIMIT_ERRORS[4]));
    }
    Ok(product)
}

fn evaluate_power_limit(
    base: &SymbolicExpr,
    exponent: &SymbolicExpr,
    original: &SymbolicExpr,
    variable: &str,
    point: f64,
    direction: LimitDirection,
    depth: usize,
) -> BuiltinResult<SymbolicExpr> {
    if let Some(exponent_value) = exponent.constant_value() {
        if exponent_value.is_sign_negative() {
            let positive_power =
                SymbolicExpr::pow_expr(base.clone(), SymbolicExpr::constant(-exponent_value));
            return evaluate_division_limit(
                &SymbolicExpr::constant(1.0),
                &positive_power,
                original,
                variable,
                point,
                direction,
                depth,
            );
        }
    }

    Ok(SymbolicExpr::pow_expr(
        evaluate_limit(base, variable, point, direction, depth)?,
        evaluate_limit(exponent, variable, point, direction, depth)?,
    ))
}

struct FractionParts {
    numerator: SymbolicExpr,
    denominator: SymbolicExpr,
    has_denominator: bool,
}

fn fraction_parts(expr: &SymbolicExpr) -> FractionParts {
    match expr {
        SymbolicExpr::Div(numerator, denominator) => FractionParts {
            numerator: (**numerator).clone(),
            denominator: (**denominator).clone(),
            has_denominator: true,
        },
        SymbolicExpr::Pow(base, exponent) => {
            if let Some(exponent_value) = exponent.constant_value() {
                if exponent_value.is_sign_negative() {
                    return FractionParts {
                        numerator: SymbolicExpr::constant(1.0),
                        denominator: SymbolicExpr::pow_expr(
                            (**base).clone(),
                            SymbolicExpr::constant(-exponent_value),
                        ),
                        has_denominator: true,
                    };
                }
            }
            FractionParts {
                numerator: expr.clone(),
                denominator: SymbolicExpr::constant(1.0),
                has_denominator: false,
            }
        }
        _ => FractionParts {
            numerator: expr.clone(),
            denominator: SymbolicExpr::constant(1.0),
            has_denominator: false,
        },
    }
}

fn evaluate_division_limit(
    numerator: &SymbolicExpr,
    denominator: &SymbolicExpr,
    original: &SymbolicExpr,
    variable: &str,
    point: f64,
    direction: LimitDirection,
    depth: usize,
) -> BuiltinResult<SymbolicExpr> {
    let point_expr = SymbolicExpr::constant(point);
    let numerator_at_point = numerator.substitute(variable, &point_expr).simplify();
    let denominator_at_point = denominator.substitute(variable, &point_expr).simplify();

    if numerator_at_point.is_zero_constant() && denominator_at_point.is_zero_constant() {
        return lhopital_limit(numerator, denominator, variable, point, direction, depth);
    }
    if let (Some(numerator_value), Some(denominator_value)) = (
        numerator_at_point.constant_value(),
        denominator_at_point.constant_value(),
    ) {
        if denominator_value.abs() > ZERO_EPSILON {
            return Ok(SymbolicExpr::constant(numerator_value / denominator_value));
        }
    }

    let numerator_limit = evaluate_limit(numerator, variable, point, direction, depth)?;
    let denominator_limit = evaluate_limit(denominator, variable, point, direction, depth)?;
    if numerator_limit.is_zero_constant() && denominator_limit.is_zero_constant() {
        return lhopital_limit(numerator, denominator, variable, point, direction, depth);
    }
    if denominator_at_point.is_zero_constant() || denominator_limit.is_zero_constant() {
        if let Some(value) = sample_simple_pole(original, variable, point, direction)
            .filter(|value| value.is_finite() || value.is_infinite())
        {
            return Ok(SymbolicExpr::constant(value));
        }
        return Err(limit_error(&LIMIT_ERRORS[4]));
    }

    Ok(SymbolicExpr::div_expr(numerator_limit, denominator_limit))
}

fn lhopital_limit(
    numerator: &SymbolicExpr,
    denominator: &SymbolicExpr,
    variable: &str,
    point: f64,
    direction: LimitDirection,
    depth: usize,
) -> BuiltinResult<SymbolicExpr> {
    if depth >= MAX_LOPITAL_DEPTH {
        return Err(limit_error(&LIMIT_ERRORS[4]));
    }
    let differentiated = SymbolicExpr::div_expr(
        numerator.derivative(variable),
        denominator.derivative(variable),
    );
    evaluate_limit(&differentiated, variable, point, direction, depth + 1)
}

fn sample_simple_pole(
    expr: &SymbolicExpr,
    variable: &str,
    point: f64,
    direction: LimitDirection,
) -> Option<f64> {
    let sample = |offset: f64| {
        expr.substitute(variable, &SymbolicExpr::constant(point + offset))
            .simplify()
            .constant_value()
    };
    match direction {
        LimitDirection::Right => sample(SAMPLE_EPSILON).map(|value| {
            if value.is_sign_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        }),
        LimitDirection::Left => sample(-SAMPLE_EPSILON).map(|value| {
            if value.is_sign_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        }),
        LimitDirection::TwoSided => {
            let left = sample(-SAMPLE_EPSILON)?;
            let right = sample(SAMPLE_EPSILON)?;
            if left.is_sign_positive() == right.is_sign_positive() {
                Some(if right.is_sign_negative() {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                })
            } else {
                None
            }
        }
    }
}

fn limit_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    limit_error_with_message(error, error.message)
}

fn limit_error_with_message(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{BuiltinIntegerInputAvailability, LIMIT_INTEGER_CAPABILITIES};
    use runmat_value::SymbolicFunction;
    use runmat_value::{IntegerStorage, Tensor};

    #[test]
    fn computes_sinc_limit() {
        let x = SymbolicExpr::variable("x");
        let expr =
            SymbolicExpr::div_expr(SymbolicExpr::function(SymbolicFunction::Sin, x.clone()), x);
        let result = evaluate_limit(&expr, "x", 0.0, LimitDirection::TwoSided, 0).expect("limit");

        assert_eq!(result, SymbolicExpr::constant(1.0));
    }

    #[test]
    fn computes_symbolic_finite_difference_limit() {
        let x = SymbolicExpr::variable("x");
        let h = SymbolicExpr::variable("h");
        let expr = SymbolicExpr::div_expr(
            SymbolicExpr::sub_expr(
                SymbolicExpr::function(
                    SymbolicFunction::Cos,
                    SymbolicExpr::add_expr(x.clone(), h.clone()),
                ),
                SymbolicExpr::function(SymbolicFunction::Cos, x),
            ),
            h,
        );
        let result = evaluate_limit(&expr, "h", 0.0, LimitDirection::TwoSided, 0).expect("limit");

        assert_eq!(result.to_string(), "-sin(x)");
    }

    #[test]
    fn computes_nested_quotient_inside_sum() {
        let x = SymbolicExpr::variable("x");
        let quotient =
            SymbolicExpr::div_expr(SymbolicExpr::function(SymbolicFunction::Sin, x.clone()), x);
        let expr = SymbolicExpr::add_expr(quotient, SymbolicExpr::constant(1.0));
        let result = evaluate_limit(&expr, "x", 0.0, LimitDirection::TwoSided, 0).expect("limit");

        assert_eq!(result, SymbolicExpr::constant(2.0));
    }

    #[test]
    fn computes_fractional_power_limit() {
        let x = SymbolicExpr::variable("x");
        let numerator = SymbolicExpr::sub_expr(
            SymbolicExpr::pow_expr(
                SymbolicExpr::function(SymbolicFunction::Cos, x.clone()),
                SymbolicExpr::constant(1.0 / 3.0),
            ),
            SymbolicExpr::constant(1.0),
        );
        let denominator = SymbolicExpr::pow_expr(x, SymbolicExpr::constant(2.0));
        let expr = SymbolicExpr::div_expr(numerator, denominator);
        let result = evaluate_limit(&expr, "x", 0.0, LimitDirection::TwoSided, 0).expect("limit");

        assert_eq!(result, SymbolicExpr::constant(-1.0 / 6.0));
    }

    #[test]
    fn preserves_formal_derivative_limit_for_unknown_symbolic_function() {
        let y = SymbolicExpr::function_reference("Y", vec!["X".to_string()]);
        let expr = SymbolicExpr::derivative_expr(y, "X", 1);

        let result = evaluate_limit(&expr, "X", 0.0, LimitDirection::TwoSided, 0).expect("limit");

        assert_eq!(result.to_string(), "diff(Y(X), X)");
    }

    #[test]
    fn computes_removable_product_quotient() {
        let x = SymbolicExpr::variable("x");
        let reciprocal = SymbolicExpr::div_expr(SymbolicExpr::constant(1.0), x.clone());
        let expr = SymbolicExpr::mul_expr(x, reciprocal);
        let result = evaluate_limit(&expr, "x", 0.0, LimitDirection::TwoSided, 0).expect("limit");

        assert_eq!(result, SymbolicExpr::constant(1.0));
    }

    #[test]
    fn computes_one_sided_reciprocal_poles() {
        let x = SymbolicExpr::variable("x");
        let expr = SymbolicExpr::pow_expr(x, SymbolicExpr::constant(-1.0));

        let right = evaluate_limit(&expr, "x", 0.0, LimitDirection::Right, 0).expect("limit");
        assert_eq!(right, SymbolicExpr::constant(f64::INFINITY));

        let left = evaluate_limit(&expr, "x", 0.0, LimitDirection::Left, 0).expect("limit");
        assert_eq!(left, SymbolicExpr::constant(f64::NEG_INFINITY));

        let two_sided = evaluate_limit(&expr, "x", 0.0, LimitDirection::TwoSided, 0)
            .expect_err("two-sided reciprocal pole should not resolve");
        assert_eq!(
            two_sided.identifier.as_deref(),
            Some("RunMat:limit:Unsupported")
        );
    }

    #[test]
    fn computes_two_sided_even_power_pole() {
        let x = SymbolicExpr::variable("x");
        let expr = SymbolicExpr::pow_expr(x, SymbolicExpr::constant(-2.0));
        let result = evaluate_limit(&expr, "x", 0.0, LimitDirection::TwoSided, 0).expect("limit");

        assert_eq!(result, SymbolicExpr::constant(f64::INFINITY));
    }

    #[test]
    fn rejects_invalid_variable_name() {
        let x = SymbolicExpr::variable("x");
        let expr =
            SymbolicExpr::div_expr(SymbolicExpr::function(SymbolicFunction::Sin, x.clone()), x);

        let args = [Value::from("x y"), Value::Num(0.0)];
        let err = parse_limit_args(&expr, FunctionArgumentView::from_slice(&args))
            .expect_err("invalid variable should error");

        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:limit:InvalidVariable")
        );
    }

    #[test]
    fn integer_points_cross_only_an_exact_binary64_boundary() {
        let x = Value::Symbolic(SymbolicExpr::variable("x"));
        let exact = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 53]), vec![1, 1])
                .expect("exact point"),
        );
        block_on(limit_builtin(
            None,
            x.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![exact]),
        ))
        .expect("exact integer point");

        let lossy = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
                .expect("lossy point"),
        );
        let error = block_on(limit_builtin(
            None,
            x,
            crate::builtins::common::test_support::builtin_arguments(vec![lossy]),
        ))
        .expect_err("lossy point must reject");
        assert_eq!(error.identifier(), Some("RunMat:limit:InvalidPoint"));
        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn numeric_integer_expression_is_gated_before_symbolic_conversion() {
        let input = || Value::Int(runmat_value::IntValue::I16(7));
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(limit_builtin(
                None,
                input(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("numeric expression is a RunMat extension");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:LimitNumericExpressionExtension")
            );
        }
        {
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            let output = block_on(limit_builtin(
                None,
                input(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("constant expression has no variable to infer");
            assert_eq!(output.identifier(), Some("RunMat:limit:InvalidVariable"));
        }
    }

    #[test]
    fn limit_integer_capabilities_distinguish_public_point_and_extension_expression() {
        assert_eq!(LIMIT_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(
            LIMIT_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert_eq!(
            LIMIT_INTEGER_CAPABILITIES[1].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
    }
}
