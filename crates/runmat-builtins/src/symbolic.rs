use crate::format_number;
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::BTreeSet;
use std::fmt;

const EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicFunction {
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    Heaviside,
    Dirac,
    DiracDerivative(u32),
}

impl SymbolicFunction {
    pub fn name(self) -> &'static str {
        match self {
            SymbolicFunction::Sin => "sin",
            SymbolicFunction::Cos => "cos",
            SymbolicFunction::Tan => "tan",
            SymbolicFunction::Exp => "exp",
            SymbolicFunction::Log => "log",
            SymbolicFunction::Sqrt => "sqrt",
            SymbolicFunction::Heaviside => "heaviside",
            SymbolicFunction::Dirac => "dirac",
            SymbolicFunction::DiracDerivative(_) => "dirac",
        }
    }

    fn apply_numeric_constant(self, value: f64) -> Option<f64> {
        let result = match self {
            SymbolicFunction::Sin => value.sin(),
            SymbolicFunction::Cos => value.cos(),
            SymbolicFunction::Tan => value.tan(),
            SymbolicFunction::Exp => value.exp(),
            SymbolicFunction::Log => value.ln(),
            SymbolicFunction::Sqrt => value.sqrt(),
            SymbolicFunction::Heaviside if value > 0.0 => 1.0,
            SymbolicFunction::Heaviside if value < 0.0 => 0.0,
            SymbolicFunction::Heaviside if value == 0.0 => 0.5,
            SymbolicFunction::Heaviside => value,
            SymbolicFunction::Dirac if value == 0.0 => f64::INFINITY,
            SymbolicFunction::Dirac => 0.0,
            SymbolicFunction::DiracDerivative(_) if value == 0.0 => f64::INFINITY,
            SymbolicFunction::DiracDerivative(_) => 0.0,
        };
        (!result.is_nan()).then_some(result)
    }

    fn apply_constant(self, value: f64) -> Option<f64> {
        self.apply_numeric_constant(value)
            .filter(|result| result.is_finite())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicExpr {
    Constant(f64),
    Rational {
        numerator: BigInt,
        denominator: BigInt,
    },
    DecimalLiteral {
        text: String,
        value: f64,
        digits: usize,
    },
    Variable(String),
    FunctionReference(String, Vec<String>),
    FunctionCall(String, Vec<SymbolicExpr>),
    Equation(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Derivative {
        expr: Box<SymbolicExpr>,
        variable: String,
        order: u32,
    },
    Neg(Box<SymbolicExpr>),
    Add(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Sub(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Mul(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Div(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Pow(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Function(SymbolicFunction, Box<SymbolicExpr>),
}

impl SymbolicExpr {
    pub fn constant(value: f64) -> Self {
        SymbolicExpr::Constant(value)
    }

    pub fn rational(numerator: impl Into<BigInt>, denominator: impl Into<BigInt>) -> Option<Self> {
        let numerator = numerator.into();
        let denominator = denominator.into();
        if denominator.is_zero() {
            return None;
        }
        let mut numerator = numerator;
        let mut denominator = denominator;
        if denominator.is_negative() {
            numerator = -numerator;
            denominator = -denominator;
        }
        let divisor = numerator.gcd(&denominator);
        Some(SymbolicExpr::Rational {
            numerator: numerator / &divisor,
            denominator: denominator / divisor,
        })
    }

    pub fn decimal_literal(text: impl Into<String>, value: f64, digits: usize) -> Self {
        SymbolicExpr::DecimalLiteral {
            text: text.into(),
            value,
            digits,
        }
    }

    pub fn variable(name: impl Into<String>) -> Self {
        SymbolicExpr::Variable(name.into())
    }

    pub fn function_reference(name: impl Into<String>, parameters: Vec<String>) -> Self {
        SymbolicExpr::FunctionReference(name.into(), parameters).simplify()
    }

    pub fn function_call(name: impl Into<String>, args: Vec<SymbolicExpr>) -> Self {
        SymbolicExpr::FunctionCall(name.into(), args).simplify()
    }

    pub fn equation(lhs: SymbolicExpr, rhs: SymbolicExpr) -> Self {
        SymbolicExpr::Equation(Box::new(lhs), Box::new(rhs)).simplify()
    }

    pub fn derivative_expr(expr: SymbolicExpr, variable: impl Into<String>, order: u32) -> Self {
        let variable = variable.into();
        let expr = expr.simplify();
        if order == 0 {
            return expr;
        }
        if !expr.contains_variable(&variable) {
            return SymbolicExpr::Constant(0.0);
        }
        SymbolicExpr::Derivative {
            expr: Box::new(expr),
            variable,
            order,
        }
        .simplify()
    }

    pub fn function(function: SymbolicFunction, argument: SymbolicExpr) -> Self {
        SymbolicExpr::Function(function, Box::new(argument)).simplify()
    }

    pub fn neg_expr(value: SymbolicExpr) -> Self {
        SymbolicExpr::Neg(Box::new(value)).simplify()
    }

    pub fn add_expr(lhs: SymbolicExpr, rhs: SymbolicExpr) -> Self {
        SymbolicExpr::Add(Box::new(lhs), Box::new(rhs)).simplify()
    }

    pub fn sub_expr(lhs: SymbolicExpr, rhs: SymbolicExpr) -> Self {
        SymbolicExpr::Sub(Box::new(lhs), Box::new(rhs)).simplify()
    }

    pub fn mul_expr(lhs: SymbolicExpr, rhs: SymbolicExpr) -> Self {
        SymbolicExpr::Mul(Box::new(lhs), Box::new(rhs)).simplify()
    }

    pub fn div_expr(lhs: SymbolicExpr, rhs: SymbolicExpr) -> Self {
        SymbolicExpr::Div(Box::new(lhs), Box::new(rhs)).simplify()
    }

    pub fn pow_expr(lhs: SymbolicExpr, rhs: SymbolicExpr) -> Self {
        SymbolicExpr::Pow(Box::new(lhs), Box::new(rhs)).simplify()
    }

    pub fn variable_name(&self) -> Option<&str> {
        match self {
            SymbolicExpr::Variable(name) => Some(name),
            _ => None,
        }
    }

    pub fn function_reference_name(&self) -> Option<&str> {
        match self {
            SymbolicExpr::FunctionReference(name, _) => Some(name),
            _ => None,
        }
    }

    pub fn function_reference_signature(&self) -> Option<(&str, &[String])> {
        match self {
            SymbolicExpr::FunctionReference(name, parameters) => Some((name, parameters)),
            _ => None,
        }
    }

    pub fn contains_variable(&self, variable: &str) -> bool {
        match self {
            SymbolicExpr::Constant(_)
            | SymbolicExpr::Rational { .. }
            | SymbolicExpr::DecimalLiteral { .. } => false,
            SymbolicExpr::Variable(name) => name == variable,
            SymbolicExpr::FunctionReference(_, parameters) => {
                parameters.iter().any(|parameter| parameter == variable)
            }
            SymbolicExpr::FunctionCall(_, args) => {
                args.iter().any(|arg| arg.contains_variable(variable))
            }
            SymbolicExpr::Equation(lhs, rhs) => {
                lhs.contains_variable(variable) || rhs.contains_variable(variable)
            }
            SymbolicExpr::Derivative { expr, .. } => expr.contains_variable(variable),
            SymbolicExpr::Neg(inner) | SymbolicExpr::Function(_, inner) => {
                inner.contains_variable(variable)
            }
            SymbolicExpr::Add(lhs, rhs)
            | SymbolicExpr::Sub(lhs, rhs)
            | SymbolicExpr::Mul(lhs, rhs)
            | SymbolicExpr::Div(lhs, rhs)
            | SymbolicExpr::Pow(lhs, rhs) => {
                lhs.contains_variable(variable) || rhs.contains_variable(variable)
            }
        }
    }

    pub fn variables(&self) -> BTreeSet<String> {
        let mut variables = BTreeSet::new();
        self.collect_variables(&mut variables);
        variables
    }

    fn collect_variables(&self, variables: &mut BTreeSet<String>) {
        match self {
            SymbolicExpr::Constant(_)
            | SymbolicExpr::Rational { .. }
            | SymbolicExpr::DecimalLiteral { .. } => {}
            SymbolicExpr::Variable(name) => {
                variables.insert(name.clone());
            }
            SymbolicExpr::FunctionReference(_, parameters) => {
                for parameter in parameters {
                    variables.insert(parameter.clone());
                }
            }
            SymbolicExpr::FunctionCall(_, args) => {
                for arg in args {
                    arg.collect_variables(variables);
                }
            }
            SymbolicExpr::Equation(lhs, rhs) => {
                lhs.collect_variables(variables);
                rhs.collect_variables(variables);
            }
            SymbolicExpr::Derivative { expr, .. } => {
                expr.collect_variables(variables);
            }
            SymbolicExpr::Neg(inner) | SymbolicExpr::Function(_, inner) => {
                inner.collect_variables(variables)
            }
            SymbolicExpr::Add(lhs, rhs)
            | SymbolicExpr::Sub(lhs, rhs)
            | SymbolicExpr::Mul(lhs, rhs)
            | SymbolicExpr::Div(lhs, rhs)
            | SymbolicExpr::Pow(lhs, rhs) => {
                lhs.collect_variables(variables);
                rhs.collect_variables(variables);
            }
        }
    }

    pub fn substitute(&self, variable: &str, replacement: &SymbolicExpr) -> SymbolicExpr {
        match self {
            SymbolicExpr::Constant(value) => SymbolicExpr::Constant(*value),
            SymbolicExpr::Rational {
                numerator,
                denominator,
            } => SymbolicExpr::Rational {
                numerator: numerator.clone(),
                denominator: denominator.clone(),
            },
            SymbolicExpr::DecimalLiteral {
                text,
                value,
                digits,
            } => SymbolicExpr::DecimalLiteral {
                text: text.clone(),
                value: *value,
                digits: *digits,
            },
            SymbolicExpr::Variable(name) if name == variable => replacement.clone(),
            SymbolicExpr::Variable(name) => SymbolicExpr::Variable(name.clone()),
            SymbolicExpr::FunctionReference(name, parameters) => {
                let mut changed = false;
                let args = parameters
                    .iter()
                    .map(|parameter| {
                        if parameter == variable {
                            changed = true;
                            replacement.clone()
                        } else {
                            SymbolicExpr::variable(parameter)
                        }
                    })
                    .collect::<Vec<_>>();
                if changed {
                    SymbolicExpr::function_call(name.clone(), args)
                } else {
                    SymbolicExpr::function_reference(name.clone(), parameters.clone())
                }
            }
            SymbolicExpr::FunctionCall(name, args) => SymbolicExpr::function_call(
                name.clone(),
                args.iter()
                    .map(|arg| arg.substitute(variable, replacement))
                    .collect(),
            ),
            SymbolicExpr::Equation(lhs, rhs) => SymbolicExpr::equation(
                lhs.substitute(variable, replacement),
                rhs.substitute(variable, replacement),
            ),
            SymbolicExpr::Derivative {
                expr,
                variable: derivative_variable,
                order,
            } => SymbolicExpr::Derivative {
                expr: Box::new(expr.substitute(variable, replacement)),
                variable: derivative_variable.clone(),
                order: *order,
            }
            .simplify(),
            SymbolicExpr::Neg(inner) => {
                SymbolicExpr::neg_expr(inner.substitute(variable, replacement))
            }
            SymbolicExpr::Add(lhs, rhs) => SymbolicExpr::add_expr(
                lhs.substitute(variable, replacement),
                rhs.substitute(variable, replacement),
            ),
            SymbolicExpr::Sub(lhs, rhs) => SymbolicExpr::sub_expr(
                lhs.substitute(variable, replacement),
                rhs.substitute(variable, replacement),
            ),
            SymbolicExpr::Mul(lhs, rhs) => SymbolicExpr::mul_expr(
                lhs.substitute(variable, replacement),
                rhs.substitute(variable, replacement),
            ),
            SymbolicExpr::Div(lhs, rhs) => SymbolicExpr::div_expr(
                lhs.substitute(variable, replacement),
                rhs.substitute(variable, replacement),
            ),
            SymbolicExpr::Pow(lhs, rhs) => SymbolicExpr::pow_expr(
                lhs.substitute(variable, replacement),
                rhs.substitute(variable, replacement),
            ),
            SymbolicExpr::Function(function, inner) => {
                SymbolicExpr::function(*function, inner.substitute(variable, replacement))
            }
        }
    }

    pub fn derivative(&self, variable: &str) -> SymbolicExpr {
        match self {
            SymbolicExpr::Constant(_)
            | SymbolicExpr::Rational { .. }
            | SymbolicExpr::DecimalLiteral { .. } => SymbolicExpr::constant(0.0),
            SymbolicExpr::Variable(name) if name == variable => SymbolicExpr::constant(1.0),
            SymbolicExpr::Variable(_) => SymbolicExpr::constant(0.0),
            SymbolicExpr::FunctionReference(_, parameters) => {
                if parameters.iter().any(|parameter| parameter == variable) {
                    SymbolicExpr::derivative_expr(self.clone(), variable, 1)
                } else {
                    SymbolicExpr::constant(0.0)
                }
            }
            SymbolicExpr::FunctionCall(_, args) => {
                if args.iter().any(|arg| arg.contains_variable(variable)) {
                    SymbolicExpr::derivative_expr(self.clone(), variable, 1)
                } else {
                    SymbolicExpr::constant(0.0)
                }
            }
            SymbolicExpr::Equation(lhs, rhs) => {
                SymbolicExpr::equation(lhs.derivative(variable), rhs.derivative(variable))
            }
            SymbolicExpr::Derivative { .. } => {
                SymbolicExpr::derivative_expr(self.clone(), variable, 1)
            }
            SymbolicExpr::Neg(inner) => SymbolicExpr::neg_expr(inner.derivative(variable)),
            SymbolicExpr::Add(lhs, rhs) => {
                SymbolicExpr::add_expr(lhs.derivative(variable), rhs.derivative(variable))
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                SymbolicExpr::sub_expr(lhs.derivative(variable), rhs.derivative(variable))
            }
            SymbolicExpr::Mul(lhs, rhs) => SymbolicExpr::add_expr(
                SymbolicExpr::mul_expr(lhs.derivative(variable), (**rhs).clone()),
                SymbolicExpr::mul_expr((**lhs).clone(), rhs.derivative(variable)),
            ),
            SymbolicExpr::Div(lhs, rhs) => {
                let numerator = SymbolicExpr::sub_expr(
                    SymbolicExpr::mul_expr(lhs.derivative(variable), (**rhs).clone()),
                    SymbolicExpr::mul_expr((**lhs).clone(), rhs.derivative(variable)),
                );
                let denominator =
                    SymbolicExpr::pow_expr((**rhs).clone(), SymbolicExpr::constant(2.0));
                SymbolicExpr::div_expr(numerator, denominator)
            }
            SymbolicExpr::Pow(base, exponent) => {
                if let SymbolicExpr::Constant(power) = **exponent {
                    let coefficient = SymbolicExpr::constant(power);
                    let lowered_power = SymbolicExpr::pow_expr(
                        (**base).clone(),
                        SymbolicExpr::constant(power - 1.0),
                    );
                    SymbolicExpr::mul_expr(
                        SymbolicExpr::mul_expr(coefficient, lowered_power),
                        base.derivative(variable),
                    )
                } else {
                    let first = SymbolicExpr::mul_expr(
                        exponent.derivative(variable),
                        SymbolicExpr::function(SymbolicFunction::Log, (**base).clone()),
                    );
                    let second = SymbolicExpr::mul_expr(
                        (**exponent).clone(),
                        SymbolicExpr::div_expr(base.derivative(variable), (**base).clone()),
                    );
                    SymbolicExpr::mul_expr(self.clone(), SymbolicExpr::add_expr(first, second))
                }
            }
            SymbolicExpr::Function(function, inner) => {
                let inner_derivative = inner.derivative(variable);
                let outer = match function {
                    SymbolicFunction::Sin => {
                        SymbolicExpr::function(SymbolicFunction::Cos, (**inner).clone())
                    }
                    SymbolicFunction::Cos => SymbolicExpr::neg_expr(SymbolicExpr::function(
                        SymbolicFunction::Sin,
                        (**inner).clone(),
                    )),
                    SymbolicFunction::Tan => SymbolicExpr::div_expr(
                        SymbolicExpr::constant(1.0),
                        SymbolicExpr::pow_expr(
                            SymbolicExpr::function(SymbolicFunction::Cos, (**inner).clone()),
                            SymbolicExpr::constant(2.0),
                        ),
                    ),
                    SymbolicFunction::Exp => {
                        SymbolicExpr::function(SymbolicFunction::Exp, (**inner).clone())
                    }
                    SymbolicFunction::Log => {
                        SymbolicExpr::div_expr(SymbolicExpr::constant(1.0), (**inner).clone())
                    }
                    SymbolicFunction::Sqrt => SymbolicExpr::div_expr(
                        SymbolicExpr::constant(1.0),
                        SymbolicExpr::mul_expr(
                            SymbolicExpr::constant(2.0),
                            SymbolicExpr::function(SymbolicFunction::Sqrt, (**inner).clone()),
                        ),
                    ),
                    SymbolicFunction::Heaviside => {
                        SymbolicExpr::function(SymbolicFunction::Dirac, (**inner).clone())
                    }
                    SymbolicFunction::Dirac => SymbolicExpr::function(
                        SymbolicFunction::DiracDerivative(1),
                        (**inner).clone(),
                    ),
                    SymbolicFunction::DiracDerivative(order) => SymbolicExpr::function(
                        SymbolicFunction::DiracDerivative(order.saturating_add(1)),
                        (**inner).clone(),
                    ),
                };
                SymbolicExpr::mul_expr(outer, inner_derivative)
            }
        }
    }

    pub fn numeric_constant_value(&self) -> Option<f64> {
        match self {
            SymbolicExpr::Constant(value) => (!value.is_nan()).then_some(*value),
            SymbolicExpr::Rational {
                numerator,
                denominator,
            } => {
                if denominator.is_zero() {
                    None
                } else {
                    Some(numerator.to_f64()? / denominator.to_f64()?)
                }
            }
            SymbolicExpr::DecimalLiteral { value, .. } => (!value.is_nan()).then_some(*value),
            SymbolicExpr::Variable(_) => None,
            SymbolicExpr::FunctionReference(_, _)
            | SymbolicExpr::FunctionCall(_, _)
            | SymbolicExpr::Equation(_, _)
            | SymbolicExpr::Derivative { .. } => None,
            SymbolicExpr::Neg(inner) => finite_or_infinite(-inner.numeric_constant_value()?),
            SymbolicExpr::Add(lhs, rhs) => {
                finite_or_infinite(lhs.numeric_constant_value()? + rhs.numeric_constant_value()?)
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                finite_or_infinite(lhs.numeric_constant_value()? - rhs.numeric_constant_value()?)
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                finite_or_infinite(lhs.numeric_constant_value()? * rhs.numeric_constant_value()?)
            }
            SymbolicExpr::Div(lhs, rhs) => {
                let denominator = rhs.numeric_constant_value()?;
                if is_zero(denominator) {
                    None
                } else {
                    finite_or_infinite(lhs.numeric_constant_value()? / denominator)
                }
            }
            SymbolicExpr::Pow(lhs, rhs) => {
                let base = lhs.numeric_constant_value()?;
                let exponent = rhs.numeric_constant_value()?;
                if is_zero(base) && exponent.is_sign_negative() {
                    None
                } else {
                    finite_or_infinite(base.powf(exponent))
                }
            }
            SymbolicExpr::Function(function, inner) => {
                function.apply_numeric_constant(inner.numeric_constant_value()?)
            }
        }
    }

    pub fn constant_value(&self) -> Option<f64> {
        self.numeric_constant_value()
            .filter(|value| value.is_finite())
    }

    pub fn has_undefined_constant_subexpression(&self) -> bool {
        match self {
            SymbolicExpr::Constant(value) => value.is_nan(),
            SymbolicExpr::Rational { denominator, .. } => denominator.is_zero(),
            SymbolicExpr::DecimalLiteral { value, .. } => value.is_nan(),
            SymbolicExpr::Variable(_) => false,
            SymbolicExpr::FunctionReference(_, _) => false,
            SymbolicExpr::FunctionCall(_, args) => args
                .iter()
                .any(SymbolicExpr::has_undefined_constant_subexpression),
            SymbolicExpr::Equation(lhs, rhs) => {
                lhs.has_undefined_constant_subexpression()
                    || rhs.has_undefined_constant_subexpression()
            }
            SymbolicExpr::Derivative { expr, .. } => expr.has_undefined_constant_subexpression(),
            SymbolicExpr::Neg(inner) => inner.has_undefined_constant_subexpression(),
            SymbolicExpr::Add(lhs, rhs)
            | SymbolicExpr::Sub(lhs, rhs)
            | SymbolicExpr::Mul(lhs, rhs) => {
                lhs.has_undefined_constant_subexpression()
                    || rhs.has_undefined_constant_subexpression()
            }
            SymbolicExpr::Div(lhs, rhs) => {
                lhs.has_undefined_constant_subexpression()
                    || rhs.has_undefined_constant_subexpression()
                    || rhs.numeric_constant_value().is_some_and(is_zero)
            }
            SymbolicExpr::Pow(lhs, rhs) => {
                lhs.has_undefined_constant_subexpression()
                    || rhs.has_undefined_constant_subexpression()
                    || matches!(
                        (lhs.numeric_constant_value(), rhs.numeric_constant_value()),
                        (Some(base), Some(exponent))
                            if is_zero(base) && exponent.is_sign_negative()
                    )
            }
            SymbolicExpr::Function(function, inner) => {
                inner.has_undefined_constant_subexpression()
                    || inner
                        .numeric_constant_value()
                        .is_some_and(|value| function.apply_numeric_constant(value).is_none())
            }
        }
    }

    pub fn has_nonfinite_constant(&self) -> bool {
        match self {
            SymbolicExpr::Constant(value) => !value.is_finite(),
            SymbolicExpr::Rational { denominator, .. } => denominator.is_zero(),
            SymbolicExpr::DecimalLiteral { value, .. } => !value.is_finite(),
            SymbolicExpr::Variable(_) => false,
            SymbolicExpr::FunctionReference(_, _) => false,
            SymbolicExpr::FunctionCall(_, args) => {
                args.iter().any(SymbolicExpr::has_nonfinite_constant)
            }
            SymbolicExpr::Equation(lhs, rhs) => {
                lhs.has_nonfinite_constant() || rhs.has_nonfinite_constant()
            }
            SymbolicExpr::Derivative { expr, .. } => expr.has_nonfinite_constant(),
            SymbolicExpr::Neg(inner) | SymbolicExpr::Function(_, inner) => {
                inner.has_nonfinite_constant()
            }
            SymbolicExpr::Add(lhs, rhs)
            | SymbolicExpr::Sub(lhs, rhs)
            | SymbolicExpr::Mul(lhs, rhs)
            | SymbolicExpr::Div(lhs, rhs)
            | SymbolicExpr::Pow(lhs, rhs) => {
                lhs.has_nonfinite_constant() || rhs.has_nonfinite_constant()
            }
        }
    }

    fn can_eliminate_zero_product_factor(&self) -> bool {
        !self.has_undefined_constant_subexpression() && !self.has_nonfinite_constant()
    }

    pub fn is_zero_constant(&self) -> bool {
        matches!(self.constant_value(), Some(value) if is_zero(value))
    }

    pub fn simplify(self) -> SymbolicExpr {
        match self {
            SymbolicExpr::Rational { .. } | SymbolicExpr::DecimalLiteral { .. } => self,
            SymbolicExpr::FunctionReference(name, parameters) => {
                SymbolicExpr::FunctionReference(name, parameters)
            }
            SymbolicExpr::FunctionCall(name, args) => SymbolicExpr::FunctionCall(
                name,
                args.into_iter()
                    .map(SymbolicExpr::simplify)
                    .collect::<Vec<_>>(),
            ),
            SymbolicExpr::Equation(lhs, rhs) => {
                SymbolicExpr::Equation(Box::new(lhs.simplify()), Box::new(rhs.simplify()))
            }
            SymbolicExpr::Derivative {
                expr,
                variable,
                order,
            } => {
                let expr = expr.simplify();
                if order == 0 {
                    return expr;
                }
                if !expr.contains_variable(&variable) {
                    return SymbolicExpr::Constant(0.0);
                }
                if let SymbolicExpr::Derivative {
                    expr: inner,
                    variable: inner_variable,
                    order: inner_order,
                } = expr
                {
                    if inner_variable == variable {
                        return SymbolicExpr::Derivative {
                            expr: inner,
                            variable,
                            order: inner_order.saturating_add(order),
                        };
                    }
                    return SymbolicExpr::Derivative {
                        expr: Box::new(SymbolicExpr::Derivative {
                            expr: inner,
                            variable: inner_variable,
                            order: inner_order,
                        }),
                        variable,
                        order,
                    };
                }
                SymbolicExpr::Derivative {
                    expr: Box::new(expr),
                    variable,
                    order,
                }
            }
            SymbolicExpr::Neg(inner) => {
                let inner = inner.simplify();
                match inner {
                    SymbolicExpr::Constant(value) => SymbolicExpr::Constant(-value),
                    SymbolicExpr::Neg(value) => *value,
                    other => SymbolicExpr::Neg(Box::new(other)),
                }
            }
            SymbolicExpr::Add(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (&lhs, &rhs) {
                    (SymbolicExpr::Constant(a), SymbolicExpr::Constant(b)) if !(a + b).is_nan() => {
                        SymbolicExpr::Constant(a + b)
                    }
                    (
                        SymbolicExpr::Rational {
                            numerator: an,
                            denominator: ad,
                        },
                        SymbolicExpr::Rational {
                            numerator: bn,
                            denominator: bd,
                        },
                    ) => SymbolicExpr::rational(an * bd + bn * ad, ad * bd)
                        .unwrap_or(SymbolicExpr::Add(Box::new(lhs), Box::new(rhs))),
                    (left, right) if left.is_zero_constant() => right.clone(),
                    (left, right) if right.is_zero_constant() => left.clone(),
                    _ => SymbolicExpr::Add(Box::new(lhs), Box::new(rhs)),
                }
            }
            SymbolicExpr::Sub(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (&lhs, &rhs) {
                    (SymbolicExpr::Constant(a), SymbolicExpr::Constant(b)) if !(a - b).is_nan() => {
                        SymbolicExpr::Constant(a - b)
                    }
                    (
                        SymbolicExpr::Rational {
                            numerator: an,
                            denominator: ad,
                        },
                        SymbolicExpr::Rational {
                            numerator: bn,
                            denominator: bd,
                        },
                    ) => SymbolicExpr::rational(an * bd - bn * ad, ad * bd)
                        .unwrap_or(SymbolicExpr::Sub(Box::new(lhs), Box::new(rhs))),
                    (left, right) if right.is_zero_constant() => left.clone(),
                    (left, right) if left.is_zero_constant() => {
                        SymbolicExpr::neg_expr(right.clone())
                    }
                    (left, right) if left == right => SymbolicExpr::Constant(0.0),
                    _ => SymbolicExpr::Sub(Box::new(lhs), Box::new(rhs)),
                }
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (&lhs, &rhs) {
                    (SymbolicExpr::Constant(a), SymbolicExpr::Constant(b)) if !(a * b).is_nan() => {
                        SymbolicExpr::Constant(a * b)
                    }
                    (
                        SymbolicExpr::Rational {
                            numerator: an,
                            denominator: ad,
                        },
                        SymbolicExpr::Rational {
                            numerator: bn,
                            denominator: bd,
                        },
                    ) => SymbolicExpr::rational(an * bn, ad * bd)
                        .unwrap_or(SymbolicExpr::Mul(Box::new(lhs), Box::new(rhs))),
                    (left, right)
                        if left.is_zero_constant() && right.can_eliminate_zero_product_factor() =>
                    {
                        SymbolicExpr::Constant(0.0)
                    }
                    (left, right)
                        if right.is_zero_constant() && left.can_eliminate_zero_product_factor() =>
                    {
                        SymbolicExpr::Constant(0.0)
                    }
                    (left, right) if is_one_expr(left) => right.clone(),
                    (left, right) if is_one_expr(right) => left.clone(),
                    _ => SymbolicExpr::Mul(Box::new(lhs), Box::new(rhs)),
                }
            }
            SymbolicExpr::Div(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (&lhs, &rhs) {
                    (_, right) if is_one_expr(right) => lhs,
                    (left, right) if left == right && !left.is_zero_constant() => {
                        SymbolicExpr::Constant(1.0)
                    }
                    (left, right)
                        if left.is_zero_constant()
                            && !right.is_zero_constant()
                            && !right.has_undefined_constant_subexpression() =>
                    {
                        SymbolicExpr::Constant(0.0)
                    }
                    (SymbolicExpr::Constant(a), SymbolicExpr::Constant(b)) if !is_zero(*b) => {
                        SymbolicExpr::Constant(a / b)
                    }
                    (
                        SymbolicExpr::Rational {
                            numerator: an,
                            denominator: ad,
                        },
                        SymbolicExpr::Rational {
                            numerator: bn,
                            denominator: bd,
                        },
                    ) if !bn.is_zero() => SymbolicExpr::rational(an * bd, ad * bn)
                        .unwrap_or(SymbolicExpr::Div(Box::new(lhs), Box::new(rhs))),
                    _ => SymbolicExpr::Div(Box::new(lhs), Box::new(rhs)),
                }
            }
            SymbolicExpr::Pow(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (&lhs, &rhs) {
                    (left, right)
                        if right.is_zero_constant()
                            && !left.has_undefined_constant_subexpression() =>
                    {
                        SymbolicExpr::Constant(1.0)
                    }
                    (left, right) if is_one_expr(right) => left.clone(),
                    (left, right)
                        if left.is_zero_constant()
                            && matches!(
                                right.numeric_constant_value(),
                                Some(exponent) if exponent.is_sign_positive()
                            ) =>
                    {
                        SymbolicExpr::Constant(0.0)
                    }
                    (left, _) if is_one_expr(left) => SymbolicExpr::Constant(1.0),
                    (SymbolicExpr::Constant(a), SymbolicExpr::Constant(b)) => {
                        let value = a.powf(*b);
                        if !(value.is_nan() || is_zero(*a) && b.is_sign_negative()) {
                            SymbolicExpr::Constant(value)
                        } else {
                            SymbolicExpr::Pow(Box::new(lhs), Box::new(rhs))
                        }
                    }
                    _ => SymbolicExpr::Pow(Box::new(lhs), Box::new(rhs)),
                }
            }
            SymbolicExpr::Function(function, inner) => {
                let inner = inner.simplify();
                if let Some(value) = inner
                    .constant_value()
                    .and_then(|value| function.apply_constant(value))
                {
                    SymbolicExpr::Constant(value)
                } else {
                    SymbolicExpr::Function(function, Box::new(inner))
                }
            }
            other => other,
        }
    }
}

fn is_zero(value: f64) -> bool {
    value.abs() <= EPSILON
}

fn finite_or_infinite(value: f64) -> Option<f64> {
    (!value.is_nan()).then_some(value)
}

fn is_one_expr(expr: &SymbolicExpr) -> bool {
    matches!(expr.constant_value(), Some(value) if (value - 1.0).abs() <= EPSILON)
}

impl fmt::Display for SymbolicExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_expr(self, f, 0, false)
    }
}

fn precedence(expr: &SymbolicExpr) -> u8 {
    match expr {
        SymbolicExpr::Constant(_)
        | SymbolicExpr::Rational { .. }
        | SymbolicExpr::DecimalLiteral { .. }
        | SymbolicExpr::Variable(_)
        | SymbolicExpr::FunctionReference(_, _)
        | SymbolicExpr::FunctionCall(_, _)
        | SymbolicExpr::Derivative { .. }
        | SymbolicExpr::Function(_, _) => 5,
        SymbolicExpr::Neg(_) => 4,
        SymbolicExpr::Pow(_, _) => 3,
        SymbolicExpr::Mul(_, _) | SymbolicExpr::Div(_, _) => 2,
        SymbolicExpr::Add(_, _) | SymbolicExpr::Sub(_, _) => 1,
        SymbolicExpr::Equation(_, _) => 0,
    }
}

fn fmt_expr(
    expr: &SymbolicExpr,
    f: &mut fmt::Formatter<'_>,
    parent_precedence: u8,
    right_child: bool,
) -> fmt::Result {
    let own_precedence = precedence(expr);
    let needs_parens =
        own_precedence < parent_precedence || (right_child && own_precedence == parent_precedence);
    if needs_parens {
        write!(f, "(")?;
    }
    match expr {
        SymbolicExpr::Constant(value) => write!(f, "{}", format_number(*value))?,
        SymbolicExpr::Rational {
            numerator,
            denominator,
        } if denominator.is_one() => write!(f, "{numerator}")?,
        SymbolicExpr::Rational {
            numerator,
            denominator,
        } => write!(f, "{numerator}/{denominator}")?,
        SymbolicExpr::DecimalLiteral { text, .. } => write!(f, "{text}")?,
        SymbolicExpr::Variable(name) => write!(f, "{name}")?,
        SymbolicExpr::FunctionReference(name, parameters) => {
            write!(f, "{name}(")?;
            for (idx, parameter) in parameters.iter().enumerate() {
                if idx > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{parameter}")?;
            }
            write!(f, ")")?;
        }
        SymbolicExpr::FunctionCall(name, args) => {
            write!(f, "{name}(")?;
            for (idx, arg) in args.iter().enumerate() {
                if idx > 0 {
                    write!(f, ", ")?;
                }
                fmt_expr(arg, f, 0, false)?;
            }
            write!(f, ")")?;
        }
        SymbolicExpr::Equation(lhs, rhs) => {
            fmt_expr(lhs, f, own_precedence, false)?;
            write!(f, " == ")?;
            fmt_expr(rhs, f, own_precedence, true)?;
        }
        SymbolicExpr::Derivative {
            expr,
            variable,
            order,
        } => {
            write!(f, "diff(")?;
            fmt_expr(expr, f, 0, false)?;
            write!(f, ", {variable}")?;
            if *order != 1 {
                write!(f, ", {order}")?;
            }
            write!(f, ")")?;
        }
        SymbolicExpr::Neg(inner) => {
            write!(f, "-")?;
            fmt_expr(inner, f, own_precedence, false)?;
        }
        SymbolicExpr::Add(lhs, rhs) => {
            fmt_expr(lhs, f, own_precedence, false)?;
            write!(f, " + ")?;
            fmt_expr(rhs, f, own_precedence, true)?;
        }
        SymbolicExpr::Sub(lhs, rhs) => {
            fmt_expr(lhs, f, own_precedence, false)?;
            write!(f, " - ")?;
            fmt_expr(rhs, f, own_precedence, true)?;
        }
        SymbolicExpr::Mul(lhs, rhs) => {
            fmt_expr(lhs, f, own_precedence, false)?;
            write!(f, "*")?;
            fmt_expr(rhs, f, own_precedence, true)?;
        }
        SymbolicExpr::Div(lhs, rhs) => {
            fmt_expr(lhs, f, own_precedence, false)?;
            write!(f, "/")?;
            fmt_expr(rhs, f, own_precedence, true)?;
        }
        SymbolicExpr::Pow(lhs, rhs) => {
            let lhs_precedence = if matches!(lhs.as_ref(), SymbolicExpr::Neg(_)) {
                precedence(lhs) + 1
            } else {
                own_precedence
            };
            fmt_expr(lhs, f, lhs_precedence, false)?;
            write!(f, "^")?;
            fmt_expr(rhs, f, own_precedence, true)?;
        }
        SymbolicExpr::Function(SymbolicFunction::DiracDerivative(order), inner) => {
            write!(f, "dirac({order}, ")?;
            fmt_expr(inner, f, 0, false)?;
            write!(f, ")")?;
        }
        SymbolicExpr::Function(function, inner) => {
            write!(f, "{}(", function.name())?;
            fmt_expr(inner, f, 0, false)?;
            write!(f, ")")?;
        }
    }
    if needs_parens {
        write!(f, ")")?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicDeclaration {
    pub name: String,
    pub parameters: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicDeclarationError {
    Empty,
    InvalidName,
    InvalidParameter,
    DuplicateParameter,
    EmptyParameterList,
    UnexpectedSyntax,
}

impl fmt::Display for SymbolicDeclarationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolicDeclarationError::Empty => write!(f, "empty symbolic declaration"),
            SymbolicDeclarationError::InvalidName => write!(f, "invalid symbolic name"),
            SymbolicDeclarationError::InvalidParameter => write!(f, "invalid symbolic parameter"),
            SymbolicDeclarationError::DuplicateParameter => {
                write!(f, "duplicate symbolic function parameter")
            }
            SymbolicDeclarationError::EmptyParameterList => {
                write!(
                    f,
                    "symbolic function declaration requires at least one parameter"
                )
            }
            SymbolicDeclarationError::UnexpectedSyntax => {
                write!(f, "invalid symbolic function declaration syntax")
            }
        }
    }
}

pub fn parse_symbolic_declaration(
    text: &str,
) -> Result<SymbolicDeclaration, SymbolicDeclarationError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(SymbolicDeclarationError::Empty);
    }

    let Some(open) = trimmed.find('(') else {
        if is_valid_symbolic_identifier(trimmed) {
            return Ok(SymbolicDeclaration {
                name: trimmed.to_string(),
                parameters: Vec::new(),
            });
        }
        return Err(SymbolicDeclarationError::InvalidName);
    };

    if !trimmed.ends_with(')') {
        return Err(SymbolicDeclarationError::UnexpectedSyntax);
    }
    let inner = &trimmed[open + 1..trimmed.len() - 1];
    if inner.contains('(') || inner.contains(')') {
        return Err(SymbolicDeclarationError::UnexpectedSyntax);
    }

    let name = trimmed[..open].trim();
    if !is_valid_symbolic_identifier(name) {
        return Err(SymbolicDeclarationError::InvalidName);
    }

    if inner.trim().is_empty() {
        return Err(SymbolicDeclarationError::EmptyParameterList);
    }

    let mut parameters = Vec::new();
    for parameter in inner.split(',') {
        let parameter = parameter.trim();
        if !is_valid_symbolic_identifier(parameter) {
            return Err(SymbolicDeclarationError::InvalidParameter);
        }
        if parameters.iter().any(|existing| existing == parameter) {
            return Err(SymbolicDeclarationError::DuplicateParameter);
        }
        parameters.push(parameter.to_string());
    }

    Ok(SymbolicDeclaration {
        name: name.to_string(),
        parameters,
    })
}

pub fn is_valid_symbolic_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphabetic() && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

pub fn symbolic_declaration_tokens(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = None;
    let mut paren_depth = 0usize;

    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() && paren_depth == 0 {
            if let Some(token_start) = start.take() {
                tokens.push(&text[token_start..idx]);
            }
            continue;
        }

        if start.is_none() {
            start = Some(idx);
        }

        match ch {
            '(' => paren_depth = paren_depth.saturating_add(1),
            ')' => paren_depth = paren_depth.saturating_sub(1),
            _ => {}
        }
    }

    if let Some(token_start) = start {
        tokens.push(&text[token_start..]);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::{
        parse_symbolic_declaration, symbolic_declaration_tokens, SymbolicDeclarationError,
        SymbolicExpr, SymbolicFunction,
    };

    #[test]
    fn substitutes_and_simplifies_symbols() {
        let x = SymbolicExpr::variable("x");
        let h = SymbolicExpr::variable("h");
        let expr = SymbolicExpr::function(
            SymbolicFunction::Cos,
            SymbolicExpr::add_expr(x.clone(), h.clone()),
        );

        assert_eq!(
            expr.substitute("h", &SymbolicExpr::constant(0.0))
                .to_string(),
            "cos(x)"
        );
    }

    #[test]
    fn differentiates_trig_expression() {
        let x = SymbolicExpr::variable("x");
        let expr = SymbolicExpr::function(SymbolicFunction::Sin, x);

        assert_eq!(expr.derivative("x").to_string(), "cos(x)");
    }

    #[test]
    fn formats_named_symbolic_function_and_derivative() {
        let expr = SymbolicExpr::function_reference("Y", vec!["X".to_string()]);

        assert_eq!(expr.to_string(), "Y(X)");
        assert_eq!(expr.derivative("X").to_string(), "diff(Y(X), X)");
    }

    #[test]
    fn substitutes_symbolic_function_reference_parameters() {
        let expr = SymbolicExpr::function_reference("Y", vec!["X".to_string()]);

        assert_eq!(
            expr.substitute("X", &SymbolicExpr::constant(0.0))
                .to_string(),
            "Y(0)"
        );
    }

    #[test]
    fn formats_symbolic_equation() {
        let lhs = SymbolicExpr::function_call("Y", vec![SymbolicExpr::constant(0.0)]);
        let rhs = SymbolicExpr::constant(0.0);

        assert_eq!(SymbolicExpr::equation(lhs, rhs).to_string(), "Y(0) == 0");
    }

    #[test]
    fn parses_symbolic_function_declarations() {
        let decl = parse_symbolic_declaration("Y(X)").expect("declaration");

        assert_eq!(decl.name, "Y");
        assert_eq!(decl.parameters, vec!["X"]);

        let decl = parse_symbolic_declaration("f(x, y)").expect("declaration");
        assert_eq!(decl.name, "f");
        assert_eq!(decl.parameters, vec!["x", "y"]);
    }

    #[test]
    fn rejects_malformed_symbolic_function_declarations() {
        assert_eq!(
            parse_symbolic_declaration("Y(").unwrap_err(),
            SymbolicDeclarationError::UnexpectedSyntax
        );
        assert_eq!(
            parse_symbolic_declaration("f()").unwrap_err(),
            SymbolicDeclarationError::EmptyParameterList
        );
        assert_eq!(
            parse_symbolic_declaration("f(x,x)").unwrap_err(),
            SymbolicDeclarationError::DuplicateParameter
        );
    }

    #[test]
    fn tokenizes_symbolic_declarations_without_splitting_parameter_lists() {
        assert_eq!(
            symbolic_declaration_tokens("x f(a, b) real g(t)"),
            vec!["x", "f(a, b)", "real", "g(t)"]
        );
    }

    #[test]
    fn formats_negative_functions_without_extra_parens() {
        let x = SymbolicExpr::variable("x");
        let expr = SymbolicExpr::neg_expr(SymbolicExpr::function(SymbolicFunction::Sin, x));

        assert_eq!(expr.to_string(), "-sin(x)");
    }

    #[test]
    fn preserves_undefined_subexpressions_under_zero_products() {
        let expr = SymbolicExpr::mul_expr(
            SymbolicExpr::constant(0.0),
            SymbolicExpr::div_expr(SymbolicExpr::constant(1.0), SymbolicExpr::constant(0.0)),
        );

        assert!(matches!(expr, SymbolicExpr::Mul(_, _)));
        assert!(expr.has_undefined_constant_subexpression());
    }

    #[test]
    fn preserves_negative_power_of_zero_as_singular() {
        let expr =
            SymbolicExpr::pow_expr(SymbolicExpr::constant(0.0), SymbolicExpr::constant(-1.0));

        assert!(matches!(expr, SymbolicExpr::Pow(_, _)));
        assert!(expr.has_undefined_constant_subexpression());
    }

    #[test]
    fn formats_negative_power_bases_with_parentheses() {
        let x = SymbolicExpr::variable("x");
        let expr = SymbolicExpr::pow_expr(SymbolicExpr::neg_expr(x), SymbolicExpr::constant(2.0));

        assert_eq!(expr.to_string(), "(-x)^2");
    }

    #[test]
    fn exposes_finite_and_infinite_symbolic_constants() {
        assert_eq!(
            SymbolicExpr::constant(4.5).numeric_constant_value(),
            Some(4.5)
        );
        assert_eq!(
            SymbolicExpr::constant(f64::INFINITY).numeric_constant_value(),
            Some(f64::INFINITY)
        );
        assert_eq!(
            SymbolicExpr::constant(f64::NAN).numeric_constant_value(),
            None
        );
    }

    #[test]
    fn oversized_rational_simplification_does_not_panic() {
        let lhs = SymbolicExpr::rational(i128::MAX, i128::MAX - 2).expect("lhs");
        let rhs = SymbolicExpr::rational(i128::MAX - 4, i128::MAX - 6).expect("rhs");
        let sum = SymbolicExpr::add_expr(lhs, rhs);

        assert!(matches!(sum, SymbolicExpr::Rational { .. }));
    }

    #[test]
    fn symbolic_heaviside_simplifies_constants_and_formats_variables() {
        let x = SymbolicExpr::variable("x");
        let expr = SymbolicExpr::function(SymbolicFunction::Heaviside, x);
        assert_eq!(expr.to_string(), "heaviside(x)");
        assert_eq!(
            SymbolicExpr::function(SymbolicFunction::Heaviside, SymbolicExpr::constant(-2.0)),
            SymbolicExpr::constant(0.0)
        );
        assert_eq!(
            SymbolicExpr::function(SymbolicFunction::Heaviside, SymbolicExpr::constant(-0.0)),
            SymbolicExpr::constant(0.5)
        );
        assert_eq!(
            SymbolicExpr::function(SymbolicFunction::Heaviside, SymbolicExpr::constant(3.0)),
            SymbolicExpr::constant(1.0)
        );
    }

    #[test]
    fn symbolic_heaviside_derivative_is_formal_dirac() {
        let x = SymbolicExpr::variable("x");
        let expr = SymbolicExpr::function(SymbolicFunction::Heaviside, x);

        assert_eq!(expr.derivative("x").to_string(), "dirac(x)");
    }

    #[test]
    fn symbolic_dirac_derivatives_preserve_order() {
        let x = SymbolicExpr::variable("x");
        let expr = SymbolicExpr::function(SymbolicFunction::Dirac, x);

        assert_eq!(expr.derivative("x").to_string(), "dirac(1, x)");
        assert_eq!(
            expr.derivative("x").derivative("x").to_string(),
            "dirac(2, x)"
        );
    }
}
