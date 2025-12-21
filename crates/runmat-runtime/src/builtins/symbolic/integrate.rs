//! Symbolic integration builtin
//!
//! Provides MATLAB-compatible `int` for symbolic integration.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{Coefficient, SymExpr, SymExprKind};

/// Compute the indefinite integral of a symbolic expression
///
/// `int(expr, x)` computes the indefinite integral of `expr` with respect to `x`
/// `int(expr)` integrates with respect to the single variable in `expr`
///
/// Currently supports:
/// - Polynomials: ∫x^n dx = x^(n+1)/(n+1)
/// - Exponentials: ∫e^x dx = e^x
/// - Trigonometric: ∫sin(x) dx = -cos(x), ∫cos(x) dx = sin(x)
/// - Logarithms: ∫1/x dx = log(x)
/// - Linear substitution: ∫f(ax+b) dx = F(ax+b)/a
#[runtime_builtin(
    name = "int",
    category = "symbolic",
    summary = "Compute the indefinite integral of a symbolic expression.",
    keywords = "int,integrate,integral,symbolic,antiderivative"
)]
fn int_builtin(expr: Value, rest: Vec<Value>) -> Result<Value, String> {
    let sym_expr = match expr {
        Value::Symbolic(e) => e,
        Value::Num(n) => SymExpr::float(n),
        Value::Int(i) => SymExpr::int(i.to_i64()),
        _ => return Err("int: first argument must be a symbolic expression".to_string()),
    };

    // Determine which variable to integrate with respect to
    let var_name = if rest.is_empty() {
        // Find the single variable
        let vars = sym_expr.free_vars();
        if vars.is_empty() {
            // Constant integration: ∫c dx = c*x (but we don't know x)
            return Err(
                "int: expression has no variables, specify integration variable".to_string(),
            );
        }
        if vars.len() > 1 {
            return Err(format!(
                "int: expression has multiple variables ({:?}), specify which to integrate",
                vars.iter().map(|s| &s.name).collect::<Vec<_>>()
            ));
        }
        vars.into_iter().next().unwrap().name
    } else {
        match &rest[0] {
            Value::Symbolic(v) => {
                if let Some(s) = v.as_var() {
                    s.name.clone()
                } else {
                    return Err("int: second argument must be a symbolic variable".to_string());
                }
            }
            Value::String(s) => s.clone(),
            Value::CharArray(ca) => ca.to_string(),
            _ => {
                return Err("int: second argument must be a symbolic variable or string".to_string())
            }
        }
    };

    let var = SymExpr::var(&var_name);
    let integral = integrate(&sym_expr, &var)?;

    // Normalize the result
    let normalizer = runmat_symbolic::StagedNormalizer::default_pipeline();
    let (simplified, _) = normalizer.normalize(integral);

    Ok(Value::Symbolic(simplified))
}

/// Integrate an expression with respect to a variable
fn integrate(expr: &SymExpr, var: &SymExpr) -> Result<SymExpr, String> {
    match expr.kind.as_ref() {
        // ∫c dx = c*x
        SymExprKind::Num(_) => Ok(SymExpr::mul(vec![expr.clone(), var.clone()])),

        // ∫x dx = x^2/2
        SymExprKind::Var(_) if expr == var => Ok(SymExpr::mul(vec![
            SymExpr::num(Coefficient::rational(1, 2)),
            SymExpr::pow(var.clone(), SymExpr::int(2)),
        ])),

        // ∫y dx = y*x (y is different variable)
        SymExprKind::Var(_) => Ok(SymExpr::mul(vec![expr.clone(), var.clone()])),

        // ∫x^n dx = x^(n+1)/(n+1) for n ≠ -1
        SymExprKind::Pow(base, exp) if base.as_ref() == var => {
            // Check if exponent is constant
            if !exp
                .free_vars()
                .iter()
                .any(|s| &SymExpr::symbol(s.clone()) == var)
            {
                // Check for x^(-1) = 1/x -> log(x)
                if let Some(c) = exp.as_coeff() {
                    if c.to_f64() == -1.0 {
                        return Ok(SymExpr::log(var.clone()));
                    }
                }

                // x^(n+1) / (n+1)
                let n_plus_1 = SymExpr::add(vec![exp.as_ref().clone(), SymExpr::int(1)]);
                let new_power = SymExpr::pow(var.clone(), n_plus_1.clone());
                Ok(SymExpr::mul(vec![
                    new_power,
                    SymExpr::pow(n_plus_1, SymExpr::int(-1)),
                ]))
            } else {
                Err("int: cannot integrate x^f(x)".to_string())
            }
        }

        // ∫(a + b + ...) dx = ∫a dx + ∫b dx + ...
        SymExprKind::Add(terms) => {
            let integrals: Result<Vec<_>, _> = terms.iter().map(|t| integrate(t, var)).collect();
            Ok(SymExpr::add(integrals?))
        }

        // ∫c*f(x) dx = c * ∫f(x) dx
        SymExprKind::Mul(factors) => {
            // Separate constant and variable-dependent factors
            let mut const_factors = Vec::new();
            let mut var_factors = Vec::new();

            for f in factors {
                if f.free_vars()
                    .iter()
                    .any(|s| &SymExpr::symbol(s.clone()) == var)
                {
                    var_factors.push(f.clone());
                } else {
                    const_factors.push(f.clone());
                }
            }

            if var_factors.is_empty() {
                // All constant: ∫c dx = c*x
                Ok(SymExpr::mul(vec![expr.clone(), var.clone()]))
            } else if var_factors.len() == 1 {
                // c * f(x): integrate f(x) and multiply by c
                let integral = integrate(&var_factors[0], var)?;
                if const_factors.is_empty() {
                    Ok(integral)
                } else {
                    const_factors.push(integral);
                    Ok(SymExpr::mul(const_factors))
                }
            } else {
                // Multiple variable factors - try to handle simple cases
                // Check if it's a polynomial term like 2*x^3
                let combined = SymExpr::mul(var_factors);
                if let Some(result) = try_integrate_product(&combined, var) {
                    if const_factors.is_empty() {
                        Ok(result)
                    } else {
                        const_factors.push(result);
                        Ok(SymExpr::mul(const_factors))
                    }
                } else {
                    Err("int: cannot integrate this product".to_string())
                }
            }
        }

        // ∫-f(x) dx = -∫f(x) dx
        SymExprKind::Neg(inner) => {
            let integral = integrate(inner, var)?;
            Ok(SymExpr::neg(integral))
        }

        // Standard function integrals
        SymExprKind::Func(name, args) if args.len() == 1 => {
            let arg = &args[0];

            // Check if argument is just the variable
            if arg == var {
                match name.as_str() {
                    // ∫sin(x) dx = -cos(x)
                    "sin" => Ok(SymExpr::neg(SymExpr::cos(var.clone()))),

                    // ∫cos(x) dx = sin(x)
                    "cos" => Ok(SymExpr::sin(var.clone())),

                    // ∫exp(x) dx = exp(x)
                    "exp" => Ok(SymExpr::exp(var.clone())),

                    // ∫sqrt(x) dx = ∫x^(1/2) dx = x^(3/2) / (3/2) = (2/3)*x^(3/2)
                    "sqrt" => Ok(SymExpr::mul(vec![
                        SymExpr::num(Coefficient::rational(2, 3)),
                        SymExpr::pow(var.clone(), SymExpr::num(Coefficient::rational(3, 2))),
                    ])),

                    _ => Err(format!("int: cannot integrate {}(x)", name)),
                }
            } else if let Some((a, _b)) = is_linear_in_var(arg, var) {
                // Linear substitution: ∫f(ax+b) dx = F(ax+b)/a
                let inner_integral = match name.as_str() {
                    "sin" => SymExpr::neg(SymExpr::cos(arg.clone())),
                    "cos" => SymExpr::sin(arg.clone()),
                    "exp" => SymExpr::exp(arg.clone()),
                    _ => return Err(format!("int: cannot integrate {}(ax+b)", name)),
                };

                // Divide by 'a'
                Ok(SymExpr::mul(vec![
                    inner_integral,
                    SymExpr::pow(a, SymExpr::int(-1)),
                ]))
            } else {
                Err(format!("int: cannot integrate {}(...)", name))
            }
        }

        _ => Err("int: cannot integrate this expression".to_string()),
    }
}

/// Try to integrate a product as a polynomial term
fn try_integrate_product(expr: &SymExpr, var: &SymExpr) -> Option<SymExpr> {
    // Check if this is x * x^n or similar
    match expr.kind.as_ref() {
        SymExprKind::Mul(factors) => {
            // Count powers of var
            let mut total_power = 0i64;
            let mut other_factors = Vec::new();

            for f in factors {
                if f == var {
                    total_power += 1;
                } else if let SymExprKind::Pow(base, exp) = f.kind.as_ref() {
                    if base.as_ref() == var {
                        if let Some(c) = exp.as_coeff() {
                            if c.is_integer() {
                                total_power += c.to_f64() as i64;
                                continue;
                            }
                        }
                        return None;
                    } else {
                        other_factors.push(f.clone());
                    }
                } else if !f
                    .free_vars()
                    .iter()
                    .any(|s| &SymExpr::symbol(s.clone()) == var)
                {
                    other_factors.push(f.clone());
                } else {
                    return None;
                }
            }

            if total_power == -1 {
                // ∫1/x dx = log(x)
                let result = SymExpr::log(var.clone());
                if other_factors.is_empty() {
                    Some(result)
                } else {
                    other_factors.push(result);
                    Some(SymExpr::mul(other_factors))
                }
            } else {
                // ∫x^n dx = x^(n+1)/(n+1)
                let n_plus_1 = total_power + 1;
                let new_power = SymExpr::pow(var.clone(), SymExpr::int(n_plus_1));
                let coeff = SymExpr::num(Coefficient::rational(1, n_plus_1));

                let result = SymExpr::mul(vec![coeff, new_power]);
                if other_factors.is_empty() {
                    Some(result)
                } else {
                    other_factors.push(result);
                    Some(SymExpr::mul(other_factors))
                }
            }
        }
        _ => None,
    }
}

/// Check if expression is linear in var: a*x + b
/// Returns (a, b) if so
fn is_linear_in_var(expr: &SymExpr, var: &SymExpr) -> Option<(SymExpr, SymExpr)> {
    match expr.kind.as_ref() {
        SymExprKind::Add(terms) => {
            let mut a = SymExpr::int(0);
            let mut b = SymExpr::int(0);

            for term in terms {
                if let SymExprKind::Mul(factors) = term.kind.as_ref() {
                    let var_idx = factors.iter().position(|f| f == var);
                    if let Some(idx) = var_idx {
                        // This is a*x term
                        let remaining: Vec<_> = factors
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != idx)
                            .map(|(_, f)| f.clone())
                            .collect();
                        let coeff = if remaining.is_empty() {
                            SymExpr::int(1)
                        } else {
                            SymExpr::mul(remaining)
                        };
                        a = a + coeff;
                    } else if !term
                        .free_vars()
                        .iter()
                        .any(|s| &SymExpr::symbol(s.clone()) == var)
                    {
                        b = b + term.clone();
                    } else {
                        return None; // Non-linear term
                    }
                } else if term == var {
                    a = a + SymExpr::int(1);
                } else if !term
                    .free_vars()
                    .iter()
                    .any(|s| &SymExpr::symbol(s.clone()) == var)
                {
                    b = b + term.clone();
                } else {
                    return None;
                }
            }

            if !a.is_zero() {
                Some((a, b))
            } else {
                None
            }
        }
        SymExprKind::Mul(factors) => {
            // a*x
            let var_idx = factors.iter().position(|f| f == var)?;
            let remaining: Vec<_> = factors
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != var_idx)
                .map(|(_, f)| f.clone())
                .collect();

            // Check remaining has no var
            for r in &remaining {
                if r.free_vars()
                    .iter()
                    .any(|s| &SymExpr::symbol(s.clone()) == var)
                {
                    return None;
                }
            }

            let a = if remaining.is_empty() {
                SymExpr::int(1)
            } else {
                SymExpr::mul(remaining)
            };

            Some((a, SymExpr::int(0)))
        }
        SymExprKind::Var(_) if expr == var => Some((SymExpr::int(1), SymExpr::int(0))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_constant() {
        // ∫5 dx = 5x
        let x = SymExpr::var("x");
        let expr = SymExpr::int(5);

        let result = integrate(&expr, &x).unwrap();
        println!("∫5 dx = {}", result);
        assert!(result.is_mul());
    }

    #[test]
    fn test_integrate_x() {
        // ∫x dx = x^2/2
        let x = SymExpr::var("x");

        let result = integrate(&x, &x).unwrap();
        println!("∫x dx = {}", result);
    }

    #[test]
    fn test_integrate_x_squared() {
        // ∫x^2 dx = x^3/3
        let x = SymExpr::var("x");
        let expr = SymExpr::pow(x.clone(), SymExpr::int(2));

        let result = integrate(&expr, &x).unwrap();
        println!("∫x^2 dx = {}", result);
    }

    #[test]
    fn test_integrate_sin() {
        // ∫sin(x) dx = -cos(x)
        let x = SymExpr::var("x");
        let expr = SymExpr::sin(x.clone());

        let result = integrate(&expr, &x).unwrap();
        println!("∫sin(x) dx = {}", result);
    }

    #[test]
    fn test_integrate_polynomial() {
        // ∫(x^2 + 2x + 1) dx = x^3/3 + x^2 + x
        let x = SymExpr::var("x");
        let expr = SymExpr::add(vec![
            SymExpr::pow(x.clone(), SymExpr::int(2)),
            SymExpr::mul(vec![SymExpr::int(2), x.clone()]),
            SymExpr::int(1),
        ]);

        let result = integrate(&expr, &x).unwrap();
        println!("∫(x^2 + 2x + 1) dx = {}", result);
    }
}
