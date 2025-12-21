//! Symbolic collect builtin
//!
//! Provides MATLAB-compatible `collect` for symbolic expressions.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{SymExpr, SymExprKind};
use std::collections::HashMap;

/// Collect coefficients of a variable in a polynomial
///
/// `collect(expr, x)` rewrites `expr` as a polynomial in `x`:
/// `a_n*x^n + ... + a_1*x + a_0`
///
/// where coefficients may contain other variables.
#[runtime_builtin(
    name = "collect",
    category = "symbolic",
    summary = "Collect coefficients of a variable in a symbolic expression.",
    keywords = "collect,polynomial,symbolic,coefficients"
)]
fn collect_builtin(expr: Value, var: Value) -> Result<Value, String> {
    let sym_expr = match expr {
        Value::Symbolic(e) => e,
        _ => return Err("collect: first argument must be a symbolic expression".to_string()),
    };

    let var_name = match var {
        Value::Symbolic(v) => {
            if let Some(s) = v.as_var() {
                s.name.clone()
            } else {
                return Err("collect: second argument must be a symbolic variable".to_string());
            }
        }
        Value::String(s) => s.clone(),
        Value::CharArray(ca) => ca.to_string(),
        _ => {
            return Err(
                "collect: second argument must be a symbolic variable or string".to_string(),
            )
        }
    };

    let var_expr = SymExpr::var(&var_name);
    let collected = collect_expr(&sym_expr, &var_expr);
    Ok(Value::Symbolic(collected))
}

/// Collect terms with respect to a variable
fn collect_expr(expr: &SymExpr, var: &SymExpr) -> SymExpr {
    // First expand the expression to get all terms
    let normalizer = runmat_symbolic::StagedNormalizer::aggressive();
    let (expanded, _) = normalizer.normalize(expr.clone());

    // Collect powers of the variable
    let mut powers: HashMap<i64, Vec<SymExpr>> = HashMap::new();

    collect_powers(&expanded, var, &mut powers);

    // Build the result: sum of coeff * var^power
    let mut result_terms = Vec::new();

    // Sort powers in descending order
    let mut sorted_powers: Vec<_> = powers.keys().copied().collect();
    sorted_powers.sort_by(|a, b| b.cmp(a));

    for power in sorted_powers {
        let coeffs = powers.get(&power).unwrap();
        let coeff_sum = if coeffs.len() == 1 {
            coeffs[0].clone()
        } else {
            SymExpr::add(coeffs.clone())
        };

        // Simplify the coefficient
        let normalizer = runmat_symbolic::StagedNormalizer::default_pipeline();
        let (simplified_coeff, _) = normalizer.normalize(coeff_sum);

        let term = match power {
            0 => simplified_coeff,
            1 => {
                if simplified_coeff.is_one() {
                    var.clone()
                } else {
                    SymExpr::mul(vec![simplified_coeff, var.clone()])
                }
            }
            _ => {
                let var_pow = SymExpr::pow(var.clone(), SymExpr::int(power));
                if simplified_coeff.is_one() {
                    var_pow
                } else {
                    SymExpr::mul(vec![simplified_coeff, var_pow])
                }
            }
        };

        result_terms.push(term);
    }

    if result_terms.is_empty() {
        SymExpr::int(0)
    } else {
        SymExpr::add(result_terms)
    }
}

/// Collect powers of a variable from an expression
fn collect_powers(expr: &SymExpr, var: &SymExpr, powers: &mut HashMap<i64, Vec<SymExpr>>) {
    match expr.kind.as_ref() {
        SymExprKind::Add(terms) => {
            for term in terms {
                collect_powers(term, var, powers);
            }
        }
        _ => {
            // Extract the power of var and the coefficient
            let (power, coeff) = extract_var_power(expr, var);
            powers.entry(power).or_default().push(coeff);
        }
    }
}

/// Extract the power of a variable and the remaining coefficient
fn extract_var_power(expr: &SymExpr, var: &SymExpr) -> (i64, SymExpr) {
    match expr.kind.as_ref() {
        // Constant: power 0, coefficient is the constant
        SymExprKind::Num(_) => (0, expr.clone()),

        // Variable: power 1 if matches, otherwise power 0
        SymExprKind::Var(_) => {
            if expr == var {
                (1, SymExpr::int(1))
            } else {
                (0, expr.clone())
            }
        }

        // Power: extract exponent if base is the variable
        SymExprKind::Pow(base, exp) => {
            if base.as_ref() == var {
                if let Some(c) = exp.as_coeff() {
                    if c.is_integer() {
                        return (c.to_f64() as i64, SymExpr::int(1));
                    }
                }
            }
            // Check if base contains the variable
            if !base
                .free_vars()
                .iter()
                .any(|s| &SymExpr::symbol(s.clone()) == var)
                && !exp
                    .free_vars()
                    .iter()
                    .any(|s| &SymExpr::symbol(s.clone()) == var)
            {
                (0, expr.clone())
            } else {
                // Complex case: treat as power 0 with full expression as coefficient
                (0, expr.clone())
            }
        }

        // Product: combine powers
        SymExprKind::Mul(factors) => {
            let mut total_power = 0i64;
            let mut coeff_factors = Vec::new();

            for f in factors {
                let (pow, coeff) = extract_var_power(f, var);
                total_power += pow;
                if !coeff.is_one() {
                    coeff_factors.push(coeff);
                }
            }

            let coeff = if coeff_factors.is_empty() {
                SymExpr::int(1)
            } else {
                SymExpr::mul(coeff_factors)
            };

            (total_power, coeff)
        }

        // Negation
        SymExprKind::Neg(inner) => {
            let (power, coeff) = extract_var_power(inner, var);
            (power, SymExpr::neg(coeff))
        }

        // Function: power 0, whole expression is coefficient
        SymExprKind::Func(_, _) => (0, expr.clone()),

        // Addition shouldn't happen here (handled at top level)
        SymExprKind::Add(_) => (0, expr.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_simple() {
        // x + 2*x should become 3*x
        let x = SymExpr::var("x");
        let expr = SymExpr::add(vec![
            x.clone(),
            SymExpr::mul(vec![SymExpr::int(2), x.clone()]),
        ]);

        let result = collect_expr(&expr, &x);
        // Result should have power 1 with coefficient 3
        println!("Result: {}", result);
    }

    #[test]
    fn test_collect_polynomial() {
        // x^2 + x + 1
        let x = SymExpr::var("x");
        let expr = SymExpr::add(vec![
            SymExpr::pow(x.clone(), SymExpr::int(2)),
            x.clone(),
            SymExpr::int(1),
        ]);

        let result = collect_expr(&expr, &x);
        println!("Collected: {}", result);
    }

    #[test]
    fn test_collect_with_other_vars() {
        // a*x + b*x = (a + b)*x
        let x = SymExpr::var("x");
        let a = SymExpr::var("a");
        let b = SymExpr::var("b");

        let expr = SymExpr::add(vec![
            SymExpr::mul(vec![a, x.clone()]),
            SymExpr::mul(vec![b, x.clone()]),
        ]);

        let result = collect_expr(&expr, &x);
        println!("Collected a*x + b*x: {}", result);
        assert!(result.is_mul() || result.is_add());
    }
}
