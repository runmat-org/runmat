//! Symbolic factorization builtin
//!
//! Provides MATLAB-compatible `factor` for symbolic expressions.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{SymExpr, SymExprKind};

/// Factor a symbolic expression
///
/// Attempts to factor polynomials into products of simpler terms.
/// Currently supports:
/// - Difference of squares: a^2 - b^2 = (a+b)(a-b)
/// - Common factor extraction: ax + ay = a(x+y)
/// - Simple quadratic factoring
#[runtime_builtin(
    name = "factor",
    category = "symbolic",
    summary = "Factor a symbolic expression into a product of simpler terms.",
    keywords = "factor,factorize,symbolic,polynomial"
)]
fn factor_builtin(expr: Value) -> Result<Value, String> {
    match expr {
        Value::Symbolic(sym_expr) => {
            let factored = factor_expr(&sym_expr);
            Ok(Value::Symbolic(factored))
        }
        Value::Num(n) => {
            // Factor integers
            if n == n.floor() && n.abs() < 1e15 {
                let factors = factor_integer(n as i64);
                if factors.len() == 1 {
                    Ok(Value::Num(n))
                } else {
                    // Return as symbolic product
                    let sym_factors: Vec<_> = factors.iter().map(|&f| SymExpr::int(f)).collect();
                    Ok(Value::Symbolic(SymExpr::mul(sym_factors)))
                }
            } else {
                Ok(Value::Num(n))
            }
        }
        Value::Int(i) => {
            let n = i.to_i64();
            let factors = factor_integer(n);
            if factors.len() == 1 {
                Ok(Value::Int(i))
            } else {
                let sym_factors: Vec<_> = factors.iter().map(|&f| SymExpr::int(f)).collect();
                Ok(Value::Symbolic(SymExpr::mul(sym_factors)))
            }
        }
        _ => Err("factor: expected symbolic expression or number".to_string()),
    }
}

/// Factor a symbolic expression
fn factor_expr(expr: &SymExpr) -> SymExpr {
    // First, try specific factoring patterns
    if let Some(factored) = try_difference_of_squares(expr) {
        return factored;
    }

    if let Some(factored) = try_common_factor(expr) {
        return factored;
    }

    if let Some(factored) = try_quadratic_factor(expr) {
        return factored;
    }

    // If no factoring applies, return normalized expression
    expr.clone()
}

/// Try to factor difference of squares: a^2 - b^2 = (a+b)(a-b)
fn try_difference_of_squares(expr: &SymExpr) -> Option<SymExpr> {
    let terms = expr.as_add()?;

    if terms.len() != 2 {
        return None;
    }

    // Look for pattern: something^2 - something^2
    let (pos_term, neg_term) = if is_negated(&terms[1]) {
        (&terms[0], get_negated_inner(&terms[1])?)
    } else if is_negated(&terms[0]) {
        (&terms[1], get_negated_inner(&terms[0])?)
    } else {
        return None;
    };

    // Check if both are squares
    let a = get_square_base(pos_term)?;
    let b = get_square_base(neg_term)?;

    // (a + b)(a - b)
    let sum = SymExpr::add(vec![a.clone(), b.clone()]);
    let diff = SymExpr::add(vec![a, SymExpr::neg(b)]);

    Some(SymExpr::mul(vec![sum, diff]))
}

/// Check if expression is negated
fn is_negated(expr: &SymExpr) -> bool {
    match expr.kind.as_ref() {
        SymExprKind::Neg(_) => true,
        SymExprKind::Mul(factors) => factors
            .iter()
            .any(|f| matches!(f.as_coeff(), Some(c) if c.is_negative())),
        _ => false,
    }
}

/// Get the inner expression of a negation
fn get_negated_inner(expr: &SymExpr) -> Option<&SymExpr> {
    match expr.kind.as_ref() {
        SymExprKind::Neg(inner) => Some(inner),
        SymExprKind::Mul(factors) => {
            // Check for -1 * something
            if factors.len() == 2 {
                if let Some(c) = factors[0].as_coeff() {
                    if c.is_neg_one() {
                        return Some(&factors[1]);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Get the base if this is expr^2
fn get_square_base(expr: &SymExpr) -> Option<SymExpr> {
    match expr.kind.as_ref() {
        SymExprKind::Pow(base, exp) => {
            if let Some(c) = exp.as_coeff() {
                if c.to_f64() == 2.0 {
                    return Some(base.as_ref().clone());
                }
            }
            None
        }
        SymExprKind::Mul(factors) => {
            // x * x = x^2
            if factors.len() == 2 && factors[0] == factors[1] {
                return Some(factors[0].clone());
            }
            None
        }
        _ => None,
    }
}

/// Try to extract common factors: ax + ay = a(x + y)
fn try_common_factor(expr: &SymExpr) -> Option<SymExpr> {
    let terms = expr.as_add()?;

    if terms.len() < 2 {
        return None;
    }

    // Find common variables in all terms
    let first_vars = terms[0].free_vars();
    if first_vars.is_empty() {
        return None;
    }

    // Check each variable to see if it appears in all terms
    for var in &first_vars {
        let all_have_var = terms.iter().all(|t| t.free_vars().contains(var));
        if !all_have_var {
            continue;
        }

        // Try to extract this variable as a common factor
        let var_expr = SymExpr::symbol(var.clone());
        let mut factored_terms = Vec::new();
        let mut can_factor = true;

        for term in terms {
            if let Some(remainder) = try_extract_factor(term, &var_expr) {
                factored_terms.push(remainder);
            } else {
                can_factor = false;
                break;
            }
        }

        if can_factor {
            return Some(SymExpr::mul(vec![var_expr, SymExpr::add(factored_terms)]));
        }
    }

    None
}

/// Try to extract a factor from an expression
fn try_extract_factor(expr: &SymExpr, factor: &SymExpr) -> Option<SymExpr> {
    match expr.kind.as_ref() {
        SymExprKind::Mul(factors) => {
            // Look for the factor in the product
            let mut found_idx = None;
            for (i, f) in factors.iter().enumerate() {
                if f == factor {
                    found_idx = Some(i);
                    break;
                }
            }

            if let Some(idx) = found_idx {
                let remaining: Vec<_> = factors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != idx)
                    .map(|(_, f)| f.clone())
                    .collect();
                Some(SymExpr::mul(remaining))
            } else {
                None
            }
        }
        _ => {
            if expr == factor {
                Some(SymExpr::int(1))
            } else {
                None
            }
        }
    }
}

/// Try to factor a quadratic: ax^2 + bx + c = a(x - r1)(x - r2)
fn try_quadratic_factor(expr: &SymExpr) -> Option<SymExpr> {
    let terms = expr.as_add()?;

    // Need exactly 3 terms for quadratic (or 2 with missing term)
    if terms.len() > 3 {
        return None;
    }

    // Find the variable (should be consistent across terms)
    let vars = expr.free_vars();
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?;
    let var_expr = SymExpr::symbol(var.clone());

    // Extract coefficients a, b, c from ax^2 + bx + c
    let mut a = 0.0_f64;
    let mut b = 0.0_f64;
    let mut c = 0.0_f64;

    for term in terms {
        if let Some((coeff, power)) = extract_term_power(term, &var_expr) {
            match power {
                0 => c += coeff,
                1 => b += coeff,
                2 => a += coeff,
                _ => return None, // Higher degree, can't factor as quadratic
            }
        } else {
            return None;
        }
    }

    if a == 0.0 {
        return None; // Not a quadratic
    }

    // Calculate discriminant
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None; // Complex roots, don't factor
    }

    // Check if roots are rational
    let sqrt_disc = discriminant.sqrt();
    if sqrt_disc.fract() != 0.0 {
        return None; // Irrational roots
    }

    let r1 = (-b + sqrt_disc) / (2.0 * a);
    let r2 = (-b - sqrt_disc) / (2.0 * a);

    // Check if roots are integers or simple fractions
    if r1.fract() != 0.0 || r2.fract() != 0.0 {
        return None;
    }

    // Build factored form: a * (x - r1) * (x - r2)
    let factor1 = if r1 == 0.0 {
        var_expr.clone()
    } else {
        SymExpr::add(vec![var_expr.clone(), SymExpr::float(-r1)])
    };

    let factor2 = if r2 == 0.0 {
        var_expr
    } else {
        SymExpr::add(vec![var_expr, SymExpr::float(-r2)])
    };

    if a == 1.0 {
        Some(SymExpr::mul(vec![factor1, factor2]))
    } else {
        Some(SymExpr::mul(vec![SymExpr::float(a), factor1, factor2]))
    }
}

/// Extract coefficient and power from a term like 3*x^2
fn extract_term_power(term: &SymExpr, var: &SymExpr) -> Option<(f64, u32)> {
    match term.kind.as_ref() {
        // Constant term
        SymExprKind::Num(c) => Some((c.to_f64(), 0)),

        // Just the variable: x (coefficient 1, power 1)
        SymExprKind::Var(_) if term == var => Some((1.0, 1)),

        // Power: x^n
        SymExprKind::Pow(base, exp) if base.as_ref() == var => {
            let e = exp.as_coeff()?.to_f64();
            if e >= 0.0 && e.fract() == 0.0 {
                Some((1.0, e as u32))
            } else {
                None
            }
        }

        // Product: coeff * x^n or coeff * x
        SymExprKind::Mul(factors) => {
            let mut coeff = 1.0;
            let mut power = 0u32;
            let mut found_var = false;

            for f in factors {
                if let Some(c) = f.as_coeff() {
                    coeff *= c.to_f64();
                } else if f == var {
                    // Accumulate power instead of overwriting
                    power += 1;
                    found_var = true;
                } else if let SymExprKind::Pow(base, exp) = f.kind.as_ref() {
                    if base.as_ref() == var {
                        let e = exp.as_coeff()?.to_f64();
                        if e >= 0.0 && e.fract() == 0.0 {
                            // Accumulate power instead of overwriting
                            power += e as u32;
                            found_var = true;
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }

            if found_var || power == 0 {
                Some((coeff, power))
            } else {
                None
            }
        }

        // Negation
        SymExprKind::Neg(inner) => {
            let (c, p) = extract_term_power(inner, var)?;
            Some((-c, p))
        }

        _ => None,
    }
}

/// Factor an integer into prime factors (with repetition)
fn factor_integer(mut n: i64) -> Vec<i64> {
    if n == 0 {
        return vec![0];
    }

    let mut factors = Vec::new();
    let negative = n < 0;
    n = n.abs();

    if n == 1 {
        return if negative { vec![-1] } else { vec![1] };
    }

    // Factor out 2s
    while n % 2 == 0 {
        factors.push(2);
        n /= 2;
    }

    // Factor odd numbers
    let mut d = 3i64;
    while d * d <= n {
        while n % d == 0 {
            factors.push(d);
            n /= d;
        }
        d += 2;
    }

    if n > 1 {
        factors.push(n);
    }

    if negative && !factors.is_empty() {
        factors[0] = -factors[0];
    }

    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_integer() {
        assert_eq!(factor_integer(12), vec![2, 2, 3]);
        assert_eq!(factor_integer(17), vec![17]); // Prime
        assert_eq!(factor_integer(-12), vec![-2, 2, 3]);
    }

    #[test]
    fn test_difference_of_squares() {
        // x^2 - 1 = (x+1)(x-1)
        let x = SymExpr::var("x");
        let x_sq = SymExpr::pow(x.clone(), SymExpr::int(2));
        let expr = SymExpr::add(vec![x_sq, SymExpr::neg(SymExpr::int(1))]);

        let result = factor_expr(&expr);
        assert!(result.is_mul(), "Expected product, got {:?}", result);
    }

    #[test]
    fn test_common_factor() {
        // x*y + x*z = x*(y + z)
        let x = SymExpr::var("x");
        let y = SymExpr::var("y");
        let z = SymExpr::var("z");

        let term1 = SymExpr::mul(vec![x.clone(), y]);
        let term2 = SymExpr::mul(vec![x, z]);
        let expr = SymExpr::add(vec![term1, term2]);

        let result = factor_expr(&expr);
        assert!(result.is_mul(), "Expected product, got {:?}", result);
    }
}
