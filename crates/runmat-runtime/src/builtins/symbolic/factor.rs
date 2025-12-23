//! Symbolic factorization builtin

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{SymExpr, SymExprKind};

/// Factor a symbolic expression
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
            if n == n.floor() && n.abs() < 1e15 {
                let factors = factor_integer(n as i64);
                if factors.len() == 1 {
                    Ok(Value::Num(n))
                } else {
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

fn factor_expr(expr: &SymExpr) -> SymExpr {
    if let Some(factored) = try_difference_of_squares(expr) {
        return factored;
    }

    if let Some(factored) = try_common_factor(expr) {
        return factored;
    }

    expr.clone()
}

fn try_difference_of_squares(expr: &SymExpr) -> Option<SymExpr> {
    let terms = expr.as_add()?;

    if terms.len() != 2 {
        return None;
    }

    let (pos_term, neg_term) = if is_negated(&terms[1]) {
        (&terms[0], get_negated_inner(&terms[1])?)
    } else if is_negated(&terms[0]) {
        (&terms[1], get_negated_inner(&terms[0])?)
    } else {
        return None;
    };

    let a = get_square_base(pos_term)?;
    let b = get_square_base(neg_term)?;

    let sum = SymExpr::add(vec![a.clone(), b.clone()]);
    let diff = SymExpr::add(vec![a, SymExpr::neg(b)]);

    Some(SymExpr::mul(vec![sum, diff]))
}

fn is_negated(expr: &SymExpr) -> bool {
    match expr.kind.as_ref() {
        SymExprKind::Neg(_) => true,
        SymExprKind::Mul(factors) => factors
            .iter()
            .any(|f| matches!(f.as_coeff(), Some(c) if c.is_negative())),
        _ => false,
    }
}

fn get_negated_inner(expr: &SymExpr) -> Option<&SymExpr> {
    match expr.kind.as_ref() {
        SymExprKind::Neg(inner) => Some(inner),
        SymExprKind::Mul(factors) => {
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
            if factors.len() == 2 && factors[0] == factors[1] {
                return Some(factors[0].clone());
            }
            None
        }
        SymExprKind::Num(c) => {
            let val = c.to_f64();
            if val >= 0.0 {
                let sqrt_val = val.sqrt();
                if (sqrt_val * sqrt_val - val).abs() < 1e-10 && sqrt_val == sqrt_val.floor() {
                    return Some(SymExpr::int(sqrt_val as i64));
                }
            }
            None
        }
        _ => None,
    }
}

fn try_common_factor(expr: &SymExpr) -> Option<SymExpr> {
    let terms = expr.as_add()?;

    if terms.len() < 2 {
        return None;
    }

    let first_vars = terms[0].free_vars();
    if first_vars.is_empty() {
        return None;
    }

    for var in &first_vars {
        let all_have_var = terms.iter().all(|t| t.free_vars().contains(var));
        if !all_have_var {
            continue;
        }

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

fn try_extract_factor(expr: &SymExpr, factor: &SymExpr) -> Option<SymExpr> {
    match expr.kind.as_ref() {
        SymExprKind::Mul(factors) => {
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

    while n % 2 == 0 {
        factors.push(2);
        n /= 2;
    }

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
        assert_eq!(factor_integer(17), vec![17]);
        assert_eq!(factor_integer(-12), vec![-2, 2, 3]);
    }

    #[test]
    fn test_difference_of_squares() {
        let x = SymExpr::var("x");
        let x_sq = SymExpr::pow(x.clone(), SymExpr::int(2));
        let expr = SymExpr::add(vec![x_sq, SymExpr::neg(SymExpr::int(1))]);

        let result = factor_expr(&expr);
        assert!(result.is_mul(), "Expected product, got {:?}", result);
    }
}
