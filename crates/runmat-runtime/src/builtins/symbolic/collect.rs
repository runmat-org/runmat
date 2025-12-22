//! Symbolic collect builtin

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{SymExpr, SymExprKind};
use std::collections::HashMap;

/// Collect coefficients of a variable in a polynomial
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

fn collect_expr(expr: &SymExpr, var: &SymExpr) -> SymExpr {
    let normalizer = runmat_symbolic::StagedNormalizer::aggressive();
    let (expanded, _) = normalizer.normalize(expr.clone());

    let mut powers: HashMap<i64, Vec<SymExpr>> = HashMap::new();

    collect_powers(&expanded, var, &mut powers);

    let mut result_terms = Vec::new();

    let mut sorted_powers: Vec<_> = powers.keys().copied().collect();
    sorted_powers.sort_by(|a, b| b.cmp(a));

    for power in sorted_powers {
        let coeffs = powers.get(&power).unwrap();
        let coeff_sum = if coeffs.len() == 1 {
            coeffs[0].clone()
        } else {
            SymExpr::add(coeffs.clone())
        };

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

fn collect_powers(expr: &SymExpr, var: &SymExpr, powers: &mut HashMap<i64, Vec<SymExpr>>) {
    match expr.kind.as_ref() {
        SymExprKind::Add(terms) => {
            for term in terms {
                collect_powers(term, var, powers);
            }
        }
        _ => {
            let (power, coeff) = extract_var_power(expr, var);
            powers.entry(power).or_default().push(coeff);
        }
    }
}

fn extract_var_power(expr: &SymExpr, var: &SymExpr) -> (i64, SymExpr) {
    match expr.kind.as_ref() {
        SymExprKind::Num(_) => (0, expr.clone()),
        SymExprKind::Var(_) => {
            if expr == var {
                (1, SymExpr::int(1))
            } else {
                (0, expr.clone())
            }
        }
        SymExprKind::Pow(base, exp) => {
            if base.as_ref() == var {
                if let Some(c) = exp.as_coeff() {
                    if c.is_integer() {
                        return (c.to_f64() as i64, SymExpr::int(1));
                    }
                }
            }
            // Variable not directly in base, treat entire expression as coefficient
            (0, expr.clone())
        }
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
        SymExprKind::Neg(inner) => {
            let (power, coeff) = extract_var_power(inner, var);
            (power, SymExpr::neg(coeff))
        }
        SymExprKind::Func(_, _) => (0, expr.clone()),
        SymExprKind::Add(_) => (0, expr.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_simple() {
        let x = SymExpr::var("x");
        let expr = SymExpr::add(vec![
            x.clone(),
            SymExpr::mul(vec![SymExpr::int(2), x.clone()]),
        ]);

        let result = collect_expr(&expr, &x);
        println!("Result: {}", result);
    }
}
