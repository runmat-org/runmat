//! Symbolic equation solver builtin

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{SymExpr, SymExprKind};

/// Solve a symbolic equation for a variable
#[runtime_builtin(
    name = "solve",
    category = "symbolic",
    summary = "Solve a symbolic equation for a variable.",
    keywords = "solve,equation,symbolic,roots"
)]
fn solve_builtin(expr: Value, rest: Vec<Value>) -> Result<Value, String> {
    let sym_expr = match expr {
        Value::Symbolic(e) => e,
        _ => return Err("solve: first argument must be a symbolic expression".to_string()),
    };

    let var_name = if rest.is_empty() {
        let vars = sym_expr.free_vars();
        if vars.is_empty() {
            return Err("solve: expression has no variables".to_string());
        }
        if vars.len() > 1 {
            return Err(format!(
                "solve: expression has multiple variables ({:?}), specify which to solve for",
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
                    return Err("solve: second argument must be a symbolic variable".to_string());
                }
            }
            Value::String(s) => s.clone(),
            Value::CharArray(ca) => ca.to_string(),
            _ => {
                return Err(
                    "solve: second argument must be a symbolic variable or string".to_string(),
                )
            }
        }
    };

    let var = SymExpr::var(&var_name);
    let solutions = solve_for_var(&sym_expr, &var)?;

    match solutions.len() {
        0 => Err("solve: no solution found".to_string()),
        1 => Ok(Value::Symbolic(solutions.into_iter().next().unwrap())),
        _ => Ok(Value::Symbolic(solutions.into_iter().next().unwrap())),
    }
}

fn solve_for_var(expr: &SymExpr, var: &SymExpr) -> Result<Vec<SymExpr>, String> {
    let degree = polynomial_degree(expr, var);

    match degree {
        Some(0) => {
            if expr.is_zero() {
                Ok(vec![var.clone()])
            } else {
                Ok(vec![])
            }
        }
        Some(1) => solve_linear(expr, var),
        Some(2) => solve_quadratic(expr, var),
        Some(n) if n <= 4 => Err(format!(
            "solve: degree {} polynomial solving not yet implemented",
            n
        )),
        _ => Err(
            "solve: cannot determine polynomial degree or expression is not polynomial".to_string(),
        ),
    }
}

fn polynomial_degree(expr: &SymExpr, var: &SymExpr) -> Option<u32> {
    match expr.kind.as_ref() {
        SymExprKind::Num(_) => Some(0),
        SymExprKind::Var(_) => {
            if expr == var {
                Some(1)
            } else {
                Some(0)
            }
        }
        SymExprKind::Pow(base, exp) => {
            if base.as_ref() == var {
                if let Some(c) = exp.as_coeff() {
                    if c.is_integer() && !c.is_negative() {
                        return Some(c.to_f64() as u32);
                    }
                }
                None
            } else if base
                .free_vars()
                .iter()
                .any(|s| &SymExpr::symbol(s.clone()) == var)
            {
                None
            } else {
                Some(0)
            }
        }
        SymExprKind::Add(terms) => {
            let mut max_degree = 0u32;
            for t in terms {
                let d = polynomial_degree(t, var)?;
                max_degree = max_degree.max(d);
            }
            Some(max_degree)
        }
        SymExprKind::Mul(factors) => {
            let mut total_degree = 0u32;
            for f in factors {
                let d = polynomial_degree(f, var)?;
                total_degree += d;
            }
            Some(total_degree)
        }
        SymExprKind::Neg(inner) => polynomial_degree(inner, var),
        SymExprKind::Func(_, _) => {
            if expr
                .free_vars()
                .iter()
                .any(|s| &SymExpr::symbol(s.clone()) == var)
            {
                None
            } else {
                Some(0)
            }
        }
    }
}

fn solve_linear(expr: &SymExpr, var: &SymExpr) -> Result<Vec<SymExpr>, String> {
    let normalizer = runmat_symbolic::StagedNormalizer::aggressive();
    let (expanded, _) = normalizer.normalize(expr.clone());

    let (a, b) = extract_linear_coeffs(&expanded, var);

    if a.is_zero() {
        if b.is_zero() {
            return Ok(vec![var.clone()]);
        } else {
            return Ok(vec![]);
        }
    }

    let neg_b = SymExpr::neg(b);
    let solution = SymExpr::mul(vec![neg_b, SymExpr::pow(a, SymExpr::int(-1))]);

    let normalizer = runmat_symbolic::StagedNormalizer::default_pipeline();
    let (simplified, _) = normalizer.normalize(solution);

    Ok(vec![simplified])
}

fn extract_linear_coeffs(expr: &SymExpr, var: &SymExpr) -> (SymExpr, SymExpr) {
    let mut coeff_a = SymExpr::int(0);
    let mut coeff_b = SymExpr::int(0);

    match expr.kind.as_ref() {
        SymExprKind::Add(terms) => {
            for term in terms {
                let (a, b) = extract_linear_coeffs(term, var);
                coeff_a = coeff_a + a;
                coeff_b = coeff_b + b;
            }
        }
        SymExprKind::Var(_) if expr == var => {
            coeff_a = SymExpr::int(1);
        }
        SymExprKind::Mul(factors) => {
            let var_idx = factors.iter().position(|f| f == var);
            if let Some(idx) = var_idx {
                let remaining: Vec<_> = factors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != idx)
                    .map(|(_, f)| f.clone())
                    .collect();
                coeff_a = if remaining.is_empty() {
                    SymExpr::int(1)
                } else {
                    SymExpr::mul(remaining)
                };
            } else {
                coeff_b = expr.clone();
            }
        }
        SymExprKind::Neg(inner) => {
            let (a, b) = extract_linear_coeffs(inner, var);
            coeff_a = SymExpr::neg(a);
            coeff_b = SymExpr::neg(b);
        }
        _ => {
            if !expr
                .free_vars()
                .iter()
                .any(|s| &SymExpr::symbol(s.clone()) == var)
            {
                coeff_b = expr.clone();
            }
        }
    }

    (coeff_a, coeff_b)
}

fn solve_quadratic(expr: &SymExpr, var: &SymExpr) -> Result<Vec<SymExpr>, String> {
    let normalizer = runmat_symbolic::StagedNormalizer::aggressive();
    let (expanded, _) = normalizer.normalize(expr.clone());

    let (a, b, c) = extract_quadratic_coeffs(&expanded, var)?;

    if a.is_zero() {
        return solve_linear(expr, var);
    }

    let b_squared = SymExpr::pow(b.clone(), SymExpr::int(2));
    let four_ac = SymExpr::mul(vec![SymExpr::int(4), a.clone(), c.clone()]);
    let discriminant = SymExpr::add(vec![b_squared, SymExpr::neg(four_ac)]);

    let normalizer = runmat_symbolic::StagedNormalizer::default_pipeline();
    let (disc_simplified, _) = normalizer.normalize(discriminant);

    let sqrt_disc = SymExpr::sqrt(disc_simplified);
    let neg_b = SymExpr::neg(b);
    let two_a = SymExpr::mul(vec![SymExpr::int(2), a]);

    let numerator1 = SymExpr::add(vec![neg_b.clone(), sqrt_disc.clone()]);
    let sol1 = SymExpr::mul(vec![
        numerator1,
        SymExpr::pow(two_a.clone(), SymExpr::int(-1)),
    ]);

    let numerator2 = SymExpr::add(vec![neg_b, SymExpr::neg(sqrt_disc)]);
    let sol2 = SymExpr::mul(vec![numerator2, SymExpr::pow(two_a, SymExpr::int(-1))]);

    let normalizer = runmat_symbolic::StagedNormalizer::default_pipeline();
    let (sol1_simplified, _) = normalizer.normalize(sol1);
    let (sol2_simplified, _) = normalizer.normalize(sol2);

    Ok(vec![sol1_simplified, sol2_simplified])
}

fn extract_quadratic_coeffs(
    expr: &SymExpr,
    var: &SymExpr,
) -> Result<(SymExpr, SymExpr, SymExpr), String> {
    let mut coeff_a = SymExpr::int(0);
    let mut coeff_b = SymExpr::int(0);
    let mut coeff_c = SymExpr::int(0);

    fn add_term(
        term: &SymExpr,
        var: &SymExpr,
        a: &mut SymExpr,
        b: &mut SymExpr,
        c: &mut SymExpr,
    ) -> Result<(), String> {
        let (power, coeff) = extract_term_var_power(term, var);
        match power {
            0 => *c = c.clone() + coeff,
            1 => *b = b.clone() + coeff,
            2 => *a = a.clone() + coeff,
            _ => return Err(format!("solve: unexpected power {} in quadratic", power)),
        }
        Ok(())
    }

    match expr.kind.as_ref() {
        SymExprKind::Add(terms) => {
            for term in terms {
                add_term(term, var, &mut coeff_a, &mut coeff_b, &mut coeff_c)?;
            }
        }
        _ => {
            add_term(expr, var, &mut coeff_a, &mut coeff_b, &mut coeff_c)?;
        }
    }

    let normalizer = runmat_symbolic::StagedNormalizer::default_pipeline();
    let (a, _) = normalizer.normalize(coeff_a);
    let (b, _) = normalizer.normalize(coeff_b);
    let (c, _) = normalizer.normalize(coeff_c);

    Ok((a, b, c))
}

fn extract_term_var_power(term: &SymExpr, var: &SymExpr) -> (u32, SymExpr) {
    match term.kind.as_ref() {
        SymExprKind::Num(_) => (0, term.clone()),
        SymExprKind::Var(_) if term == var => (1, SymExpr::int(1)),
        SymExprKind::Var(_) => (0, term.clone()),
        SymExprKind::Pow(base, exp) if base.as_ref() == var => {
            if let Some(c) = exp.as_coeff() {
                if c.is_integer() && !c.is_negative() {
                    return (c.to_f64() as u32, SymExpr::int(1));
                }
            }
            (0, term.clone())
        }
        SymExprKind::Mul(factors) => {
            let mut power = 0u32;
            let mut coeff_factors = Vec::new();

            for f in factors {
                if f == var {
                    power += 1;
                } else if let SymExprKind::Pow(base, exp) = f.kind.as_ref() {
                    if base.as_ref() == var {
                        if let Some(c) = exp.as_coeff() {
                            if c.is_integer() && !c.is_negative() {
                                power += c.to_f64() as u32;
                                continue;
                            }
                        }
                    }
                    coeff_factors.push(f.clone());
                } else {
                    coeff_factors.push(f.clone());
                }
            }

            let coeff = if coeff_factors.is_empty() {
                SymExpr::int(1)
            } else {
                SymExpr::mul(coeff_factors)
            };

            (power, coeff)
        }
        SymExprKind::Neg(inner) => {
            let (power, coeff) = extract_term_var_power(inner, var);
            (power, SymExpr::neg(coeff))
        }
        _ => (0, term.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_linear() {
        let x = SymExpr::var("x");
        let expr = SymExpr::add(vec![
            SymExpr::mul(vec![SymExpr::int(2), x.clone()]),
            SymExpr::int(4),
        ]);

        let solutions = solve_for_var(&expr, &x).unwrap();
        assert_eq!(solutions.len(), 1);
    }
}
