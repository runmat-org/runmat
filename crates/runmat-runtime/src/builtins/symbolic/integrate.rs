//! Symbolic integration builtin

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{Coefficient, SymExpr, SymExprKind};

/// Compute the indefinite integral of a symbolic expression
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

    let var_name = if rest.is_empty() {
        let vars = sym_expr.free_vars();
        if vars.is_empty() {
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

    let normalizer = runmat_symbolic::StagedNormalizer::default_pipeline();
    let (simplified, _) = normalizer.normalize(integral);

    Ok(Value::Symbolic(simplified))
}

fn integrate(expr: &SymExpr, var: &SymExpr) -> Result<SymExpr, String> {
    match expr.kind.as_ref() {
        SymExprKind::Num(_) => Ok(SymExpr::mul(vec![expr.clone(), var.clone()])),

        SymExprKind::Var(_) if expr == var => Ok(SymExpr::mul(vec![
            SymExpr::num(Coefficient::rational(1, 2)),
            SymExpr::pow(var.clone(), SymExpr::int(2)),
        ])),

        SymExprKind::Var(_) => Ok(SymExpr::mul(vec![expr.clone(), var.clone()])),

        SymExprKind::Pow(base, exp) if base.as_ref() == var => {
            if !exp
                .free_vars()
                .iter()
                .any(|s| &SymExpr::symbol(s.clone()) == var)
            {
                if let Some(c) = exp.as_coeff() {
                    if c.to_f64() == -1.0 {
                        return Ok(SymExpr::log(var.clone()));
                    }
                }

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

        SymExprKind::Add(terms) => {
            let integrals: Result<Vec<_>, _> = terms.iter().map(|t| integrate(t, var)).collect();
            Ok(SymExpr::add(integrals?))
        }

        SymExprKind::Mul(factors) => {
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
                Ok(SymExpr::mul(vec![expr.clone(), var.clone()]))
            } else if var_factors.len() == 1 {
                let integral = integrate(&var_factors[0], var)?;
                if const_factors.is_empty() {
                    Ok(integral)
                } else {
                    const_factors.push(integral);
                    Ok(SymExpr::mul(const_factors))
                }
            } else {
                Err("int: cannot integrate this product".to_string())
            }
        }

        SymExprKind::Neg(inner) => {
            let integral = integrate(inner, var)?;
            Ok(SymExpr::neg(integral))
        }

        SymExprKind::Func(name, args) if args.len() == 1 => {
            let arg = &args[0];

            if arg == var {
                match name.as_str() {
                    "sin" => Ok(SymExpr::neg(SymExpr::cos(var.clone()))),
                    "cos" => Ok(SymExpr::sin(var.clone())),
                    "exp" => Ok(SymExpr::exp(var.clone())),
                    "sqrt" => Ok(SymExpr::mul(vec![
                        SymExpr::num(Coefficient::rational(2, 3)),
                        SymExpr::pow(var.clone(), SymExpr::num(Coefficient::rational(3, 2))),
                    ])),
                    _ => Err(format!("int: cannot integrate {}(x)", name)),
                }
            } else {
                Err(format!("int: cannot integrate {}(...)", name))
            }
        }

        _ => Err("int: cannot integrate this expression".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_constant() {
        let x = SymExpr::var("x");
        let expr = SymExpr::int(5);

        let result = integrate(&expr, &x).unwrap();
        assert!(result.is_mul());
    }

    #[test]
    fn test_integrate_x() {
        let x = SymExpr::var("x");

        let result = integrate(&x, &x).unwrap();
        assert!(result.is_mul());
    }
}
