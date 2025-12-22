//! MATLAB-compatible `subs` builtin for symbolic substitution.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{StagedNormalizer, SymExpr};

/// Substitute values in a symbolic expression
#[runtime_builtin(
    name = "subs",
    category = "symbolic",
    summary = "Substitute values in a symbolic expression.",
    keywords = "subs,substitute,symbolic,replace"
)]
fn subs_builtin(expr: Value, rest: Vec<Value>) -> Result<Value, String> {
    let sym_expr = match expr {
        Value::Symbolic(e) => e,
        _ => return Err("subs: first argument must be a symbolic expression".to_string()),
    };

    if rest.len() < 2 {
        return Err("subs: requires at least 3 arguments (expr, var, value)".to_string());
    }

    let var_name = extract_var_name(&rest[0])?;

    let replacement = match &rest[1] {
        Value::Num(n) => SymExpr::float(*n),
        Value::Int(i) => SymExpr::int(i.to_i64()),
        Value::Bool(b) => SymExpr::int(if *b { 1 } else { 0 }),
        Value::Symbolic(e) => e.clone(),
        Value::String(s) => SymExpr::var(s),
        Value::CharArray(ca) => SymExpr::var(ca.data.iter().collect::<String>()),
        other => {
            return Err(format!(
                "subs: replacement must be numeric or symbolic, got {:?}",
                other
            ))
        }
    };

    let result = sym_expr.substitute(&var_name, &replacement);

    let normalizer = StagedNormalizer::default_pipeline();
    let (normalized, _) = normalizer.normalize(result);

    Ok(Value::Symbolic(normalized))
}

fn extract_var_name(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) => Ok(ca.data.iter().collect()),
        Value::Symbolic(expr) => {
            if let Some(s) = expr.as_var() {
                Ok(s.name.clone())
            } else {
                Err("subs: expected symbolic variable".to_string())
            }
        }
        _ => Err("subs: expected variable name or symbolic variable".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subs_numeric() {
        let x = SymExpr::var("x");
        let x_squared = SymExpr::pow(x.clone(), SymExpr::int(2));

        let result = subs_builtin(
            Value::Symbolic(x_squared),
            vec![Value::Symbolic(x), Value::Num(3.0)],
        )
        .unwrap();

        match result {
            Value::Symbolic(expr) => {
                if let Some(c) = expr.as_coeff() {
                    assert!((c.to_f64() - 9.0).abs() < 1e-10);
                }
            }
            _ => panic!("Expected symbolic result"),
        }
    }
}
