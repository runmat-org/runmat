//! MATLAB-compatible `subs` builtin for symbolic substitution.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{StagedNormalizer, SymExpr};

/// Substitute values in a symbolic expression
///
/// # Examples
/// ```matlab
/// x = sym('x');
/// expr = x^2 + 3*x + 1;
/// result = subs(expr, x, 2);      % Returns 11 (as symbolic)
/// result = subs(expr, 'x', 2);    % Same result
/// ```
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

    // Parse variable name
    let var_name = extract_var_name(&rest[0])?;

    // Parse replacement value
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

    // Perform substitution
    let result = sym_expr.substitute(&var_name, &replacement);

    // Normalize the result
    let normalizer = StagedNormalizer::default_pipeline();
    let (normalized, _) = normalizer.normalize(result);

    Ok(Value::Symbolic(normalized))
}

/// Extract variable name from a Value
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
        // x^2 with x=3 should give 9
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

    #[test]
    fn test_subs_symbolic() {
        // x + y with x=z should give z + y
        let x = SymExpr::var("x");
        let y = SymExpr::var("y");
        let expr = x.clone() + y;

        let result = subs_builtin(
            Value::Symbolic(expr),
            vec![
                Value::String("x".to_string()),
                Value::String("z".to_string()),
            ],
        )
        .unwrap();

        match result {
            Value::Symbolic(expr) => {
                let vars = expr.free_vars();
                let names: Vec<_> = vars.iter().map(|s| s.name.as_str()).collect();
                assert!(names.contains(&"z"));
                assert!(names.contains(&"y"));
                assert!(!names.contains(&"x"));
            }
            _ => panic!("Expected symbolic result"),
        }
    }
}
