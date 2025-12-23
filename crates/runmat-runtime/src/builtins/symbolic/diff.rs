//! MATLAB-compatible `diff` builtin for symbolic differentiation.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{StagedNormalizer, SymExpr};

/// Differentiate a symbolic expression
#[runtime_builtin(
    name = "diff",
    category = "symbolic",
    summary = "Differentiate a symbolic expression.",
    keywords = "diff,differentiate,derivative,symbolic,calculus"
)]
fn diff_builtin(expr: Value, rest: Vec<Value>) -> Result<Value, String> {
    let sym_expr = match expr {
        Value::Symbolic(e) => e,
        _ => return Err("diff: first argument must be a symbolic expression".to_string()),
    };

    let var_name = if rest.is_empty() {
        let free_vars = sym_expr.free_vars();
        if free_vars.is_empty() {
            return Ok(Value::Symbolic(SymExpr::int(0)));
        }
        let mut names: Vec<_> = free_vars.iter().map(|s| s.name.clone()).collect();
        names.sort();
        names.into_iter().next().unwrap()
    } else {
        extract_var_name(&rest[0])?
    };

    let order = if rest.len() > 1 {
        match &rest[1] {
            Value::Num(n) => *n as u32,
            Value::Int(i) => i.to_i64() as u32,
            _ => return Err("diff: order must be a positive integer".to_string()),
        }
    } else {
        1
    };

    let mut result = sym_expr;
    for _ in 0..order {
        result = result.diff(&var_name);
    }

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
                Err("diff: expected symbolic variable".to_string())
            }
        }
        _ => Err("diff: expected variable name or symbolic variable".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_constant() {
        let five = SymExpr::int(5);

        let result =
            diff_builtin(Value::Symbolic(five), vec![Value::String("x".to_string())]).unwrap();

        match result {
            Value::Symbolic(expr) => {
                assert!(expr.is_zero());
            }
            _ => panic!("Expected symbolic result"),
        }
    }
}
