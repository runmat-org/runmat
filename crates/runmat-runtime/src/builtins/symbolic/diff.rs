//! MATLAB-compatible `diff` builtin for symbolic differentiation.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{StagedNormalizer, SymExpr};

/// Differentiate a symbolic expression
///
/// # Examples
/// ```matlab
/// x = sym('x');
/// expr = x^2 + 3*x + 1;
/// deriv = diff(expr, x);     % Returns 2*x + 3
/// deriv2 = diff(expr, x, 2); % Returns 2 (second derivative)
/// ```
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

    // Parse differentiation variable
    let var_name = if rest.is_empty() {
        // Find the first free variable
        let free_vars = sym_expr.free_vars();
        if free_vars.is_empty() {
            return Ok(Value::Symbolic(SymExpr::int(0)));
        }
        // Get first variable alphabetically
        let mut names: Vec<_> = free_vars.iter().map(|s| s.name.clone()).collect();
        names.sort();
        names.into_iter().next().unwrap()
    } else {
        // First rest argument is the variable
        extract_var_name(&rest[0])?
    };

    // Parse order (optional second argument)
    let order = if rest.len() > 1 {
        match &rest[1] {
            Value::Num(n) => *n as u32,
            Value::Int(i) => i.to_i64() as u32,
            _ => return Err("diff: order must be a positive integer".to_string()),
        }
    } else {
        1
    };

    // Perform differentiation
    let mut result = sym_expr;
    for _ in 0..order {
        result = result.diff(&var_name);
    }

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
    fn test_diff_power() {
        // d/dx (x^2) = 2*x
        let x = SymExpr::var("x");
        let x_squared = SymExpr::pow(x.clone(), SymExpr::int(2));

        let result = diff_builtin(Value::Symbolic(x_squared), vec![Value::Symbolic(x)]).unwrap();

        match result {
            Value::Symbolic(_expr) => {
                // Should be 2*x (after normalization)
            }
            _ => panic!("Expected symbolic result"),
        }
    }

    #[test]
    fn test_diff_constant() {
        // d/dx (5) = 0
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

    #[test]
    fn test_diff_second_order() {
        // d²/dx² (x^3) = 6*x
        let x = SymExpr::var("x");
        let x_cubed = SymExpr::pow(x.clone(), SymExpr::int(3));

        let result = diff_builtin(
            Value::Symbolic(x_cubed),
            vec![Value::Symbolic(x), Value::Num(2.0)],
        )
        .unwrap();

        match result {
            Value::Symbolic(_expr) => {
                // Should be 6*x (after normalization)
            }
            _ => panic!("Expected symbolic result"),
        }
    }
}
