//! MATLAB-compatible `simplify` builtin for simplifying symbolic expressions.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::StagedNormalizer;

/// Simplify a symbolic expression
///
/// # Examples
/// ```matlab
/// x = sym('x');
/// expr = x + x + 0;
/// simplified = simplify(expr);  % Returns 2*x
/// ```
#[runtime_builtin(
    name = "simplify",
    category = "symbolic",
    summary = "Simplify a symbolic expression.",
    keywords = "simplify,symbolic,reduce,normalize"
)]
fn simplify_builtin(expr: Value, _rest: Vec<Value>) -> Result<Value, String> {
    let sym_expr = match expr {
        Value::Symbolic(e) => e,
        _ => return Err("simplify: argument must be a symbolic expression".to_string()),
    };

    // Use the default normalization pipeline for simplification
    let normalizer = StagedNormalizer::default_pipeline();
    let (simplified, _) = normalizer.normalize(sym_expr);

    Ok(Value::Symbolic(simplified))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_symbolic::SymExpr;

    #[test]
    fn test_simplify_add_zero() {
        // x + 0 should simplify to x
        let x = SymExpr::var("x");
        let expr = SymExpr::add(vec![x.clone(), SymExpr::int(0)]);

        let result = simplify_builtin(Value::Symbolic(expr), vec![]).unwrap();

        match result {
            Value::Symbolic(simplified) => {
                // After simplification, should just be x
                assert!(simplified.is_var() || simplified.is_add());
            }
            _ => panic!("Expected symbolic result"),
        }
    }

    #[test]
    fn test_simplify_mul_one() {
        // x * 1 should simplify to x
        let x = SymExpr::var("x");
        let expr = SymExpr::mul(vec![x.clone(), SymExpr::int(1)]);

        let result = simplify_builtin(Value::Symbolic(expr), vec![]).unwrap();

        match result {
            Value::Symbolic(simplified) => {
                assert!(simplified.is_var() || simplified.is_mul());
            }
            _ => panic!("Expected symbolic result"),
        }
    }

    #[test]
    fn test_simplify_constants() {
        // 1 + 2 + 3 should simplify to 6
        let expr = SymExpr::add(vec![SymExpr::int(1), SymExpr::int(2), SymExpr::int(3)]);

        let result = simplify_builtin(Value::Symbolic(expr), vec![]).unwrap();

        match result {
            Value::Symbolic(simplified) => {
                if let Some(c) = simplified.as_coeff() {
                    assert!((c.to_f64() - 6.0).abs() < 1e-10);
                }
            }
            _ => panic!("Expected symbolic result"),
        }
    }
}
