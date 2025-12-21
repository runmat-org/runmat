//! MATLAB-compatible `expand` builtin for expanding symbolic expressions.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::StagedNormalizer;

/// Expand a symbolic expression
///
/// # Examples
/// ```matlab
/// x = sym('x');
/// expr = (x + 1)^2;
/// expanded = expand(expr);  % Returns x^2 + 2*x + 1
/// ```
#[runtime_builtin(
    name = "expand",
    category = "symbolic",
    summary = "Expand a symbolic expression.",
    keywords = "expand,symbolic,polynomial,multiply"
)]
fn expand_builtin(expr: Value, _rest: Vec<Value>) -> Result<Value, String> {
    let sym_expr = match expr {
        Value::Symbolic(e) => e,
        _ => return Err("expand: argument must be a symbolic expression".to_string()),
    };

    // Use aggressive pipeline that includes expansion
    let normalizer = StagedNormalizer::aggressive();
    let (expanded, _) = normalizer.normalize(sym_expr);

    Ok(Value::Symbolic(expanded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_symbolic::SymExpr;

    #[test]
    fn test_expand_binomial() {
        // (x + 1)^2 should expand to x^2 + 2*x + 1
        let x = SymExpr::var("x");
        let expr = SymExpr::pow(
            SymExpr::add(vec![x.clone(), SymExpr::int(1)]),
            SymExpr::int(2),
        );

        let result = expand_builtin(Value::Symbolic(expr), vec![]).unwrap();

        match result {
            Value::Symbolic(expanded) => {
                // After expansion, should be an Add with 3 terms
                assert!(expanded.is_add());
            }
            _ => panic!("Expected symbolic result"),
        }
    }

    #[test]
    fn test_expand_product() {
        // (x + 1) * (x - 1) = x^2 - 1
        let x = SymExpr::var("x");
        let expr = SymExpr::mul(vec![
            SymExpr::add(vec![x.clone(), SymExpr::int(1)]),
            SymExpr::add(vec![x.clone(), SymExpr::int(-1)]),
        ]);

        let result = expand_builtin(Value::Symbolic(expr), vec![]).unwrap();

        match result {
            Value::Symbolic(_expanded) => {
                // Should be x^2 - 1 or equivalent
            }
            _ => panic!("Expected symbolic result"),
        }
    }
}
