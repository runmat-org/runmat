//! MATLAB-compatible `expand` builtin for expanding symbolic expressions.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::StagedNormalizer;

/// Expand a symbolic expression
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
        let x = SymExpr::var("x");
        let expr = SymExpr::pow(
            SymExpr::add(vec![x.clone(), SymExpr::int(1)]),
            SymExpr::int(2),
        );

        let result = expand_builtin(Value::Symbolic(expr), vec![]).unwrap();

        match result {
            Value::Symbolic(expanded) => {
                assert!(expanded.is_add());
            }
            _ => panic!("Expected symbolic result"),
        }
    }
}
