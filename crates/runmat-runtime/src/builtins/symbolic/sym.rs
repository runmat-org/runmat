//! MATLAB-compatible `sym` builtin for creating symbolic variables and expressions.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::SymExpr;

/// Create a symbolic variable or expression
#[runtime_builtin(
    name = "sym",
    category = "symbolic",
    summary = "Create a symbolic variable or expression.",
    keywords = "sym,symbolic,variable,symbol"
)]
fn sym_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let expr = match value {
        Value::String(name) => {
            let attrs = parse_assumptions(&rest)?;
            create_sym_var(&name, attrs)
        }
        Value::CharArray(ca) => {
            let name: String = ca.data.iter().collect();
            let attrs = parse_assumptions(&rest)?;
            create_sym_var(&name, attrs)
        }
        Value::Num(n) => SymExpr::float(n),
        Value::Int(i) => SymExpr::int(i.to_i64()),
        Value::Bool(b) => SymExpr::int(if b { 1 } else { 0 }),
        Value::Symbolic(expr) => expr,
        other => {
            return Err(format!(
                "sym: expected string, number, or symbolic expression, got {:?}",
                other
            ))
        }
    };

    Ok(Value::Symbolic(expr))
}

fn parse_assumptions(args: &[Value]) -> Result<runmat_symbolic::SymbolAttrs, String> {
    let mut attrs = runmat_symbolic::SymbolAttrs::default();

    for arg in args {
        let assumption = match arg {
            Value::String(s) => s.to_lowercase(),
            Value::CharArray(ca) => ca.data.iter().collect::<String>().to_lowercase(),
            _ => continue,
        };

        match assumption.as_str() {
            "real" => attrs.real = true,
            "positive" => {
                attrs.positive = true;
                attrs.real = true;
            }
            "integer" => {
                attrs.integer = true;
                attrs.real = true;
            }
            "nonnegative" => {
                attrs.nonnegative = true;
                attrs.real = true;
            }
            _ => {
                return Err(format!("sym: unknown assumption '{}'", assumption));
            }
        }
    }

    Ok(attrs)
}

fn create_sym_var(name: &str, attrs: runmat_symbolic::SymbolAttrs) -> SymExpr {
    let symbol = runmat_symbolic::Symbol::with_attrs(name.to_string(), attrs);
    SymExpr::symbol(symbol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sym_string() {
        let result = sym_builtin(Value::String("x".to_string()), vec![]).unwrap();
        match result {
            Value::Symbolic(expr) => {
                assert!(expr.is_var());
                assert_eq!(format!("{}", expr), "x");
            }
            _ => panic!("Expected symbolic value"),
        }
    }

    #[test]
    fn test_sym_number() {
        let result = sym_builtin(Value::Num(42.0), vec![]).unwrap();
        match result {
            Value::Symbolic(expr) => {
                assert!(expr.is_num());
            }
            _ => panic!("Expected symbolic value"),
        }
    }
}
