//! MATLAB-compatible `syms` builtin for creating multiple symbolic variables.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::SymExpr;

/// Create multiple symbolic variables at once
///
/// Returns a cell array of symbolic variables.
///
/// # Examples
/// ```matlab
/// vars = syms('x', 'y', 'z');  % Creates x, y, z as symbolic variables
/// ```
#[runtime_builtin(
    name = "syms",
    category = "symbolic",
    summary = "Create multiple symbolic variables.",
    keywords = "syms,symbolic,variable,symbol,multiple"
)]
fn syms_builtin(rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return Err("syms: at least one variable name required".to_string());
    }

    let mut symbols = Vec::with_capacity(rest.len());

    for arg in &rest {
        let name = match arg {
            Value::String(s) => s.clone(),
            Value::CharArray(ca) => ca.data.iter().collect(),
            _ => {
                return Err(format!(
                    "syms: expected string variable name, got {:?}",
                    arg
                ))
            }
        };

        // Validate variable name (simple check)
        if name.is_empty() || !name.chars().next().unwrap().is_alphabetic() {
            return Err(format!("syms: invalid variable name '{}'", name));
        }

        let expr = SymExpr::var(&name);
        symbols.push(Value::Symbolic(expr));
    }

    // Return as cell array if multiple, or single value if one
    if symbols.len() == 1 {
        Ok(symbols.pop().unwrap())
    } else {
        // Create a cell array
        let handles: Vec<runmat_gc_api::GcPtr<Value>> = symbols
            .into_iter()
            .map(|v| runmat_gc::gc_allocate(v).expect("gc alloc"))
            .collect();
        let cell = runmat_builtins::CellArray::new_handles(handles, 1, rest.len())
            .map_err(|e| format!("syms: {}", e))?;
        Ok(Value::Cell(cell))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syms_single() {
        let result = syms_builtin(vec![Value::String("x".to_string())]).unwrap();
        match result {
            Value::Symbolic(expr) => {
                assert!(expr.is_var());
            }
            _ => panic!("Expected symbolic value"),
        }
    }

    #[test]
    fn test_syms_multiple() {
        let result = syms_builtin(vec![
            Value::String("x".to_string()),
            Value::String("y".to_string()),
        ])
        .unwrap();
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.cols, 2);
            }
            _ => panic!("Expected cell array"),
        }
    }

    #[test]
    fn test_syms_empty_error() {
        let result = syms_builtin(vec![]);
        assert!(result.is_err());
    }
}
