//! MATLAB-compatible `matlabFunction` builtin for converting symbolic expressions to numeric functions.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{compile_with_vars, SymExpr};

/// Convert a symbolic expression to a numeric function
///
/// Returns a function handle that can be called with numeric arguments.
/// The function evaluates the symbolic expression with the given values.
///
/// # Examples
/// ```matlab
/// x = sym('x');
/// expr = x^2 + 3*x + 1;
/// f = matlabFunction(expr);
/// result = f(2);  % Returns 11
/// ```
#[runtime_builtin(
    name = "matlabFunction",
    category = "symbolic",
    summary = "Convert a symbolic expression to a numeric function.",
    keywords = "matlabFunction,symbolic,function,numeric,compile"
)]
fn matlabfunction_builtin(expr: Value, rest: Vec<Value>) -> Result<Value, String> {
    let sym_expr = match expr {
        Value::Symbolic(e) => e,
        _ => return Err("matlabFunction: argument must be a symbolic expression".to_string()),
    };

    // Collect free variables from expression
    let free_vars = sym_expr.free_vars();
    let mut var_names: Vec<String> = free_vars.iter().map(|s| s.name.clone()).collect();
    var_names.sort(); // Ensure deterministic ordering

    // Check for explicit 'Vars' option
    for i in 0..rest.len() {
        if is_vars_keyword(&rest[i]) && i + 1 < rest.len() {
            var_names = extract_var_list(&rest[i + 1])?;
            break;
        }
    }

    // Compile the expression
    let var_refs: Vec<&str> = var_names.iter().map(|s| s.as_str()).collect();
    let compiled = compile_with_vars(&sym_expr, &var_refs);

    // Store as a function handle with the compiled bytecode
    // The function name encodes the symbolic expression info
    let func_name = format!("__sym_fn_{}", compiled.variables.join("_"));

    // Store compiled expression in a static registry for later evaluation
    register_compiled_function(&func_name, compiled, sym_expr);

    Ok(Value::FunctionHandle(func_name))
}

// Static storage for compiled functions
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[allow(dead_code)]
struct CompiledFunctionEntry {
    compiled: runmat_symbolic::CompiledExpr,
    original: SymExpr,
}

static COMPILED_FUNCTIONS: OnceLock<Mutex<HashMap<String, CompiledFunctionEntry>>> =
    OnceLock::new();

fn compiled_functions() -> &'static Mutex<HashMap<String, CompiledFunctionEntry>> {
    COMPILED_FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_compiled_function(
    name: &str,
    compiled: runmat_symbolic::CompiledExpr,
    original: SymExpr,
) {
    let mut map = compiled_functions().lock().unwrap();
    map.insert(
        name.to_string(),
        CompiledFunctionEntry { compiled, original },
    );
}

/// Evaluate a compiled symbolic function by name
pub fn eval_compiled_function(name: &str, args: &[f64]) -> Result<f64, String> {
    let map = compiled_functions().lock().unwrap();
    let entry = map
        .get(name)
        .ok_or_else(|| format!("Unknown compiled function: {}", name))?;
    entry.compiled.eval(args)
}

/// Check if value is the 'Vars' keyword
fn is_vars_keyword(v: &Value) -> bool {
    match v {
        Value::String(s) => s.eq_ignore_ascii_case("vars"),
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().collect();
            s.eq_ignore_ascii_case("vars")
        }
        _ => false,
    }
}

/// Extract variable list from a cell array or string
fn extract_var_list(v: &Value) -> Result<Vec<String>, String> {
    match v {
        Value::Cell(cell) => {
            let mut names = Vec::with_capacity(cell.data.len());
            for handle in &cell.data {
                let val = &**handle;
                match val {
                    Value::String(s) => names.push(s.clone()),
                    Value::CharArray(ca) => names.push(ca.data.iter().collect()),
                    Value::Symbolic(expr) => {
                        if let Some(s) = expr.as_var() {
                            names.push(s.name.clone());
                        } else {
                            return Err(
                                "matlabFunction: Vars must contain variable names".to_string()
                            );
                        }
                    }
                    _ => return Err("matlabFunction: Vars must contain variable names".to_string()),
                }
            }
            Ok(names)
        }
        Value::String(s) => Ok(vec![s.clone()]),
        Value::CharArray(ca) => Ok(vec![ca.data.iter().collect()]),
        Value::Symbolic(expr) => {
            if let Some(s) = expr.as_var() {
                Ok(vec![s.name.clone()])
            } else {
                Err("matlabFunction: expected variable".to_string())
            }
        }
        _ => Err("matlabFunction: Vars must be a cell array of variable names".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matlabfunction_compile() {
        // Create x^2 + 1
        let x = SymExpr::var("x");
        let expr = SymExpr::add(vec![
            SymExpr::pow(x.clone(), SymExpr::int(2)),
            SymExpr::int(1),
        ]);

        let result = matlabfunction_builtin(Value::Symbolic(expr), vec![]);
        assert!(result.is_ok());
        match result.unwrap() {
            Value::FunctionHandle(name) => {
                assert!(name.starts_with("__sym_fn_"));
            }
            _ => panic!("Expected function handle"),
        }
    }

    #[test]
    fn test_extract_var_list_string() {
        let result = extract_var_list(&Value::String("x".to_string())).unwrap();
        assert_eq!(result, vec!["x"]);
    }

    #[test]
    fn test_matlabfunction_compile_and_eval() {
        // Create x^2 + 3*x + 1
        let x = SymExpr::var("x");
        let expr = SymExpr::add(vec![
            SymExpr::pow(x.clone(), SymExpr::int(2)),
            SymExpr::mul(vec![SymExpr::int(3), x.clone()]),
            SymExpr::int(1),
        ]);

        // Compile directly using the symbolic crate (bypasses global registry)
        let compiled = runmat_symbolic::compile(&expr);

        // Evaluate at x = 2: 4 + 6 + 1 = 11
        let eval_result = compiled.eval(&[2.0]).unwrap();
        assert!(
            (eval_result - 11.0).abs() < 1e-10,
            "Expected 11.0, got {}",
            eval_result
        );

        // Evaluate at x = 0: 0 + 0 + 1 = 1
        let eval_result = compiled.eval(&[0.0]).unwrap();
        assert!(
            (eval_result - 1.0).abs() < 1e-10,
            "Expected 1.0, got {}",
            eval_result
        );

        // Evaluate at x = -1: 1 - 3 + 1 = -1
        let eval_result = compiled.eval(&[-1.0]).unwrap();
        assert!(
            (eval_result - (-1.0)).abs() < 1e-10,
            "Expected -1.0, got {}",
            eval_result
        );
    }

    #[test]
    fn test_matlabfunction_multivar() {
        // Create x^2 + y^2
        let x = SymExpr::var("x");
        let y = SymExpr::var("y");
        let expr = SymExpr::add(vec![
            SymExpr::pow(x.clone(), SymExpr::int(2)),
            SymExpr::pow(y.clone(), SymExpr::int(2)),
        ]);

        // Compile directly with explicit variable order
        let compiled = runmat_symbolic::compile_with_vars(&expr, &["x", "y"]);

        // Evaluate at x=3, y=4: 9 + 16 = 25
        let eval_result = compiled.eval(&[3.0, 4.0]).unwrap();
        assert!(
            (eval_result - 25.0).abs() < 1e-10,
            "Expected 25.0, got {}",
            eval_result
        );
    }

    #[test]
    fn test_matlabfunction_with_transcendental() {
        // Create sin(x) + cos(x)
        let x = SymExpr::var("x");
        let expr = SymExpr::add(vec![SymExpr::sin(x.clone()), SymExpr::cos(x.clone())]);

        // Compile directly
        let compiled = runmat_symbolic::compile(&expr);

        // Evaluate at x=0: sin(0) + cos(0) = 0 + 1 = 1
        let eval_result = compiled.eval(&[0.0]).unwrap();
        assert!(
            (eval_result - 1.0).abs() < 1e-10,
            "Expected 1.0, got {}",
            eval_result
        );

        // Evaluate at x=π/2: sin(π/2) + cos(π/2) = 1 + 0 = 1
        let eval_result = compiled.eval(&[std::f64::consts::FRAC_PI_2]).unwrap();
        assert!(
            (eval_result - 1.0).abs() < 1e-10,
            "Expected 1.0, got {}",
            eval_result
        );
    }
}
