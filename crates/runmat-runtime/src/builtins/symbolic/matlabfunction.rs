//! MATLAB-compatible `matlabFunction` builtin

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::{compile_with_vars, SymExpr};

/// Convert a symbolic expression to a numeric function
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

    let free_vars = sym_expr.free_vars();
    let mut var_names: Vec<String> = free_vars.iter().map(|s| s.name.clone()).collect();
    var_names.sort();

    for i in 0..rest.len() {
        if is_vars_keyword(&rest[i]) && i + 1 < rest.len() {
            var_names = extract_var_list(&rest[i + 1])?;
            break;
        }
    }

    let var_refs: Vec<&str> = var_names.iter().map(|s| s.as_str()).collect();
    let compiled = compile_with_vars(&sym_expr, &var_refs);

    let func_name = format!("__sym_fn_{}", compiled.variables.join("_"));

    register_compiled_function(&func_name, compiled, sym_expr);

    Ok(Value::FunctionHandle(func_name))
}

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
}
