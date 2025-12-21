//! Arithmetic operations for symbolic expressions
//!
//! Provides overloaded +, -, *, /, ^ operations for symbolic values.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;
use runmat_symbolic::SymExpr;

/// Symbolic addition: sym + sym, sym + num, num + sym
fn symbolic_add(a: SymExpr, b: SymExpr) -> SymExpr {
    SymExpr::add(vec![a, b])
}

/// Symbolic subtraction: sym - sym, sym - num, num - sym
fn symbolic_sub(a: SymExpr, b: SymExpr) -> SymExpr {
    SymExpr::add(vec![a, SymExpr::neg(b)])
}

/// Symbolic multiplication: sym * sym, sym * num, num * sym
fn symbolic_mul(a: SymExpr, b: SymExpr) -> SymExpr {
    SymExpr::mul(vec![a, b])
}

/// Symbolic division: sym / sym
fn symbolic_div(a: SymExpr, b: SymExpr) -> SymExpr {
    SymExpr::mul(vec![a, SymExpr::pow(b, SymExpr::int(-1))])
}

/// Symbolic power: sym ^ sym
fn symbolic_pow(a: SymExpr, b: SymExpr) -> SymExpr {
    SymExpr::pow(a, b)
}

/// Convert a numeric value to a symbolic expression
pub fn value_to_sym(v: &Value) -> Option<SymExpr> {
    match v {
        Value::Symbolic(e) => Some(e.clone()),
        Value::Num(n) => Some(SymExpr::float(*n)),
        Value::Int(i) => Some(SymExpr::int(i.to_i64())),
        Value::Bool(b) => Some(SymExpr::int(if *b { 1 } else { 0 })),
        _ => None,
    }
}

/// Check if either value is symbolic
pub fn is_symbolic_operation(a: &Value, b: &Value) -> bool {
    matches!(a, Value::Symbolic(_)) || matches!(b, Value::Symbolic(_))
}

/// Symbolic plus builtin
#[runtime_builtin(
    name = "plus",
    category = "symbolic",
    summary = "Symbolic addition.",
    keywords = "plus,add,symbolic"
)]
fn plus_symbolic(a: Value, b: Value) -> Result<Value, String> {
    if !is_symbolic_operation(&a, &b) {
        return Err("plus_symbolic: requires at least one symbolic argument".to_string());
    }

    let sym_a =
        value_to_sym(&a).ok_or_else(|| format!("plus: cannot convert {:?} to symbolic", a))?;
    let sym_b =
        value_to_sym(&b).ok_or_else(|| format!("plus: cannot convert {:?} to symbolic", b))?;

    Ok(Value::Symbolic(symbolic_add(sym_a, sym_b)))
}

/// Symbolic minus builtin
#[runtime_builtin(
    name = "minus",
    category = "symbolic",
    summary = "Symbolic subtraction.",
    keywords = "minus,subtract,symbolic"
)]
fn minus_symbolic(a: Value, b: Value) -> Result<Value, String> {
    if !is_symbolic_operation(&a, &b) {
        return Err("minus_symbolic: requires at least one symbolic argument".to_string());
    }

    let sym_a =
        value_to_sym(&a).ok_or_else(|| format!("minus: cannot convert {:?} to symbolic", a))?;
    let sym_b =
        value_to_sym(&b).ok_or_else(|| format!("minus: cannot convert {:?} to symbolic", b))?;

    Ok(Value::Symbolic(symbolic_sub(sym_a, sym_b)))
}

/// Symbolic times builtin
#[runtime_builtin(
    name = "times",
    category = "symbolic",
    summary = "Symbolic multiplication.",
    keywords = "times,multiply,symbolic"
)]
fn times_symbolic(a: Value, b: Value) -> Result<Value, String> {
    if !is_symbolic_operation(&a, &b) {
        return Err("times_symbolic: requires at least one symbolic argument".to_string());
    }

    let sym_a =
        value_to_sym(&a).ok_or_else(|| format!("times: cannot convert {:?} to symbolic", a))?;
    let sym_b =
        value_to_sym(&b).ok_or_else(|| format!("times: cannot convert {:?} to symbolic", b))?;

    Ok(Value::Symbolic(symbolic_mul(sym_a, sym_b)))
}

/// Symbolic mtimes builtin (matrix multiplication for symbolic = same as times for scalars)
#[runtime_builtin(
    name = "mtimes",
    category = "symbolic",
    summary = "Symbolic matrix multiplication.",
    keywords = "mtimes,multiply,symbolic"
)]
fn mtimes_symbolic(a: Value, b: Value) -> Result<Value, String> {
    // For scalar symbolic, mtimes = times
    times_symbolic(a, b)
}

/// Symbolic rdivide builtin
#[runtime_builtin(
    name = "rdivide",
    category = "symbolic",
    summary = "Symbolic right division.",
    keywords = "rdivide,divide,symbolic"
)]
fn rdivide_symbolic(a: Value, b: Value) -> Result<Value, String> {
    if !is_symbolic_operation(&a, &b) {
        return Err("rdivide_symbolic: requires at least one symbolic argument".to_string());
    }

    let sym_a =
        value_to_sym(&a).ok_or_else(|| format!("rdivide: cannot convert {:?} to symbolic", a))?;
    let sym_b =
        value_to_sym(&b).ok_or_else(|| format!("rdivide: cannot convert {:?} to symbolic", b))?;

    Ok(Value::Symbolic(symbolic_div(sym_a, sym_b)))
}

/// Symbolic mrdivide builtin (matrix right division = rdivide for scalars)
#[runtime_builtin(
    name = "mrdivide",
    category = "symbolic",
    summary = "Symbolic matrix right division.",
    keywords = "mrdivide,divide,symbolic"
)]
fn mrdivide_symbolic(a: Value, b: Value) -> Result<Value, String> {
    rdivide_symbolic(a, b)
}

/// Symbolic power builtin
#[runtime_builtin(
    name = "power",
    category = "symbolic",
    summary = "Symbolic element-wise power.",
    keywords = "power,pow,symbolic"
)]
fn power_symbolic(a: Value, b: Value) -> Result<Value, String> {
    if !is_symbolic_operation(&a, &b) {
        return Err("power_symbolic: requires at least one symbolic argument".to_string());
    }

    let sym_a =
        value_to_sym(&a).ok_or_else(|| format!("power: cannot convert {:?} to symbolic", a))?;
    let sym_b =
        value_to_sym(&b).ok_or_else(|| format!("power: cannot convert {:?} to symbolic", b))?;

    Ok(Value::Symbolic(symbolic_pow(sym_a, sym_b)))
}

/// Symbolic mpower builtin (matrix power = power for scalars)
#[runtime_builtin(
    name = "mpower",
    category = "symbolic",
    summary = "Symbolic matrix power.",
    keywords = "mpower,pow,symbolic"
)]
fn mpower_symbolic(a: Value, b: Value) -> Result<Value, String> {
    power_symbolic(a, b)
}

/// Symbolic uminus builtin
#[runtime_builtin(
    name = "uminus",
    category = "symbolic",
    summary = "Symbolic unary minus.",
    keywords = "uminus,negate,symbolic"
)]
fn uminus_symbolic(a: Value) -> Result<Value, String> {
    match a {
        Value::Symbolic(expr) => Ok(Value::Symbolic(SymExpr::neg(expr))),
        _ => Err("uminus_symbolic: requires symbolic argument".to_string()),
    }
}

/// Symbolic uplus builtin (identity)
#[runtime_builtin(
    name = "uplus",
    category = "symbolic",
    summary = "Symbolic unary plus.",
    keywords = "uplus,symbolic"
)]
fn uplus_symbolic(a: Value) -> Result<Value, String> {
    match a {
        Value::Symbolic(expr) => Ok(Value::Symbolic(expr)),
        _ => Err("uplus_symbolic: requires symbolic argument".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_add() {
        let x = SymExpr::var("x");
        let y = SymExpr::var("y");

        let result = plus_symbolic(Value::Symbolic(x), Value::Symbolic(y)).unwrap();
        match result {
            Value::Symbolic(expr) => assert!(expr.is_add()),
            _ => panic!("Expected symbolic result"),
        }
    }

    #[test]
    fn test_symbolic_mul_with_numeric() {
        let x = SymExpr::var("x");

        let result = times_symbolic(Value::Symbolic(x), Value::Num(2.0)).unwrap();
        match result {
            Value::Symbolic(expr) => assert!(expr.is_mul()),
            _ => panic!("Expected symbolic result"),
        }
    }

    #[test]
    fn test_symbolic_power() {
        let x = SymExpr::var("x");

        let result = power_symbolic(Value::Symbolic(x), Value::Num(2.0)).unwrap();
        match result {
            Value::Symbolic(expr) => assert!(expr.is_pow()),
            _ => panic!("Expected symbolic result"),
        }
    }

    #[test]
    fn test_symbolic_uminus() {
        let x = SymExpr::var("x");

        let result = uminus_symbolic(Value::Symbolic(x)).unwrap();
        match result {
            Value::Symbolic(_expr) => {}
            _ => panic!("Expected symbolic result"),
        }
    }
}
