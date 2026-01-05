//! Global constants registered into the runtime (variables, not functions).
//! This replaces legacy registrations in `src/constants.rs`.

use runmat_builtins::Value;

// Numeric constants
runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "pi", value: Value::Num(std::f64::consts::PI) }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "e", value: Value::Num(std::f64::consts::E) }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "eps", value: Value::Num(f64::EPSILON) }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "sqrt2", value: Value::Num(std::f64::consts::SQRT_2) }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "i", value: Value::Complex(0.0, 1.0) }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "j", value: Value::Complex(0.0, 1.0) }
}

// Infinity and NaN (both lowercase and MATLAB-style capitalised names)
runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "inf", value: Value::Num(f64::INFINITY) }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "Inf", value: Value::Num(f64::INFINITY) }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "nan", value: Value::Num(f64::NAN) }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "NaN", value: Value::Num(f64::NAN) }
}

// Logical constants
runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "true", value: Value::Bool(true) }
}

runmat_builtins::inventory::submit! {
    runmat_builtins::Constant { name: "false", value: Value::Bool(false) }
}
