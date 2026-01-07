//! Global constants registered into the runtime (variables, not functions).
//! This replaces legacy registrations in `src/constants.rs`.

use runmat_builtins::Value;
use runmat_macros::register_constant;

// Numeric constants
register_constant!(
    "pi",
    Value::Num(std::f64::consts::PI),
    "crate::builtins::constants"
);

register_constant!(
    "e",
    Value::Num(std::f64::consts::E),
    "crate::builtins::constants"
);

register_constant!(
    "eps",
    Value::Num(f64::EPSILON),
    "crate::builtins::constants"
);

register_constant!(
    "sqrt2",
    Value::Num(std::f64::consts::SQRT_2),
    "crate::builtins::constants"
);

register_constant!("i", Value::Complex(0.0, 1.0), "crate::builtins::constants");

register_constant!("j", Value::Complex(0.0, 1.0), "crate::builtins::constants");

// Infinity and NaN (both lowercase and MATLAB-style capitalised names)
register_constant!(
    "inf",
    Value::Num(f64::INFINITY),
    "crate::builtins::constants"
);

register_constant!(
    "Inf",
    Value::Num(f64::INFINITY),
    "crate::builtins::constants"
);

register_constant!("nan", Value::Num(f64::NAN), "crate::builtins::constants");

register_constant!("NaN", Value::Num(f64::NAN), "crate::builtins::constants");

// Logical constants
register_constant!("true", Value::Bool(true), "crate::builtins::constants");

register_constant!("false", Value::Bool(false), "crate::builtins::constants");
