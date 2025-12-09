//! Global constants registered into the runtime (variables, not functions).
//! This replaces legacy registrations in `src/constants.rs`.

#[cfg(not(target_arch = "wasm32"))]
use runmat_builtins::Value;
use runmat_macros::register_constant;

register_constant!("pi", Value::Num(std::f64::consts::PI));
register_constant!("e", Value::Num(std::f64::consts::E));
register_constant!("eps", Value::Num(f64::EPSILON));
register_constant!("sqrt2", Value::Num(std::f64::consts::SQRT_2));
register_constant!("inf", Value::Num(f64::INFINITY));
register_constant!("Inf", Value::Num(f64::INFINITY));
register_constant!("nan", Value::Num(f64::NAN));
register_constant!("NaN", Value::Num(f64::NAN));
register_constant!("true", Value::Bool(true));
register_constant!("false", Value::Bool(false));
