use runmat_value::Value;

runmat_macros::register_constant!(
    "pi",
    Value::Num(std::f64::consts::PI),
    "crate::builtins::constants::core::pi"
);
