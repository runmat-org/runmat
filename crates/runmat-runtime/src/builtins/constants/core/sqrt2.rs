use runmat_value::Value;

runmat_macros::register_constant!(
    "sqrt2",
    Value::Num(std::f64::consts::SQRT_2),
    "crate::builtins::constants::core::sqrt2"
);
