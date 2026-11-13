use runmat_value::Value;

runmat_macros::register_constant!(
    "eps",
    Value::Num(f64::EPSILON),
    "crate::builtins::constants::core::eps"
);
