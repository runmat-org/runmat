use runmat_value::Value;

runmat_macros::register_constant!(
    "j",
    Value::Complex(0.0, 1.0),
    "crate::builtins::constants::core::j"
);
