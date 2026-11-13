use runmat_value::Value;

runmat_macros::register_constant!(
    "i",
    Value::Complex(0.0, 1.0),
    "crate::builtins::constants::core::i"
);
