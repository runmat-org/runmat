use runmat_value::Value;

/// Resolve a named constant through its runtime registration.
///
/// Names are exact: alternate spellings must have their own registration. Both
/// bytecode and native execution use this lookup, so a symbolic MIR constant
/// cannot accidentally become a string or acquire a second value authority.
pub fn builtin_constant_value(name: &str) -> Option<Value> {
    #[cfg(target_arch = "wasm32")]
    crate::builtins::wasm_registry::register_all();

    runmat_builtins::constants()
        .into_iter()
        .find(|constant| constant.name == name)
        .map(|constant| constant.value.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_uses_exact_registered_names() {
        assert!(
            matches!(builtin_constant_value("pi"), Some(Value::Num(value))
            if value.to_bits() == std::f64::consts::PI.to_bits())
        );
        assert!(
            matches!(builtin_constant_value("NaN"), Some(Value::Num(value))
            if value.is_nan())
        );
        assert!(builtin_constant_value("PI").is_none());
        assert!(builtin_constant_value("definitely_missing_constant").is_none());
    }
}
