#[test]
fn runtime_and_static_constant_identity_sets_match() {
    let runtime = runmat_builtins::constants()
        .into_iter()
        .map(|constant| constant.name)
        .collect::<std::collections::BTreeSet<_>>();
    let catalog = runmat_builtins::builtin_constant_catalog_entries()
        .iter()
        .map(|constant| constant.name)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(runtime, catalog);
}

#[test]
fn constants_preserve_exact_values_and_case_spellings() {
    use runmat_value::Value;
    let values = runmat_builtins::constants();
    let lookup = |name| {
        &values
            .iter()
            .find(|entry| entry.name == name)
            .expect("constant")
            .value
    };
    assert_eq!(lookup("pi"), &Value::Num(std::f64::consts::PI));
    assert_eq!(lookup("eps"), &Value::Num(f64::EPSILON));
    assert_eq!(lookup("sqrt2"), &Value::Num(std::f64::consts::SQRT_2));
    assert_eq!(lookup("i"), &Value::Complex(0.0, 1.0));
    assert_eq!(lookup("j"), &Value::Complex(0.0, 1.0));
    assert_eq!(lookup("true"), &Value::Bool(true));
    assert_eq!(lookup("false"), &Value::Bool(false));
    for name in ["inf", "Inf"] {
        assert!(
            matches!(lookup(name), Value::Num(value) if value.is_infinite() && value.is_sign_positive())
        );
    }
    for name in ["nan", "NaN"] {
        assert!(matches!(lookup(name), Value::Num(value) if value.is_nan()));
    }
}
