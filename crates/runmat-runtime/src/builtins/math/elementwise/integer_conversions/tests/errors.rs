use runmat_value::Value;

#[test]
fn every_identity_exposes_its_catalog_signature() {
    for (name, descriptor) in [
        ("int8", &runmat_builtins::INT8_DESCRIPTOR),
        ("int16", &runmat_builtins::INT16_DESCRIPTOR),
        ("int32", &runmat_builtins::INT32_DESCRIPTOR),
        ("int64", &runmat_builtins::INT64_DESCRIPTOR),
        ("uint8", &runmat_builtins::UINT8_DESCRIPTOR),
        ("uint16", &runmat_builtins::UINT16_DESCRIPTOR),
        ("uint32", &runmat_builtins::UINT32_DESCRIPTOR),
        ("uint64", &runmat_builtins::UINT64_DESCRIPTOR),
    ] {
        let label = format!("Y = {name}(X)");
        assert!(descriptor
            .signatures
            .iter()
            .any(|signature| signature.label == label));
    }
}

#[test]
fn unsupported_input_and_arity_errors_keep_identity_specific_identifiers() {
    for name in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let input = crate::dispatcher::call_builtin(name, &[Value::String("x".into())])
            .expect_err("string input must fail");
        let expected_input = format!("RunMat:{name}:InvalidInput");
        assert_eq!(input.identifier.as_deref(), Some(expected_input.as_str()));

        let arity = crate::dispatcher::call_builtin(name, &[Value::Num(1.0), Value::Num(2.0)])
            .expect_err("extra input must fail");
        let expected_arity = format!("RunMat:{name}:InvalidArgument");
        assert_eq!(arity.identifier.as_deref(), Some(expected_arity.as_str()));
    }
}
