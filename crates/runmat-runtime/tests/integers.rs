#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use runmat_builtins::{IntValue, IntegerStorage, Tensor, Value};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_scalar_variants_arithmetic() {
    let i8v = Value::Int(IntValue::I8(5));
    let i16v = Value::Int(IntValue::I16(-2));
    let u32v = Value::Int(IntValue::U32(7));
    let u64v = Value::Int(IntValue::U64(3));
    assert!(runmat_runtime::call_builtin("plus", &[i8v.clone(), i16v.clone()]).is_err());
    assert!(runmat_runtime::call_builtin("minus", &[u32v.clone(), u64v.clone()]).is_err());
    assert!(runmat_runtime::call_builtin("times", &[i8v.clone(), u32v.clone()]).is_err());
    assert_eq!(
        runmat_runtime::call_builtin("rdivide", &[u32v.clone(), i16v.clone()]).unwrap(),
        Value::Num(7.0 / (-2.0))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_promotion_with_double() {
    let i = Value::Int(IntValue::I32(4));
    let d = Value::Num(2.5);
    assert_eq!(
        runmat_runtime::call_builtin("plus", &[i.clone(), d.clone()]).unwrap(),
        Value::Int(IntValue::I32(7))
    );
    assert_eq!(
        runmat_runtime::call_builtin("times", &[d.clone(), i.clone()]).unwrap(),
        Value::Int(IntValue::I32(10))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_array_elementwise_arithmetic_preserves_native_storage() {
    let signed = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I8(vec![100, -100]), vec![1, 2]).expect("signed input"),
    );
    let signed_rhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I8(vec![30, 30]), vec![1, 2]).expect("signed rhs"),
    );
    let unsigned = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 2]), vec![1, 2])
            .expect("unsigned input"),
    );
    let unsigned_rhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![1, 3]), vec![1, 2]).expect("unsigned rhs"),
    );

    assert_eq!(
        runmat_runtime::call_builtin("plus", &[signed.clone(), signed_rhs.clone()]).expect("plus"),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::I8(vec![127, -70]), vec![1, 2])
                .expect("plus expected"),
        )
    );
    assert_eq!(
        runmat_runtime::call_builtin("minus", &[signed.clone(), signed_rhs]).expect("minus"),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::I8(vec![70, -128]), vec![1, 2])
                .expect("minus expected"),
        )
    );
    assert_eq!(
        runmat_runtime::call_builtin("times", &[unsigned, unsigned_rhs]).expect("times"),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 6]), vec![1, 2])
                .expect("times expected"),
        )
    );
    assert_eq!(
        runmat_runtime::call_builtin("times", &[signed.clone(), Value::Num(2.6)])
            .expect("integer array with scalar double"),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::I8(vec![127, -128]), vec![1, 2])
                .expect("scalar double expected"),
        )
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_relations_keep_uint64_and_signed_values_exact() {
    let max_unsigned = Value::Int(IntValue::U64(u64::MAX));
    let max_signed = Value::Int(IntValue::I64(i64::MAX));
    assert_eq!(
        runmat_runtime::call_builtin("eq", &[max_unsigned.clone(), max_signed.clone()])
            .expect("eq"),
        Value::Bool(false)
    );
    assert_eq!(
        runmat_runtime::call_builtin("ne", &[max_unsigned.clone(), max_signed.clone()])
            .expect("ne"),
        Value::Bool(true)
    );
    assert_eq!(
        runmat_runtime::call_builtin("gt", &[max_unsigned.clone(), max_signed.clone()])
            .expect("gt"),
        Value::Bool(true)
    );
    assert_eq!(
        runmat_runtime::call_builtin("ge", &[max_unsigned, max_signed]).expect("ge"),
        Value::Bool(true)
    );
    assert_eq!(
        runmat_runtime::call_builtin(
            "lt",
            &[Value::Int(IntValue::I8(-1)), Value::Int(IntValue::U8(0))],
        )
        .expect("lt"),
        Value::Bool(true)
    );
    assert_eq!(
        runmat_runtime::call_builtin(
            "le",
            &[Value::Int(IntValue::I8(-1)), Value::Int(IntValue::U8(0))],
        )
        .expect("le"),
        Value::Bool(true)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_relation_size_errors_keep_builtin_identifier() {
    let lhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![1, 2]).expect("lhs"),
    );
    let rhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U8(vec![1, 2, 3]), vec![1, 3]).expect("rhs"),
    );
    let error = runmat_runtime::call_builtin("eq", &[lhs, rhs]).expect_err("size mismatch");
    assert_eq!(error.identifier(), Some("RunMat:eq:SizeMismatch"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sum_native_preserves_exact_integer_storage_while_default_is_double() {
    let input = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1, 5, 7]), vec![2, 2])
            .expect("input"),
    );
    assert_eq!(
        runmat_runtime::call_builtin("sum", &[input.clone(), Value::from("native")])
            .expect("native sum"),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 12]), vec![1, 2])
                .expect("native output"),
        )
    );
    assert_eq!(
        runmat_runtime::call_builtin(
            "sum",
            &[input.clone(), Value::Num(2.0), Value::from("native")],
        )
        .expect("native row sum"),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 8]), vec![2, 1])
                .expect("native row output"),
        )
    );
    assert_eq!(
        runmat_runtime::call_builtin(
            "sum",
            &[Value::Int(IntValue::U64(u64::MAX)), Value::from("native")],
        )
        .expect("native scalar sum"),
        Value::Int(IntValue::U64(u64::MAX))
    );
    match runmat_runtime::call_builtin("sum", &[input]).expect("default sum") {
        Value::Tensor(tensor) => {
            assert!(tensor.integer_storage().is_none());
            assert_eq!(tensor.shape, vec![1, 2]);
        }
        other => panic!("expected double tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_native_preserves_integer_storage_while_default_is_double() {
    let input = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U8(vec![2, 200, 3, 2]), vec![2, 2]).expect("input"),
    );
    assert_eq!(
        runmat_runtime::call_builtin("prod", &[input.clone(), Value::from("native")])
            .expect("native product"),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(vec![255, 6]), vec![1, 2])
                .expect("native output"),
        )
    );
    assert_eq!(
        runmat_runtime::call_builtin(
            "prod",
            &[Value::Int(IntValue::I64(i64::MIN)), Value::from("native")],
        )
        .expect("native scalar product"),
        Value::Int(IntValue::I64(i64::MIN))
    );
    match runmat_runtime::call_builtin("prod", &[input]).expect("default product") {
        Value::Tensor(tensor) => assert!(tensor.integer_storage().is_none()),
        other => panic!("expected double tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_class_and_string() {
    let i = Value::Int(IntValue::U16(42));
    let cls = runmat_runtime::call_builtin("class", [i.clone()].as_slice()).unwrap();
    if let Value::String(s) = cls {
        assert_eq!(s, "uint16");
    } else {
        panic!();
    }
    let s = runmat_runtime::call_builtin("string", [i].as_slice()).unwrap();
    if let Value::StringArray(sa) = s {
        assert_eq!(sa.data[0], "42");
    } else {
        panic!();
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn uint16_builtin_dispatches_and_casts_array() {
    let input = Value::Tensor(Tensor::new(vec![3.49, -2.0, 70000.0], vec![1, 3]).unwrap());
    let output = runmat_runtime::call_builtin("uint16", &[input]).expect("uint16 builtin");
    match output {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 3]);
            assert_eq!(tensor.data, vec![3.0, 0.0, u16::MAX as f64]);
        }
        other => panic!("expected tensor output, got {other:?}"),
    }
}
