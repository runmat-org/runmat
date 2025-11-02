use runmat_builtins::{IntValue, Value};

#[test]
fn integer_scalar_variants_arithmetic() {
    let i8v = Value::Int(IntValue::I8(5));
    let i16v = Value::Int(IntValue::I16(-2));
    let u32v = Value::Int(IntValue::U32(7));
    let u64v = Value::Int(IntValue::U64(3));
    assert_eq!(runmat_runtime::call_builtin("plus", &[i8v.clone(), i16v.clone()]).unwrap(), Value::Num(3.0));
    assert_eq!(runmat_runtime::call_builtin("minus", &[u32v.clone(), u64v.clone()]).unwrap(), Value::Num(4.0));
    assert_eq!(runmat_runtime::call_builtin("times", &[i8v.clone(), u32v.clone()]).unwrap(), Value::Num(35.0));
    assert_eq!(
        runmat_runtime::call_builtin("rdivide", &[u32v.clone(), i16v.clone()]).unwrap(),
        Value::Num(7.0 / (-2.0))
    );
}

#[test]
fn integer_promotion_with_double() {
    let i = Value::Int(IntValue::I32(4));
    let d = Value::Num(2.5);
    assert_eq!(runmat_runtime::call_builtin("plus", &[i.clone(), d.clone()]).unwrap(), Value::Num(6.5));
    assert_eq!(runmat_runtime::call_builtin("times", &[d.clone(), i.clone()]).unwrap(), Value::Num(10.0));
}

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
