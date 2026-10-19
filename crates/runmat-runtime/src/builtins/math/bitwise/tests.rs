use super::{
    bitcmp::bitcmp_builtin, bitget::bitget_builtin, bitset::bitset_builtin,
    bitshift::bitshift_builtin,
};
use futures::executor::block_on;
use runmat_value::{IntValue, Value};

#[test]
fn direct_operations_preserve_integer_classes() {
    let Value::Int(complement) =
        block_on(bitcmp_builtin(vec![Value::Int(IntValue::U8(15))])).expect("bitcmp")
    else {
        panic!("expected integer")
    };
    assert_eq!(complement, IntValue::U8(240));
    let Value::Int(bit) = block_on(bitget_builtin(vec![
        Value::Int(IntValue::U16(5)),
        Value::Num(3.0),
    ]))
    .expect("bitget") else {
        panic!("expected integer")
    };
    assert_eq!(bit, IntValue::U16(1));
    let Value::Int(set) = block_on(bitset_builtin(vec![
        Value::Int(IntValue::U16(0)),
        Value::Num(4.0),
    ]))
    .expect("bitset") else {
        panic!("expected integer")
    };
    assert_eq!(set, IntValue::U16(8));
    let Value::Int(shifted) = block_on(bitshift_builtin(vec![
        Value::Int(IntValue::I16(-8)),
        Value::Num(-2.0),
    ]))
    .expect("bitshift") else {
        panic!("expected integer")
    };
    assert_eq!(shifted, IntValue::I16(-2));
}
