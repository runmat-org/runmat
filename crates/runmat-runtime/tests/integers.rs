use runmat_builtins::{IntValue, Value};
use runmat_runtime::{elementwise_add, elementwise_div, elementwise_mul, elementwise_sub};

#[test]
fn integer_scalar_variants_arithmetic() {
	let i8v = Value::Int(IntValue::I8(5));
	let i16v = Value::Int(IntValue::I16(-2));
	let u32v = Value::Int(IntValue::U32(7));
	let u64v = Value::Int(IntValue::U64(3));
	assert_eq!(elementwise_add(&i8v, &i16v).unwrap(), Value::Num(3.0));
	assert_eq!(elementwise_sub(&u32v, &u64v).unwrap(), Value::Num(4.0));
	assert_eq!(elementwise_mul(&i8v, &u32v).unwrap(), Value::Num(35.0));
	assert_eq!(elementwise_div(&u32v, &i16v).unwrap(), Value::Num(7.0/(-2.0)));
}

#[test]
fn integer_promotion_with_double() {
	let i = Value::Int(IntValue::I32(4));
	let d = Value::Num(2.5);
	assert_eq!(elementwise_add(&i, &d).unwrap(), Value::Num(6.5));
	assert_eq!(elementwise_mul(&d, &i).unwrap(), Value::Num(10.0));
}

#[test]
fn integer_class_and_string() {
	let i = Value::Int(IntValue::U16(42));
	let cls = runmat_runtime::call_builtin("class", [i.clone()].as_slice()).unwrap();
	if let Value::String(s) = cls { assert_eq!(s, "uint16"); } else { panic!(); }
	let s = runmat_runtime::call_builtin("string", [i].as_slice()).unwrap(); if let Value::StringArray(sa) = s { assert_eq!(sa.data[0], "42"); } else { panic!(); }
}
