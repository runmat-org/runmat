use runmat_types::IntegerClass;
use runmat_value::{
    CharArray, IntValue, IntegerStorage, SymbolicArray, SymbolicExpr, Tensor, Value,
};

use crate::builtins::common::integer_conversion::IntegerClassExt;

fn call(name: &str, value: Value) -> Value {
    crate::dispatcher::call_builtin(name, &[value]).expect("integer conversion")
}

fn classes() -> [(IntegerClass, &'static str); 8] {
    [
        (IntegerClass::Int8, "int8"),
        (IntegerClass::Int16, "int16"),
        (IntegerClass::Int32, "int32"),
        (IntegerClass::Int64, "int64"),
        (IntegerClass::UInt8, "uint8"),
        (IntegerClass::UInt16, "uint16"),
        (IntegerClass::UInt32, "uint32"),
        (IntegerClass::UInt64, "uint64"),
    ]
}

#[test]
fn every_identity_rounds_saturates_and_retains_its_typed_scalar() {
    for (class, name) in classes() {
        assert_eq!(
            call(name, Value::Num(3.5)),
            Value::Int(class.cast_scalar(3.5))
        );
        assert_eq!(
            call(name, Value::Num(f64::INFINITY)),
            Value::Int(class.cast_scalar(f64::INFINITY))
        );
        assert_eq!(
            call(name, Value::Num(f64::NEG_INFINITY)),
            Value::Int(class.cast_scalar(f64::NEG_INFINITY))
        );
    }
}

#[test]
fn every_identity_preserves_tensor_shape_and_uses_authoritative_storage() {
    let input = Tensor::new(vec![-2.5, 1.5, 300.0, 4.0], vec![2, 2]).expect("input");
    for (class, name) in classes() {
        let Value::Tensor(output) = call(name, Value::Tensor(input.clone())) else {
            panic!("{name} must return a tensor");
        };
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(
            output.integer_storage().map(IntegerStorage::integer_class),
            Some(class)
        );
    }
}

#[test]
fn exact_wide_integer_input_never_rounds_through_binary64() {
    let input = Tensor::new_integer(
        IntegerStorage::U64(vec![0, 1_u64 << 63, u64::MAX]),
        vec![1, 3],
    )
    .expect("input");
    let Value::Tensor(output) = call("int64", Value::Tensor(input)) else {
        panic!("int64 must return a tensor");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::I64(vec![0, i64::MAX, i64::MAX]))
    );
}

#[test]
fn logical_and_character_arrays_convert_for_every_identity() {
    let logical = runmat_value::LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical");
    let chars = CharArray::new_row("Az");
    for (class, name) in classes() {
        for (input, expected) in [
            (Value::LogicalArray(logical.clone()), vec![1.0, 0.0]),
            (Value::CharArray(chars.clone()), vec![65.0, 122.0]),
        ] {
            let Value::Tensor(output) = call(name, input) else {
                panic!("{name} must return a tensor");
            };
            assert_eq!(output.shape, vec![1, 2]);
            assert_eq!(
                output.integer_storage().map(IntegerStorage::integer_class),
                Some(class)
            );
            assert_eq!(output.materialize_f64(), expected);
        }
    }
}

#[test]
fn symbolic_constants_convert_and_symbolic_variables_are_rejected() {
    let constants = SymbolicArray::new(
        vec![SymbolicExpr::constant(-2.4), SymbolicExpr::constant(3.6)],
        vec![1, 2],
    )
    .expect("constants");
    for (class, name) in classes() {
        let Value::Tensor(output) = call(name, Value::SymbolicArray(constants.clone())) else {
            panic!("{name} must return a tensor");
        };
        assert_eq!(
            output.integer_storage().map(IntegerStorage::integer_class),
            Some(class)
        );
        assert!(crate::dispatcher::call_builtin(
            name,
            &[Value::Symbolic(SymbolicExpr::variable("x"))]
        )
        .is_err());
    }
}

#[test]
fn paired_complex_input_stays_complex_for_every_identity() {
    for (class, name) in classes() {
        let Value::ComplexTensor(output) = call(name, Value::Complex(1.0, 1e-48)) else {
            panic!("{name} must preserve the complex domain");
        };
        let storage = output.integer_storage().expect("integer complex storage");
        assert_eq!(storage.real.integer_class(), class);
        assert_eq!(storage.imag.integer_class(), class);
        assert_eq!(storage.real.value_at(0), Some(class.cast_scalar(1.0)));
        assert_eq!(storage.imag.value_at(0), Some(class.cast_scalar(0.0)));
    }
}

#[test]
fn integer_class_is_the_authoritative_source_type() {
    let target: IntegerClass = IntValue::U32(1).integer_class();
    assert_eq!(target, IntegerClass::UInt32);
}
