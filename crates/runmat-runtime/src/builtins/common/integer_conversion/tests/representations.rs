use futures::executor::block_on;
use runmat_types::IntegerClass;
use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage, Value};

use super::super::{cast_value, IntegerClassExt};

#[test]
fn typed_complex_conversion_preserves_wide_components_before_saturation() {
    let input = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
            IntegerStorage::U64(vec![1, 2]),
        )
        .expect("matching components"),
        vec![1, 2],
    )
    .expect("input");
    let Value::ComplexTensor(output) =
        block_on(cast_value(Value::ComplexTensor(input), IntegerClass::Int64)).expect("conversion")
    else {
        panic!("conversion must preserve complex storage");
    };
    assert_eq!(
        output.integer_storage().cloned(),
        Some(
            IntegerComplexStorage::new(
                IntegerStorage::I64(vec![i64::MAX, i64::MAX]),
                IntegerStorage::I64(vec![1, 2]),
            )
            .expect("matching components")
        )
    );
}

#[test]
fn every_integer_class_preserves_empty_complex_shape_and_special_values() {
    for target in [
        IntegerClass::Int8,
        IntegerClass::Int16,
        IntegerClass::Int32,
        IntegerClass::Int64,
        IntegerClass::UInt8,
        IntegerClass::UInt16,
        IntegerClass::UInt32,
        IntegerClass::UInt64,
    ] {
        let empty = ComplexTensor::new(Vec::new(), vec![2, 0, 3]).expect("empty");
        let Value::ComplexTensor(output) =
            block_on(cast_value(Value::ComplexTensor(empty), target)).expect("empty conversion")
        else {
            panic!("conversion must preserve complex storage");
        };
        assert_eq!(output.shape, vec![2, 0, 3]);
        let storage = output.integer_storage().expect("storage");
        assert_eq!(storage.real, target.storage(Vec::new()));
        assert_eq!(storage.imag, target.storage(Vec::new()));

        let special = ComplexTensor::new(
            vec![(f64::NAN, 0.0), (f64::INFINITY, f64::NEG_INFINITY)],
            vec![1, 2],
        )
        .expect("special values");
        let Value::ComplexTensor(output) =
            block_on(cast_value(Value::ComplexTensor(special), target)).expect("conversion")
        else {
            panic!("conversion must preserve complex storage");
        };
        let storage = output.integer_storage().expect("storage");
        assert_eq!(
            storage.real,
            target.storage(vec![
                target.cast_scalar(f64::NAN),
                target.cast_scalar(f64::INFINITY),
            ])
        );
        assert_eq!(
            storage.imag,
            target.storage(vec![
                target.cast_scalar(0.0),
                target.cast_scalar(f64::NEG_INFINITY),
            ])
        );
    }
}
