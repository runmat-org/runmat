use futures::executor::block_on;
use runmat_types::IntegerClass;
use runmat_value::{ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, Tensor, Value};

use super::super::{cast_value, IntegerClassExt};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn low_rank_real_and_complex_conversions_preserve_existing_shape_policy() {
    let rank_zero = Tensor::new(vec![7.0], Vec::new()).expect("rank-zero real");
    assert_eq!(
        block_on(cast_value(Value::Tensor(rank_zero), IntegerClass::Int16))
            .expect("rank-zero conversion"),
        Value::Int(IntValue::I16(7))
    );

    let rank_one = Tensor::new(vec![7.0, 8.0], vec![2]).expect("rank-one real");
    let Value::Tensor(rank_one) =
        block_on(cast_value(Value::Tensor(rank_one), IntegerClass::Int16))
            .expect("rank-one conversion")
    else {
        panic!("rank-one conversion must remain an array")
    };
    assert_eq!(rank_one.shape(), &[2]);
    assert_eq!(
        rank_one.integer_storage(),
        Some(&IntegerStorage::I16(vec![7, 8]))
    );

    for (shape, values) in [
        (Vec::new(), vec![(1.0, 2.0)]),
        (vec![2], vec![(1.0, 2.0), (3.0, 4.0)]),
    ] {
        let input = ComplexTensor::new(values, shape.clone()).expect("low-rank complex");
        let Value::ComplexTensor(output) =
            block_on(cast_value(Value::ComplexTensor(input), IntegerClass::Int16))
                .expect("complex conversion")
        else {
            panic!("complex conversion must remain an array")
        };
        assert_eq!(output.shape(), shape);
        let storage = output.integer_storage().expect("integer-complex storage");
        let expected_real = if shape.is_empty() {
            vec![1]
        } else {
            vec![1, 3]
        };
        let expected_imag = if shape.is_empty() {
            vec![2]
        } else {
            vec![2, 4]
        };
        assert_eq!(storage.real(), &IntegerStorage::I16(expected_real));
        assert_eq!(storage.imag(), &IntegerStorage::I16(expected_imag));
    }
}

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
        assert_eq!(output.shape(), &[2, 0, 3]);
        let storage = output.integer_storage().expect("storage");
        assert_eq!(storage.real(), &target.storage(Vec::new()));
        assert_eq!(storage.imag(), &target.storage(Vec::new()));

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
            storage.real(),
            &target.storage(vec![
                target.cast_scalar(f64::NAN),
                target.cast_scalar(f64::INFINITY),
            ])
        );
        assert_eq!(
            storage.imag(),
            &target.storage(vec![
                target.cast_scalar(0.0),
                target.cast_scalar(f64::NEG_INFINITY),
            ])
        );
    }
}
