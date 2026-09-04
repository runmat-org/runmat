use super::*;
use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage};

fn storages() -> [IntegerStorage; 8] {
    [
        IntegerStorage::I8(vec![4, 7, i8::MAX]),
        IntegerStorage::I16(vec![4, 700, i16::MAX]),
        IntegerStorage::I32(vec![4, i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![4, i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![4, 7, u8::MAX]),
        IntegerStorage::U16(vec![4, 700, u16::MAX]),
        IntegerStorage::U32(vec![4, 9_007_199, u32::MAX]),
        IntegerStorage::U64(vec![4, 9_007_199_254_740_993, u64::MAX]),
    ]
}

#[test]
fn every_integer_class_preserves_exact_scalar_and_vector_storage() {
    for storage in storages() {
        let values = storage.exact_values();
        let scalar = Tensor::new_integer(
            storage
                .from_exact_values_like(vec![values[0].clone()])
                .unwrap(),
            vec![1, 1],
        )
        .unwrap();
        let selection = Tensor::new_integer(
            storage
                .from_exact_values_like(vec![one_like(&values[0])])
                .unwrap(),
            vec![1, 1],
        )
        .unwrap();
        assert_eq!(
            call(Value::Tensor(scalar), Value::Tensor(selection)).unwrap(),
            Value::Int(values[0].clone())
        );

        let expected = storage
            .from_exact_values_like(vec![
                values[0].clone(),
                values[0].clone(),
                values[1].clone(),
                values[1].clone(),
                values[2].clone(),
                values[2].clone(),
            ])
            .unwrap();
        let input = Tensor::new_integer(storage.clone(), vec![1, 3]).unwrap();
        let Value::Tensor(output) = call(Value::Tensor(input), Value::Num(2.0)).unwrap() else {
            panic!("expected tensor")
        };
        assert_eq!(output.integer_storage(), Some(&expected));

        let empty = storage.from_exact_values_like(Vec::new()).unwrap();
        for (selection, shape) in [(0.0, vec![1, 0]), (4.0, vec![0, 4])] {
            let input = Tensor::new_integer(storage.clone(), vec![1, 3]).unwrap();
            let Value::Tensor(output) = call(Value::Tensor(input), Value::Num(selection)).unwrap()
            else {
                panic!("expected empty tensor")
            };
            assert_eq!(output.shape, shape);
            assert_eq!(output.integer_storage(), Some(&empty));
        }
    }
}

#[test]
fn every_complex_integer_class_preserves_both_components_exactly() {
    for real in storages() {
        let real_values = real.exact_values();
        let imaginary = real
            .from_exact_values_like(vec![
                real_values[2].clone(),
                real_values[1].clone(),
                real_values[0].clone(),
            ])
            .unwrap();
        let imaginary_values = imaginary.exact_values();
        let expected = IntegerComplexStorage::new(
            real.from_exact_values_like(vec![
                real_values[0].clone(),
                real_values[0].clone(),
                real_values[1].clone(),
                real_values[1].clone(),
                real_values[2].clone(),
                real_values[2].clone(),
            ])
            .unwrap(),
            imaginary
                .from_exact_values_like(vec![
                    imaginary_values[0].clone(),
                    imaginary_values[0].clone(),
                    imaginary_values[1].clone(),
                    imaginary_values[1].clone(),
                    imaginary_values[2].clone(),
                    imaginary_values[2].clone(),
                ])
                .unwrap(),
        )
        .unwrap();
        let input = ComplexTensor::new_integer(
            IntegerComplexStorage::new(real, imaginary).unwrap(),
            vec![1, 3],
        )
        .unwrap();
        let Value::ComplexTensor(output) =
            call(Value::ComplexTensor(input), Value::Num(2.0)).unwrap()
        else {
            panic!("expected complex tensor")
        };
        assert_eq!(output.shape, vec![3, 2]);
        assert_eq!(output.integer_storage(), Some(&expected));
    }
}
