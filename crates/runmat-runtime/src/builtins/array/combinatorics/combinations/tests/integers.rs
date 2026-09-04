use super::*;
use runmat_value::{IntegerStorage, Tensor};

fn integer_cases() -> Vec<IntegerStorage> {
    vec![
        IntegerStorage::I8(vec![-1, 2]),
        IntegerStorage::I16(vec![-1, 2]),
        IntegerStorage::I32(vec![-1, 2]),
        IntegerStorage::I64(vec![-1, 2]),
        IntegerStorage::U8(vec![1, 2]),
        IntegerStorage::U16(vec![1, 2]),
        IntegerStorage::U32(vec![1, 2]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
    ]
}

#[test]
fn every_integer_class_is_repeated_without_numeric_conversion() {
    for storage in integer_cases() {
        let exact = storage.exact_values();
        let expected = storage
            .from_exact_values_like(vec![
                exact[0].clone(),
                exact[0].clone(),
                exact[1].clone(),
                exact[1].clone(),
            ])
            .unwrap();
        let output = table(
            call(
                Value::Tensor(Tensor::new_integer(storage, vec![2, 1]).unwrap()),
                vec![Value::Tensor(
                    Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap(),
                )],
            )
            .unwrap(),
        );
        let variables = table_variables(&output).unwrap();
        let Value::Tensor(column) = &variables.fields["Var1"] else {
            panic!("expected integer column")
        };
        assert_eq!(column.integer_storage(), Some(&expected));
        assert_eq!(column.shape, vec![4, 1]);
    }
}

#[test]
fn empty_cartesian_product_preserves_all_integer_classes() {
    for storage in integer_cases() {
        let empty = storage.zeros_like(0);
        let expected_other = storage.zeros_like(0);
        let output = table(
            call(
                Value::Tensor(Tensor::new_integer(empty.clone(), vec![0, 1]).unwrap()),
                vec![Value::Tensor(
                    Tensor::new_integer(storage, vec![1, 2]).unwrap(),
                )],
            )
            .unwrap(),
        );
        let variables = table_variables(&output).unwrap();
        let Value::Tensor(first) = &variables.fields["Var1"] else {
            panic!("expected integer column")
        };
        let Value::Tensor(second) = &variables.fields["Var2"] else {
            panic!("expected integer column")
        };
        assert_eq!(first.integer_storage(), Some(&empty));
        assert_eq!(second.integer_storage(), Some(&expected_other));
    }
}
