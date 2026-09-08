use super::*;
use runmat_value::{IntValue, IntegerStorage};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn typed_integer_blocks_and_partitions_preserve_exact_values() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]),
        vec![2, 1],
    )
    .unwrap();
    let partitions = Tensor::new_integer(IntegerStorage::U8(vec![1, 1]), vec![2, 1]).unwrap();
    let result = run(Value::Tensor(input), vec![Value::Tensor(partitions)]).unwrap();
    let Value::Cell(cells) = result else {
        panic!("expected cell array");
    };
    assert_eq!(cells.shape, vec![2, 1]);
    assert_eq!(cells.data[0], Value::Int(IntValue::U64(u64::MAX - 1)));
    assert_eq!(cells.data[1], Value::Int(IntValue::U64(u64::MAX)));
}

#[test]
fn typed_integer_partition_vectors_follow_extension_policy() {
    let input = || Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();
    let partition = || Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = run(Value::Tensor(input()), vec![Value::Tensor(partition())])
            .expect_err("typed partitions are extension-gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:Mat2cellIntegerPartitionsExtension")
        );
    }
    {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run(Value::Tensor(input()), vec![Value::Tensor(partition())])
            .expect("RunMat extension");
        let Value::Cell(cells) = result else {
            panic!("expected cell array");
        };
        assert_eq!(cells.data, vec![Value::Int(IntValue::U64(u64::MAX))]);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn typed_partition_entries_are_checked_without_f64_truncation() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let partitions = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();
    let err = run(Value::Tensor(input), vec![Value::Tensor(partitions)])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("18446744073709551615"),
        "unexpected error message: {err}"
    );

    let input = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let partitions = Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).unwrap();
    let err = run(Value::Tensor(input), vec![Value::Tensor(partitions)])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("non-negative"),
        "unexpected error message: {err}"
    );

    let input = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };
    let err = run(Value::Tensor(input), vec![row_vector(&[boundary])])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("platform limits"),
        "unexpected error message: {err}"
    );
}
