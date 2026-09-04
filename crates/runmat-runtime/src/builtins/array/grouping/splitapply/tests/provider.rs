use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn resident_data_is_materialized_once_before_group_slicing() {
    test_support::with_test_provider(|provider| {
        let source =
            Tensor::new_integer(IntegerStorage::U64(vec![9, 7, 11, 8]), vec![4, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &source).unwrap();
        let output = call(
            "max",
            Value::GpuTensor(handle),
            vec![tensor(vec![1.0, 2.0, 1.0, 2.0], vec![4, 1])],
        )
        .unwrap();
        let Value::Tensor(output) = output else {
            panic!("expected integer output");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![11, 8]))
        );
    });
}
