use super::super::typecast_builtin;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, Tensor, Value};

use crate::builtins::common::{gpu_helpers, test_support};

#[test]
fn preserves_owner_residency_and_exact_integer_bytes() {
    test_support::with_test_provider(|provider| {
        let source = Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 63) + 9, u64::MAX]),
            vec![1, 2],
        )
        .expect("source");
        let handle = gpu_helpers::upload_tensor(provider, &source).expect("upload");
        let output = block_on(typecast_builtin(vec![
            Value::GpuTensor(handle),
            Value::String("uint8".to_string()),
        ]))
        .expect("resident typecast");
        assert!(matches!(output, Value::GpuTensor(_)));
        let output = test_support::gather(output).expect("gather result");
        let mut expected = Vec::new();
        for value in [(1_u64 << 63) + 9, u64::MAX] {
            expected.extend_from_slice(&value.to_ne_bytes());
        }
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U8(expected))
        );
    });
}
