use futures::executor::block_on;
use runmat_types::IntegerClass;
use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage, Value};

use super::super::cast_value;
use crate::builtins::common::test_support;

#[test]
fn typed_complex_gpu_conversion_stays_exact_and_resident() {
    test_support::with_test_provider(|provider| {
        let input = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                IntegerStorage::U64(vec![1, u64::MAX]),
            )
            .expect("storage"),
            vec![1, 2],
        )
        .expect("input");
        let handle = crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &input)
            .expect("upload");
        let Value::GpuTensor(output) =
            block_on(cast_value(Value::GpuTensor(handle), IntegerClass::Int64))
                .expect("conversion")
        else {
            panic!("conversion must remain resident");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_class(&output),
            Some(IntegerClass::Int64)
        );
        let Value::ComplexTensor(gathered) = block_on(
            crate::builtins::common::gpu_helpers::gather_value_async(&Value::GpuTensor(output)),
        )
        .expect("gather") else {
            panic!("expected complex integer storage");
        };
        let storage = gathered.integer_storage().expect("storage");
        assert_eq!(
            storage.real,
            IntegerStorage::I64(vec![9_007_199_254_740_993, i64::MAX])
        );
        assert_eq!(storage.imag, IntegerStorage::I64(vec![1, i64::MAX]));
    });
}
