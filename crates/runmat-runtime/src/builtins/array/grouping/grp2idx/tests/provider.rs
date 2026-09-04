use runmat_value::{IntegerStorage, Tensor, Value};

use super::super::execute;
use super::outputs;

#[test]
fn resident_integer_outputs_keep_owner_and_do_not_alias() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        futures::executor::block_on(async {
            let input = Tensor::new_integer(
                IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, u64::MAX]),
                vec![3, 1],
            )
            .unwrap();
            let prototype = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input)
                .expect("upload grouping input");
            let values = outputs(
                execute::apply(Value::GpuTensor(prototype.clone()))
                    .await
                    .unwrap(),
            );
            let Value::GpuTensor(g) = &values[0] else {
                panic!("expected resident g")
            };
            let Value::GpuTensor(levels) = &values[2] else {
                panic!("expected resident levels")
            };
            assert_ne!(
                (g.device_id, g.buffer_id),
                (prototype.device_id, prototype.buffer_id)
            );
            assert_ne!(
                (levels.device_id, levels.buffer_id),
                (prototype.device_id, prototype.buffer_id)
            );
            assert_ne!(
                (g.device_id, g.buffer_id),
                (levels.device_id, levels.buffer_id)
            );
            assert!(std::ptr::eq(
                runmat_accelerate_api::provider_for_handle(g).unwrap(),
                provider
            ));
            assert!(std::ptr::eq(
                runmat_accelerate_api::provider_for_handle(levels).unwrap(),
                provider
            ));
        });
    });
}
