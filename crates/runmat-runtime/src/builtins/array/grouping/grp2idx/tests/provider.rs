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
            assert!(!g.shares_allocation_with(&prototype));
            assert!(!levels.shares_allocation_with(&prototype));
            assert!(!g.shares_allocation_with(levels));
            assert!(runmat_accelerate_api::same_provider_instance(
                runmat_accelerate_api::provider_for_handle(g).unwrap(),
                provider
            ));
            assert!(runmat_accelerate_api::same_provider_instance(
                runmat_accelerate_api::provider_for_handle(levels).unwrap(),
                provider
            ));
        });
    });
}
