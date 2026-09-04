use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn resident_integer_input_is_gathered_without_losing_identity() {
    test_support::with_test_provider(|provider| {
        let base = 1_u64 << 53;
        let source = Tensor::new_integer(
            IntegerStorage::U64(vec![base + 1, base, base + 1]),
            vec![3, 1],
        )
        .unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &source).unwrap();
        let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
        let _outputs = crate::output_count::push_output_count(Some(2));
        let outputs = output_list(call(Value::GpuTensor(handle), Vec::new()).unwrap());
        assert!(
            matches!(&outputs[1], Value::Tensor(ids) if ids.integer_storage() == Some(&IntegerStorage::U64(vec![base, base + 1])))
        );
    });
}
