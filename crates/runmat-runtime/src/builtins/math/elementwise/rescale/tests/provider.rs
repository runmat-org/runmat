use futures::executor::block_on;
use runmat_value::{Tensor, Value};

use crate::builtins::common::{gpu_helpers, test_support};

use super::super::rescale_builtin;
use super::support::assert_close;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn restores_output_to_exact_input_owner() {
    test_support::with_test_provider(|provider| {
        let source = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("source");
        let handle = gpu_helpers::upload_tensor(provider, &source).expect("upload");
        let device_id = handle.device_id();
        let input_lease = handle.allocation_lease().downgrade();
        let result = block_on(rescale_builtin(Value::GpuTensor(handle.clone()), vec![]))
            .expect("resident rescale");
        let Value::GpuTensor(output) = &result else {
            panic!("resident output")
        };
        assert_eq!(output.device_id(), device_id);
        let output_lease = output.allocation_lease().downgrade();
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), &[1, 3]);
        assert_close(&gathered.materialize_f64(), &[0.0, 0.5, 1.0]);
        assert!(input_lease.acquire().is_some());
        drop(handle);
        assert!(input_lease.acquire().is_none());
        assert!(output_lease.acquire().is_none());
    });
}
