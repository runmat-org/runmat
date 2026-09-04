use runmat_value::{IntegerStorage, NumericStorage, Tensor, Value};

use crate::builtins::common::{gpu_helpers, test_support};

use super::call;

#[test]
fn unary_and_binary_floating_results_remain_resident() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![-1.0, 0.0, 3.0], vec![3, 1]).expect("input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let result = call(Value::GpuTensor(handle), vec![]).expect("unary");
        assert!(matches!(result, Value::GpuTensor(_)));
        let output = test_support::gather(result).expect("gather");
        assert_eq!(output.shape, vec![3, 1]);
        assert_eq!(output.materialize_f64(), vec![0.5, 1.0, 8.0]);

        let left = Tensor::new(vec![0.5, 1.5], vec![2, 1]).expect("left");
        let right = Tensor::new(vec![3.0, 4.0], vec![2, 1]).expect("right");
        let left = gpu_helpers::upload_tensor(provider, &left).expect("upload left");
        let right = gpu_helpers::upload_tensor(provider, &right).expect("upload right");
        let result = call(Value::GpuTensor(left), vec![Value::GpuTensor(right)]).expect("binary");
        assert!(matches!(result, Value::GpuTensor(_)));
        assert_eq!(
            test_support::gather(result)
                .expect("gather")
                .materialize_f64(),
            vec![4.0, 24.0]
        );
    });
}

#[test]
fn integer_unary_restores_double_while_binary_fallback_is_host() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let input =
            Tensor::new_integer(IntegerStorage::U64(vec![0, 64]), vec![1, 2]).expect("input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let result = call(Value::GpuTensor(handle), vec![]).expect("unary");
        assert!(matches!(result, Value::GpuTensor(_)));
        assert_eq!(
            test_support::gather(result)
                .expect("gather")
                .into_numeric_storage()
                .expect("storage"),
            NumericStorage::F64(vec![1.0, 2.0_f64.powi(64)])
        );

        let left = Tensor::new_integer(IntegerStorage::U16(vec![4, 3]), vec![1, 2]).expect("left");
        let right =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 4]), vec![1, 2]).expect("right");
        let left = gpu_helpers::upload_tensor(provider, &left).expect("upload left");
        let right = gpu_helpers::upload_tensor(provider, &right).expect("upload right");
        let Value::Tensor(output) =
            call(Value::GpuTensor(left), vec![Value::GpuTensor(right)]).expect("binary")
        else {
            panic!("binary fallback must be host-resident");
        };
        assert_eq!(output.materialize_f64(), vec![8.0, 48.0]);

        let wide =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("wide");
        let wide = gpu_helpers::upload_tensor(provider, &wide).expect("upload wide");
        let error = call(Value::GpuTensor(wide), vec![]).expect_err("lossy input must reject");
        assert!(error.message().contains("exactly representable as double"));
    });
}
