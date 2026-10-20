use super::*;
use crate::builtins::common::{gpu_helpers, test_support};

#[test]
fn idivide_preserves_exact_supported_gpu_residency() {
    test_support::with_test_provider(|provider| {
        let dividend = Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 63) + 12, u64::MAX - 1]),
            vec![1, 2],
        )
        .expect("dividend");
        let divisor =
            Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![1, 2]).expect("divisor");
        let dividend = gpu_helpers::upload_tensor(provider, &dividend).expect("dividend upload");
        let divisor = gpu_helpers::upload_tensor(provider, &divisor).expect("divisor upload");
        let result = block_on(idivide_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(dividend),
                Value::GpuTensor(divisor),
            ]),
        ))
        .expect("resident idivide");
        assert!(matches!(result, Value::GpuTensor(_)));
        assert_eq!(
            test_support::gather(result)
                .expect("result gather")
                .integer_storage(),
            Some(&IntegerStorage::U64(vec![
                ((1_u64 << 63) + 12) / 2,
                (u64::MAX - 1) / 3,
            ]))
        );
    });
}

#[test]
fn resident_singleton_retains_full_nd_shape() {
    test_support::with_test_provider(|provider| {
        let input =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX - 1]), vec![1, 1, 1]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &input).unwrap();
        let output = block_on(idivide_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(source.clone()),
                Value::Int(IntValue::U64(2)),
            ]),
        ))
        .unwrap();
        let Value::GpuTensor(output) = output else {
            panic!("resident singleton");
        };
        assert_eq!(output.shape(), &[1, 1, 1]);
        assert_eq!(
            block_on(provider.download_integer(&output)).unwrap().data,
            runmat_accelerate_api::HostIntegerDataOwned::U64(vec![(u64::MAX - 1) / 2])
        );
        for handle in [source, output] {
            let physical = handle.allocation_lease().downgrade();
            handle.release().unwrap();
            assert!(physical.acquire().is_none());
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn idivide_preserves_actual_wgpu_residency() {
    let _guard = test_support::accel_test_lock();
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let dividend =
        Tensor::new_integer(IntegerStorage::U8(vec![10, 11]), vec![1, 2]).expect("dividend");
    let divisor = Tensor::new_integer(IntegerStorage::U8(vec![3]), vec![1, 1]).expect("divisor");
    let dividend = gpu_helpers::upload_tensor(provider, &dividend).expect("dividend upload");
    let divisor = gpu_helpers::upload_tensor(provider, &divisor).expect("divisor upload");
    let quotient = block_on(idivide_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(dividend),
            Value::GpuTensor(divisor),
        ]),
    ))
    .expect("idivide");
    assert!(matches!(quotient, Value::GpuTensor(_)));
    assert_eq!(
        test_support::gather(quotient)
            .expect("quotient gather")
            .integer_storage(),
        Some(&IntegerStorage::U8(vec![3, 3]))
    );
}
