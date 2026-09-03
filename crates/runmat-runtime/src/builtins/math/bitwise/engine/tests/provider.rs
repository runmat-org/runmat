use super::super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;

#[test]
fn scalar_or_exact_size_plan_ignores_only_trailing_singletons() {
    let plan = scalar_or_exact_size_plan(&[2, 1], &[2, 1, 1]).expect("same MATLAB size");
    assert_eq!(plan.output_shape(), &[2, 1, 1]);
    assert!(scalar_or_exact_size_plan(&[2, 1], &[2, 1, 2]).is_err());
    assert!(scalar_or_exact_size_plan(&[0, 2], &[0, 2]).is_ok());
    assert!(scalar_or_exact_size_plan(&[0, 2], &[1, 2]).is_err());
}

#[test]
fn sparse_bit_position_and_count_operations_reject_general_singleton_expansion() {
    let sparse = runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 5.0])
        .expect("sparse");
    let column = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("column operand");
    for error in [
        block_on(bitshift_builtin(vec![
            Value::SparseTensor(sparse.clone()),
            Value::Tensor(column.clone()),
            Value::String("uint8".to_string()),
        ]))
        .expect_err("sparse bitshift singleton expansion must reject"),
        block_on(bitget_builtin(vec![
            Value::SparseTensor(sparse.clone()),
            Value::Tensor(column.clone()),
            Value::String("uint8".to_string()),
        ]))
        .expect_err("sparse bitget singleton expansion must reject"),
        block_on(bitset_builtin(vec![
            Value::SparseTensor(sparse),
            Value::Num(1.0),
            Value::Tensor(column),
            Value::String("uint8".to_string()),
        ]))
        .expect_err("sparse bitset value expansion must reject"),
    ] {
        assert_eq!(error.identifier(), ERROR_SIZE_MISMATCH.identifier);
        assert!(error.message().contains("exactly the same size"));
    }
}

#[test]
fn gathered_gpu_bit_position_shapes_use_scalar_or_exact_size_rules() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).expect("input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let row = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("row operand");
        let error = block_on(bitget_builtin(vec![
            Value::GpuTensor(handle),
            Value::Tensor(row),
        ]))
        .expect_err("gathered GPU singleton expansion must reject");
        assert_eq!(error.identifier(), ERROR_SIZE_MISMATCH.identifier);

        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let shifts = Tensor::new(vec![1.0, -1.0], vec![2, 1]).expect("same-size shifts");
        let output = block_on(bitshift_builtin(vec![
            Value::GpuTensor(handle),
            Value::Tensor(shifts),
        ]))
        .expect("gathered GPU same-size bitshift");
        assert!(matches!(output, Value::GpuTensor(_)));
        let output = test_support::gather(output).expect("gather resident result");
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U8(vec![2, 1]))
        );
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn wgpu_gathered_bit_position_shapes_use_scalar_or_exact_size_rules() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let input = Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).expect("input");
    let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
    let row = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("row operand");
    let error = block_on(bitget_builtin(vec![
        Value::GpuTensor(handle),
        Value::Tensor(row),
    ]))
    .expect_err("wgpu singleton expansion must reject");
    assert_eq!(error.identifier(), ERROR_SIZE_MISMATCH.identifier);

    let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
    let positions = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("same-size positions");
    let output = block_on(bitget_builtin(vec![
        Value::GpuTensor(handle),
        Value::Tensor(positions),
    ]))
    .expect("wgpu same-size bitget");
    assert!(matches!(output, Value::GpuTensor(_)));
    let output = test_support::gather(output).expect("gather bitget result");
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U8(vec![1, 1]))
    );
}

#[test]
fn remaining_direct_bit_functions_preserve_supported_gpu_residency() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(IntegerStorage::U8(vec![0b1010, 0b0101]), vec![1, 2])
            .expect("input");
        let other = Tensor::new_integer(IntegerStorage::U8(vec![0b1100, 0b0011]), vec![1, 2])
            .expect("other");

        let left = gpu_helpers::upload_tensor(provider, &input).expect("left upload");
        let right = gpu_helpers::upload_tensor(provider, &other).expect("right upload");
        let xor = block_on(bitxor_builtin(vec![
            Value::GpuTensor(left),
            Value::GpuTensor(right),
        ]))
        .expect("bitxor");
        assert!(matches!(xor, Value::GpuTensor(_)));
        assert_eq!(
            test_support::gather(xor)
                .expect("xor gather")
                .integer_storage(),
            Some(&IntegerStorage::U8(vec![0b0110, 0b0110]))
        );

        let source = gpu_helpers::upload_tensor(provider, &input).expect("cmp upload");
        let complement = block_on(bitcmp_builtin(vec![Value::GpuTensor(source)])).expect("bitcmp");
        assert!(matches!(complement, Value::GpuTensor(_)));
        assert_eq!(
            test_support::gather(complement)
                .expect("complement gather")
                .integer_storage(),
            Some(&IntegerStorage::U8(vec![0b1111_0101, 0b1111_1010]))
        );

        let source = gpu_helpers::upload_tensor(provider, &input).expect("get upload");
        let bits = block_on(bitget_builtin(vec![
            Value::GpuTensor(source),
            Value::Num(2.0),
        ]))
        .expect("bitget");
        assert!(matches!(bits, Value::GpuTensor(_)));
        assert_eq!(
            test_support::gather(bits)
                .expect("bitget gather")
                .integer_storage(),
            Some(&IntegerStorage::U8(vec![1, 0]))
        );

        let source = gpu_helpers::upload_tensor(provider, &input).expect("set upload");
        let cleared = block_on(bitset_builtin(vec![
            Value::GpuTensor(source),
            Value::Num(2.0),
            Value::Bool(false),
        ]))
        .expect("bitset");
        assert!(matches!(cleared, Value::GpuTensor(_)));
        assert_eq!(
            test_support::gather(cleared)
                .expect("bitset gather")
                .integer_storage(),
            Some(&IntegerStorage::U8(vec![0b1000, 0b0101]))
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn direct_bit_functions_preserve_actual_wgpu_residency() {
    let _guard = test_support::accel_test_lock();
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let input =
        Tensor::new_integer(IntegerStorage::U8(vec![0b1010, 0b0101]), vec![1, 2]).expect("input");
    let other =
        Tensor::new_integer(IntegerStorage::U8(vec![0b1100, 0b0011]), vec![1, 2]).expect("other");

    let left = gpu_helpers::upload_tensor(provider, &input).expect("left upload");
    let right = gpu_helpers::upload_tensor(provider, &other).expect("right upload");
    let xor = block_on(bitxor_builtin(vec![
        Value::GpuTensor(left),
        Value::GpuTensor(right),
    ]))
    .expect("bitxor");
    assert!(matches!(xor, Value::GpuTensor(_)));
    assert_eq!(
        test_support::gather(xor)
            .expect("xor gather")
            .integer_storage(),
        Some(&IntegerStorage::U8(vec![0b0110, 0b0110]))
    );

    let source = gpu_helpers::upload_tensor(provider, &input).expect("cmp upload");
    let complement = block_on(bitcmp_builtin(vec![Value::GpuTensor(source)])).expect("bitcmp");
    assert!(matches!(complement, Value::GpuTensor(_)));

    let source = gpu_helpers::upload_tensor(provider, &input).expect("get upload");
    let bits = block_on(bitget_builtin(vec![
        Value::GpuTensor(source),
        Value::Num(2.0),
    ]))
    .expect("bitget");
    assert!(matches!(bits, Value::GpuTensor(_)));

    let source = gpu_helpers::upload_tensor(provider, &input).expect("set upload");
    let cleared = block_on(bitset_builtin(vec![
        Value::GpuTensor(source),
        Value::Num(2.0),
        Value::Bool(false),
    ]))
    .expect("bitset");
    assert!(matches!(cleared, Value::GpuTensor(_)));
}
