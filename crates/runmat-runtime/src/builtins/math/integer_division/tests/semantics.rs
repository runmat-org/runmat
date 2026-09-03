use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage, NumericDType, Tensor, Value};

const ERROR_INVALID_INPUT: runmat_builtins::BuiltinErrorDescriptor =
    runmat_builtins::IDIVIDE_ERROR_INVALID_INPUT;
const ERROR_DIVIDE_BY_ZERO: runmat_builtins::BuiltinErrorDescriptor =
    runmat_builtins::IDIVIDE_ERROR_DIVIDE_BY_ZERO;
const ERROR_OVERFLOW: runmat_builtins::BuiltinErrorDescriptor =
    runmat_builtins::IDIVIDE_ERROR_OVERFLOW;

#[test]
fn idivide_preserves_scalar_integer_class_and_rounding_modes() {
    assert_eq!(
        block_on(idivide_builtin(vec![
            Value::Int(IntValue::I16(-7)),
            Value::Int(IntValue::I16(3)),
        ]))
        .expect("idivide fix"),
        Value::Int(IntValue::I16(-2))
    );
    assert_eq!(
        block_on(idivide_builtin(vec![
            Value::Int(IntValue::I16(-7)),
            Value::Int(IntValue::I16(3)),
            Value::String("floor".to_string()),
        ]))
        .expect("idivide floor"),
        Value::Int(IntValue::I16(-3))
    );
    assert_eq!(
        block_on(idivide_builtin(vec![
            Value::Int(IntValue::I16(-7)),
            Value::Int(IntValue::I16(3)),
            Value::String("ceil".to_string()),
        ]))
        .expect("idivide ceil"),
        Value::Int(IntValue::I16(-2))
    );
    assert_eq!(
        block_on(idivide_builtin(vec![
            Value::Int(IntValue::I16(5)),
            Value::Int(IntValue::I16(2)),
            Value::String("round".to_string()),
        ]))
        .expect("idivide round"),
        Value::Int(IntValue::I16(3))
    );
}

#[test]
fn idivide_broadcasts_uint_tensor_and_preserves_dtype() {
    let lhs = Tensor::new_with_dtype(vec![9.0, 10.0, 11.0], vec![1, 3], NumericDType::U16).unwrap();
    let out = block_on(idivide_builtin(vec![
        Value::Tensor(lhs),
        Value::Int(IntValue::U16(3)),
    ]))
    .expect("idivide tensor");
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape, vec![1, 3]);
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U16(vec![3, 3, 3]))
    );
}

#[test]
fn idivide_native_signed_and_unsigned_64_bit_arrays_stay_exact() {
    let signed = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, -7, 7]), vec![1, 3])
        .expect("signed input");
    let signed_out = block_on(idivide_builtin(vec![
        Value::Tensor(signed),
        Value::Int(IntValue::I64(2)),
        Value::from("floor"),
    ]))
    .expect("signed idivide");
    let Value::Tensor(signed_out) = signed_out else {
        panic!("expected signed tensor");
    };
    assert_eq!(
        signed_out.integer_storage(),
        Some(&IntegerStorage::I64(vec![i64::MIN / 2, -4, 3]))
    );

    let unsigned = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX, (1_u64 << 63) + 1]),
        vec![1, 2],
    )
    .expect("unsigned input");
    let unsigned_out = block_on(idivide_builtin(vec![
        Value::Tensor(unsigned),
        Value::Int(IntValue::U64(2)),
    ]))
    .expect("unsigned idivide");
    let Value::Tensor(unsigned_out) = unsigned_out else {
        panic!("expected unsigned tensor");
    };
    assert_eq!(
        unsigned_out.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX / 2, (1_u64 << 62)]))
    );
}

#[test]
fn idivide_allows_scalar_double_with_non64_integer_class() {
    assert_eq!(
        block_on(idivide_builtin(vec![
            Value::Num(10.0),
            Value::Int(IntValue::U16(3)),
        ]))
        .expect("double dividend"),
        Value::Int(IntValue::U16(3))
    );
    assert_eq!(
        block_on(idivide_builtin(vec![
            Value::Int(IntValue::I32(-7)),
            Value::Num(3.0),
            Value::String("floor".to_string()),
        ]))
        .expect("double divisor"),
        Value::Int(IntValue::I32(-3))
    );
}

#[test]
fn idivide_rejects_zero_mixed_class_and_invalid_double_inputs() {
    let zero = block_on(idivide_builtin(vec![
        Value::Int(IntValue::U8(1)),
        Value::Int(IntValue::U8(0)),
    ]))
    .expect_err("zero divisor should fail");
    assert_eq!(zero.identifier(), ERROR_DIVIDE_BY_ZERO.identifier);

    let mixed = block_on(idivide_builtin(vec![
        Value::Int(IntValue::U8(1)),
        Value::Int(IntValue::U16(1)),
    ]))
    .expect_err("mixed class should fail");
    assert_eq!(mixed.identifier(), ERROR_INVALID_INPUT.identifier);

    let two_doubles = block_on(idivide_builtin(vec![Value::Num(4.0), Value::Num(2.0)]))
        .expect_err("two doubles should fail");
    assert_eq!(two_doubles.identifier(), ERROR_INVALID_INPUT.identifier);

    let int64_double = block_on(idivide_builtin(vec![
        Value::Int(IntValue::I64(4)),
        Value::Num(2.0),
    ]))
    .expect_err("int64 plus double should fail");
    assert_eq!(int64_double.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn idivide_rejects_the_unrepresentable_signed_minimum_quotient() {
    let overflow = block_on(idivide_builtin(vec![
        Value::Int(IntValue::I64(i64::MIN)),
        Value::Int(IntValue::I64(-1)),
    ]))
    .expect_err("int64 minimum divided by negative one must not wrap");
    assert_eq!(overflow.identifier(), ERROR_OVERFLOW.identifier);
}

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
        let result = block_on(idivide_builtin(vec![
            Value::GpuTensor(dividend),
            Value::GpuTensor(divisor),
        ]))
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
    let quotient = block_on(idivide_builtin(vec![
        Value::GpuTensor(dividend),
        Value::GpuTensor(divisor),
    ]))
    .expect("idivide");
    assert!(matches!(quotient, Value::GpuTensor(_)));
    assert_eq!(
        test_support::gather(quotient)
            .expect("quotient gather")
            .integer_storage(),
        Some(&IntegerStorage::U8(vec![3, 3]))
    );
}
