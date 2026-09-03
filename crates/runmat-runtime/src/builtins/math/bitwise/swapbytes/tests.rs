use super::*;
use futures::executor::block_on;
use runmat_builtins::{SWAPBYTES_EXPLICIT_GPU_EXTENSION, SWAPBYTES_INTEGER_CAPABILITIES};
use runmat_value::IntegerStorage;

#[test]
fn preserves_every_integer_scalar_class() {
    let cases = [
        (IntValue::I8(-2), IntValue::I8(-2)),
        (IntValue::I16(0x0102), IntValue::I16(0x0201)),
        (IntValue::I32(0x0102_0304), IntValue::I32(0x0403_0201)),
        (
            IntValue::I64(0x0102_0304_0506_0708),
            IntValue::I64(0x0807_0605_0403_0201),
        ),
        (IntValue::U8(0xfe), IntValue::U8(0xfe)),
        (IntValue::U16(0x0102), IntValue::U16(0x0201)),
        (IntValue::U32(0x0102_0304), IntValue::U32(0x0403_0201)),
        (
            IntValue::U64(0x0102_0304_0506_0708),
            IntValue::U64(0x0807_0605_0403_0201),
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(
            block_on(swapbytes_builtin(Value::Int(input))).expect("swapbytes"),
            Value::Int(expected)
        );
    }
}

#[test]
fn preserves_native_integer_array_storage_and_shape() {
    let input = Tensor::new_integer(
        IntegerStorage::U64(vec![0x0102_0304_0506_0708, u64::MAX]),
        vec![2, 1],
    )
    .unwrap();
    let Value::Tensor(output) =
        block_on(swapbytes_builtin(Value::Tensor(input))).expect("swapbytes")
    else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape, vec![2, 1]);
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![0x0807_0605_0403_0201, u64::MAX,]))
    );
}

#[test]
fn swaps_float_bits_without_widening_single_storage() {
    let f64_bits = [0x3ff4_0000_0000_0000_u64, 0x7ff8_0000_0000_1234];
    let f32_bits = [0x3fa0_0000_u32, 0x7fc0_1234];
    let double = Tensor::from_numeric_storage(
        NumericStorage::F64(f64_bits.map(f64::from_bits).to_vec()),
        vec![1, 2],
    )
    .unwrap();
    let single = Tensor::from_numeric_storage(
        NumericStorage::F32(f32_bits.map(f32::from_bits).to_vec()),
        vec![1, 2],
    )
    .unwrap();

    let Value::Tensor(double) = block_on(swapbytes_builtin(Value::Tensor(double))).expect("double")
    else {
        panic!("expected double tensor")
    };
    let Value::Tensor(single) = block_on(swapbytes_builtin(Value::Tensor(single))).expect("single")
    else {
        panic!("expected single tensor")
    };
    let NumericStorage::F64(double) = double.into_numeric_storage().unwrap() else {
        panic!("expected double storage")
    };
    let NumericStorage::F32(single) = single.into_numeric_storage().unwrap() else {
        panic!("expected single storage")
    };
    assert_eq!(
        double
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        f64_bits.map(u64::swap_bytes)
    );
    assert_eq!(
        single
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        f32_bits.map(u32::swap_bytes)
    );
}

#[test]
fn rejects_unsupported_values_with_the_catalog_error() {
    let error = block_on(swapbytes_builtin(Value::Bool(true))).expect_err("logical input");
    assert_eq!(error.identifier(), SWAPBYTES_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn explicit_gpu_fallback_is_compatibility_gated_before_provider_access() {
    assert_eq!(SWAPBYTES_INTEGER_CAPABILITIES.len(), 1);
    assert_eq!(SWAPBYTES_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    let handle = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX - 454,
        descriptor: Default::default(),
    }
    .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(swapbytes_builtin(Value::GpuTensor(handle)))
        .expect_err("strict mode rejects explicit fallback before provider access");
    assert_eq!(
        error.identifier(),
        SWAPBYTES_EXPLICIT_GPU_EXTENSION.error_identifier
    );
}
