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
    assert_eq!(output.shape(), &[2, 1]);
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![0x0807_0605_0403_0201, u64::MAX,]))
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
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident numeric input");
    let handle = allocation
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("explicit resident view");
    drop(allocation.handle);
    let caller_alias = handle.clone();
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(swapbytes_builtin(Value::GpuTensor(handle)))
        .expect_err("strict mode rejects explicit fallback before provider access");
    assert_eq!(
        error.identifier(),
        SWAPBYTES_EXPLICIT_GPU_EXTENSION.error_identifier
    );
    assert_eq!(allocation.probe.retirements(), 0);
    caller_alias.allocation_lease().validate_live().unwrap();
    drop(error);
    drop(caller_alias);
    assert_eq!(allocation.probe.retirements(), 1);
}

mod storage;
