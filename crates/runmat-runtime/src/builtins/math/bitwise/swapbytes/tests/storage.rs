use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn preserves_low_rank_shapes_and_exact_integer_and_single_bits() {
    let integer = Tensor::new_integer(IntegerStorage::U64(vec![0x0102_0304_0506_0708]), vec![])
        .expect("rank-zero integer");
    let Value::Tensor(integer) =
        block_on(swapbytes_builtin(Value::Tensor(integer))).expect("swap integer")
    else {
        panic!("expected integer tensor");
    };
    assert!(integer.shape().is_empty());
    assert_eq!(
        integer.integer_storage(),
        Some(&IntegerStorage::U64(vec![0x0807_0605_0403_0201]))
    );

    let bits = [0x3fa0_0000_u32, 0x7fc0_1234];
    let single = Tensor::from_numeric_storage(
        NumericStorage::F32(bits.map(f32::from_bits).to_vec()),
        vec![2],
    )
    .expect("rank-one single");
    let Value::Tensor(single) =
        block_on(swapbytes_builtin(Value::Tensor(single))).expect("swap single")
    else {
        panic!("expected single tensor");
    };
    assert_eq!(single.shape(), &[2]);
    let NumericStorage::F32(single) = single.into_numeric_storage().expect("single storage") else {
        panic!("expected single storage");
    };
    assert_eq!(
        single
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        bits.map(u32::swap_bytes)
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
