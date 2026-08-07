use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;

#[test]
fn bitand_double_scalars_return_double() {
    let out = block_on(bitand_builtin(vec![Value::Num(6.0), Value::Num(3.0)])).expect("bitand");
    assert_eq!(out, Value::Num(2.0));
}

#[test]
fn bitand_preserves_documented_logical_output_class() {
    assert_eq!(
        block_on(bitand_builtin(vec![Value::Bool(true), Value::Bool(false)]))
            .expect("logical scalar bitand"),
        Value::Bool(false)
    );
    let left = LogicalArray::new(vec![1, 1, 0, 0], vec![2, 2]).expect("left logical");
    let right = LogicalArray::new(vec![1, 0], vec![2, 1]).expect("right logical");
    let Value::LogicalArray(output) = block_on(bitand_builtin(vec![
        Value::LogicalArray(left),
        Value::LogicalArray(right),
    ]))
    .expect("logical array bitand") else {
        panic!("logical inputs must produce a logical array");
    };
    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(output.data, vec![1, 0, 0, 0]);
}

#[test]
fn bitand_single_input_is_a_mode_gated_extension() {
    let single = Tensor::from_f32(vec![6.0, 3.0], vec![1, 2]).expect("single input");
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(bitand_builtin(vec![
            Value::Tensor(single.clone()),
            Value::Num(3.0),
        ]))
        .expect_err("MATLAB mode rejects undocumented single input");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:BitandSingleInputExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let Value::Tensor(output) =
            block_on(bitand_builtin(vec![Value::Tensor(single), Value::Num(3.0)]))
                .expect("RunMat mode retains single extension")
        else {
            panic!("array input must produce an array");
        };
        assert_eq!(output.materialize_f64(), vec![2.0, 3.0]);
    }
}

#[test]
fn bitand_rejects_unsupported_resident_forms_before_provider_access() {
    let signed = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX - 70,
    };
    runmat_accelerate_api::set_handle_integer_type(
        &signed,
        runmat_accelerate_api::IntegerElementType::I32,
    );
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(bitand_builtin(vec![
        Value::GpuTensor(signed),
        Value::Int(IntValue::I32(1)),
    ]))
    .expect_err("MATLAB mode rejects signed resident input before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:BitandGpuUndocumentedInputExtension")
    );

    let unsigned = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX - 71,
    };
    runmat_accelerate_api::set_handle_integer_type(
        &unsigned,
        runmat_accelerate_api::IntegerElementType::U8,
    );
    let error = block_on(bitand_builtin(vec![
        Value::GpuTensor(unsigned),
        Value::Int(IntValue::U8(1)),
        Value::from("uint8"),
    ]))
    .expect_err("MATLAB mode rejects resident assumedtype before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:BitandGpuAssumedTypeExtension")
    );
}

#[test]
fn bitand_documented_gpu_integer_subset_restores_exact_resident_output() {
    test_support::with_test_provider(|provider| {
        let cases = [
            (
                IntegerStorage::U8(vec![0xff, 0x81]),
                IntegerStorage::U8(vec![0x0f, 0x01]),
                IntegerStorage::U8(vec![0x0f, 0x01]),
                runmat_accelerate_api::IntegerElementType::U8,
            ),
            (
                IntegerStorage::U16(vec![0xff00, 0x8001]),
                IntegerStorage::U16(vec![0x0ff0, 0x0001]),
                IntegerStorage::U16(vec![0x0f00, 0x0001]),
                runmat_accelerate_api::IntegerElementType::U16,
            ),
            (
                IntegerStorage::U32(vec![0xffff_0000, 0x8000_0001]),
                IntegerStorage::U32(vec![0x0fff_f000, 0x0000_0001]),
                IntegerStorage::U32(vec![0x0fff_0000, 0x0000_0001]),
                runmat_accelerate_api::IntegerElementType::U32,
            ),
        ];
        for (left, right, expected, element_type) in cases {
            let left = Tensor::new_integer(left, vec![1, 2]).expect("left documented GPU integer");
            let right =
                Tensor::new_integer(right, vec![1, 2]).expect("right documented GPU integer");
            let left = gpu_helpers::upload_tensor(provider, &left).expect("upload left");
            let right = gpu_helpers::upload_tensor(provider, &right).expect("upload right");
            let result = block_on(bitand_builtin(vec![
                Value::GpuTensor(left),
                Value::GpuTensor(right),
            ]))
            .expect("documented GPU bitand");
            let Value::GpuTensor(handle) = &result else {
                panic!("documented GPU bitand must preserve residency");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(handle),
                Some(element_type)
            );
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.integer_storage(), Some(&expected));
        }
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn bitand_documented_gpu_integer_subset_round_trips_through_wgpu() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let cases = [
        (
            IntegerStorage::U8(vec![0xff, 0x81]),
            IntegerStorage::U8(vec![0x0f, 0x01]),
            IntegerStorage::U8(vec![0x0f, 0x01]),
        ),
        (
            IntegerStorage::U16(vec![0xff00, 0x8001]),
            IntegerStorage::U16(vec![0x0ff0, 0x0001]),
            IntegerStorage::U16(vec![0x0f00, 0x0001]),
        ),
        (
            IntegerStorage::U32(vec![0xffff_0000, 0x8000_0001]),
            IntegerStorage::U32(vec![0x0fff_f000, 0x0000_0001]),
            IntegerStorage::U32(vec![0x0fff_0000, 0x0000_0001]),
        ),
    ];
    for (left, right, expected) in cases {
        let left = Tensor::new_integer(left, vec![1, 2]).expect("left WGPU integer");
        let right = Tensor::new_integer(right, vec![1, 2]).expect("right WGPU integer");
        let left = gpu_helpers::upload_tensor(provider, &left).expect("upload left WGPU");
        let right = gpu_helpers::upload_tensor(provider, &right).expect("upload right WGPU");
        let result = block_on(bitand_builtin(vec![
            Value::GpuTensor(left),
            Value::GpuTensor(right),
        ]))
        .expect("WGPU bitand");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather WGPU result");
        assert_eq!(gathered.integer_storage(), Some(&expected));
    }
}

#[test]
fn bitwise_uint32_scalars_preserve_uint32() {
    let out = block_on(bitor_builtin(vec![
        Value::Int(IntValue::U32(0b0101)),
        Value::Int(IntValue::U32(0b0011)),
    ]))
    .expect("bitor");
    assert_eq!(out, Value::Int(IntValue::U32(0b0111)));
}

#[test]
fn bitand_broadcasts_tensor_and_scalar() {
    let tensor =
        Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::U32).unwrap();
    let out = block_on(bitand_builtin(vec![
        Value::Tensor(tensor),
        Value::Int(IntValue::U32(1)),
    ]))
    .expect("bitand");
    match out {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(
                t.integer_storage(),
                Some(&IntegerStorage::U32(vec![1, 0, 1, 0]))
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn binary_bitwise_sparse_operands_preserve_or_materialize_by_zero_semantics() {
    let left = runmat_builtins::SparseTensor::new(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![0b0110 as f64, 0b1010 as f64],
    )
    .expect("left sparse");
    let right = runmat_builtins::SparseTensor::new(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 0],
        vec![0b0011 as f64, 0b0101 as f64],
    )
    .expect("right sparse");

    let Value::SparseTensor(and) = block_on(bitand_builtin(vec![
        Value::SparseTensor(left.clone()),
        Value::SparseTensor(right.clone()),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse bitand") else {
        panic!("bitand of sparse operands preserves CSC storage");
    };
    assert_eq!(and.get(0, 0), Some(2.0));
    assert_eq!(and.nnz(), 1);

    let Value::SparseTensor(or) = block_on(bitor_builtin(vec![
        Value::SparseTensor(left.clone()),
        Value::SparseTensor(right),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse bitor") else {
        panic!("bitor of sparse operands preserves CSC storage");
    };
    assert_eq!(or.get(0, 0), Some(7.0));
    assert_eq!(or.get(1, 1), Some(10.0));
    assert_eq!(or.nnz(), 3);

    let Value::Tensor(xor) = block_on(bitxor_builtin(vec![
        Value::SparseTensor(left),
        Value::Num(1.0),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse xor nonzero scalar") else {
        panic!("xor with a nonzero scalar materializes implicit zeros");
    };
    assert_eq!(
        xor.as_f64_slice().expect("double bitxor output"),
        &[7.0, 1.0, 1.0, 11.0]
    );
}

#[test]
fn binary_bitwise_sparse_dense_and_broadcast_forms_use_zero_aware_output_storage() {
    let sparse = runmat_builtins::SparseTensor::new(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![0b0110 as f64, 0b1010 as f64],
    )
    .expect("sparse");
    let dense =
        Tensor::new(vec![0b0011 as f64, 0.0, 0.0, 0b1111 as f64], vec![2, 2]).expect("dense");
    let Value::SparseTensor(and) = block_on(bitand_builtin(vec![
        Value::SparseTensor(sparse.clone()),
        Value::Tensor(dense),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse dense bitand") else {
        panic!("bitand keeps sparse output because zero is annihilating");
    };
    assert_eq!(and.get(0, 0), Some(2.0));
    assert_eq!(and.get(1, 1), Some(10.0));
    assert_eq!(and.nnz(), 2);

    let Value::Tensor(or) = block_on(bitor_builtin(vec![
        Value::SparseTensor(sparse),
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).expect("broadcast dense")),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse dense bitor") else {
        panic!("bitor materializes when broadcast dense values make implicit zeros nonzero");
    };
    assert_eq!(or.shape, vec![2, 2]);
    assert_eq!(
        or.as_f64_slice().expect("double bitor output"),
        &[6.0, 1.0, 0.0, 11.0]
    );
}

#[test]
fn binary_bitwise_rejects_runmat_typed_sparse_integer_extension() {
    let typed = runmat_builtins::SparseTensor::new_integer(
        1,
        1,
        vec![0, 1],
        vec![0],
        IntegerStorage::U8(vec![1]),
    )
    .expect("typed sparse");
    let error = block_on(bitand_builtin(vec![
        Value::SparseTensor(typed),
        Value::Num(1.0),
    ]))
    .expect_err("typed sparse is not a MATLAB sparse integer representation");
    assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn binary_bitwise_rejects_mixed_integer_classes() {
    let forward = block_on(bitor_builtin(vec![
        Value::Int(IntValue::U8(1)),
        Value::Int(IntValue::U32(256)),
    ]))
    .expect_err("mixed integer classes must fail");
    let reverse = block_on(bitor_builtin(vec![
        Value::Int(IntValue::U32(256)),
        Value::Int(IntValue::U8(1)),
    ]))
    .expect_err("mixed integer classes must fail");

    assert_eq!(forward.identifier(), ERROR_INVALID_INPUT.identifier);
    assert_eq!(reverse.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn bitwise_preserves_all_native_integer_scalar_classes() {
    let cases = [
        (IntValue::I8(-5), IntValue::I8(6), IntValue::I8(2)),
        (IntValue::I16(-5), IntValue::I16(6), IntValue::I16(2)),
        (IntValue::I32(-5), IntValue::I32(6), IntValue::I32(2)),
        (IntValue::I64(-5), IntValue::I64(6), IntValue::I64(2)),
        (
            IntValue::U8(0b1010),
            IntValue::U8(0b0110),
            IntValue::U8(0b0010),
        ),
        (
            IntValue::U16(0b1010),
            IntValue::U16(0b0110),
            IntValue::U16(0b0010),
        ),
        (
            IntValue::U32(0b1010),
            IntValue::U32(0b0110),
            IntValue::U32(0b0010),
        ),
        (
            IntValue::U64(0b1010),
            IntValue::U64(0b0110),
            IntValue::U64(0b0010),
        ),
    ];
    for (left, right, expected) in cases {
        let actual =
            block_on(bitand_builtin(vec![Value::Int(left), Value::Int(right)])).expect("bitand");
        assert_eq!(actual, Value::Int(expected));
    }

    let high = block_on(bitor_builtin(vec![
        Value::Int(IntValue::U64(1_u64 << 63)),
        Value::Int(IntValue::U64(1_u64 << 60)),
    ]))
    .expect("uint64 high-bit bitor");
    assert_eq!(
        high,
        Value::Int(IntValue::U64((1_u64 << 63) | (1_u64 << 60)))
    );
}

#[test]
fn bitxor_preserves_all_native_integer_scalar_classes() {
    let cases = [
        (IntValue::I8(-5), IntValue::I8(6), IntValue::I8(-3)),
        (IntValue::I16(-5), IntValue::I16(6), IntValue::I16(-3)),
        (IntValue::I32(-5), IntValue::I32(6), IntValue::I32(-3)),
        (IntValue::I64(-5), IntValue::I64(6), IntValue::I64(-3)),
        (
            IntValue::U8(0b1010),
            IntValue::U8(0b0110),
            IntValue::U8(0b1100),
        ),
        (
            IntValue::U16(0b1010),
            IntValue::U16(0b0110),
            IntValue::U16(0b1100),
        ),
        (
            IntValue::U32(0b1010),
            IntValue::U32(0b0110),
            IntValue::U32(0b1100),
        ),
        (
            IntValue::U64(0b1010),
            IntValue::U64(0b0110),
            IntValue::U64(0b1100),
        ),
    ];
    for (left, right, expected) in cases {
        let actual =
            block_on(bitxor_builtin(vec![Value::Int(left), Value::Int(right)])).expect("bitxor");
        assert_eq!(actual, Value::Int(expected));
    }
}

#[test]
fn bitcmp_preserves_all_native_integer_scalar_classes() {
    let cases = [
        (IntValue::I8(-11), IntValue::I8(10)),
        (IntValue::I16(-11), IntValue::I16(10)),
        (IntValue::I32(-11), IntValue::I32(10)),
        (IntValue::I64(-11), IntValue::I64(10)),
        (IntValue::U8(0b0101), IntValue::U8(0b1111_1010)),
        (IntValue::U16(0b0101), IntValue::U16(0xfffa)),
        (IntValue::U32(0b0101), IntValue::U32(0xffff_fffa)),
        (IntValue::U64(0b0101), IntValue::U64(u64::MAX - 5)),
    ];
    for (input, expected) in cases {
        let actual = block_on(bitcmp_builtin(vec![Value::Int(input)])).expect("bitcmp");
        assert_eq!(actual, Value::Int(expected));
    }
}

#[test]
fn bitcmp_preserves_exact_integer_arrays_and_default_double_behavior() {
    let input =
        Tensor::new_integer(IntegerStorage::U64(vec![0, 1_u64 << 63]), vec![1, 2]).expect("input");
    let Value::Tensor(output) =
        block_on(bitcmp_builtin(vec![Value::Tensor(input)])).expect("bitcmp")
    else {
        panic!("expected tensor result");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX, !(1_u64 << 63)]))
    );

    assert_eq!(
        block_on(bitcmp_builtin(vec![Value::Num(0.0)])).expect("double bitcmp"),
        Value::Num(u64::MAX as f64)
    );
}

#[test]
fn bitcmp_sparse_double_materializes_when_complementing_implicit_zeros() {
    let sparse =
        runmat_builtins::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![5.0, 1.0])
            .expect("sparse");
    let Value::Tensor(output) = block_on(bitcmp_builtin(vec![
        Value::SparseTensor(sparse),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse bitcmp") else {
        panic!("bitcmp must materialize a complement of sparse implicit zeros");
    };
    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(
        output.as_f64_slice().expect("double bitcmp output"),
        &[250.0, 255.0, 255.0, 254.0]
    );
}

#[test]
fn bitcmp_is_registered_and_dispatches() {
    assert!(runmat_builtins::builtin_function_by_name(BITCMP_NAME).is_some());
    assert_eq!(
        crate::dispatcher::call_builtin(BITCMP_NAME, &[Value::Int(IntValue::U8(0b0101))])
            .expect("runtime dispatch"),
        Value::Int(IntValue::U8(0b1111_1010))
    );
}

#[test]
fn bitget_preserves_all_native_integer_scalar_classes() {
    let cases = [
        (IntValue::I8(-1), IntValue::I8(1)),
        (IntValue::I16(-1), IntValue::I16(1)),
        (IntValue::I32(-1), IntValue::I32(1)),
        (IntValue::I64(-1), IntValue::I64(1)),
        (IntValue::U8(0b1010), IntValue::U8(0)),
        (IntValue::U16(0b1010), IntValue::U16(0)),
        (IntValue::U32(0b1010), IntValue::U32(0)),
        (IntValue::U64(0b1010), IntValue::U64(0)),
    ];
    for (input, expected) in cases {
        let actual =
            block_on(bitget_builtin(vec![Value::Int(input), Value::Num(1.0)])).expect("bitget");
        assert_eq!(actual, Value::Int(expected));
    }
}

#[test]
fn bitget_broadcasts_positions_and_preserves_uint64_storage() {
    let input =
        Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63]), vec![1, 1]).expect("input");
    let positions = Tensor::new(vec![1.0, 63.0, 64.0], vec![1, 3]).expect("positions");
    let Value::Tensor(output) = block_on(bitget_builtin(vec![
        Value::Tensor(input),
        Value::Tensor(positions),
    ]))
    .expect("bitget") else {
        panic!("expected tensor result");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![0, 0, 1]))
    );
}

#[test]
fn bitget_handles_signed_bits_double_output_and_invalid_positions() {
    assert_eq!(
        block_on(bitget_builtin(vec![
            Value::Int(IntValue::I8(-29)),
            Value::Num(8.0)
        ]))
        .expect("signed bit"),
        Value::Int(IntValue::I8(1))
    );
    assert_eq!(
        block_on(bitget_builtin(vec![Value::Num(8.0), Value::Num(4.0)])).expect("double bit"),
        Value::Num(1.0)
    );
    for position in [0.0, -1.0, 9.0] {
        let error = block_on(bitget_builtin(vec![
            Value::Int(IntValue::U8(1)),
            Value::Num(position),
        ]))
        .expect_err("invalid bit position");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
    }
}

#[test]
fn bitget_sparse_double_scalar_position_preserves_sparse_storage() {
    let sparse =
        runmat_builtins::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![5.0, 2.0])
            .expect("sparse");
    let Value::SparseTensor(output) = block_on(bitget_builtin(vec![
        Value::SparseTensor(sparse),
        Value::Num(1.0),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse bitget") else {
        panic!("bitget must preserve sparse storage for zero-valued implicit entries");
    };
    assert_eq!(output.shape(), vec![2, 2]);
    assert_eq!(output.get(0, 0), Some(1.0));
    assert_eq!(output.nnz(), 1);
}

#[test]
fn sparse_bitwise_position_and_value_arrays_require_matching_sizes() {
    let sparse =
        runmat_builtins::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 5.0])
            .expect("sparse");

    let Value::SparseTensor(shifted) = block_on(bitshift_builtin(vec![
        Value::SparseTensor(sparse.clone()),
        Value::Tensor(Tensor::new(vec![1.0, -1.0, 1.0, -1.0], vec![2, 2]).expect("shifts")),
        Value::String("uint8".to_string()),
    ]))
    .expect("same-size sparse bitshift") else {
        panic!("bitshift leaves implicit zeros sparse for every shift");
    };
    assert_eq!(shifted.get(0, 0), Some(6.0));
    assert_eq!(shifted.get(1, 1), Some(2.0));

    let Value::SparseTensor(got) = block_on(bitget_builtin(vec![
        Value::SparseTensor(sparse.clone()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]).expect("positions")),
        Value::String("uint8".to_string()),
    ]))
    .expect("same-size sparse bitget") else {
        panic!("bitget leaves implicit zeros sparse for every position");
    };
    assert_eq!(got.get(0, 0), Some(1.0));
    assert_eq!(got.nnz(), 1);

    let Value::SparseTensor(cleared) = block_on(bitset_builtin(vec![
        Value::SparseTensor(sparse.clone()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]).expect("positions")),
        Value::Tensor(Tensor::new(vec![0.0; 4], vec![2, 2]).expect("clear values")),
        Value::String("uint8".to_string()),
    ]))
    .expect("same-size sparse bitset clear") else {
        panic!("clearing broadcast positions preserves implicit zeros");
    };
    assert_eq!(cleared.get(0, 0), Some(2.0));
    assert_eq!(cleared.get(1, 1), Some(5.0));

    let Value::Tensor(set) = block_on(bitset_builtin(vec![
        Value::SparseTensor(sparse),
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]).expect("positions")),
        Value::Tensor(Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).expect("set values")),
        Value::String("uint8".to_string()),
    ]))
    .expect("same-size sparse bitset set") else {
        panic!("setting an implicit position materializes the result");
    };
    assert_eq!(
        set.as_f64_slice().expect("double bitset output"),
        &[2.0, 0.0, 1.0, 7.0]
    );
}

#[test]
fn bitget_is_registered_and_dispatches() {
    assert!(runmat_builtins::builtin_function_by_name(BITGET_NAME).is_some());
    assert_eq!(
        crate::dispatcher::call_builtin(
            BITGET_NAME,
            &[Value::Int(IntValue::U8(0b1010)), Value::Num(2.0)],
        )
        .expect("runtime dispatch"),
        Value::Int(IntValue::U8(1))
    );
}

#[test]
fn bitset_preserves_all_native_integer_scalar_classes() {
    let cases = [
        (IntValue::I8(0), IntValue::I8(2)),
        (IntValue::I16(0), IntValue::I16(2)),
        (IntValue::I32(0), IntValue::I32(2)),
        (IntValue::I64(0), IntValue::I64(2)),
        (IntValue::U8(0), IntValue::U8(2)),
        (IntValue::U16(0), IntValue::U16(2)),
        (IntValue::U32(0), IntValue::U32(2)),
        (IntValue::U64(0), IntValue::U64(2)),
    ];
    for (input, expected) in cases {
        let actual =
            block_on(bitset_builtin(vec![Value::Int(input), Value::Num(2.0)])).expect("bitset");
        assert_eq!(actual, Value::Int(expected));
    }
}

#[test]
fn bitset_supports_explicit_set_clear_and_uint64_high_bits() {
    assert_eq!(
        block_on(bitset_builtin(vec![
            Value::Int(IntValue::U8(0b1111)),
            Value::Num(3.0),
            Value::Num(0.0),
        ]))
        .expect("clear bit"),
        Value::Int(IntValue::U8(0b1011))
    );
    assert_eq!(
        block_on(bitset_builtin(vec![
            Value::Int(IntValue::I8(0)),
            Value::Num(8.0),
            Value::Bool(true),
        ]))
        .expect("set signed high bit"),
        Value::Int(IntValue::I8(i8::MIN))
    );
    assert_eq!(
        block_on(bitset_builtin(vec![
            Value::Int(IntValue::U64(0)),
            Value::Num(64.0),
            Value::Num(0.5),
        ]))
        .expect("set uint64 high bit"),
        Value::Int(IntValue::U64(1_u64 << 63))
    );
}

#[test]
fn bitset_sparse_scalar_forms_preserve_or_materialize_from_zero_semantics() {
    let sparse =
        runmat_builtins::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 2.0])
            .expect("sparse");
    let Value::SparseTensor(cleared) = block_on(bitset_builtin(vec![
        Value::SparseTensor(sparse.clone()),
        Value::Num(1.0),
        Value::Num(0.0),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse clear") else {
        panic!("clearing preserves sparse storage");
    };
    assert_eq!(cleared.get(0, 0), Some(2.0));
    assert_eq!(cleared.get(1, 1), Some(2.0));

    let Value::Tensor(set) = block_on(bitset_builtin(vec![
        Value::SparseTensor(sparse),
        Value::Num(1.0),
        Value::Num(1.0),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse set") else {
        panic!("setting implicit zero materializes");
    };
    assert_eq!(
        set.as_f64_slice().expect("double bitset output"),
        &[3.0, 1.0, 1.0, 3.0]
    );
}

#[test]
fn bitset_accepts_same_size_inputs_and_scalar_values() {
    let input = Tensor::new_integer(IntegerStorage::U16(vec![0, 0]), vec![1, 2]).expect("input");
    let positions = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("positions");
    let values = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("values");
    let Value::Tensor(output) = block_on(bitset_builtin(vec![
        Value::Tensor(input),
        Value::Tensor(positions),
        Value::Tensor(values),
    ]))
    .expect("bitset") else {
        panic!("expected tensor result");
    };
    assert_eq!(output.shape, vec![1, 2]);
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U16(vec![1, 0]))
    );

    let input = Tensor::new_integer(IntegerStorage::U16(vec![0, 0]), vec![1, 2]).expect("input");
    let Value::Tensor(output) = block_on(bitset_builtin(vec![
        Value::Tensor(input),
        Value::Num(2.0),
        Value::Bool(true),
    ]))
    .expect("scalar-expanded bitset") else {
        panic!("expected tensor result");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U16(vec![2, 2]))
    );
}

#[test]
fn bit_position_and_count_operations_reject_general_singleton_expansion() {
    let input = Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).expect("input");
    let row = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("row operand");
    for error in [
        block_on(bitshift_builtin(vec![
            Value::Tensor(input.clone()),
            Value::Tensor(row.clone()),
        ]))
        .expect_err("bitshift row/column expansion must reject"),
        block_on(bitget_builtin(vec![
            Value::Tensor(input.clone()),
            Value::Tensor(row.clone()),
        ]))
        .expect_err("bitget row/column expansion must reject"),
        block_on(bitset_builtin(vec![
            Value::Tensor(input.clone()),
            Value::Tensor(row.clone()),
            Value::Bool(true),
        ]))
        .expect_err("bitset position expansion must reject"),
        block_on(bitset_builtin(vec![
            Value::Tensor(input),
            Value::Num(1.0),
            Value::Tensor(row),
        ]))
        .expect_err("bitset value expansion must reject"),
    ] {
        assert_eq!(error.identifier(), ERROR_SIZE_MISMATCH.identifier);
        assert!(error.message().contains("exactly the same size"));
    }
}

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
    let sparse =
        runmat_builtins::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 5.0])
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
        let Value::Tensor(output) = block_on(bitshift_builtin(vec![
            Value::GpuTensor(handle),
            Value::Tensor(shifts),
        ]))
        .expect("gathered GPU same-size bitshift") else {
            panic!("expected typed tensor result");
        };
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
    let Value::Tensor(output) = block_on(bitget_builtin(vec![
        Value::GpuTensor(handle),
        Value::Tensor(positions),
    ]))
    .expect("wgpu same-size bitget") else {
        panic!("expected typed tensor result");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U8(vec![1, 1]))
    );
}

#[test]
fn bitset_rejects_invalid_positions_nonfinite_values_and_dispatches() {
    for position in [0.0, -1.0, 9.0] {
        let error = block_on(bitset_builtin(vec![
            Value::Int(IntValue::U8(0)),
            Value::Num(position),
        ]))
        .expect_err("invalid position");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
    }
    let error = block_on(bitset_builtin(vec![
        Value::Int(IntValue::U8(0)),
        Value::Num(1.0),
        Value::Num(f64::NAN),
    ]))
    .expect_err("nonfinite bit value");
    assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);

    assert!(runmat_builtins::builtin_function_by_name(BITSET_NAME).is_some());
    assert_eq!(
        crate::dispatcher::call_builtin(
            BITSET_NAME,
            &[Value::Int(IntValue::U8(0)), Value::Num(1.0)],
        )
        .expect("runtime dispatch"),
        Value::Int(IntValue::U8(1))
    );
}

#[test]
fn bitxor_broadcasts_exact_uint64_storage_without_losing_high_bits() {
    let left = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
        .expect("left");
    let Value::Tensor(output) = block_on(bitxor_builtin(vec![
        Value::Tensor(left),
        Value::Int(IntValue::U64(1_u64 << 60)),
    ]))
    .expect("bitxor") else {
        panic!("expected tensor result");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![
            (1_u64 << 63) | (1_u64 << 60),
            u64::MAX ^ (1_u64 << 60),
        ]))
    );
}

#[test]
fn bitxor_follows_binary_integer_class_rules_and_is_registered() {
    assert!(runmat_builtins::builtin_function_by_name(BITXOR_NAME).is_some());
    assert_eq!(
        crate::dispatcher::call_builtin(
            BITXOR_NAME,
            &[
                Value::Int(IntValue::U8(0b1010)),
                Value::Int(IntValue::U8(0b0110))
            ],
        )
        .expect("runtime dispatch"),
        Value::Int(IntValue::U8(0b1100))
    );
    assert_eq!(
        block_on(bitxor_builtin(vec![
            Value::Int(IntValue::U16(0b1010)),
            Value::Num(6.0),
        ]))
        .expect("scalar double bitxor"),
        Value::Int(IntValue::U16(0b1100))
    );
    let error = block_on(bitxor_builtin(vec![
        Value::Int(IntValue::U8(1)),
        Value::Int(IntValue::U16(1)),
    ]))
    .expect_err("mixed integer classes must fail");
    assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn bitwise_native_integer_arrays_preserve_exact_64_bit_storage() {
    let left = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![1, 2])
        .expect("left");
    let right = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 60, u64::MAX]), vec![1, 2])
        .expect("right");
    let Value::Tensor(output) = block_on(bitand_builtin(vec![
        Value::Tensor(left),
        Value::Tensor(right),
    ]))
    .expect("bitand") else {
        panic!("expected tensor result");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![1_u64 << 60, 1_u64 << 63]))
    );
}

#[test]
fn bitshift_supports_positive_and_negative_counts() {
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Int(IntValue::U32(3)),
            Value::Num(2.0)
        ]))
        .expect("left shift"),
        Value::Int(IntValue::U32(12))
    );
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Int(IntValue::U32(8)),
            Value::Num(-1.0)
        ]))
        .expect("right shift"),
        Value::Int(IntValue::U32(4))
    );
}

#[test]
fn bitshift_preserves_integer_width() {
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Int(IntValue::U8(255)),
            Value::Num(1.0)
        ]))
        .expect("left shift"),
        Value::Int(IntValue::U8(254))
    );

    let tensor = Tensor::new_with_dtype(vec![255.0, 128.0], vec![1, 2], NumericDType::U8).unwrap();
    let out = block_on(bitshift_builtin(vec![
        Value::Tensor(tensor),
        Value::Num(1.0),
    ]))
    .expect("tensor shift");
    match out {
        Value::Tensor(t) => {
            assert_eq!(t.integer_storage(), Some(&IntegerStorage::U8(vec![254, 0])));
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn bitshift_sparse_double_preserves_sparse_implicit_zeros() {
    let sparse =
        runmat_builtins::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 5.0])
            .expect("sparse");
    let Value::SparseTensor(output) = block_on(bitshift_builtin(vec![
        Value::SparseTensor(sparse),
        Value::Num(1.0),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse bitshift") else {
        panic!("bitshift must preserve sparse storage when zero shifts to zero");
    };
    assert_eq!(output.shape(), vec![2, 2]);
    assert_eq!(output.get(0, 0), Some(6.0));
    assert_eq!(output.get(1, 1), Some(10.0));
    assert_eq!(output.nnz(), 2);
}

#[test]
fn bitshift_preserves_signed_arithmetic_and_64_bit_results() {
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Int(IntValue::I64(-4)),
            Value::Int(IntValue::I64(-1)),
        ]))
        .expect("signed arithmetic right shift"),
        Value::Int(IntValue::I64(-2))
    );
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Int(IntValue::U64(1_u64 << 63)),
            Value::Int(IntValue::I64(-63)),
        ]))
        .expect("uint64 right shift"),
        Value::Int(IntValue::U64(1))
    );
}

#[test]
fn bitwise_rejects_fractional_double() {
    let err = block_on(bitand_builtin(vec![Value::Num(1.5), Value::Num(1.0)]))
        .expect_err("fractional inputs should fail");
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn assumedtype_interprets_double_inputs_with_signed_bits_and_keeps_double_output() {
    let int8 = Value::String("int8".to_string());
    assert_eq!(
        block_on(bitand_builtin(vec![
            Value::Num(-5.0),
            Value::Num(6.0),
            int8.clone()
        ]))
        .expect("bitand assumedtype"),
        Value::Num(2.0)
    );
    assert_eq!(
        block_on(bitor_builtin(vec![
            Value::Num(-5.0),
            Value::Num(6.0),
            int8.clone()
        ]))
        .expect("bitor assumedtype"),
        Value::Num(-1.0)
    );
    assert_eq!(
        block_on(bitxor_builtin(vec![
            Value::Num(-5.0),
            Value::Num(6.0),
            int8.clone()
        ]))
        .expect("bitxor assumedtype"),
        Value::Num(-3.0)
    );
    assert_eq!(
        block_on(bitcmp_builtin(vec![Value::Num(-29.0), int8.clone()]))
            .expect("bitcmp assumedtype"),
        Value::Num(28.0)
    );
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Num(-4.0),
            Value::Num(-1.0),
            int8.clone()
        ]))
        .expect("bitshift assumedtype"),
        Value::Num(-2.0)
    );
    assert_eq!(
        block_on(bitget_builtin(vec![
            Value::Num(-29.0),
            Value::Num(8.0),
            int8.clone()
        ]))
        .expect("bitget assumedtype"),
        Value::Num(1.0)
    );
    assert_eq!(
        block_on(bitset_builtin(vec![
            Value::Num(0.0),
            Value::Num(8.0),
            Value::Num(1.0),
            int8,
        ]))
        .expect("bitset assumedtype"),
        Value::Num(-128.0)
    );
}

#[test]
fn assumedtype_enforces_integer_classes_and_numeric_ranges() {
    let mismatch = block_on(bitxor_builtin(vec![
        Value::Int(IntValue::I8(1)),
        Value::Int(IntValue::I8(2)),
        Value::String("uint8".to_string()),
    ]))
    .expect_err("mismatched assumedtype");
    assert_eq!(mismatch.identifier(), ERROR_INVALID_INPUT.identifier);

    let out_of_range = block_on(bitset_builtin(vec![
        Value::Num(128.0),
        Value::Num(1.0),
        Value::String("int8".to_string()),
    ]))
    .expect_err("out of range assumedtype value");
    assert_eq!(out_of_range.identifier(), ERROR_INVALID_INPUT.identifier);

    let uint64_limit = block_on(bitcmp_builtin(vec![
        Value::Num(2_f64.powi(64)),
        Value::String("uint64".to_string()),
    ]))
    .expect_err("uint64 assumedtype excludes 2^64");
    assert_eq!(uint64_limit.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn assumedtype_preserves_typed_integer_output_and_dispatches_all_arities() {
    assert_eq!(
        block_on(bitand_builtin(vec![
            Value::Int(IntValue::I8(-5)),
            Value::Int(IntValue::I8(6)),
            Value::String("int8".to_string()),
        ]))
        .expect("typed bitand assumedtype"),
        Value::Int(IntValue::I8(2))
    );
    assert_eq!(
        crate::dispatcher::call_builtin(
            BITSET_NAME,
            &[
                Value::Num(0.0),
                Value::Num(8.0),
                Value::String("int8".to_string()),
            ],
        )
        .expect("bitset third assumedtype dispatch"),
        Value::Num(-128.0)
    );
}

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
fn swapbytes_preserves_integer_scalar_classes() {
    assert_eq!(
        block_on(swapbytes_builtin(Value::Int(IntValue::U16(0x1234)))).expect("swapbytes"),
        Value::Int(IntValue::U16(0x3412))
    );
    assert_eq!(
        block_on(swapbytes_builtin(Value::Int(IntValue::I32(0x01020304)))).expect("swapbytes"),
        Value::Int(IntValue::I32(0x04030201))
    );
}

#[test]
fn swapbytes_preserves_tensor_dtype() {
    let tensor = Tensor::new_with_dtype(
        vec![0x1234_u16 as f64, 0x00ff_u16 as f64],
        vec![1, 2],
        NumericDType::U16,
    )
    .unwrap();
    let out = block_on(swapbytes_builtin(Value::Tensor(tensor))).expect("swapbytes");
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.numeric_dtype(), NumericDType::U16);
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U16(vec![0x3412, 0xff00]))
    );
}

#[test]
fn swapbytes_preserves_native_integer_storage_for_all_widths() {
    let input = Tensor::new_integer(
        IntegerStorage::I64(vec![0x0102_0304_0506_0708_i64, -2]),
        vec![1, 2],
    )
    .expect("input");
    let Value::Tensor(output) =
        block_on(swapbytes_builtin(Value::Tensor(input))).expect("swapbytes")
    else {
        panic!("expected tensor");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::I64(vec![
            0x0807_0605_0403_0201_i64,
            (-2_i64).swap_bytes(),
        ]))
    );

    let input = Tensor::new_integer(
        IntegerStorage::U64(vec![0x0102_0304_0506_0708_u64, u64::MAX]),
        vec![1, 2],
    )
    .expect("input");
    let Value::Tensor(output) =
        block_on(swapbytes_builtin(Value::Tensor(input))).expect("swapbytes")
    else {
        panic!("expected tensor");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![
            0x0807_0605_0403_0201_u64,
            u64::MAX
        ]))
    );
}

#[test]
fn swapbytes_reinterprets_floating_point_bytes() {
    let value = 1.25_f64;
    let out = block_on(swapbytes_builtin(Value::Num(value))).expect("swapbytes");
    assert_eq!(
        out,
        Value::Num(f64::from_bits(value.to_bits().swap_bytes()))
    );
}
