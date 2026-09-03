use super::super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;

#[test]
fn bitshift_single_and_logical_inputs_are_independently_gated() {
    let single = Tensor::from_f32(vec![3.0], vec![1, 1]).expect("single input");
    let logical = LogicalArray::new(vec![1], vec![1, 1]).expect("logical input");
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for (args, identifier) in [
        (
            vec![Value::Tensor(single.clone()), Value::Num(1.0)],
            "RunMat:compatibility:BitshiftSingleValueInputExtension",
        ),
        (
            vec![Value::Num(3.0), Value::Tensor(single)],
            "RunMat:compatibility:BitshiftSingleCountInputExtension",
        ),
        (
            vec![Value::LogicalArray(logical.clone()), Value::Num(1.0)],
            "RunMat:compatibility:BitshiftLogicalValueInputExtension",
        ),
        (
            vec![Value::Num(3.0), Value::LogicalArray(logical)],
            "RunMat:compatibility:BitshiftLogicalCountInputExtension",
        ),
    ] {
        let error =
            block_on(bitshift_builtin(args)).expect_err("MATLAB mode rejects undocumented input");
        assert_eq!(error.identifier(), Some(identifier));
    }
}

#[test]
fn bitshift_malformed_arity_precedes_compatibility_gates() {
    let single = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single input");
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(bitshift_builtin(vec![Value::Tensor(single)]))
        .expect_err("malformed arity must retain the ordinary invalid-input path");
    assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn bitshift_rejects_unsupported_resident_forms_before_provider_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

    let signed_value = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX - 74,
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(
        runmat_accelerate_api::NumericElementType::I32,
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    let error = block_on(bitshift_builtin(vec![
        Value::GpuTensor(signed_value),
        Value::Num(1.0),
    ]))
    .expect_err("MATLAB mode rejects signed A before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:BitshiftGpuUndocumentedInputExtension")
    );

    let unsigned_value = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX - 75,
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(
        runmat_accelerate_api::NumericElementType::U8,
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    let error = block_on(bitshift_builtin(vec![
        Value::GpuTensor(unsigned_value.clone()),
        Value::Int(IntValue::U64(1)),
    ]))
    .expect_err("MATLAB mode rejects 64-bit k before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:BitshiftGpuUndocumentedInputExtension")
    );

    let double_value = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX - 76,
        descriptor: Default::default(),
    };
    let error = block_on(bitshift_builtin(vec![
        Value::GpuTensor(double_value),
        Value::Num(1.0),
    ]))
    .expect_err("MATLAB mode requires at least one integer-array input before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:BitshiftGpuUndocumentedInputExtension")
    );

    let error = block_on(bitshift_builtin(vec![
        Value::GpuTensor(unsigned_value),
        Value::Num(1.0),
        Value::from("uint8"),
    ]))
    .expect_err("MATLAB mode rejects resident assumedtype before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:BitshiftGpuAssumedTypeExtension")
    );
}

#[test]
fn bitshift_documented_gpu_domain_restores_exact_resident_output() {
    test_support::with_test_provider(|provider| {
        let cases = [
            (
                IntegerStorage::U8(vec![0x81, 0x80]),
                IntegerStorage::U8(vec![0x02, 0x40]),
                runmat_accelerate_api::IntegerElementType::U8,
            ),
            (
                IntegerStorage::U16(vec![0x8001, 0x8000]),
                IntegerStorage::U16(vec![0x0002, 0x4000]),
                runmat_accelerate_api::IntegerElementType::U16,
            ),
            (
                IntegerStorage::U32(vec![0x8000_0001, 0x8000_0000]),
                IntegerStorage::U32(vec![0x0000_0002, 0x4000_0000]),
                runmat_accelerate_api::IntegerElementType::U32,
            ),
        ];
        for (values, expected, element_type) in cases {
            let values = Tensor::new_integer(values, vec![1, 2]).expect("documented values");
            let counts = Tensor::new_integer(IntegerStorage::I32(vec![1, -1]), vec![1, 2])
                .expect("documented counts");
            let values = gpu_helpers::upload_tensor(provider, &values).expect("upload values");
            let counts = gpu_helpers::upload_tensor(provider, &counts).expect("upload counts");
            let result = block_on(bitshift_builtin(vec![
                Value::GpuTensor(values),
                Value::GpuTensor(counts),
            ]))
            .expect("documented GPU bitshift");
            let Value::GpuTensor(handle) = &result else {
                panic!("documented GPU bitshift must preserve residency");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(handle),
                Some(element_type)
            );
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.integer_storage(), Some(&expected));
        }

        let counts = Tensor::new_integer(IntegerStorage::I8(vec![1, -1]), vec![1, 2])
            .expect("resident integer counts");
        let counts = gpu_helpers::upload_tensor(provider, &counts).expect("upload counts");
        let values = Tensor::new(vec![3.0, 8.0], vec![1, 2]).expect("host double values");
        let result = block_on(bitshift_builtin(vec![
            Value::Tensor(values),
            Value::GpuTensor(counts),
        ]))
        .expect("documented double A and integer GPU k");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather double result");
        assert_eq!(gathered.as_f64_slice(), Some(&[6.0, 4.0][..]));
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn bitshift_documented_gpu_domain_round_trips_through_wgpu() {
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
            IntegerStorage::U8(vec![0x81, 0x80]),
            IntegerStorage::U8(vec![0x02, 0x40]),
        ),
        (
            IntegerStorage::U16(vec![0x8001, 0x8000]),
            IntegerStorage::U16(vec![0x0002, 0x4000]),
        ),
        (
            IntegerStorage::U32(vec![0x8000_0001, 0x8000_0000]),
            IntegerStorage::U32(vec![0x0000_0002, 0x4000_0000]),
        ),
    ];
    for (values, expected) in cases {
        let values = Tensor::new_integer(values, vec![1, 2]).expect("WGPU values");
        let counts = Tensor::new_integer(IntegerStorage::I32(vec![1, -1]), vec![1, 2])
            .expect("documented counts");
        let values = gpu_helpers::upload_tensor(provider, &values).expect("upload values");
        let counts = gpu_helpers::upload_tensor(provider, &counts).expect("upload counts");
        let result = block_on(bitshift_builtin(vec![
            Value::GpuTensor(values),
            Value::GpuTensor(counts),
        ]))
        .expect("WGPU bitshift");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather WGPU result");
        assert_eq!(gathered.integer_storage(), Some(&expected));
    }
}

#[test]
fn bitshift_accepts_every_integer_count_class_without_changing_a_class() {
    for count in [
        IntValue::I8(1),
        IntValue::I16(1),
        IntValue::I32(1),
        IntValue::I64(1),
        IntValue::U8(1),
        IntValue::U16(1),
        IntValue::U32(1),
        IntValue::U64(1),
    ] {
        assert_eq!(
            block_on(bitshift_builtin(vec![
                Value::Int(IntValue::U16(3)),
                Value::Int(count),
            ]))
            .expect("documented integer count"),
            Value::Int(IntValue::U16(6))
        );
    }
}
