use super::super::*;
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
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(
        runmat_accelerate_api::NumericElementType::I32,
        runmat_accelerate_api::GpuTensorStorage::Real,
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
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(
        runmat_accelerate_api::NumericElementType::U8,
        runmat_accelerate_api::GpuTensorStorage::Real,
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
